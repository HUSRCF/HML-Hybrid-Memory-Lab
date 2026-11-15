"""CPU controls for one copied Query and five causally evolving native Queries."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from sas_memory.runtime import fork_cache, cache_tensors, TextRuntime
from scripts.fresh_query_runtime import capture_question, ReaderContext
from scripts.query_layer_runtime import SingleLayerQueryExchange, measure_single_layer
from scripts.query_layer_data import read_plan, sources, self_control_ids, MODELS, LAYERS, DIRECTIONS, REFERENCES
from scripts.query_layer_metrics import summarize
from scripts.analyze_query_layer import audit_single_exchange
from scripts.post_encoding_data import build_registry, selected_contexts, context_queries
from scripts.fresh_query_data import read_plan as previous_plan
from scripts.post_encoding_runtime import CacheGuard
from tests.test_fresh_query import TinyRuntime
from tests.test_post_encoding import CharRuntime, ToyCache, prior_index


class SingleQueryRuntimeTests(unittest.TestCase):
    def setUp(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as native
        self.native, self.original = native, native.eager_attention_forward
        self.runtime, self.source = TinyRuntime(), ToyCache()
        self.bank, self.capture = capture_question(self.runtime, 'abc', self.source, 'x', ReaderContext('full', '110', 1))

    def tearDown(self):
        self.assertIs(self.native.eager_attention_forward, self.original)
        self.assertFalse(self.runtime.model._forward_pre_hooks)

    def test_each_actual_single_layer_self_copy_replays_output_cache_and_decode(self):
        natural, original_logits, _ = self.runtime.advance('Q:x?', self.source, 'abc')
        for layer in LAYERS:
            with SingleLayerQueryExchange(self.runtime.model, 3, self.bank.question_ids, self.bank, layer) as exchange:
                actual, logits, _ = self.runtime.advance('Q:x?', self.source, 'abc')
                self.runtime.model(input_ids=torch.tensor([[65]]), past_key_values=fork_cache(actual))
                report = exchange.report()
            torch.testing.assert_close(logits, original_logits, rtol=0, atol=0)
            for a, b in zip(cache_tensors(actual), cache_tensors(natural)):
                torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
            for index, stat in report['layers'].items():
                self.assertEqual(stat['actual_replacement'], index == layer)
                self.assertEqual(stat['original_query_object_preserved'], index != layer)
                self.assertEqual(stat['passthrough_calls'], 1)
                self.assertEqual(stat['native_query'], self.bank.record()['tensors'][str(index)])

    def test_early_cross_patch_leaves_later_actual_queries_unclamped(self):
        other = fork_cache(self.source)
        other.layers[0].recurrent_states[0].add_(1000)
        recipient, _ = capture_question(self.runtime, 'abc', other, 'x', ReaderContext('full', '110', 1))
        guard = CacheGuard(other)
        with SingleLayerQueryExchange(self.runtime.model, 3, self.bank.question_ids, self.bank, 3) as exchange:
            self.runtime.advance('Q:x?', other, 'abc')
            report = exchange.report()
        guard.check()
        self.assertEqual(report['layers'][3]['used_query'], self.bank.record()['tensors']['3'])
        self.assertNotEqual(report['layers'][7]['native_query'], recipient.record()['tensors']['7'])
        self.assertNotEqual(report['layers'][7]['native_query'], self.bank.record()['tensors']['7'])
        self.assertEqual(report['layers'][7]['native_query'], report['layers'][7]['used_query'])
        self.assertTrue(report['layers'][7]['original_query_object_preserved'])

    def test_last_layer_cross_patch_preserves_all_upstream_native_queries(self):
        other = fork_cache(self.source)
        other.layers[0].recurrent_states[0].add_(1000)
        recipient, _ = capture_question(self.runtime, 'abc', other, 'x', ReaderContext('full', '110', 1))
        with SingleLayerQueryExchange(self.runtime.model, 3, self.bank.question_ids, self.bank, 23) as exchange:
            self.runtime.advance('Q:x?', other, 'abc')
            report = exchange.report()
        for index in LAYERS:
            self.assertEqual(report['layers'][index]['native_query'], recipient.record()['tensors'][str(index)])

    def test_oracle_and_unregistered_layers_rejected_before_installing_hooks(self):
        oracle, _ = capture_question(self.runtime, 'abc', self.source, 'x', ReaderContext('oracle', '110', 1))
        for donor, layer in ((oracle, 3), (self.bank, 2), (self.bank, True)):
            with self.assertRaises(ValueError):
                SingleLayerQueryExchange(self.runtime.model, 3, self.bank.question_ids, donor, layer)
        with self.assertRaisesRegex(ValueError, 'Full-only'):
            measure_single_layer(self.runtime, {'id': 'ctx'}, self.source,
                                 {'prompt_ids_sha256': oracle.prompt_sha}, oracle, 3)

    def test_wrong_known_question_or_nonquery_mutation_fails_and_cleans_up(self):
        with self.assertRaises(ValueError):
            with SingleLayerQueryExchange(self.runtime.model, 3, self.bank.question_ids, self.bank, 3):
                self.runtime.advance('Q:y?', self.source, 'abc')
        def corrupt(module, query, key, value, mask, **kwargs):
            result = self.original(module, query, key, value, mask, **kwargs)
            key.add_(1)
            return result
        self.native.eager_attention_forward = corrupt
        try:
            with self.assertRaisesRegex(ValueError, 'non-Query'):
                with SingleLayerQueryExchange(self.runtime.model, 3, self.bank.question_ids, self.bank, 3):
                    self.runtime.advance('Q:x?', self.source, 'abc')
            self.assertIs(self.native.eager_attention_forward, corrupt)
        finally:
            self.native.eager_attention_forward = self.original

    def test_auditor_rejects_false_identity_extra_replacement_and_clamped_query(self):
        with SingleLayerQueryExchange(self.runtime.model, 3, self.bank.question_ids, self.bank, 3) as exchange:
            self.runtime.advance('Q:x?', self.source, 'abc')
            report = json.loads(json.dumps(exchange.report()))
        q = {'ledger_tokens': 3, 'input_ids': self.runtime.ids('abcQ:x?')}
        record = self.bank.record()
        reader = {'attention': {str(i): {'cached_calls': 1} for i in LAYERS}}
        audit_single_exchange(report, q, record, record, 3, reader)
        for kind in ('extra', 'identity', 'clamp', 'decode', 'layer', 'upstream'):
            bad = copy.deepcopy(report)
            if kind == 'extra': bad['layers']['7']['actual_replacement'] = True
            elif kind == 'identity': bad['layers']['7']['original_query_object_preserved'] = False
            elif kind == 'clamp': bad['layers']['7']['used_query']['sha256'] = '0' * 64
            elif kind == 'decode': bad['decode_queries_unmodified'] = False
            elif kind == 'layer': bad['selected_layer'] = 7
            else: bad['layers']['3']['native_query']['sha256'] = '0' * 64
            with self.assertRaises(ValueError, msg=kind):
                audit_single_exchange(bad, q, record, record, 3, reader)


class SingleQueryAccountingTests(unittest.TestCase):
    def setUp(self):
        self.cases, self.registry, self.queries = build_registry(CharRuntime(), previous_plan(), prior_index())

    def test_complete230source_closure(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(len(sources()), 230)
        for name in sources():
            path = root / name
            self.assertTrue(path.is_file(), name)
            if path.suffix != '.py': continue
            for node in ast.walk(ast.parse(path.read_text())):
                modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                           [n.name for n in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    target = module.replace('.', '/') + '.py'
                    if (root / target).is_file(): self.assertIn(target, sources(), f'{name}: {target}')

    def test_control_ids_and_full_smoke_counts_are_output_independent(self):
        for smoke in (False, True):
            contexts = selected_contexts(self.registry, smoke)
            ids = self_control_ids(contexts, self.queries, smoke)
            self.assertEqual(len(ids), 2 if smoke else 16)
            changed = copy.deepcopy(self.queries)
            for q in changed.values(): q['answer'] = 'changed'
            self.assertEqual(ids, self_control_ids(contexts, changed, smoke))
            count = sum(len(context_queries(c, self.queries, smoke)) for c in contexts if c['episode'] != 'before')
            self.assertEqual(count * 2 * 6 * 2, 288 if smoke else 4608)
            self.assertEqual(len(ids) * 2 * 6 * 2, 48 if smoke else 384)
        self.assertFalse(read_plan()['smoke']['accuracy_gate'])

    def synthetic(self):
        rows, refs = {}, {'full_readouts': {}}
        for model in MODELS:
            rows[model] = [{'id': key, 'condition': direction, 'layer': layer, 'greedy_exact': direction == 'HN', 'answer_nll': 1.}
                          for key, q in self.queries.items() if q['episode'] != 'before' for direction in DIRECTIONS for layer in LAYERS]
            refs['full_readouts'][model] = [{'id': key, 'condition': c, 'greedy_exact': False, 'answer_nll': 2.}
                          for key, q in self.queries.items() if q['episode'] != 'before' for c in REFERENCES]
        return rows, refs

    def test_all24new_and8archived_cells_preserve_baseline_failures(self):
        rows, refs = self.synthetic()
        report = summarize(rows, self.queries, refs)
        self.assertEqual((report['new_primary_readouts'], report['archived_references'], report['cell_count']), (4608, 1536, 32))
        self.assertEqual(report['cells']['native']['HN.L03']['joint6'], {'correct': 32, 'total': 32})
        self.assertEqual(report['cells']['native']['NN.all6']['greedy_exact'], 0)
        self.assertEqual(report['paired_contrasts']['native']['HN.L03_vs_HH.all6']['all']['gain'], 192)
        self.assertFalse(report['fresh_confirmation'])
        self.assertFalse(report['architecture_effectiveness_claim'])

    def test_missing_or_duplicate_rows_fail_complete_accounting(self):
        for duplicate in (False, True):
            rows, refs = self.synthetic()
            if duplicate: rows['native'][-1] = rows['native'][0]
            else: rows['native'].pop()
            with self.assertRaises(ValueError): summarize(rows, self.queries, refs)

    def test_real_runner_cpu_smoke_matches_predecessor_and_all_guard_counts(self):
        from scripts import run_fresh_query as previous, run_query_layer as runner
        class RunnerRuntime(TinyRuntime):
            @torch.inference_mode()
            def advance(self, text, cache=None, prefix_text=''):
                return super().advance(text, ToyCache(0) if cache is None else cache, prefix_text)
            probe = TextRuntime.probe
            def candidate_scores(self, prompt, query_cache, logits, candidates, batch=False):
                self.model(input_ids=torch.tensor([[65]]), past_key_values=fork_cache(query_cache))
                return {c: {'nll_sum': float(logits.abs().mean()), 'nll_mean': float(logits.abs().mean())} for c in candidates}
            def greedy_answer(self, cache, logits):
                self.model(input_ids=torch.tensor([[65]]), past_key_values=fork_cache(cache))
                return '123'
        def tail(runtime, prefix, prefix_text, text, tokens, mode):
            cache, logits, count = runtime.advance(text, prefix, prefix_text)
            for key, tensor in cache_tensors(cache):
                if key[1] in ('recurrent_states', 'conv_states'): tensor.add_(1)
            return cache, logits, {'unit_fixture': True}
        def state():
            return {'token_policy': 'none', 'completed_primary': 0, 'completed_same_source_replays': 0,
                    'completed_captures': 0, 'completed_reference_replays': 0, 'native_before_attach_checks': [],
                    'attached_none_checks': [], 'legacy_full_checks': [], 'native_restore_checks': [], 'prefixes': {}, 'sources': {}}
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            runtime = RunnerRuntime()
            _, registry, queries = build_registry(runtime, previous_plan(), prior_index())
            control = {'answer': '123', 'answer_nll': 1., 'greedy_exact': True, 'generated': '123', 'query_tokens': 1}
            ids = [str(i) for i in range(12)]
            anchors = {k: control for k in ids}
            refs = {'legacy_ids': ids, 'legacy_queries': {k: {'ledger_tokens': 3} for k in ids},
                    'legacy_full': {'native': anchors}, 'model_states': {'native': {}}}
            with tempfile.TemporaryDirectory() as folder, torch.inference_mode():
                oldpath, newpath = Path(folder) / 'old', Path(folder) / 'new'
                oldpath.mkdir(); newpath.mkdir()
                oldstate, newstate = state(), state()
                with patch.object(previous, 'measure', return_value=control), patch.object(previous, 'actual_prompt', return_value='control'), patch.object(previous, 'advance_tail', side_effect=tail):
                    previous.run_model(runtime, 'native', previous_plan(), registry, queries, refs, True, oldpath, oldstate, anchors, lambda: None)
                refs['source_caches'] = {'native': json.loads(json.dumps({k: oldstate[k] for k in ('prefixes', 'sources')}))}
                refs['full_readouts'] = {'native': [json.loads(v) for v in (oldpath / 'native.primary.jsonl').read_text().splitlines()
                                      if json.loads(v)['reader'] == 'full' and json.loads(v)['condition'] != 'before']}
                refs['full_donors'] = {'native': [json.loads(v) for v in (oldpath / 'native.donors.jsonl').read_text().splitlines() if json.loads(v)['reader'] == 'full']}
                with patch.object(runner, 'measure', return_value=control), patch.object(runner, 'actual_prompt', return_value='control'), patch.object(runner, 'advance_tail', side_effect=tail):
                    runner.run_model(runtime, 'native', read_plan(), registry, queries, refs, True, newpath, newstate, anchors, lambda: None)
                self.assertEqual([newstate[k] for k in ('completed_primary', 'completed_reference_replays', 'completed_same_source_replays', 'completed_captures')], [144, 48, 24, 24])
                self.assertEqual([v['source_guard_checks'] for v in newstate['prefixes'].values()], [250])
                control_ids = self_control_ids(selected_contexts(registry, True), queries, True)
                for cid, value in newstate['sources'].items():
                    count = sum(queries[k]['context_id'] == cid for k in control_ids)
                    self.assertEqual(value['caches']['N']['source_guard_checks'], 57 + 12 * count)
                    self.assertEqual(value['caches']['H']['source_guard_checks'], 56 + 12 * count)
                donors = [json.loads(v) for v in (newpath / 'native.donors.jsonl').read_text().splitlines()]
                for donor in donors:
                    self.assertEqual(donor['bank']['guard_checks'], (20 if donor['origin'] == 'N' else 19) + (12 if donor['id'] in control_ids else 0))
                self.assertTrue(newstate['hooks_restored'])
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
