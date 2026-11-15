"""Truth, matched inputs and full two-model auditor checks without model loading."""
import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import natural_load_data as data
from scripts import analyze_natural_load as analysis
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.run_controller_trial import write_json
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash


class NaturalLoadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = data.read_plan(); cls.refs = data.reference_inputs()
        # Tokenizer-only feasibility must never instantiate a model or read weights.
        with patch('sas_memory.runtime.TextRuntime.__init__', side_effect=AssertionError('No model constructor')), \
             patch('transformers.Qwen3_5ForCausalLM.from_pretrained', side_effect=AssertionError('No model loader')), \
             patch('safetensors.torch.load_file', side_effect=AssertionError('No weight loader')):
            cls.values = data.build_registry(tokenizer_runtime(cls.plan['model_path']), cls.plan, cls.refs)
        cls.cases, cls.registry, cls.queries = cls.values

    def test_complete_fresh_grid_and_matched_tokens(self):
        self.assertEqual(data.select_seeds(self.plan,self.refs), list(range(76000,76008)))
        self.assertEqual((len(self.cases),len(self.registry['contexts']),len(self.queries)),(16,144,864))
        self.assertEqual({c['ledger_tokens'] for c in self.registry['contexts']},{351,1247,7519})
        self.assertEqual(len({q['prompt_ids_sha256'] for q in self.queries.values()}),864)
        self.assertEqual(len(data.selected_contexts(self.registry,True)),9)
        self.assertEqual(sum(len(data.context_queries(c,self.queries,True))*4 for c in data.selected_contexts(self.registry,True)),108)

    def test_executable_independence_update_and_support(self):
        for case in self.cases:
            self.assertEqual(len(case['prefix_events']),19)
            before = {r:data.truth(case,case['episodes'][0],r)[0] for r in data.ROLES}
            for episode in case['episodes'][1:]:
                tail = episode['tail']; count = episode['record_count']; load = episode['active_entities']
                self.assertEqual(len(tail),count)
                unrelated = next(e for e in case['episodes'] if e['id']==f'unrelated-{count}-load{load}')
                counts = {e['target']:0 for e in unrelated['tail']}
                for e in unrelated['tail']:counts[e['target']] += 1
                self.assertEqual(len(counts),load);self.assertEqual(set(counts.values()),{count//load})
                self.assertTrue(set(counts).isdisjoint({case['target'],case['protected']}))
                for role in data.ROLES:
                    answer,support = data.truth(case,episode,role)
                    self.assertEqual(answer,str((case['prefix_events']+tail)[support]['value']))
                    if episode['kind']=='related' and role=='target_current':self.assertNotEqual(answer,before[role])
                    else:self.assertEqual(answer,before[role])

    def test_bijective_renaming_preserves_every_operation(self):
        for original,permuted in zip(self.cases[::2],self.cases[1::2]):
            self.assertEqual(original['seed'],permuted['seed']); mapping = permuted['account_mapping']
            self.assertEqual(len(set(mapping.values())),26)
            for left,right in zip(original['episodes'],permuted['episodes']):
                for a,b in zip(original['prefix_events']+left['tail'],permuted['prefix_events']+right['tail']):
                    self.assertEqual({**a,'target':mapping[a['target']]},b)

    def test_oracle_support_and_question_boundaries(self):
        contexts = {c['id']:c for c in self.registry['contexts']}
        for q in self.queries.values():
            c = contexts[q['context_id']]; b = c['records'][q['support_event_index']]
            selected = {i for i,v in enumerate(q['history_keep_bits']) if v=='1'}
            self.assertEqual(selected,set(range(q['prelude_tokens'])) | set(range(b['start'],b['stop'])))
            self.assertEqual(q['answer_ids'].__len__(),3)
            self.assertTrue(q['input_ids'][q['ledger_tokens']:])

    def test_scope_and_all_prior_sources_unchanged(self):
        self.assertEqual(len(data.sources()),240);self.assertEqual(len(data.predecessor_sources()),234)
        for name in data.NEW_SOURCES:
            if not name.endswith('.py'):continue
            tree = ast.parse((data.ROOT/name).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith(('scripts.','sas_memory.')):
                    self.assertIn(node.module.replace('.','/')+'.py',data.sources())

    def test_summary_keeps_initial_failures_and_all_cells(self):
        rows = [{'id':q['id'],'reader':'full','greedy_exact':True,'generated':q['answer'],'answer_nll':.5}
            for q in self.queries.values()]
        baseline = next(q for q in self.queries.values() if q['episode']=='before')
        next(r for r in rows if r['id']==baseline['id']).update(greedy_exact=False,generated='0')
        result = analysis.summarize({'native':rows},self.queries)
        self.assertEqual(len(result['cells']),9);self.assertEqual(sum(c['questions'] for c in result['cells']),864)
        self.assertEqual(result['independent_streams'],8)
        gains = [c for c in result['per_stream_paired_contrasts'] if c['left_id']==baseline['id']]
        self.assertEqual(len(gains),8);self.assertTrue(all(c['gains']==1 for c in gains))

    def fixture(self, destination):
        """Synthetic two-model smoke: exercise every audit branch, no inference."""
        root = Path(destination); run = root/'run'; run.mkdir(); packet = root/'packet'; packet.mkdir()
        write_json(packet/'manifest.json',{'fixture':True})
        old = json.loads((Path(self.plan['source_run'])/'manifest.json').read_text())
        now = datetime.now(timezone.utc); digests = {n:streaming_hash(data.ROOT/n) for n in sorted(data.sources())}
        registered = {'created_utc':now.isoformat(),'source_sha256':digests}
        m = deepcopy(old)
        m.update(status='completed',started_utc=(now+timedelta(seconds=1)).isoformat(),completed_utc=(now+timedelta(seconds=2)).isoformat(),
            smoke=True,reverse_model_order=False,model_order=list(data.MODELS),completed_models=list(data.MODELS),
            packet=str(packet),packet_manifest_sha256=streaming_hash(packet/'manifest.json'),protocol_sha256=streaming_hash(data.PROTOCOL),
            source_sha256=digests,smoke_equivalence_gate=None,models={})
        for n in ('plan','precision_addendum','kernel_validation_addendum'):
            value = self.plan if n=='plan' else json.loads((Path(self.plan['source_run'])/(n+'.json')).read_text())
            write_json(run/(n+'.json'),value)
        (run/'installed_modeling_source.py').write_bytes(Path(self.plan['installed_modeling_source']).read_bytes())
        for name in digests:
            p = run/'sources'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((data.ROOT/name).read_bytes())
        contexts = data.selected_contexts(self.registry,True)
        template = next(iter(old['models']['native']['prefixes'].values()))
        def cache_record(tokens,checks):
            c = deepcopy(template);c['metadata']['original_metadata']['sequence_length']=tokens;c['source_guard_checks']=checks
            for key,tensor in c['tensors'].items():
                if "'keys'" in key or "'values'" in key:tensor['shape'][-2]=tokens
            return c
        before = contexts[0]; parent = cache_record(351,64)
        fresh = Path('/media/990Pro/Code/SAS/runs/memory_update/fresh-query-primary-20261115-01/native.primary.jsonl')
        proofs = {}
        for line in fresh.read_text().splitlines():
            r = json.loads(line);proofs.setdefault(r['reader'],r['proof'])
            if len(proofs)==2:break
        for name in data.MODELS:
            state = deepcopy(old['models'][name]);state.update(completed_primary=54,prefixes={before['prefix_key']:deepcopy(parent)},sources={})
            for c in contexts[1:]:
                t = next(iter(old['models'][name]['sources'].values()))['proofs']['normal_tail'];proof = deepcopy(t)
                proof.update(tail_tokens=c['tail_tokens'],historical_kv_sha256={k:v['sha256'] for k,v in parent['tensors'].items() if "'keys'" in k or "'values'" in k},
                    write_keep_sha256=data_fingerprint('1'*351+'0'*c['tail_tokens']))
                for stat in proof['gates']['layers'].values():stat.update(rows=c['tail_tokens'],unheld_rows=c['tail_tokens'])
                state['sources'][c['id']] = {'cache':cache_record(c['ledger_tokens'],8),'tail_proof':proof,
                    'all_sources_built_before_any_question':True,'pairwise_storage_isolated':True}
            rows = []
            for c in contexts:
                for q in data.context_queries(c,self.queries,True):
                    for reader in data.READERS:
                        proof=deepcopy(proofs[reader]);proof['reader_keep_sha256']=data_fingerprint(q['history_keep_bits'])
                        rows.append({'id':q['id'],'model':name,'reader':reader,'answer':q['answer'],'answer_nll':.5,'generated':q['answer'],
                            'greedy_exact':True,'query_tokens':len(q['input_ids'])-q['ledger_tokens'],'registration_sha256':data_fingerprint(q),
                            'source_key':q['source_key'],'prompt_ids_sha256':q['prompt_ids_sha256'],'prefill':'cached',
                            'measurement_origin':'normally_encoded_without_intervention','proof':proof})
            raw=run/f'{name}.jsonl';raw.write_text(''.join(json.dumps(r)+'\n' for r in rows));state['results_sha256']=streaming_hash(raw)
            m['models'][name]=state
        write_json(run/'manifest.json',m)
        return run,packet,(registered,self.cases,self.registry,self.queries,self.refs)

    def test_complete_two_model_audit_without_native_inference(self):
        with tempfile.TemporaryDirectory() as d:
            run,packet,registered = self.fixture(d)
            with patch.object(analysis,'DESTINATION',packet),patch.object(analysis,'registered_inputs',return_value=registered), \
                 patch('sas_memory.runtime.TextRuntime.__init__',side_effect=AssertionError('No inference')):
                result=analysis.audit_run(run,allow_smoke=True)
            self.assertEqual(result['primary_readouts_audited'],108);self.assertEqual(len(result['metrics']['cells']),36)
            self.assertEqual(result['metrics']['independent_streams'],1)

    def test_partial_grid_rejected_even_with_updated_raw_hash(self):
        with tempfile.TemporaryDirectory() as d:
            run,packet,registered = self.fixture(d);m=json.loads((run/'manifest.json').read_text())
            raw=run/'gdn_all_tokens.jsonl';raw.write_text('\n'.join(raw.read_text().splitlines()[:-1])+'\n')
            m['models']['gdn_all_tokens']['results_sha256']=streaming_hash(raw);write_json(run/'manifest.json',m)
            with patch.object(analysis,'DESTINATION',packet),patch.object(analysis,'registered_inputs',return_value=registered):
                with self.assertRaisesRegex(ValueError,'unfiltered'):analysis.audit_run(run,True)

    def test_cache_guard_count_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            run,packet,registered = self.fixture(d);m=json.loads((run/'manifest.json').read_text())
            next(iter(m['models']['gdn_all_tokens']['sources'].values()))['cache']['source_guard_checks']-=1
            write_json(run/'manifest.json',m)
            with patch.object(analysis,'DESTINATION',packet),patch.object(analysis,'registered_inputs',return_value=registered):
                with self.assertRaisesRegex(ValueError,'guarded48'):analysis.audit_run(run,True)


if __name__ == '__main__':unittest.main()
