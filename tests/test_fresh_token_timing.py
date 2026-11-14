"""Timing interventions must preserve inactive rows and all registered causal controls."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
import torch

from tests.test_fresh_layer_localization import Model
from scripts.analyze_fresh_token_timing import audit_query
from scripts.fresh_timing_controls import (LAYERS, TIMINGS, TimingReadout, TimingWeights,
    active_positions, audit_operator, audit_weights, input_scope, sources)
from scripts.run_write_locality_load_screen import hook_state


class TimingHookTests(unittest.TestCase):
    def test_only_registered_question_rows_are_masked_and_future_rows_are_natural(self):
        keep = [True, False, True, False]; ids = torch.arange(9)[None]
        for timing, rows in zip(TIMINGS, ([4, 5, 6], [6], [4, 5])):
            model = Model(); baseline = model(input_ids=ids); hooks = hook_state(model)
            with TimingReadout(model, keep, [7, 23], 3, timing) as op:
                actual = model(input_ids=ids)
                for i, values, native in zip(LAYERS, actual, baseline):
                    changed = values[..., 1] == 0
                    if i in (7, 23):
                        self.assertTrue(all(changed[..., j].all().item() for j in rows))
                        inactive = [j for j in range(9) if j not in rows]
                        self.assertTrue(torch.equal(values[..., inactive, :], native[..., inactive, :]))
                    else:
                        self.assertTrue(torch.equal(values, native))
                self.assertEqual(op.batches[0]['active_positions'], rows)
            self.assertEqual(hook_state(model), hooks)

    def test_actual_selected_and_other_question_probabilities_are_observed_separately(self):
        keep = [True, False, True, False]
        for timing in TIMINGS:
            model = Model(); q = {'ledger_tokens': 4, 'question_tokens': 3}
            with TimingReadout(model, keep, [11], 3, timing) as op:
                with TimingWeights(model, keep, 1, 3, timing) as obs:
                    model(input_ids=torch.arange(7)[None])
            audit_operator(op.calls, op.receipts, op.batches, [11], q, timing, [7])
            audit_weights(obs.stats, [11], q, timing)
            self.assertEqual(obs.stats[11]['selected_max_suppressed_probability'], 0)
            self.assertGreater(obs.stats[23]['selected_max_suppressed_probability'], 0)
            if timing != 'complete_question':
                self.assertGreater(obs.stats[11]['other_max_suppressed_probability'], 0)

    def test_last_token_mask_preserves_preceding_question_and_cached_answer_row(self):
        model = Model(); keep = [True, False, True, False]
        preceding = model(input_ids=torch.arange(6)[None])
        cached = model(input_ids=torch.tensor([[7]]), past_key_values=SimpleNamespace(get_seq_length=lambda: 7))
        with TimingReadout(model, keep, LAYERS, 3, 'last_question_token') as op:
            actual = model(input_ids=torch.arange(6)[None])
            answer = model(input_ids=torch.tensor([[7]]), past_key_values=SimpleNamespace(get_seq_length=lambda: 7))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, preceding)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(answer, cached)))
        self.assertEqual([b['active_positions'] for b in op.batches], [[], []])

    def test_all_history_identity_invalid_timing_and_exception_cleanup(self):
        model = Model(); hooks = hook_state(model); ids = torch.arange(7)[None]
        expected = model(input_ids=ids)
        for timing in TIMINGS:
            with TimingReadout(model, [True] * 4, LAYERS, 3, timing):
                actual = model(input_ids=ids)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
        for timing, tokens in (('unknown', 3), ('last_question_token', 1)):
            with self.assertRaises(ValueError): TimingReadout(model, [True, False], [3], tokens, timing)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with TimingReadout(model, [True, False, True, False], [7], 3, 'complete_question'):
                with TimingWeights(model, [True, False, True, False], 1, 3, 'complete_question'):
                    raise RuntimeError('injected')
        self.assertEqual(hook_state(model), hooks)


class TimingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = input_scope()

    @staticmethod
    def receipt(layers, lengths):
        return {i: {'prefill_calls': len(lengths), 'cached_calls': 0,
            'overridden_calls': len(lengths) if i in layers else 0,
            'untouched_calls': 0 if i in layers else len(lengths),
            'mask_identity_exact': True, 'native_causal_mask_exact': True, 'inactive_rows_exact': True,
            'nonhistorical_columns_exact': True} for i in LAYERS}

    @staticmethod
    def batches(q, timing, lengths):
        h, n = q['ledger_tokens'], q['prompt_tokens']
        selected = [n - 1] if timing == 'last_question_token' else list(range(h, n - (timing == 'earlier_question_tokens')))
        return [{'past_length': 0, 'query_length': length, 'active_positions': [p for p in selected if p < length]} for length in lengths]

    def fixture(self, index=0):
        plan = self.scope['plan']; entry = plan['queries'][index]; q = self.scope['queries'][entry['id']]
        originals = {p['condition']: p for p in plan['prior_references'] if (p['model'], p['id']) == (entry['model'], q['id'])}
        full = originals['full']; ref = {'id': q['id'], **{k: full[k] for k in
            ('prefix_sha256', 'full_question_cache_sha256', 'attention23_input_sha256', 'first_token')},
            'preceding_question_cache_sha256': 'a' * 64, 'preceding_question_logits_sha256': 'b' * 64}
        rows, proofs, controls = [], [], []
        for timing in plan['timing_order']:
            controls.append({'id': q['id'], 'timing': timing, 'complete_cache_input_logits_exact': True,
                'calls': {'prefill': 1, 'cached': 0}, 'receipts': self.receipt(LAYERS, [q['prompt_tokens']]),
                'batches': self.batches(q, timing, [q['prompt_tokens']])})
            for label, layers in plan['conditions'].items():
                expected = originals[label] if timing == 'complete_question' else full
                rows.append({'model': entry['model'], 'id': q['id'], 'condition': label, 'timing': timing,
                    'source_regime': entry['regime'], 'direction': entry['direction'], 'selected_layers': layers,
                    'token_policy': 'question', 'prefill': 'full', 'arithmetic_profile': plan['arithmetic_profile'],
                    'registration_sha256': entry['registration_sha256'], 'prompt_ids_sha256': q['prompt_ids_sha256'],
                    'mask_registration_sha256': entry['mask_registration_sha256'], 'keep_bits_sha256': entry['keep_bits_sha256'],
                    'first_token': deepcopy(expected['first_token'])})
                lengths = [q['prompt_tokens'], q['ledger_tokens']] + ([q['prompt_tokens'] - 1] if timing == 'last_question_token' else [])
                active = q['question_tokens'] if timing == 'complete_question' else 1 if timing == 'last_question_token' else q['question_tokens'] - 1
                stats = {i: {'calls': 1, 'selected_query_rows': active, 'other_query_rows': q['question_tokens'] - active,
                    'selected_max_suppressed_probability': 0. if i in layers else .1,
                    'selected_max_suppressed_mass': 0. if i in layers else .2,
                    'other_max_suppressed_probability': .1 if timing != 'complete_question' else 0.,
                    'other_max_suppressed_mass': .2 if timing != 'complete_question' else 0.,
                    'max_kept_record_mass': .3} for i in LAYERS}
                proofs.append({'id': q['id'], 'condition': label, 'timing': timing, 'selected_layers': layers,
                    'calls': {'prefill': len(lengths), 'cached': 0}, 'receipts': self.receipt(layers, lengths),
                    'batches': self.batches(q, timing, lengths), 'attention_stats': stats,
                    'complete_prefix_memory_exact': True, 'prefix_logits_exact': True, 'prefix_sha256': full['prefix_sha256'],
                    'masked_prefix_sha256': full['prefix_sha256'], 'historical_kv_fields_exact': 12,
                    'full_question_cache_sha256': expected['full_question_cache_sha256'],
                    'full_question_cache_equals_unmasked': expected['full_question_cache_sha256'] == full['full_question_cache_sha256'],
                    'attention23_input_sha256': expected['attention23_input_sha256'],
                    'attention23_input_exact_to_unmasked': expected['attention23_input_sha256'] == full['attention23_input_sha256'],
                    'preceding_question_proof': {'complete_memory_exact': True, 'last_logits_exact': True,
                        'cache_sha256': 'a' * 64, 'logits_sha256': 'b' * 64, 'earlier_question_kv_fields_exact': 12}
                        if timing == 'last_question_token' else None, 'hooks_restored': True})
        return [self.scope, entry, rows, proofs, ref, controls, full['prefix_sha256']]

    def test_actual_registered_scope_keeps_all32references_all_layers_and_repair(self):
        plan = self.scope['plan']; self.assertEqual(len(sources()), 121)
        self.assertEqual(len(plan['prior_references']), 32)
        self.assertEqual([q['direction'] for q in plan['queries']], ['harm', 'harm', 'repair', 'harm'])
        for i in range(4):
            result = audit_query(*self.fixture(i)); self.assertEqual(len(result['conditions']), 8)

    def test_missing_case_condition_timing_or_probability_layer_is_rejected(self):
        for part in ('row', 'proof', 'control', 'probability'):
            args = self.fixture(2)
            if part == 'row': args[2].pop()
            if part == 'proof': args[3].pop()
            if part == 'control': args[5].pop()
            if part == 'probability': args[3][8]['attention_stats'].pop(19)
            with self.subTest(part=part), self.assertRaises(ValueError): audit_query(*args)

    def test_even_consistent_small_first_logit_drift_cannot_replace_prior_reference(self):
        args = self.fixture(); first = args[2][1]['first_token']
        first['gold_logit'] += .03125; first['gold_margin'] += .03125
        with self.assertRaises(ValueError): audit_query(*args)

    def test_absolute_row_timing_inactive_rows_and_selected_suppression_required(self):
        for field in ('active_position', 'inactive_mask', 'nonhistory', 'probability'):
            args = self.fixture(); proof = args[3][9]
            if field == 'active_position': proof['batches'][0]['active_positions'] = [self.scope['queries'][args[1]['id']]['ledger_tokens']]
            if field == 'inactive_mask': proof['receipts'][3]['inactive_rows_exact'] = False
            if field == 'nonhistory': proof['receipts'][3]['nonhistorical_columns_exact'] = False
            if field == 'probability': proof['attention_stats'][3]['selected_max_suppressed_probability'] = .001
            with self.subTest(field=field), self.assertRaises(ValueError): audit_query(*args)

    def test_complete_preceding_question_memory_and_only_final_layer_identity_required(self):
        for field in ('preceding', 'preceding_logits', 'prior_questionKV', 'final_input', 'final_cache', 'prefix'):
            args = self.fixture()
            if field == 'preceding': args[3][9]['preceding_question_proof']['complete_memory_exact'] = False
            if field == 'preceding_logits': args[3][9]['preceding_question_proof']['logits_sha256'] = 'c' * 64
            if field == 'prior_questionKV': args[3][9]['preceding_question_proof']['earlier_question_kv_fields_exact'] = 11
            if field == 'final_input': args[3][15]['attention23_input_sha256'] = 'd' * 64
            if field == 'final_cache': args[3][15]['full_question_cache_sha256'] = 'e' * 64
            if field == 'prefix': args[3][8]['masked_prefix_sha256'] = 'f' * 64
            with self.subTest(field=field), self.assertRaises(ValueError): audit_query(*args)

    def test_unselected_or_other_question_probabilities_may_change(self):
        args = self.fixture()
        args[3][10]['attention_stats'][23]['selected_max_suppressed_probability'] = .31
        args[3][10]['attention_stats'][7]['other_max_suppressed_probability'] = .29
        audit_query(*args)


if __name__ == '__main__':
    unittest.main()
