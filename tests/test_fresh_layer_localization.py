"""Exercise actual layer hooks and reject incomplete or contradictory causal proofs."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from sas_memory.kv_readout import QuestionKVReadout, question_kv_mask
from scripts.analyze_fresh_layer_localization import audit_query, audit_first
from scripts.copy_readout_runtime import SelectionWeights
from scripts.fresh_layer_controls import (LAYERS, LayerReadout, Attention23Input,
    audit_receipts, audit_probabilities, input_scope, sources, first_token)
from scripts.run_write_locality_load_screen import hook_state


class Attention(torch.nn.Module):
    def forward(self, hidden_states, attention_mask):
        self.received = attention_mask
        return hidden_states, torch.softmax(attention_mask.float(), -1)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation='eager',
            layer_types=(['linear_attention'] * 3 + ['full_attention']) * 6)
        self.embedding = torch.nn.Embedding(10, 2)
        self.model = torch.nn.Module(); self.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(24)])
        for i in LAYERS:
            self.model.layers[i].self_attn = Attention()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, past_key_values=None, attention_mask=None, position_ids=None):
        past = past_key_values.get_seq_length() if past_key_values else 0
        self.native = question_kv_mask([True] * 4, past, input_ids.shape[1], 1, torch.float32, 'cpu')
        mask = self.native if attention_mask is None else attention_mask['full_attention']
        hidden = self.embedding(input_ids)
        result = []
        for i in LAYERS:
            _, probabilities = self.model.layers[i].self_attn(hidden_states=hidden, attention_mask=mask)
            result.append(probabilities)
        return result


class LayerHookTests(unittest.TestCase):
    def test_selected_masks_only_question_rows_and_cached_decode(self):
        model = Model(); hooks = hook_state(model); keep = [True, False, True, False]
        with LayerReadout(model, keep, [7, 23]) as operator:
            with SelectionWeights(model, {'original': keep}, 1) as observer:
                full = model(input_ids=torch.arange(7)[None])
                native = model.native
                for i in LAYERS:
                    actual = model.model.layers[i].self_attn.received
                    self.assertIs(actual, operator.mask if i in (7, 23) else native)
                    self.assertTrue(torch.equal(actual[..., :4, :], native[..., :4, :]))
                cached = model(input_ids=torch.tensor([[7]]),
                    past_key_values=SimpleNamespace(get_seq_length=lambda: 7))
            audit_receipts(operator.calls, operator.receipts, [7, 23], 1)
            audit_probabilities(observer.stats['original'], [7, 23], operator.calls)
            for i, p, c in zip(LAYERS, full, cached):
                self.assertEqual(p[..., 4:, 1].max().item() == 0, i in (7, 23))
                self.assertEqual(c[..., 1].max().item() == 0, i in (7, 23))
        self.assertEqual(hook_state(model), hooks)

    def test_global_and_all_layer_masks_agree_and_all_history_is_identity(self):
        model = Model(); ids = torch.arange(7)[None]; keep = [True, False, True, False]
        baseline = model(input_ids=ids)
        with LayerReadout(model, [True] * 4, LAYERS): all_history = model(input_ids=ids)
        with LayerReadout(model, keep, LAYERS): actual = model(input_ids=ids)
        with QuestionKVReadout(model, keep): original = model(input_ids=ids)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, original)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(all_history, baseline)))

    def test_capture_full_question_once_and_cleanup_on_exception(self):
        model = Model(); hooks = hook_state(model)
        with Attention23Input(model, 7) as capture:
            model(input_ids=torch.arange(4)[None]); model(input_ids=torch.arange(7)[None])
            model(input_ids=torch.tensor([[7]]), past_key_values=SimpleNamespace(get_seq_length=lambda: 7))
        self.assertEqual(capture.calls, 1)
        self.assertTrue(torch.equal(capture.hidden, model.embedding(torch.arange(7)[None])))
        with self.assertRaisesRegex(RuntimeError, 'test exception'):
            with LayerReadout(model, [True, False], [23]): raise RuntimeError('test exception')
        self.assertEqual(hook_state(model), hooks)

    def test_invalid_scope_and_custom_global_inputs_are_rejected(self):
        model = Model(); hooks = hook_state(model)
        for keep, selected in (([], [3]), ([True, 0], [3]), ([True, False], [3, 3]), ([True, False], [2])):
            with self.subTest(keep=keep, selected=selected), self.assertRaises(ValueError):
                LayerReadout(model, keep, selected)
        with self.assertRaises(ValueError):
            with LayerReadout(model, [True, False], [23]):
                model(input_ids=torch.tensor([[1]]), position_ids=torch.tensor([[1]]))
        self.assertEqual(hook_state(model), hooks)


class LayerAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = input_scope()

    @staticmethod
    def receipts(layers, prefill=1, cached=1):
        return {i: {'prefill_calls': prefill, 'cached_calls': cached,
            'overridden_calls': prefill + cached if i in layers else 0,
            'untouched_calls': 0 if i in layers else prefill + cached, 'mask_identity_exact': True} for i in LAYERS}

    def fixture(self, index=0):
        entry = self.scope['plan']['queries'][index]; q = self.scope['queries'][entry['id']]
        first = {'argmax_id': q['answer_ids'][0], 'argmax_text': 'test', 'gold_id': q['answer_ids'][0],
            'gold_logit': 3., 'non_gold_id': q['answer_ids'][0] + 1, 'non_gold_logit': 2.,
            'gold_margin': 1., 'logits_sha256': 'd' * 64}
        ref = {'id': q['id'], 'prefix_sha256': 'a' * 64, 'full_question_cache_sha256': 'b' * 64,
            'attention23_input_sha256': 'c' * 64, 'first_token': deepcopy(first)}
        rows, proofs = [], []
        for label, layers in self.scope['plan']['conditions'].items():
            all_layers = label == 'all_layers'
            row = {'model': entry['model'], 'id': q['id'], 'condition': label, 'source_regime': entry['regime'],
                'direction': entry['direction'], 'selected_layers': layers, 'token_policy': 'question', 'prefill': 'full',
                'arithmetic_profile': self.scope['plan']['arithmetic_profile'], 'registration_sha256': entry['registration_sha256'],
                'mask_registration_sha256': entry['mask_registration_sha256'], 'keep_bits_sha256': entry['keep_bits_sha256'],
                **entry['expected_intervened' if all_layers else 'expected_full'], 'first_token': deepcopy(first)}
            if all_layers: row['first_token']['logits_sha256'] = '9' * 64
            rows.append(row)
            proofs.append({'id': q['id'], 'condition': label, 'selected_layers': layers,
                'measure_calls': {'prefill': 1, 'cached': 1}, 'measure_receipts': self.receipts(layers),
                'attention_stats': {i: {'prefill_calls': 1, 'cached_calls': 1, 'query_rows': 4,
                    'max_suppressed_probability': 0. if i in layers else .1,
                    'max_suppressed_mass': 0. if i in layers else .2, 'max_kept_record_mass': .3} for i in LAYERS},
                'proof_calls': {'prefill': 2, 'cached': 0}, 'proof_receipts': self.receipts(layers, 2, 0),
                'complete_prefix_memory_exact': True, 'prefix_logits_exact': True,
                'prefix_sha256': 'a' * 64, 'masked_prefix_sha256': 'a' * 64, 'historical_kv_fields_exact': 12,
                'full_question_cache_sha256': ('e' if all_layers else 'b') * 64,
                'full_question_cache_equals_unmasked': not all_layers,
                'attention23_input_sha256': ('f' if all_layers else 'c') * 64,
                'attention23_input_exact_to_unmasked': not all_layers, 'hooks_restored': True})
        original = {'id': q['id'], 'readout': entry['expected_intervened'], 'historical_kv_fields_exact': 12,
            'calls': {'prefill': 3, 'cached': 1}, 'full_question_cache_sha256': 'e' * 64,
            'attention23_input_sha256': 'f' * 64, 'logits_sha256': '9' * 64}
        all_history = {'id': q['id'], 'readout': entry['expected_full'], 'complete_cache_input_logits_exact': True,
            'calls': {'prefill': 3, 'cached': 1}, 'receipts': self.receipts(LAYERS, 3, 1)}
        equivalence = {'id': q['id'], 'cache_input_logits_readout_exact': True}
        return [self.scope, entry, rows, proofs, ref, original, all_history, equivalence, 'a' * 64]

    def test_registered_four_cases_all_layers_and_repair_retained(self):
        self.assertEqual(len(sources()), 117)
        self.assertEqual([q['direction'] for q in self.scope['plan']['queries']], ['harm', 'harm', 'repair', 'harm'])
        for i in range(4):
            result = audit_query(*self.fixture(i))
            self.assertEqual(len(result['single_layer']), 6)
            self.assertEqual(len(result['conditional_leave_one_out']), 6)

    def test_no_condition_or_probability_observation_may_be_omitted(self):
        for missing in ('row', 'proof', 'receipt', 'probability'):
            args = self.fixture()
            if missing == 'row': args[2].pop()
            if missing == 'proof': args[3].pop()
            if missing == 'receipt': args[3][0]['measure_receipts'].pop(7)
            if missing == 'probability': args[3][0]['attention_stats'].pop(7)
            with self.subTest(missing=missing), self.assertRaises(ValueError): audit_query(*args)

    def test_exact_replay_mask_receipt_and_selected_suppression_required(self):
        for change in ('nll', 'global', 'all_history', 'receipt', 'probability'):
            args = self.fixture()
            if change == 'nll': args[2][0]['answer_nll'] += 1e-12
            if change == 'global': args[5]['logits_sha256'] = '8' * 64
            if change == 'all_history': args[6]['complete_cache_input_logits_exact'] = False
            if change == 'receipt': args[3][1]['measure_receipts'][3]['mask_identity_exact'] = False
            if change == 'probability': args[3][1]['attention_stats'][3]['max_suppressed_probability'] = .001
            with self.subTest(change=change), self.assertRaises(ValueError): audit_query(*args)

    def test_complete_prefix_and_only_final_layer_question_identity_required(self):
        for field, value in (('historical_kv_fields_exact', 11), ('masked_prefix_sha256', 'b' * 64),
                             ('complete_prefix_memory_exact', False), ('prefix_logits_exact', False),
                             ('full_question_cache_sha256', 'e' * 64), ('attention23_input_sha256', 'f' * 64)):
            args = self.fixture(); args[3][7][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): audit_query(*args)

    def test_unselected_probability_changes_do_not_invalidate_localization(self):
        args = self.fixture(); args[3][2]['attention_stats'][23]['max_suppressed_probability'] = .27
        audit_query(*args)

    def test_first_token_margin_and_identity_are_not_mean_answer_nll(self):
        runtime = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: str(ids[0])))
        q = {'answer_ids': [2]}
        actual = first_token(runtime, torch.tensor([[1., 4., 3.5]]), q)
        audit_first(actual, q); self.assertEqual(actual['gold_margin'], -.5)
        for key, value in (('argmax_id', 2), ('gold_margin', 0.), ('gold_id', 1), ('non_gold_logit', float('nan'))):
            bad = {**actual, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError): audit_first(bad, q)


if __name__ == '__main__':
    unittest.main()
