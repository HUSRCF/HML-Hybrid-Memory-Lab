"""Reject selective replay, numerical drift and incomplete native-memory proofs."""

from copy import deepcopy
import unittest

from scripts.analyze_fresh_control_replay import audit_model
from scripts.copy_readout_runtime import readout
from scripts.run_fresh_control_replay import input_scope, trials, sources


class FreshControlReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = input_scope()

    def fixture(self, name='local_question'):
        from tests.test_copy_readout_sufficiency import ReadoutSufficiencyTests
        template, _, _ = ReadoutSufficiencyTests().proof_fixture()
        plan = self.scope['plan']
        selected = [q for q in plan['queries'] if q['model'] == name]
        originals = {q['id']: q for q in selected}
        rows, proofs = [], []
        for trial in trials(plan, name):
            entry = originals[trial['id']]
            for regime, actual in (('full', entry['expected_full']), (entry['regime'], entry['expected_intervened'])):
                rows.append({'model': name, **trial, 'regime': regime, 'token_policy': 'question', 'prefill': 'full',
                    'arithmetic_profile': plan['arithmetic_profile'], 'registration_sha256': entry['registration_sha256'],
                    'mask_registration_sha256': None if regime == 'full' else entry['mask_registration_sha256'],
                    'keep_bits_sha256': None if regime == 'full' else entry['keep_bits_sha256'],
                    'hook_calls': None if regime == 'full' else {'prefill': 1, 'cached': 1},
                    'exact_original_replay': True, **actual})
            proof = deepcopy(template)
            proof.update(id=entry['id'], unmasked_readout=readout(rows[-2]),
                unmasked_weights={entry['regime']: deepcopy(template['unmasked_weights']['oracle'])})
            masked = deepcopy(template['regimes']['oracle'])
            masked.update(readout=readout(rows[-1]), measure_hook_calls=rows[-1]['hook_calls'])
            proof['regimes'] = {entry['regime']: masked}
            proofs.append({**trial, 'proof': proof})
        prefix_checks = []
        for entry in selected:
            q = self.scope['queries'][entry['id']]
            prefix_checks.append({'id': q['id'], 'source_key': q['source_key'], 'ledger_tokens': q['ledger_tokens'],
                'complete_KV_recurrent_conv_bookkeeping_exact': True, 'last_prefix_logits_exact': True,
                'prefix_sha256': 'a' * 64})
        return rows, {'token_policy': 'question', 'completed_trials': 6, 'recorded_rows': 12,
            'proofs': proofs, 'prefix_checks': prefix_checks}

    def test_actual_all_four_changes_and_fixed_order_are_retained(self):
        plan = self.scope['plan']
        self.assertEqual([q['direction'] for q in plan['queries']], ['harm', 'harm', 'repair', 'harm'])
        self.assertEqual(len({q['id'] for q in plan['queries']}), 4)
        self.assertEqual(len(sources()), 113)
        self.assertTrue(all(not gate['passed'] for gate in self.scope['primary_gates'].values()))
        for name in plan['models']:
            expected = [q['id'] for q in plan['queries'] if q['model'] == name]
            self.assertEqual([t['id'] for t in trials(plan, name)], expected + expected[::-1] + expected)
            rows, state = self.fixture(name)
            counts = audit_model(self.scope, name, rows, state)
            self.assertEqual(counts['recorded_readouts'], 12)
            self.assertEqual(counts['actual_masked_attention_and_complete_prefix_proofs'], 6)

    def test_omitted_repair_or_repetition_cannot_be_accepted(self):
        rows, state = self.fixture('gdn_question_only')
        repair = next(q['id'] for q in self.scope['plan']['queries'] if q['direction'] == 'repair')
        for bad_rows in (rows[:-2], [r for r in rows if r['id'] != repair], rows[2:] + rows[:2]):
            with self.subTest(rows=len(bad_rows)), self.assertRaises(ValueError):
                audit_model(self.scope, 'gdn_question_only', bad_rows, state)
        bad = deepcopy(state); bad['proofs'].pop()
        with self.assertRaises(ValueError): audit_model(self.scope, 'gdn_question_only', rows, bad)

    def test_even_small_nll_drift_or_false_exact_flag_is_rejected(self):
        rows, state = self.fixture()
        for field, value in (('answer_nll', rows[1]['answer_nll'] + 1e-12), ('exact_original_replay', False),
                             ('generated', '784\n'), ('keep_bits_sha256', 'bad')):
            bad = deepcopy(rows); bad[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit_model(self.scope, 'local_question', bad, state)

    def test_memory_and_actual_attention_proofs_cannot_be_substituted(self):
        rows, state = self.fixture()
        for field, value in (('historical_kv_fields_exact', 11), ('masked_prefix_sha256', 'b' * 64),
                             ('complete_prefix_memory_exact', False), ('prefix_logits_exact', False)):
            bad = deepcopy(state); bad['proofs'][0]['proof']['regimes']['oracle'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit_model(self.scope, 'local_question', rows, bad)
        bad = deepcopy(state)
        bad['proofs'][0]['proof']['regimes']['oracle']['attention_stats'][3]['max_suppressed_probability'] = .001
        with self.assertRaises(ValueError): audit_model(self.scope, 'local_question', rows, bad)
        bad = deepcopy(state); bad['proofs'][0]['proof']['all_history_cache_logits_readout_exact'] = False
        with self.assertRaises(ValueError): audit_model(self.scope, 'local_question', rows, bad)

    def test_every_masked_prefix_must_match_actual_native_state(self):
        rows, state = self.fixture()
        for change in ('omit', 'hash', 'logits', 'conv'):
            bad = deepcopy(state)
            if change == 'omit': bad['prefix_checks'].pop()
            if change == 'hash': bad['prefix_checks'][0]['prefix_sha256'] = 'b' * 64
            if change == 'logits': bad['prefix_checks'][0]['last_prefix_logits_exact'] = False
            if change == 'conv': bad['prefix_checks'][0]['complete_KV_recurrent_conv_bookkeeping_exact'] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit_model(self.scope, 'local_question', rows, bad)


if __name__ == '__main__':
    unittest.main()
