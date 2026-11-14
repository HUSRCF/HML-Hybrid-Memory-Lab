from copy import deepcopy
import unittest

from sas_memory.training_data import data_fingerprint
from scripts.analyze_write_locality_adapter_transfer import audit_rows
from scripts.locality_transfer_data import MODELS, deduplicate_queries, read_plan
from scripts.locality_transfer_metrics import summarize


class LocalityAdapterTransferTests(unittest.TestCase):
    def test_dedup_keeps_distinct_source_worlds_and_every_binding(self):
        left = {'source_key': 'case/A0_C1', 'question_ids_sha256': 'same', 'answer': '111'}
        right = {**left, 'source_key': 'case/A1_C1', 'answer': '222'}
        candidates = [(left, {'pair_id': 'p1', 'side': 'left', 'label': 'current_A'}),
                      (right, {'pair_id': 'p1', 'side': 'right', 'label': 'current_A'}),
                      (right, {'pair_id': 'p2', 'side': 'right', 'label': 'current_A'})]
        physical, bindings = deduplicate_queries(candidates)
        self.assertEqual(len(physical), 2)
        self.assertEqual(len(bindings), 3)
        self.assertNotEqual(bindings[0]['physical_id'], bindings[1]['physical_id'])
        self.assertEqual(bindings[1]['physical_id'], bindings[2]['physical_id'])

    def test_repeated_query_conflicting_truth_and_duplicate_binding_rejected(self):
        q = {'source_key': 'same', 'question_ids_sha256': 'same', 'answer': '111'}
        b = {'pair_id': 'p', 'side': 'left', 'label': 'preservation'}
        with self.assertRaises(ValueError):
            deduplicate_queries([(q, b), ({**q, 'answer': '222'}, {**b, 'pair_id': 'other'})])
        with self.assertRaises(ValueError):
            deduplicate_queries([(q, b), (q, b)])

    def test_registered_final_checkpoints_are_fixed_and_paired(self):
        plan = read_plan()
        self.assertEqual([plan['models'][m]['token_policy'] for m in MODELS], ['all', 'question'])
        self.assertEqual({plan['models'][m]['final_step'] for m in MODELS}, {432})
        self.assertEqual(len({plan['models'][m]['initial_state_sha256'] for m in MODELS}), 1)

    def test_auditor_rejects_wrong_world_and_missing_queries(self):
        q = {'id': 'q', 'question_tokens': 26, 'answer': '111'}
        row = {'id': 'q', 'model': MODELS[0], 'prefill': 'full', 'reader_policy': 'full', 'token_policy': 'all',
               'registration_sha256': data_fingerprint(q), 'query_tokens': 26, 'answer': '111',
               'generated': '111\n', 'greedy_exact': True, 'answer_nll': .1}
        audit_rows([row], [q], MODELS[0], {'token_policy': 'all'})
        with self.assertRaises(ValueError):
            audit_rows([{**row, 'answer': '222', 'generated': '222\n'}], [q], MODELS[0], {'token_policy': 'all'})
        with self.assertRaises(ValueError):
            audit_rows([], [q], MODELS[0], {'token_policy': 'all'})

    def test_repairs_distinguish_update_regression_and_joint_update_failure(self):
        pairs, bindings, physical = [], [], []
        models = {name: [] for name in MODELS}
        for i in range(2):
            pair = {'id': f'p{i}', 'cohort': 'replica', 'stratum': 'B', 'case_id': f'case{i}',
                    'base_stream_id': f'stream{i}', 'source_keys': {'left': f'L{i}', 'right': f'R{i}'}}
            pairs.append(pair)
            for side in ('left', 'right'):
                for label in ('preservation', 'current_A'):
                    key = f'{i}/{side}/{label}'
                    native_ok = label == 'current_A' or (side == 'left' and i == 0)
                    physical.append({'id': key, 'native_full_readout': {'greedy_exact': native_ok, 'answer_nll': .1}})
                    bindings.append({'pair_id': pair['id'], 'side': side, 'label': label, 'physical_id': key})
                    for model in MODELS:
                        ok = not (model == MODELS[0] and side == 'right' and label == 'current_A')
                        models[model].append({'id': key, 'greedy_exact': ok, 'answer_nll': .05})
        registry = {'pairs': pairs, 'bindings': bindings, 'physical_queries': physical}
        result = summarize(registry, models)
        group = result['cohort_stratum_results'][0]
        self.assertEqual(group['native_full_update_regression_cases'], ['case0'])
        self.assertEqual(group['native_full_update_regression_repairs'][MODELS[1]]['all_roles_joint_repaired'], ['case0'])
        self.assertEqual(group['native_full_update_regression_repairs'][MODELS[0]]['preservation_repaired'], ['case0'])
        self.assertEqual(group['native_full_update_regression_repairs'][MODELS[0]]['all_roles_joint_repaired'], [])
        self.assertEqual(group['counts'][MODELS[0]]['right']['all_roles_joint']['correct'], 0)
        self.assertEqual(group['counts'][MODELS[1]]['right']['all_roles_joint']['correct'], 2)


if __name__ == '__main__':
    unittest.main()
