from copy import deepcopy
import unittest

from sas_memory.write_locality_data import CELLS, ROLES, build_data, read_protocol, validate_data
from sas_memory.write_locality_metrics import paired, summarize


class WriteLocalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = read_protocol()
        cls.data = build_data(cls.plan)

    def test_four_cells_preserve_protected_and_initial_truth(self):
        cases = {example['case']['id']: example['case'] for example in self.data['validation']}
        self.assertEqual(len(cases), 48)
        self.assertEqual(validate_data(self.data, self.plan)['examples'], 576)
        for case in cases.values():
            old_a = str(case['prefix'][0]['value'])
            old_b = str(case['prefix'][1]['value'])
            new_a = str(case['variants']['A1_C0']['tail'][0]['value'])
            self.assertNotEqual(old_a, new_a)
            for cell in CELLS:
                actual = {query['role']: query['answer'] for query in case['variants'][cell]['queries']}
                self.assertEqual(actual, {'target_current': new_a if cell[1] == '1' else old_a,
                                          'protected_current': old_b, 'target_initial': old_a})

    def test_data_audit_rejects_missing_labels_and_wrong_write_scope(self):
        missing = deepcopy(self.data)
        missing['validation'].pop()
        wrong_label = deepcopy(self.data)
        wrong_label['validation'][0]['query']['answer'] = '000'
        wrong_scope = deepcopy(self.data)
        case = wrong_scope['validation'][0]['case']
        case['variants']['A1_C1']['tail'][1]['target'] = case['protected_entity']
        for altered in (missing, wrong_label, wrong_scope):
            with self.subTest(), self.assertRaises(ValueError):
                validate_data(altered, self.plan)

    def test_signal_requires_two_base_streams_not_two_value_siblings(self):
        rows = []
        for example in self.data['validation']:
            case = example['case']
            rows.append({**{field: case[field] for field in ('seed', 'active_accounts', 'value_version', 'base_case_id', 'base_stream_id')},
                         'case_id': case['id'], 'variant': example['variant'], 'role': example['query']['role'],
                         'answer': example['query']['answer'], 'greedy_exact': True, 'answer_nll': 0.1})
        for row in rows:
            if (row['seed'] == 97000 and row['active_accounts'] == 8 and row['value_version'] in (0, 1) and
                row['variant'] == 'A1_C0' and row['role'] == ROLES[1]):
                row['greedy_exact'] = False
        self.assertFalse(summarize(rows)['screen_signal']['present'])
        for row in rows:
            if (row['seed'] == 97001 and row['active_accounts'] == 8 and row['value_version'] == 0 and
                row['variant'] == 'A1_C0' and row['role'] == ROLES[1]):
                row['greedy_exact'] = False
        report = summarize(rows)
        self.assertTrue(report['screen_signal']['present'])
        signal = report['screen_signal']['contrasts'][0]
        self.assertEqual(signal['regressed'], 3)
        self.assertEqual(len(signal['regressed_base_streams']), 2)
        self.assertEqual(report['examples'], 576)

    def test_paired_table_keeps_baseline_failures_and_rescues(self):
        rows = []
        for case_id, before, after in [('a', True, False), ('b', False, True)]:
            for cell, correct in [('A0_C0', before), ('A1_C0', after)]:
                rows.append({'case_id': case_id, 'base_stream_id': case_id, 'variant': cell,
                             'answer': '123', 'greedy_exact': correct, 'answer_nll': 0.1})
        result = paired(rows, 'A0_C0', 'A1_C0')
        self.assertEqual((result['units'], result['rescued'], result['regressed']), (2, 1, 1))
        self.assertEqual(result['secondary_baseline_correct_regression_rate']['baseline_correct'], 1)
        changed_answer = deepcopy(rows)
        changed_answer[1]['answer'] = '321'
        with self.assertRaises(ValueError):
            paired(changed_answer, 'A0_C0', 'A1_C0')


if __name__ == '__main__':
    unittest.main()
