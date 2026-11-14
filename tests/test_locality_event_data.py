from copy import deepcopy
import unittest

from sas_memory.locality_replication import read_plan
from sas_memory.write_locality_data import build_data
from scripts.locality_event_data import (STRATA, cases_for_stratum, check_source_alignment,
                                         paired_queries, read_protocols)


class LocalityEventDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_data(read_plan())

    def test_all_registered_strata_preserve_stream_and_value_coverage(self):
        self.assertEqual(len(read_protocols()), 3)
        for stratum in STRATA:
            cases = cases_for_stratum(self.data, stratum)
            self.assertEqual(len(cases), 16)
            self.assertEqual({c['seed'] for c in cases}, {97100, 97101, 97102, 97103})
            self.assertEqual({c['value_version'] for c in cases}, {0, 1, 2, 3})
        partial = {'validation': [r for r in self.data['validation'] if r['case']['value_version'] != 3]}
        with self.assertRaises(ValueError):
            cases_for_stratum(partial, STRATA[0])

    def test_companions_keep_separate_legitimate_worlds(self):
        for stratum in STRATA:
            for case in cases_for_stratum(self.data, stratum):
                queries = paired_queries(case, stratum)
                self.assertEqual(queries['preservation']['left'], queries['preservation']['right'])
                a = queries['current_A']
                self.assertEqual(a['left']['answer'] != a['right']['answer'], stratum['axis'] == 'A')
                self.assertEqual('current_C' in queries, stratum['axis'] == 'C')
                if stratum['axis'] == 'C':
                    c = queries['current_C']
                    self.assertNotEqual(c['left']['answer'], c['right']['answer'])
                    self.assertNotEqual(c['left']['question'], a['left']['question'])
        case = deepcopy(cases_for_stratum(self.data, STRATA[0])[0])
        case['variants']['A1_C1']['tail'][0]['value'] = case['prefix'][0]['value']
        with self.assertRaises(ValueError):
            paired_queries(case, STRATA[0])

    def test_alignment_rejects_unrelated_source_or_query_changes(self):
        records = [{'token_start': i * 3, 'token_end': i * 3 + 3,
                    'value_tokens': [i * 3, i * 3 + 1, i * 3 + 2]} for i in range(162)]
        prefix = {'ledger_tokens': 486, 'records': records}
        left = list(range(486))
        right = left.copy()
        right[480] = 999
        result = check_source_alignment(left, right, prefix, STRATA[0])
        self.assertEqual(result['support_record'], 1)
        self.assertEqual(result['support_span'], [3, 6])
        self.assertEqual(result['changed_token_positions'], [480])
        with self.assertRaises(ValueError):
            check_source_alignment(left, right + [7], prefix, STRATA[0])
        right[4] = 998
        with self.assertRaises(ValueError):
            check_source_alignment(left, right, prefix, STRATA[0])
        with self.assertRaises(ValueError):
            check_source_alignment(left, left, prefix, STRATA[0])


if __name__ == '__main__':
    unittest.main()
