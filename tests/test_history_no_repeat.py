from copy import deepcopy
import unittest

from sas_memory.tasks import Operation, execute
from sas_memory.training_data import data_fingerprint
from scripts.analyze_semantic_history_no_repeat import audit_rows
from scripts.history_no_repeat_data import (CELLS, ROLES, build_data, evaluated_members, identity,
                                          read_plan, safety_samples, validate_data)
from scripts.history_no_repeat_metrics import summarize


class HistoryNoRepeatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = read_plan()
        cls.data = build_data(cls.plan)

    def rows(self):
        return [{**{key: ex['case'][key] for key in ('cohort', 'base_stream_id', 'base_case_id', 'seed', 'active_accounts', 'value_version')},
                 'case_id': ex['case']['id'], 'variant': ex['variant'], 'role': ex['query']['role'],
                 'answer': ex['query']['answer'], 'generated': ex['query']['answer']+'\n',
                 'greedy_exact': True, 'answer_nll': .1} for ex in self.data['validation']]

    @staticmethod
    def mark_error(rows, seeds, load=128, role=ROLES[0], versions=(0,), cell=CELLS[1]):
        for row in rows:
            if row['seed'] in seeds and row['active_accounts'] == load and row['value_version'] in versions and row['variant'] == cell and row['role'] == role:
                row.update(greedy_exact=False, generated='000\n')

    def test_initial_evidence_once_and_both_current_worlds_require_distant_history(self):
        self.assertEqual(validate_data(self.data, self.plan)['examples'], 384)
        cases = {ex['case']['id']: ex['case'] for ex in self.data['validation']}
        self.assertEqual(len(cases), 64)
        for case in cases.values():
            initial, middle = case['prefix'][0]['value'], case['prefix'][3]['value']
            self.assertNotEqual(initial, middle)
            for cell in CELLS:
                events = case['prefix'] + case['variants'][cell]['tail']
                truth = execute([Operation(**event) for event in events])
                self.assertEqual(sum(event['value'] == initial for event in events), 1)
                self.assertEqual(truth['initial'][case['target_entity']], initial)
                self.assertNotEqual(truth['current'][case['target_entity']], initial)
                self.assertEqual(events[-1], case['variants'][CELLS[0]]['tail'][-1])

    def test_data_audit_rejects_recency_shortcut_wrong_truth_and_filtered_pairs(self):
        shortcut, wrong, missing = deepcopy(self.data), deepcopy(self.data), deepcopy(self.data)
        case = shortcut['validation'][0]['case']
        case['variants'][CELLS[0]]['tail'][0]['value'] = case['prefix'][0]['value']
        wrong['validation'][1]['query']['answer'] = '000'
        missing['validation'].pop()
        for altered in (shortcut, wrong, missing):
            with self.subTest(), self.assertRaises(ValueError):
                validate_data(altered, self.plan)

    def test_smoke_and_history_samples_cover_both_cohorts_loads_and_worlds(self):
        smoke = evaluated_members(self.data, True)
        self.assertEqual(len(smoke), 24)
        self.assertEqual({(ex['case']['cohort'], ex['case']['active_accounts']) for ex in smoke},
                         {(cohort, load) for cohort in ('discovery', 'replica') for load in (32, 128)})
        self.assertEqual({ex['case']['value_version'] for ex in smoke}, {0})
        self.assertEqual(len(safety_samples(smoke)), 8)
        self.assertEqual(len(safety_samples(self.data['validation'])), 16)
        self.assertEqual({key[2] for key in safety_samples(smoke)}, {ROLES[0]})

    def test_primary_requires_both_cohorts_and_all_legitimate_current_values(self):
        rows = self.rows()
        self.mark_error(rows, (97400, 97401))
        self.assertFalse(summarize(rows, self.data)['primary_signal']['present_in_both_cohorts'])
        self.mark_error(rows, (97500, 97501))
        self.assertTrue(summarize(rows, self.data)['primary_signal']['present_in_both_cohorts'])
        self.mark_error(rows, (97501,), role=ROLES[2])
        report = summarize(rows, self.data)
        self.assertFalse(report['primary_signal']['present_in_both_cohorts'])
        group = next(g for g in report['cohort_load_results'] if g['cohort']=='replica' and g['active_accounts']==128)
        self.assertEqual(group['paired_roles'][ROLES[0]]['regressed'], 2)
        self.assertEqual(len(group['commit_qualified_regression_cases']), 1)
        self.assertEqual(len(group['unqualified_regression_cases']), 1)

    def test_four_value_failures_in_one_stream_do_not_satisfy_two_stream_gate(self):
        rows = self.rows()
        self.mark_error(rows, (97400, 97500), versions=(0, 1, 2, 3))
        report = summarize(rows, self.data)
        self.assertFalse(report['primary_signal']['present_in_both_cohorts'])
        self.assertEqual({len(v) for v in report['primary_signal']['cohort_qualified_base_streams'].values()}, {1})

    def test_secondary_load_cannot_replace_failed_primary(self):
        rows = self.rows()
        self.mark_error(rows, (97400, 97401, 97500, 97501), load=32)
        report = summarize(rows, self.data)
        self.assertTrue(all(g['descriptive_two_stream_signal'] for g in report['cohort_load_results'] if g['active_accounts']==32))
        self.assertFalse(report['primary_signal']['present_in_both_cohorts'])

    def test_low_baseline_history_is_flagged_with_all_failures_retained(self):
        rows = self.rows()
        self.mark_error(rows, (97400, 97401, 97402, 97500, 97501, 97502), versions=(0, 1, 2, 3), cell=CELLS[0])
        report = summarize(rows, self.data)
        self.assertTrue(all(report['primary_signal']['cohort_baseline_floor'].values()))
        self.assertEqual(report['examples'], 384)
        for group in report['cohort_load_results']:
            if group['primary']:
                self.assertEqual(group['paired_roles'][ROLES[0]]['reference_correct'], 4)
                self.assertEqual(group['paired_roles'][ROLES[0]]['rescued'], 12)

    def test_auditor_rejects_other_world_answer_and_missing_observation(self):
        ex = self.data['validation'][1]
        row = self.rows()[1]
        registration = {'question_ids': [1, 2, 3]}
        row.update(prefill='full', reader_policy='full', registration_sha256=data_fingerprint(registration), query_tokens=3)
        registrations = {tuple(identity(ex)): registration}
        audit_rows([row], [ex], registrations)
        other_answer = ex['case']['variants'][CELLS[1]]['queries'][1]['answer']
        with self.assertRaises(ValueError):
            audit_rows([{**row, 'answer': other_answer, 'generated': other_answer+'\n'}], [ex], registrations)
        with self.assertRaises(ValueError):
            audit_rows([], [ex], registrations)


if __name__ == '__main__':
    unittest.main()
