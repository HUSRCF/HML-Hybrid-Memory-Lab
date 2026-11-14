from copy import deepcopy
import unittest

from sas_memory.tasks import Operation, execute
from sas_memory.training_data import data_fingerprint
from scripts.analyze_semantic_history_no_repeat import audit_rows
from scripts.overwrite_history_data import (CELLS, CONTRASTS, ROLES, build_data, evaluated_members,
    events_for, identity, prompt_for, read_plan, safety_samples, validate_data)
from scripts.overwrite_history_metrics import summarize
from scripts.run_semantic_overwrite_history import bound_engine


class OverwriteHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = read_plan()
        cls.data = build_data(cls.plan)

    def rows(self):
        return [{**{k: ex['case'][k] for k in ('cohort', 'base_stream_id', 'base_case_id', 'seed', 'active_accounts', 'value_version')},
                 'case_id': ex['case']['id'], 'variant': ex['variant'], 'role': ex['query']['role'],
                 'answer': ex['query']['answer'], 'generated': ex['query']['answer']+'\n',
                 'greedy_exact': True, 'answer_nll': .1} for ex in evaluated_members(self.data)]

    @staticmethod
    def mark_error(rows, seeds, cell='XX', versions=(0,), role=ROLES[1]):
        for row in rows:
            if row['seed'] in seeds and row['variant'] == cell and row['value_version'] in versions and row['role'] == role:
                row.update(greedy_exact=False, generated='000\n')

    def test_complete_executable_states_match_and_final_value_sensitivity_remains(self):
        self.assertEqual(validate_data(self.data, self.plan)['same_final_pairs'], 64)
        for case in self.data['cases']:
            truths = {c: execute([Operation(**e) for e in events_for(case, c)]) for c in CELLS}
            for novel, repeat in CONTRASTS.values():
                self.assertEqual(truths[novel], truths[repeat])
                self.assertEqual(events_for(case, novel)[160:], events_for(case, repeat)[160:])
            self.assertNotEqual(truths['XX']['current'][case['target_entity']], truths['XY']['current'][case['target_entity']])

    def test_compact_data_and_actual_renderer_change_only_the_earlier_value(self):
        self.assertEqual(len(self.data['cases']), 32)
        self.assertEqual(len(self.data['validation']), 384)
        self.assertFalse(any('case' in r for r in self.data['validation']))
        class Runtime:
            prompt_start = ''
            @staticmethod
            def query_suffix(question):
                return '\nQuestion: '+question
        for case in self.data['cases']:
            for novel, repeat in CONTRASTS.values():
                qn = case['variants'][novel]['queries'][1]
                qr = case['variants'][repeat]['queries'][1]
                self.assertEqual(qn, qr)
                ln, pn = prompt_for(Runtime(), case, novel, qn)
                lr, pr = prompt_for(Runtime(), case, repeat, qr)
                self.assertNotEqual(ln, lr)
                self.assertEqual(len(ln), len(lr))
                en, er = events_for(case, novel), events_for(case, repeat)
                self.assertEqual([i for i in range(162) if en[i] != er[i]], [3])

    def test_missing_selector_and_wrong_final_state_are_rejected(self):
        missing, wrong = deepcopy(self.data), deepcopy(self.data)
        missing['validation'].pop()
        wrong['cases'][0]['variants']['XX']['tail'][1]['value'] = wrong['cases'][0]['X']
        for altered in (missing, wrong):
            with self.subTest(), self.assertRaises(ValueError):
                validate_data(altered, self.plan)

    def test_private_engine_preserves_old_module_and_unchanged_execute_bytecode(self):
        import scripts.run_semantic_history_no_repeat as old
        original = old.PROTOCOL, old.evaluated_members, old.prompt_for, old.safety_samples
        engine = bound_engine()
        self.assertEqual(engine.execute.__code__.co_code, old.execute.__code__.co_code)
        self.assertIs(engine.execute.__globals__, vars(engine))
        self.assertIs(engine.prompt_for, prompt_for)
        self.assertEqual((old.PROTOCOL, old.evaluated_members, old.prompt_for, old.safety_samples), original)

    def test_smoke_and_samples_cover_both_cohorts_and_all_four_cells(self):
        smoke = evaluated_members(self.data, True)
        self.assertEqual(len(smoke), 24)
        self.assertEqual({r['variant'] for r in smoke}, set(CELLS))
        self.assertEqual({r['case']['value_version'] for r in smoke}, {0})
        self.assertEqual(len(safety_samples(smoke)), 8)
        self.assertEqual(len(safety_samples(evaluated_members(self.data))), 16)
        self.assertEqual({k[2] for k in safety_samples(smoke)}, {ROLES[1]})

    def test_primary_requires_both_cohorts_and_correct_preservation_guards(self):
        rows = self.rows()
        self.mark_error(rows, (97600, 97601))
        self.assertFalse(summarize(rows, self.data)['primary_signal']['present_in_both_cohorts'])
        self.mark_error(rows, (97700, 97701))
        self.assertTrue(summarize(rows, self.data)['primary_signal']['present_in_both_cohorts'])
        self.mark_error(rows, (97701,), role=ROLES[0])
        result = summarize(rows, self.data)
        self.assertFalse(result['primary_signal']['present_in_both_cohorts'])
        replica = result['cohort_results'][1]
        self.assertEqual(replica['raw']['losses'], 2)
        self.assertEqual(replica['guard_qualified']['losses'], 1)

    def test_value_siblings_cannot_replace_two_positive_streams(self):
        rows = self.rows()
        self.mark_error(rows, (97600, 97700), versions=(0, 1, 2, 3))
        result = summarize(rows, self.data)
        self.assertFalse(result['primary_signal']['present_in_both_cohorts'])
        self.assertEqual([len(c['positive_qualified_base_streams']) for c in result['cohort_results']], [1, 1])

    def test_rescues_cancel_losses_even_with_two_positive_streams(self):
        rows = self.rows()
        self.mark_error(rows, (97600, 97601, 97700, 97701))
        self.mark_error(rows, (97602, 97603, 97702, 97703), cell='XY')
        result = summarize(rows, self.data)
        self.assertFalse(result['primary_signal']['present_in_both_cohorts'])
        for cohort in result['cohort_results']:
            self.assertEqual((cohort['raw']['losses'], cohort['raw']['rescues'], cohort['raw']['signed_net_loss']), (2, 2, 0))
            self.assertEqual(len(cohort['positive_qualified_base_streams']), 2)

    def test_insufficient_novel_baseline_fails_despite_positive_signed_effect(self):
        rows = self.rows()
        self.mark_error(rows, (97600, 97601))
        for cell in ('XY', 'YY'):
            self.mark_error(rows, (97600, 97601, 97602), cell=cell, versions=(0, 1, 2, 3))
        cohort = summarize(rows, self.data)['cohort_results'][0]
        self.assertEqual(cohort['raw']['novel_correct'], 20)
        self.assertFalse(cohort['primary_gate_checks']['novel_baseline_at_least24_of32'])
        self.assertTrue(all(v for k,v in cohort['primary_gate_checks'].items() if k!='novel_baseline_at_least24_of32'))
        self.assertFalse(cohort['primary_passed'])

    def test_auditor_rejects_other_final_answer_and_missing_observation(self):
        ex = evaluated_members(self.data)[1]
        row = self.rows()[1]
        registration = {'question_ids': [1, 2, 3]}
        row.update(prefill='full', reader_policy='full', registration_sha256=data_fingerprint(registration), query_tokens=3)
        registrations = {tuple(identity(ex)): registration}
        audit_rows([row], [ex], registrations)
        other = ex['case']['variants']['XY']['queries'][1]['answer']
        with self.assertRaises(ValueError):
            audit_rows([{**row, 'answer': other, 'generated': other+'\n'}], [ex], registrations)
        with self.assertRaises(ValueError):
            audit_rows([], [ex], registrations)


if __name__ == '__main__':
    unittest.main()
