"""Complete schedule factorial and meaningful matched-timing contrast controls."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.matched_shared_timing_common import read_plan
from scripts.trained_shared_timing_data import DATA_SOURCES,ORDER,SHARED_MODELS,sources
from scripts.trained_shared_timing_metrics import summarize,QUESTION


class TrainedTimingTests(unittest.TestCase):
    def test_full_registered_factorial_and_preserved_baselines(self):
        p=read_plan();c=p['confirmation']
        self.assertEqual(c['evaluation_order'],list(ORDER));self.assertEqual(len(ORDER),16)
        self.assertEqual(SHARED_MODELS,('shared','shared_question','shared_ledger'))
        for name in SHARED_MODELS:
            self.assertEqual({a['token_policy'] for a in c['evaluation_arms'].values() if a['model']==name},{'all','question','ledger'})
        self.assertEqual(c['counts']['primary_physical_full_answer_readouts'],5376)
        self.assertEqual(c['smoke_physical_readouts_per_order'],288)

    def test_complete_new_runtime_import_closure_preserves155_training_files(self):
        self.assertEqual((len(DATA_SOURCES),len(sources())),(157,160))
        for name in sources():
            if not name.endswith('.py'):continue
            for node in ast.walk(ast.parse(Path(name).read_text())):
                modules=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                    [a.name for a in node.names] if isinstance(node,ast.Import) else [])
                for module in modules:
                    target=module.replace('.','/')+'.py'
                    if module.startswith(('scripts.','sas_memory.')) and Path(target).is_file():self.assertIn(target,sources())

    def fixture(self):
        # Existing62000 fixtures only: do not generate the prospective63000 cohort in tests.
        root=Path('/media/990Pro/Code/SAS/datasets/memory_update/fresh-shared-timing-v1')
        cases=json.loads((root/'cases.json').read_text());registry=json.loads((root/'registry.json').read_text())
        rows=[{'id':q['id'],'greedy_exact':True,'answer_nll':0.0} for q in registry['physical']]
        registered={'models':{'native':None,'shared':{'teacher_train_fit':{'examples':864}}},
            'references':{'prior_fresh_gates':{'old_failed':False}}}
        return {name:deepcopy(rows) for name in ORDER},cases,registry,registered

    def test_zero_differences_cannot_pass_positive_gain_gates(self):
        args=self.fixture();m=summarize(*args)
        self.assertEqual(list(m['models']),list(ORDER))
        self.assertFalse(m['registered_criteria']['matched_question_positive_joint_gain_over_native'])
        self.assertFalse(m['registered_criteria']['matched_question_positive_joint_gain_over_all_trained_question_deployed'])
        self.assertTrue(m['registered_criteria']['matched_question_no_all_trained_all_deployed_correct_regression'])
        self.assertEqual(m['models'][QUESTION]['physical_queries'],336)
        self.assertEqual(m['models'][QUESTION]['bindings'],432)

    def test_one_regression_is_preserved_and_fails_strict_preservation(self):
        args=self.fixture();args[0][QUESTION][0]['greedy_exact']=False
        m=summarize(*args);p=m['matched_question_functional_preservation']
        self.assertEqual((p['regression'],p['repair'],p['physical_queries']),(1,0,336))
        self.assertFalse(m['registered_criteria']['matched_question_no_all_trained_all_deployed_correct_regression'])
        self.assertEqual(sum(row['regression'] for row in p['individual_physical_pairs']),1)

    def test_opposite_smokes_compare_all288_readouts_and_actual_scope(self):
        from scripts.analyze_trained_shared_timing import compare_observations
        ids=[str(i) for i in range(18)]
        first={'model_order':list(ORDER),'selected_physical_ids':ids,'native_anchors':[],
            'models':{n:{'prefix_checks':[]} for n in ORDER}}
        second=deepcopy(first);second['model_order'].reverse()
        rows={n:[{'id':i,'policy_calls':[{'observed':'same'}]} for i in ids] for n in ORDER}
        self.assertEqual(compare_observations(first,second,rows,deepcopy(rows))['exact_readouts'],288)
        changed=deepcopy(rows);changed[QUESTION][0]['policy_calls'][0]['observed']='different'
        with self.assertRaises(ValueError):compare_observations(first,second,rows,changed)

    def test_missing_cell_or_physical_row_is_rejected(self):
        args=self.fixture();args[0].pop('shared_ledger_trained_all')
        with self.assertRaises(ValueError):summarize(*args)
        args=self.fixture();args[0][QUESTION].pop()
        with self.assertRaises(ValueError):summarize(*args)


if __name__=='__main__':unittest.main()
