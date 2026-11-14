from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.analyze_gdn_question_timing_control import ARM, check_plan_pair, check_prefix_samples


ROOT = Path(__file__).resolve().parents[1]


class TimingControlTests(unittest.TestCase):
    def test_pair_rejects_scope_budget_and_initialization_changes(self):
        plan = json.loads((ROOT/'configs/gdn_question_timing_control_plan.json').read_text())
        original = json.loads((ROOT/'configs/semantic_single_path_controls_plan.json').read_text())
        manifest = {'arms':{'gdn_only':{'initial_state_sha256':plan['initial_state_sha256'][ARM]}}}
        check_plan_pair(plan,original,manifest)
        mutations = [(('maximum_optimizer_steps',),431),(('arms',ARM,'token_policy'),'all'),
                     (('arms',ARM,'rank'),83),(('initial_state_sha256',ARM),'0'*64)]
        for keys,value in mutations:
            changed = deepcopy(plan)
            target = changed
            for key in keys[:-1]:target=target[key]
            target[keys[-1]]=value
            with self.subTest(keys=keys),self.assertRaises(ValueError):
                check_plan_pair(changed,original,manifest)

    def test_prefix_controls_require_both_prescribed_extremes(self):
        registry = [{'case_id':'a','variant':'before','ledger_tokens':100},
                    {'case_id':'b','variant':'control','ledger_tokens':150},
                    {'case_id':'c','variant':'treatment','ledger_tokens':200}]
        short = {**registry[0],'kv_recurrent_conv_and_logits_exact':True}
        long = {**registry[-1],'kv_recurrent_conv_and_logits_exact':True}
        check_prefix_samples([short,long],registry)
        invalid = [[short,short],[long,short],[short,{**long,'ledger_tokens':199}],
                   [short,{**long,'kv_recurrent_conv_and_logits_exact':False}]]
        for checks in invalid:
            with self.subTest(checks=checks),self.assertRaises(ValueError):
                check_prefix_samples(checks,registry)


if __name__ == '__main__':
    unittest.main()
