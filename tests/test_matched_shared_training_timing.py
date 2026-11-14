"""Fixed optimization budget, paired initialization and prospective factorial."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.matched_shared_timing_common import (read_plan,validate_plan,references,initial_fingerprint,
    sources,scope_samples,ARMS)
from scripts.analyze_matched_shared_training_timing import audit_guard
from scripts.unseen_shared_timing_runtime import sources as previous_sources


class MatchedSharedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=read_plan();cls.refs=references(cls.plan)
        cls.base=json.loads(Path('configs/order_augmented_semantic_plan.json').read_text())
        cls.data=json.loads((Path(cls.plan['data_source'])/'data.json').read_text())
        cls.registry=json.loads((Path(cls.plan['data_source'])/'registry.json').read_text())

    def test_same_architecture_initialization_and_original_optimization_budget(self):
        p=self.plan
        self.assertEqual(initial_fingerprint(self.refs['shared_description'],p['initialization_seed']),
            self.base['initial_state_sha256']['shared'])
        self.assertEqual(set(p['initial_state_sha256'].values()),{self.base['initial_state_sha256']['shared']})
        self.assertEqual((len(self.data['train']),p['maximum_optimizer_steps'],p['gradient_accumulation']),(864,432,2))
        self.assertEqual(p['optimized_input_token_budgets']['432'],527808)
        self.assertEqual(p['supervised_answer_token_budgets']['432'],3456)

    def test_rank_loss_budget_seed_and_heldout_grid_cannot_be_tuned(self):
        for field in ['steps','lr','rank','loss','seed','heldout','baseline','deployment','count']:
            p=deepcopy(self.plan)
            if field=='steps':p['maximum_optimizer_steps']=864
            elif field=='lr':p['learning_rate']=.002
            elif field=='rank':p['arms']['shared_question']['rank']=128
            elif field=='loss':p['objective']='extra consistency supervision'
            elif field=='seed':p['initialization_seed']=1702
            elif field=='heldout':p['confirmation']['fresh_seeds'][0]=62000
            elif field=='baseline':p['confirmation']['evaluation_arms'].pop('local')
            elif field=='deployment':p['confirmation']['evaluation_arms']['shared_ledger_trained_question']['token_policy']='ledger'
            else:p['confirmation']['counts']['primary_physical_full_answer_readouts']=3360
            with self.assertRaises(ValueError):validate_plan(p,self.base)

    def test_entire3x3factorial_and_all_old_baselines_are_registered_before_training(self):
        c=self.plan['confirmation'];self.assertEqual(len(c['evaluation_order']),16)
        self.assertEqual(c['counts']['primary_physical_full_answer_readouts'],16*336)
        for arm in ARMS:
            self.assertEqual({c['evaluation_arms'][f'{arm}_trained_{p}']['token_policy'] for p in ['all','question','ledger']},
                {'all','question','ledger'})
        self.assertEqual(c['smoke_physical_readouts_per_order'],16*18)
        self.assertEqual(len(sources()),140);self.assertTrue(previous_sources()<sources())
        self.assertTrue(all(not v['passed'] for v in self.refs['prior_failed_gates'].values()))
        self.assertTrue(all(not v for v in self.refs['previous_timing_criteria'].values()))

    def test_fixed_scope_controls_cover_all_variant_roles_and_last_registered_example(self):
        ids=scope_samples(self.data)
        self.assertEqual({(self.data['validation'][i]['variant'],self.data['validation'][i]['query']['role']) for i in ids},
            {(v,r) for v in ['before','control','treatment'] for r in ['target_current','target_initial','protected_current']})
        self.assertIn(431,ids)
        changed=deepcopy(self.data)
        for e in changed['validation']:e['query']['answer']='999'
        self.assertEqual(scope_samples(changed),ids)

    def test_guard_rejects_changed_native_weights_unfrozen_backbone_or_training_arithmetic(self):
        refs=self.refs;desc=refs['shared_description'];arm='shared_ledger';p=self.plan
        guard={'arm':arm,'policy':'ledger','description':desc,'enabled_outputs':sorted(desc['output_shapes']),
            'native_weights':{'sha256':p['native_loaded_weights_sha256'],'all_parameters_frozen':True,'no_parameter_gradients':True},
            'adapter_state_before':'a'*64,'adapter_state_after':'a'*64,'hooks_restored':True,'rng_unchanged':True,
            'adapter_linear_profile':'original_training_fp32_adapter_linears','all_adapter_linear_weights_fp32':True,
            'no_adapter_arithmetic_hooks':True,'scope_checks':[{'index':i} for i in scope_samples(self.data)]}
        for mutation in ['native','gradients','arithmetic','hooks','scope','adapter']:
            broken=deepcopy(guard)
            if mutation=='native':broken['native_weights']['sha256']='b'*64
            elif mutation=='gradients':broken['native_weights']['no_parameter_gradients']=False
            elif mutation=='arithmetic':broken['adapter_linear_profile']='fp64_training'
            elif mutation=='hooks':broken['no_adapter_arithmetic_hooks']=False
            elif mutation=='scope':broken['enabled_outputs'].pop()
            else:broken['adapter_state_after']='c'*64
            with self.assertRaises(ValueError):audit_guard(broken,arm,p,self.data,self.registry,refs)


if __name__=='__main__':unittest.main()
