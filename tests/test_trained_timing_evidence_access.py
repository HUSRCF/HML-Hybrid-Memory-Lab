"""Original-cohort completeness, fixed masks and strict same-profile references."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

from scripts.trained_timing_evidence_access_data import (MODELS,REGIMES,DATA_SOURCES,sources,read_plan,
    source_inputs,build_masks,tokenizer_runtime,sample_ids,selected_inputs,unique_prefix_queries,require_exact_full)
from scripts.trained_timing_evidence_access_metrics import summarize


class EvidenceAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=read_plan();cls.cases,cls.registry,cls.references,cls.refs=source_inputs(cls.plan)

    def test_five_fixed_models_preserve_both_deployments_of_same_question_checkpoint(self):
        p=self.plan
        self.assertEqual(list(p['models']),list(MODELS));self.assertEqual(len(self.refs['all16_raw_sha256']),16)
        self.assertEqual(p['models']['shared_question_trained_all']['checkpoint_sha256'],p['models']['shared_question_trained_question']['checkpoint_sha256'])
        self.assertEqual(p['counts']['primary_physical_readouts'],5040)
        self.assertTrue(all(v is False for v in self.refs['prior_timing_criteria_unchanged'].values()))
        self.assertFalse(p['prospective_unseen_confirmation']);self.assertFalse(p['architecture_effectiveness_claim'])

    def test_source_inventory_is_closed_and_preserves_all160_measurement_sources(self):
        self.assertEqual((len(DATA_SOURCES),len(sources())),(163,166))
        for name in sources():
            if not name.endswith('.py'):continue
            for node in ast.walk(ast.parse(Path(name).read_text())):
                modules=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                    [v.name for v in node.names] if isinstance(node,ast.Import) else [])
                for module in modules:
                    target=module.replace('.','/')+'.py'
                    if module.startswith(('scripts.','sas_memory.')) and Path(target).is_file():self.assertIn(target,sources())

    def test_original336_program_supports_and_equal_budget_disjoint_masks(self):
        masks=build_masks(self.cases,self.registry,tokenizer_runtime(self.plan['model_path']))
        self.assertEqual(list(masks),[q['id'] for q in self.registry['physical']])
        for q in self.registry['physical']:
            record=masks[q['id']];a,b=[record['keep_bits'][r] for r in REGIMES[1:]];cut=q['prelude_tokens']
            self.assertEqual(len(a),q['ledger_tokens']);self.assertEqual(len(b),len(a))
            self.assertEqual(a.count('1'),b.count('1'));self.assertEqual(a[:cut],'1'*cut);self.assertEqual(b[:cut],a[:cut])
            self.assertFalse(any(x==y=='1' for x,y in zip(a[cut:],b[cut:])))
            self.assertGreater(a[cut:].count('1'),0);self.assertGreater(a.count('0'),0)

    def test_operational_smoke_covers_all50_memory_controls_and_native_prefixes(self):
        physical,bindings=selected_inputs(self.registry,True);keys={q['id'] for q in physical}
        self.assertEqual((len(physical),len(bindings)),(18,18));self.assertTrue(set(sample_ids(self.cases,self.registry))<=keys)
        self.assertEqual(len(unique_prefix_queries(physical)),6)
        self.assertEqual(len(unique_prefix_queries(self.registry['physical'])),56)

    def test_supplemental_controls_cover_depth2_short_long_and_both_styles(self):
        from scripts.trained_timing_evidence_access_metrics import supplemental_depth2_queries
        queries=supplemental_depth2_queries(self.cases,self.registry);ids={q['id'] for q in queries}
        self.assertEqual(len(ids),2)
        selected=[b for b in self.registry['bindings'] if b['physical_id'] in ids and b['role']=='target_current' and b['variant']!='before']
        self.assertEqual({b['dependency_depth'] for b in selected},{2})
        self.assertEqual({b['question_style'] for b in selected},{'original','entity_first'})
        self.assertEqual({b['distractor_updates'] for b in selected},{8,64})
        base=set(sample_ids(self.cases,self.registry));self.assertFalse(base&ids)
        combined=[b for b in self.registry['bindings'] if b['physical_id'] in base|ids and b['role']=='target_current' and b['variant']!='before']
        self.assertEqual({b['dependency_depth'] for b in combined},{1,2,3})

    def test_cross_order_equivalence_rejects_supplemental_depth2_control_drift(self):
        from scripts.analyze_trained_timing_evidence_access import compare_observations
        ids=[str(i) for i in range(18)]
        first={'model_order':list(MODELS),'selected_physical_ids':ids,'native_anchors':[],
            'models':{n:{'depth2_proofs':[{'observed':'same'}]} for n in MODELS}}
        second=deepcopy(first);second['model_order'].reverse()
        rows={n:{r:[{'id':key} for key in ids] for r in REGIMES} for n in MODELS}
        self.assertEqual(compare_observations(first,second,rows,deepcopy(rows))['exact_readouts'],270)
        second['models']['native']['depth2_proofs'][0]['observed']='changed'
        with self.assertRaises(ValueError):compare_observations(first,second,rows,deepcopy(rows))

    def test_full_reference_rejects_any_of_five_field_changes_even_small_nll(self):
        expected={'answer':'123','answer_nll':1.,'greedy_exact':True,'generated':'123','query_tokens':10}
        require_exact_full(expected,expected,'native','x')
        for field,value in [('answer','321'),('answer_nll',1.+1e-12),('greedy_exact',False),('generated','123 '),('query_tokens',11)]:
            changed={**expected,field:value}
            with self.assertRaises(ValueError):require_exact_full(changed,expected,'native','x')

    def test_all_null_contrasts_stay_null_and_missing_model_or_query_rejected(self):
        rows=[{'id':q['id'],'greedy_exact':True,'answer_nll':0.} for q in self.registry['physical']]
        all_rows={n:{r:deepcopy(rows) for r in REGIMES} for n in MODELS}
        m=summarize(all_rows,self.cases,self.registry,self.plan)
        self.assertIsNone(m['success_thresholds']);self.assertFalse(m['architecture_effectiveness_claim'])
        for name in MODELS:
            self.assertEqual(m['models'][name]['full']['physical_queries'],336)
            self.assertEqual(m['oracle_paired_contrasts'][name]['full']['physical']['regression'],0)
            self.assertFalse(m['oracle_paired_contrasts'][name]['full']['joint_cases']['candidate_passed'])
        changed=deepcopy(all_rows);changed.pop('gdn_all_tokens')
        with self.assertRaises(ValueError):summarize(changed,self.cases,self.registry,self.plan)
        changed=deepcopy(all_rows);changed['native']['oracle'].pop()
        with self.assertRaises(ValueError):summarize(changed,self.cases,self.registry,self.plan)


if __name__=='__main__':unittest.main()
