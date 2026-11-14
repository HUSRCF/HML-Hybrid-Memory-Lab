"""Fresh actual-input novelty, whole records and independent scientific gates."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.analyze_fresh_copy_confirmation import audit_proofs
from scripts.compare_fresh_copy_smokes import compare_observations
from scripts.fresh_copy_data import (DESTINATION, MODELS, PRIMARY, MASKED_REGIMES, check_novelty,
    registered_inputs, read_plan, model_order)
from scripts.fresh_copy_metrics import summarize
from scripts.fresh_copy_runtime import verify_prior_index


class FreshCopyConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registration, cls.source_plan, cls.cases, cls.registry, cls.masks = registered_inputs()
        cls.plan = read_plan()
        cls.index = json.loads((DESTINATION / 'prior_inputs.json').read_text())

    def test_actual_prior_inputs_independently_reconstruct_and_have_no_overlap(self):
        proof = verify_prior_index(DESTINATION, self.registration)
        self.assertEqual(proof['actual_prior_index_sha256'], self.registration['prior_inputs_sha256'])
        self.assertGreater(proof['complete_prompts'], 6000)
        self.assertGreater(proof['complete_prefixes'], 1900)
        self.assertEqual(self.registration['freshness']['overlap'], {'seeds': [], 'identifiers': [], 'complete_prompts': [], 'complete_prefixes': []})
        for field, value in [('seeds', self.cases[0]['seed']), ('identifiers', self.cases[0]['base_case_id']),
                             ('complete_prompt_ids_sha256', self.registry['physical'][0]['id']),
                             ('complete_prefix_ids_sha256', self.registry['physical'][0]['source_key'])]:
            wrong = deepcopy(self.index); wrong[field].append(value)
            with self.assertRaises(ValueError): check_novelty(self.cases, self.registry, wrong)

    def test_all_fixed_fresh_cases_and_whole_record_masks_are_preserved(self):
        self.assertEqual({c['seed'] for c in self.cases}, set(range(61000, 61008)))
        self.assertEqual(len(self.cases), 48); self.assertEqual(len(self.registry['physical']), 336)
        self.assertEqual(len(self.registry['bindings']), 432)
        self.assertEqual({m['removal_registration']['removed_record_tokens'] for m in self.masks.values()}, {17})
        self.assertEqual({m['oracle_registration']['retained_record_tokens'] for m in self.masks.values()}, {17,38,59,80})
        for mask in self.masks.values():
            self.assertEqual(list(mask['keep_bits']), list(MASKED_REGIMES))
            reg = mask['removal_registration']
            for regime, span in zip(MASKED_REGIMES[1:], (reg['support_token_span'],reg['other_token_span'])):
                self.assertEqual([i for i,b in enumerate(mask['keep_bits'][regime]) if b == '0'], list(range(*span)))
            self.assertTrue(reg['support_leaf_symbolically_necessary'])
        self.assertEqual(sum(len(v) for v in self.plan['regimes_by_model'].values()), 14)

    def synthetic_rows(self):
        rows = {m: {r: [{'id':q['id'], 'greedy_exact':True,'answer_nll':.01} for q in self.registry['physical']]
                    for r in self.plan['regimes_by_model'][m]} for m in MODELS}
        targets = {b['physical_id'] for b in self.registry['bindings'] if b['variant'] != 'before' and b['role'] == 'target_current'}
        control = {b['physical_id'] for b in self.registry['bindings'] if b['variant'] == 'control' and b['role'] == 'target_current'}
        for m in ('independent', *PRIMARY):
            for row in rows[m]['full']:
                if row['id'] in targets: row.update(greedy_exact=False,answer_nll=2.)
        for m in PRIMARY:
            for row in rows[m]['remove_support']:
                if row['id'] in targets: row.update(greedy_exact=False,answer_nll=2.)
            for row in rows[m]['remove_matched_other']:
                if row['id'] in targets - control: row.update(greedy_exact=False,answer_nll=2.)
        return rows, targets

    def test_architecture_gate_is_independent_of_both_readout_gates(self):
        rows, targets = self.synthetic_rows()
        result = summarize(rows,self.cases,self.registry,self.source_plan,self.plan)
        for gate in ('generic_shared_controller','frozen_prefix_sufficiency','whole_record_necessity'):
            self.assertTrue(result[gate]['passed'],gate)
        for row in rows['independent']['full']: row.update(greedy_exact=True,answer_nll=.01)
        result = summarize(rows,self.cases,self.registry,self.source_plan,self.plan)
        self.assertFalse(result['generic_shared_controller']['passed'])
        self.assertTrue(result['frozen_prefix_sufficiency']['passed']);self.assertTrue(result['whole_record_necessity']['passed'])
        rows, targets = self.synthetic_rows()
        failures=set(sorted(targets)[:7])
        for row in rows['local_question']['oracle']:
            if row['id'] in failures: row.update(greedy_exact=False,answer_nll=2.)
        result=summarize(rows,self.cases,self.registry,self.source_plan,self.plan)
        self.assertFalse(result['frozen_prefix_sufficiency']['passed']); self.assertTrue(result['whole_record_necessity']['passed'])
        self.assertTrue(result['generic_shared_controller']['passed'])
        rows,_=self.synthetic_rows();rows.pop('native')
        with self.assertRaises(ValueError): summarize(rows,self.cases,self.registry,self.source_plan,self.plan)

    def test_necessity_requires_both_models_unrelated_control_preservation(self):
        rows,_=self.synthetic_rows()
        key=next(b['physical_id'] for b in self.registry['bindings'] if b['role']=='protected_current' and b['variant']=='control')
        next(r for r in rows['gdn_question_only']['remove_matched_other'] if r['id']==key)['greedy_exact']=False
        result=summarize(rows,self.cases,self.registry,self.source_plan,self.plan)
        self.assertFalse(result['whole_record_necessity']['passed']);self.assertTrue(result['frozen_prefix_sufficiency']['passed'])
        self.assertTrue(result['prior_shared_history_repeat_primaries_remain_failed'])

    def test_full_only_and_three_mask_proof_scopes_cannot_be_substituted(self):
        from tests.test_copy_readout_sufficiency import ReadoutSufficiencyTests
        p,full,indexed=ReadoutSufficiencyTests().proof_fixture()
        single=p['regimes']['oracle'];weight=p['unmasked_weights']['oracle'];row=indexed['oracle']
        p['regimes']={r:deepcopy(single) for r in MASKED_REGIMES}
        p['unmasked_weights']={r:deepcopy(weight) for r in MASKED_REGIMES}
        indexed={r:deepcopy(row) for r in MASKED_REGIMES}
        audit_proofs([p],['q'],full,indexed,MASKED_REGIMES)
        wrong=deepcopy(p);wrong['regimes']['remove_matched_other']['historical_kv_fields_exact']=11
        with self.assertRaises(ValueError): audit_proofs([wrong],['q'],full,indexed,MASKED_REGIMES)
        p['regimes']={};p['unmasked_weights']={}
        audit_proofs([p],['q'],full,{},[])
        wrong=deepcopy(p);wrong['all_history_cache_logits_readout_exact']=False
        with self.assertRaises(ValueError): audit_proofs([wrong],['q'],full,{},[])
        with self.assertRaises(ValueError): audit_proofs([p],['q'],full,indexed,MASKED_REGIMES)

    def order_fixture(self):
        ids=[f'q{i}' for i in range(18)]
        template={'answer':'123','answer_nll':.1,'greedy_exact':True,'generated':'123\n','query_tokens':20}
        rows={m:{r:[{'id':q,**template} for q in ids] for r in self.plan['regimes_by_model'][m]} for m in MODELS}
        first={'model_order':model_order(),'selected_physical_ids':ids,'native_anchors':[{'id':q} for q in ids[:10]],'models':{}}
        for m in MODELS:
            first['models'][m]={'proofs':[{'id':q,'regimes':{r:{'prefix_sha256':'a'*64,'masked_prefix_sha256':'a'*64}
                for r in (MASKED_REGIMES if m in PRIMARY else ())}} for q in ids[:10]],
                'prefix_checks':[{'source_key':q,'prefix_sha256':'a'*64} for q in ids[:6]] if m in PRIMARY else []}
        second=deepcopy(first);second['model_order']=model_order(True)
        return first,second,rows,deepcopy(rows)

    def test_all252_smoke_readouts_and60_masked_hash_pairs_are_required(self):
        a,b,x,y=self.order_fixture();counts=compare_observations(a,b,x,y)
        self.assertEqual(counts['exact_readouts'],252);self.assertEqual(counts['exact_masked_prefix_hash_pairs'],60)
        y['native']['full'].pop()
        with self.assertRaises(ValueError): compare_observations(a,b,x,y)
        a,b,x,y=self.order_fixture();y['gdn_question_only']['oracle'][-1]['answer_nll']+=1e-12
        with self.assertRaises(ValueError): compare_observations(a,b,x,y)
        a,b,x,y=self.order_fixture();b['models']['local_question']['proofs'][-1]['regimes']['remove_support']['masked_prefix_sha256']='b'*64
        with self.assertRaises(ValueError): compare_observations(a,b,x,y)


if __name__=='__main__': unittest.main()
