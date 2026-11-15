"""Matched whole-record selections and independently fixed evidence reading."""
from collections import Counter
from types import SimpleNamespace
import random
import unittest
from unittest.mock import patch
import torch
from tests.test_oracle_write_hold import FakeModel
from scripts.update_budget_data import choose_records,read_plan,FAMILIES,MODELS,REGIMES,DATA_SOURCES,sources
from scripts.update_budget_runtime import measure_condition,GDN_LAYERS
from sas_memory.training_data import data_fingerprint

RECORDS=[{'index':i,'kind':kind,'length':length} for i,(kind,length) in enumerate(
    [('SET',17),('SET',17),('COPY',21),('SET',17),('COPY',21),('SET',17)])]

class BudgetTests(unittest.TestCase):
    def test_every_family_matches_record_types_lengths_and_total_tokens(self):
        budget={('SET',17):2,('COPY',21):1}
        for family in FAMILIES:
            selected=choose_records(RECORDS,budget,'fixed-prefix',family)
            chosen=[b for b in RECORDS if b['index'] in selected]
            self.assertEqual(Counter((b['kind'],b['length']) for b in chosen),budget)
            self.assertEqual(sum(b['length'] for b in chosen),55)
            self.assertEqual(len(set(selected)),3)
    def test_position_controls_select_within_strata(self):
        budget={('SET',17):1,('COPY',21):1}
        self.assertEqual(choose_records(RECORDS,budget,'p','early'),[0,2])
        self.assertEqual(choose_records(RECORDS,budget,'p','recent'),[4,5])
    def test_hash_ranking_is_deterministic_without_global_random_state(self):
        state=random.getstate();budget={('SET',17):1,('COPY',21):2}
        for family in ('random0','random1'):
            a=choose_records(RECORDS,budget,'one-prefix-budget',family)
            self.assertEqual(a,choose_records(RECORDS,budget,'one-prefix-budget',family))
        self.assertEqual(random.getstate(),state)
    def test_semantic_coincidences_are_retained_without_resampling(self):
        budget={('SET',17):4,('COPY',21):2}
        for family in FAMILIES:self.assertEqual(choose_records(RECORDS,budget,'any-prefix',family),list(range(6)))
    def test_invalid_budgets_and_unknown_policy_are_rejected(self):
        for budget in ({('SET',17):5},{('SET',17):-1},{('SET',17):True},{('ADD',17):1}):
            with self.assertRaises(ValueError):choose_records(RECORDS,budget,'p','early')
        with self.assertRaises(ValueError):choose_records(RECORDS,{('SET',17):1},'p','tuned')
    def test_writer_mask_never_replaces_reader_mask_or_native_decay(self):
        model=FakeModel();runtime=SimpleNamespace(model=model);captured=[]
        class Reader:
            def __init__(self,model,keep):captured.append(keep);self.calls={'prefill':1,'cached':1}
            def __enter__(self):return self
            def __exit__(self,*args):pass
        def measure(*args):model(input_ids=torch.ones((1,6),dtype=torch.long));return {'result':'constant'}
        policy={'history_keep_bits':'1100','write_keep_bits':{'random0':'1010'}}
        with patch('sas_memory.kv_readout.QuestionKVReadout',Reader),patch('scripts.run_oracle_kv_readout.measure',measure):
            value,proof=measure_condition(runtime,'fixed-ledger',{'prelude_tokens':1},policy,'random0_oracle')
        self.assertEqual(captured,[[True,True,False,False]])
        self.assertEqual(proof['write_keep_sha256'],data_fingerprint('1010'))
        self.assertEqual(proof['reader_keep_sha256'],data_fingerprint('1100'))
        for i in GDN_LAYERS:
            g,b=model.model.layers[i].linear_attn.last
            self.assertTrue((g<0).all())
            self.assertTrue((b[:,[1,3]]==0).all())
            self.assertTrue((b[:,[0,2,4,5]]>0).all())
        self.assertFalse(model._forward_pre_hooks)
    def test_registered_complete_scope_keeps_all_existing_models(self):
        p=read_plan();self.assertEqual(len(MODELS),5);self.assertEqual(len(REGIMES),8)
        self.assertEqual(len(DATA_SOURCES),192);self.assertEqual(len(sources()),197)
        self.assertEqual(p['counts']['new_primary_physical_readouts'],13440)
        self.assertFalse(p['training']);self.assertFalse(p['architecture_effectiveness_claim'])

if __name__=='__main__':unittest.main()
