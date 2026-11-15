"""Single-gate control semantics and whole-cohort protocol coverage."""
import copy
import unittest
import torch
from tests.test_oracle_write_hold import FakeModel
from scripts.gate_component_runtime import ComponentHold,GDN_LAYERS,SETTINGS
from scripts.gate_component_data import read_plan,MODELS,REGIMES,ARCHIVED,UNIFIED,DATA_SOURCES,sources

class ComponentTests(unittest.TestCase):
    def test_single_components_leave_other_native_gate_exact(self):
        for mode in ('none','decay','delta','both'):
            model=FakeModel();ids=torch.tensor([[1,2,3,4,5,6]])
            model(input_ids=ids);before={i:copy.deepcopy(model.model.layers[i].linear_attn.last) for i in GDN_LAYERS}
            with ComponentHold(model,[True,False,True,False],mode) as hold:
                model(input_ids=ids);proof=hold.report()
            for i in GDN_LAYERS:
                for kind,(expected,actual) in enumerate(zip(before[i],model.model.layers[i].linear_attn.last)):
                    expected=expected.clone()
                    active=mode=='both' or (kind==0 and mode=='decay') or (kind==1 and mode=='delta')
                    if active:expected[:,[1,3]]=0
                    self.assertTrue(torch.equal(expected,actual),(i,mode,kind))
                stat=proof['layers'][i]
                self.assertEqual(stat['forced_decay_rows'],2 if mode in ('decay','both') else 0)
                self.assertEqual(stat['forced_delta_rows'],2 if mode in ('delta','both') else 0)
            self.assertFalse(model._forward_pre_hooks)
    def test_invalid_mode_rejected_before_hooks(self):
        model=FakeModel()
        with self.assertRaises(ValueError):ComponentHold(model,[True,False],'guess')
        self.assertFalse(model._forward_pre_hooks)
    def test_both_endpoint_matches_old_implementation_exactly(self):
        from scripts.oracle_write_hold_runtime import RecurrentHold
        model=FakeModel();ids=torch.ones((1,6),dtype=torch.long)
        with RecurrentHold(model,[True,False,True,False]) as old:model(input_ids=ids);old.report()
        expected={i:copy.deepcopy(model.model.layers[i].linear_attn.last) for i in GDN_LAYERS}
        with ComponentHold(model,[True,False,True,False],'both') as new:model(input_ids=ids);new.report()
        for i in GDN_LAYERS:
            for a,b in zip(expected[i],model.model.layers[i].linear_attn.last):self.assertTrue(torch.equal(a,b))
    def test_all_models_and_eight_cells_registered_without_new_training(self):
        p=read_plan();self.assertEqual(len(MODELS),5);self.assertEqual(len(REGIMES),4);self.assertEqual(len(ARCHIVED),4)
        self.assertEqual(set(UNIFIED),set(REGIMES)|set(ARCHIVED));self.assertEqual(len(DATA_SOURCES),183);self.assertEqual(len(sources()),188)
        self.assertEqual(p['counts']['new_primary_physical_readouts'],6720);self.assertEqual(p['counts']['archived_physical_readouts'],6720)
        self.assertFalse(p['training']);self.assertFalse(p['architecture_effectiveness_claim'])
    def test_reader_and_gate_axes_are_separate(self):
        self.assertEqual(SETTINGS['decay_full'],('decay',False));self.assertEqual(SETTINGS['decay_oracle'],('decay',True))
        self.assertEqual(SETTINGS['delta_full'],('delta',False));self.assertEqual(SETTINGS['delta_oracle'],('delta',True))
        self.assertEqual(SETTINGS['hold_oracle'],('both',True));self.assertEqual(SETTINGS['full'],('none',False))

if __name__=='__main__':unittest.main()
