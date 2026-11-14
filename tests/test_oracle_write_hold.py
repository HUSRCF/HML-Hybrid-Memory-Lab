"""Gate/control semantics independent of downstream task predictions."""
import copy
from types import SimpleNamespace
import unittest
import torch
from scripts.oracle_write_hold_runtime import RecurrentHold,GDN_LAYERS
from scripts.oracle_write_hold_data import build_policies


class FakeGDN(torch.nn.Module):
    def __init__(self):
        super().__init__();self.in_proj_a=torch.nn.Linear(2,2,bias=False);self.in_proj_b=torch.nn.Linear(2,2,bias=False)
        self.A_log=torch.nn.Parameter(torch.zeros(2));self.dt_bias=torch.nn.Parameter(torch.zeros(2))
        self.chunk_gated_delta_rule=self.capture;self.recurrent_gated_delta_rule=self.capture;self.last=None
    def capture(self,*args,**kwargs):self.last=(kwargs['g'].detach().clone(),kwargs['beta'].detach().clone());return None
    def forward(self,x):
        b=self.in_proj_b(x).sigmoid();a=self.in_proj_a(x)
        g=-self.A_log.float().exp()*torch.nn.functional.softplus(a.float()+self.dt_bias)
        function=self.recurrent_gated_delta_rule if x.shape[1]==1 else self.chunk_gated_delta_rule
        function(x,x,x,g=g,beta=b)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.device=torch.device('cpu')
        self.model=SimpleNamespace(layers=[SimpleNamespace(linear_attn=FakeGDN()) for _ in range(24)])
    def forward(self,input_ids,past_key_values=None,**kwargs):
        x=torch.ones((1,input_ids.shape[1],2))
        for i in GDN_LAYERS:self.model.layers[i].linear_attn(x)


class HoldTests(unittest.TestCase):
    def test_only_selected_prefix_positions_held(self):
        model=FakeModel();keep=[True,False,True,False]
        with RecurrentHold(model,keep) as hold:
            model(input_ids=torch.tensor([[1,2,3,4,5,6]]));proof=hold.report()
        for i in GDN_LAYERS:
            g,b=model.model.layers[i].linear_attn.last
            self.assertTrue(torch.equal(g[:,[1,3]],torch.zeros_like(g[:,[1,3]])))
            self.assertTrue(torch.equal(b[:,[1,3]],torch.zeros_like(b[:,[1,3]])))
            self.assertTrue((g[:,[0,2,4,5]]<0).all());self.assertTrue((b[:,[0,2,4,5]]>0).all())
            self.assertEqual(proof['layers'][i]['held_rows'],2)
    def test_disabled_and_all_retained_are_exact(self):
        for keep,enabled in [([True,False],False),([True,True],True)]:
            model=FakeModel();model(input_ids=torch.ones((1,4),dtype=torch.long))
            before=[copy.deepcopy(model.model.layers[i].linear_attn.last) for i in GDN_LAYERS]
            originals=[model.model.layers[i].linear_attn.chunk_gated_delta_rule for i in GDN_LAYERS]
            with RecurrentHold(model,keep,enabled) as hold:
                model(input_ids=torch.ones((1,4),dtype=torch.long));hold.report()
            for j,i in enumerate(GDN_LAYERS):
                for a,b in zip(before[j],model.model.layers[i].linear_attn.last):self.assertTrue(torch.equal(a,b))
                self.assertEqual(originals[j],model.model.layers[i].linear_attn.chunk_gated_delta_rule)
                self.assertFalse(model.model.layers[i].linear_attn.in_proj_a._forward_hooks)
    def test_cached_absolute_position(self):
        model=FakeModel();cache=SimpleNamespace(get_seq_length=lambda:3)
        with RecurrentHold(model,[True,False,True,False]) as hold:
            model(input_ids=torch.tensor([[7]]),past_key_values=cache);proof=hold.report()
        self.assertTrue(all(s['held_rows']==1 and s['recurrent_calls']==1 for s in proof['layers'].values()))
    def test_cleanup_on_failure(self):
        model=FakeModel();original=model.model.layers[0].linear_attn.chunk_gated_delta_rule
        with self.assertRaises(RuntimeError):
            with RecurrentHold(model,[True,False]):raise RuntimeError('test')
        self.assertEqual(original,model.model.layers[0].linear_attn.chunk_gated_delta_rule)
        self.assertFalse(model._forward_pre_hooks)
    def test_gate_corruption_is_detected(self):
        model=FakeModel()
        with RecurrentHold(model,[True,False]) as hold:
            model(input_ids=torch.tensor([[1,2]]));hold.valid &= False
            with self.assertRaises(ValueError):hold.report()
    def test_policy_excludes_prelude_and_question(self):
        q={'id':'a','ledger_tokens':5,'prelude_tokens':2,'prompt_ids_sha256':'x'}
        p=build_policies({'a':q},{'a':{'keep_bits':{'oracle':'11100'},'oracle_event_indices':[0],'event_count':3}})['a']
        self.assertEqual(p['held_positions'],[3,4]);self.assertEqual(p['first_held_position'],3)
        with self.assertRaises(ValueError):build_policies({'a':q},{'a':{'keep_bits':{'oracle':'01100'}}})


if __name__=='__main__':unittest.main()
