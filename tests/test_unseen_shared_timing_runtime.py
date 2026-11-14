"""Actual passive timing boundaries, rejection controls and paired inference units."""

from copy import deepcopy
import json
import unittest

import torch
from torch import nn

from sas_memory.training_data import data_fingerprint
from scripts.unseen_shared_timing_data import DESTINATION,ORDER
from scripts.unseen_shared_timing_runtime import PolicyTrace,audit_policy,active_interval
from scripts.unseen_shared_timing_metrics import summarize
from scripts.analyze_unseen_shared_timing import compare_observations


class Cache:
    def __init__(self,length):self.length=length
    def get_seq_length(self):return self.length


class Rotary(nn.Module):
    def forward(self,hidden,positions):return hidden


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Module();self.model.rotary_emb=Rotary()
        self.projection=nn.Linear(2,2,bias=False,dtype=torch.bfloat16)
        with torch.no_grad():self.projection.weight.copy_(torch.eye(2))
    def forward(self,input_ids,use_cache,logits_to_keep,past_key_values=None):
        start=past_key_values.get_seq_length() if past_key_values is not None else 0
        positions=torch.arange(start,start+input_ids.shape[1]).view(1,1,-1).expand(3,1,-1)
        hidden=self.model.rotary_emb(input_ids[...,None].expand(-1,-1,2).to(torch.bfloat16),positions)
        output=self.projection(hidden)
        return type('Result',(),{'past_key_values':Cache(start+input_ids.shape[1]),'output':output})()


class ToyControl:
    def __init__(self,model,policy,leak=False):
        self._specs=[(model.projection,'p','shared',False)];self._enabled_outputs={'p'}
        self._token_policy=policy;self._token_boundary=2;self._position_start=0
        self.handles=[model.register_forward_pre_hook(self.position,with_kwargs=True),model.projection.register_forward_hook(self.change)]
        self.leak=leak
    def position(self,module,args,kwargs):
        cache=kwargs.get('past_key_values');self._position_start=cache.get_seq_length() if cache else 0
    def change(self,module,args,output):
        positions=torch.arange(self._position_start,self._position_start+output.shape[1])
        low,high=active_interval(self._position_start,output.shape[1],2,self._token_policy)
        active=(positions>=low)&(positions<high)
        if self.leak:active[:]=True
        return output+active[None,:,None].to(output.dtype)
    def close(self):
        for handle in self.handles:handle.remove()


class TimingRuntimeTests(unittest.TestCase):
    def query(self):
        return {'ledger_tokens':2,'prompt_tokens':4,'prompt_ids_sha256':data_fingerprint([1,2,3,4]),'answer_ids':[7,8,9]}

    def trace(self,policy):
        model=ToyModel();controller=ToyControl(model,policy)
        with torch.inference_mode(),PolicyTrace(model,controller,self.query(),policy) as trace:
            result=model(input_ids=torch.tensor([[1,2,3,4]]),use_cache=True,logits_to_keep=1)
            model(input_ids=torch.tensor([[7,8]]),past_key_values=result.past_key_values,use_cache=True,logits_to_keep=2)
            model(input_ids=torch.tensor([[7]]),past_key_values=result.past_key_values,use_cache=True,logits_to_keep=1)
            model(input_ids=torch.tensor([[8]]),past_key_values=Cache(5),use_cache=True,logits_to_keep=1)
        controller.close();self.assertFalse(model._forward_pre_hooks);self.assertFalse(model.projection._forward_hooks)
        return trace.calls

    def test_actual_full_prefill_gold_and_greedy_intervals_and_observer_cleanup(self):
        for policy,expected in [('all',[4,2,1,1]),('question',[2,2,1,1]),('ledger',[2,0,0,0]),('none',[0,0,0,0])]:
            calls=self.trace(policy);audit_policy(calls,self.query(),policy,{'p':2})
            self.assertEqual([c['projections']['p']['changed_active_rows'] for c in calls],expected)

    def test_actual_projection_leakage_is_rejected_and_passive_hooks_removed(self):
        model=ToyModel();controller=ToyControl(model,'ledger',leak=True)
        with self.assertRaisesRegex(ValueError,'leaked'):
            with torch.inference_mode(),PolicyTrace(model,controller,self.query(),'ledger'):
                model(input_ids=torch.tensor([[1,2,3,4]]),use_cache=True,logits_to_keep=1)
        controller.close();self.assertFalse(model._forward_hooks);self.assertFalse(model.projection._forward_hooks)

    def test_audit_rejects_position_gold_scope_and_greedy_mutations(self):
        original=self.trace('question')
        for mutation in ['rope','gold','interval','projection','inactive','greedy']:
            calls=deepcopy(original)
            if mutation=='rope':calls[0]['rope_positions_sha256']='0'*64
            elif mutation=='gold':calls[1]['cached_input_ids'][0]=0
            elif mutation=='interval':calls[0]['active_interval']=[0,4]
            elif mutation=='projection':calls[0]['projections']={}
            elif mutation=='inactive':calls[0]['projections']['p']['inactive_rows_exact']=False
            else:calls[-1]['start']=4
            with self.assertRaises(ValueError):audit_policy(calls,self.query(),'question',{'p':2})

    def test_gold_only_eos_stop_and_single_token_gold_are_valid(self):
        calls=self.trace('question');audit_policy(calls[:2],self.query(),'question',{'p':2})
        q=self.query();q['answer_ids']=[7]
        audit_policy(calls[:1]+calls[2:],q,'question',{'p':2})

    def test_registered_metrics_reject_a_single_regression_despite_net_repair(self):
        cases=json.loads((DESTINATION/'cases.json').read_text());registry=json.loads((DESTINATION/'registry.json').read_text())
        from scripts.unseen_shared_timing_data import selected_inputs
        physical,_=selected_inputs(registry)
        rows={name:[{'id':q['id'],'greedy_exact':name!='native','answer_nll':0.0} for q in physical] for name in ORDER}
        rows['shared_question_timing'][0]['greedy_exact']=False
        registered=json.loads((DESTINATION/'manifest.json').read_text())
        result=summarize(rows,cases,registry,registered)
        self.assertFalse(result['registered_criteria']['question_only_no_shared_correct_regression'])
        self.assertEqual(result['within_checkpoint_question_functional_preservation']['regression'],1)
        self.assertEqual(result['within_checkpoint_question_functional_preservation']['physical_queries'],336)
        for factor in ('seed','question_style','distractor_updates','variant'):
            groups=result['within_checkpoint_question_functional_preservation']['breakdowns'][factor]
            self.assertEqual(sum(g['physical_queries'] for g in groups),336)
            self.assertEqual(sum(g['regression'] for g in groups),1)
        self.assertEqual(len(result['models']),10)
        self.assertTrue(all(not gate['passed'] for gate in result['prior_fresh_gates_unchanged'].values()))

    def test_opposite_smokes_cannot_omit_or_change_any_readout(self):
        ids=[str(i) for i in range(18)];rows={name:[{'id':i,'score':1} for i in ids] for name in ORDER}
        first={'model_order':list(ORDER),'selected_physical_ids':ids,'native_anchors':[],
            'models':{name:{'prefix_checks':[]} for name in ORDER}}
        second=deepcopy(first);second['model_order']=list(ORDER[::-1])
        self.assertEqual(compare_observations(first,second,rows,deepcopy(rows))['exact_readouts'],180)
        for mode in ['drop','change']:
            broken=deepcopy(rows)
            if mode=='drop':broken['shared_question_timing'].pop()
            else:broken['shared_question_timing'][-1]['score']=2
            with self.assertRaises(ValueError):compare_observations(first,second,rows,broken)


if __name__=='__main__':unittest.main()
