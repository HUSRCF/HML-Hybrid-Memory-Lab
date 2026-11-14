"""Passive proof of actual adapter timing, prefix identity and smoke ordering."""

import json
from pathlib import Path

from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.fresh_layer_controls import tensor_hash
from scripts.run_write_locality_load_screen import hook_state
from scripts.semantic_order_path_common import assert_complete_cache_equal,cache_fingerprint
from scripts.state_continuation_controls import complete_metadata
from scripts.unseen_shared_timing_data import DATA_SOURCES


def sources():
    return DATA_SOURCES|{'scripts/unseen_shared_timing_runtime.py','scripts/unseen_shared_timing_metrics.py',
        'scripts/run_unseen_shared_timing.py','scripts/analyze_unseen_shared_timing.py'}


def active_interval(start,length,boundary,policy):
    end=start+length
    if policy=='all':return [start,end]
    if policy=='none':return [start,start]
    if policy=='question':return [min(end,max(start,boundary)),end]
    if policy=='ledger':return [start,max(start,min(end,boundary))]
    raise ValueError('Unknown original controller token policy')


class PolicyTrace:
    """Observe projection outputs immediately before and after the existing adapter."""
    def __init__(self,model,controller,query,policy):
        self.controller,self.query,self.policy=controller,query,policy
        self.calls=[];self.before_values={};self.handles=[]
        self.targets={name:projection.out_features for projection,name,_,_ in controller._specs} if controller else {}
        if controller and set(controller._enabled_outputs)!=set(self.targets):raise ValueError('Original enabled outputs changed')
        self.handles.append(model.register_forward_pre_hook(self.before,with_kwargs=True))
        self.handles.append(model.model.rotary_emb.register_forward_pre_hook(self.positions,with_kwargs=True))
        if controller:
            for projection,name,_,_ in controller._specs:
                self.handles.append(projection.register_forward_hook(self.capture(name),prepend=True))
                self.handles.append(projection.register_forward_hook(self.observe(name)))
        self.handles.append(model.register_forward_hook(self.after,with_kwargs=True))
    def before(self,module,args,kwargs):
        ids=kwargs['input_ids'];cache=kwargs.get('past_key_values');start=int(cache.get_seq_length()) if cache is not None else 0
        if (ids.ndim!=2 or ids.shape[0]!=1 or not kwargs.get('use_cache') or
            kwargs.get('attention_mask') is not None or kwargs.get('position_ids') is not None or self.before_values):
            raise ValueError('Original unrestricted singleton forward required')
        if self.controller and (self.controller._token_policy!=self.policy or
            self.controller._token_boundary!=self.query['ledger_tokens']):raise ValueError('Actual controller timing differs from registered arm')
        self.calls.append({'start':start,'length':ids.shape[1],'prefill':cache is None,
            'input_ids_sha256':data_fingerprint(ids.tolist()[0]),
            'cached_input_ids':ids.tolist()[0] if cache is not None else None,
            'logits_to_keep':kwargs['logits_to_keep'],
            'active_interval':active_interval(start,ids.shape[1],self.query['ledger_tokens'],self.policy),
            'unrestricted_inputs':True,'rope_positions_exact':False,'projections':{}})
    def positions(self,module,args,kwargs):
        import torch
        positions=kwargs.get('position_ids',args[1] if len(args)>1 else None);call=self.calls[-1]
        expected=torch.arange(call['start'],call['start']+call['length'],device=positions.device).view(1,1,-1).expand(3,1,-1)
        if not torch.equal(positions,expected) or call['rope_positions_exact']:raise ValueError('Original actual RoPE positions changed')
        call.update(rope_positions_exact=True,rope_positions_sha256=tensor_hash(positions),rope_positions_shape=list(positions.shape))
    def capture(self,name):
        def hook(module,args,output):
            if name in self.before_values:raise ValueError('Projection capture overlap')
            self.before_values[name]=output.detach().clone()
        return hook
    def observe(self,name):
        def hook(module,args,output):
            import torch
            original=self.before_values.pop(name);call=self.calls[-1];start=call['start'];length=call['length']
            if list(output.shape)!=[1,length,self.targets[name]] or output.dtype!=original.dtype:
                raise ValueError('Actual original projection layout changed')
            if self.policy!='all' and int(self.controller._position_start)!=start:raise ValueError('Controller uses a different cached position')
            low,high=call['active_interval'];positions=torch.arange(start,start+length,device=output.device)
            active=(positions>=low)&(positions<high);inactive=~active
            if not torch.equal(original[:,inactive],output[:,inactive]):raise ValueError('Adapter leaked outside registered token positions')
            changed=(original!=output).any(-1)[0]
            if (changed&inactive).any():raise ValueError('Actual inactive projection rows changed')
            call['projections'][name]={'shape':list(output.shape),'dtype':str(output.dtype),
                'active_rows':int(active.sum().item()),'inactive_rows':int(inactive.sum().item()),
                'inactive_rows_exact':True,'changed_active_rows':int((changed&active).sum().item())}
        return hook
    def after(self,module,args,kwargs,result):
        call=self.calls[-1]
        if (not call['rope_positions_exact'] or set(call['projections'])!=set(self.targets) or self.before_values or
            int(result.past_key_values.get_seq_length())!=call['start']+call['length']):
            raise ValueError('Actual projection/position/cache-forward observation incomplete')
        call['end_seq_length']=int(result.past_key_values.get_seq_length())
    def __enter__(self):return self
    def __exit__(self,*_):
        for handle in self.handles:handle.remove()
        self.before_values.clear()


def audit_policy(calls,query,policy,targets):
    import torch
    from scripts.analyze_fresh_layer_localization import sha
    if not calls:raise ValueError('Actual unrestricted forward receipts required')
    gold=query['answer_ids'][:-1]
    if not 0<=len(calls)-1-int(bool(gold))<=12:raise ValueError('Original greedy generation limit changed')
    expected_prefill={'start':0,'length':query['prompt_tokens'],'prefill':True,
        'input_ids_sha256':query['prompt_ids_sha256'],'cached_input_ids':None,'logits_to_keep':1}
    if any(calls[0][k]!=v for k,v in expected_prefill.items()):raise ValueError('Original full-prompt prefill changed')
    if gold and (len(calls)<2 or calls[1]['cached_input_ids']!=gold or calls[1]['start']!=query['prompt_tokens'] or
        calls[1]['length']!=len(gold) or calls[1]['logits_to_keep']!=len(gold)):
        raise ValueError('Original singleton gold-scoring forward changed')
    for i,call in enumerate(calls):
        start,length=call['start'],call['length'];low,high=active_interval(start,length,query['ledger_tokens'],policy)
        sha(call['input_ids_sha256']);sha(call['rope_positions_sha256'])
        expected_positions=torch.arange(start,start+length).view(1,1,-1).expand(3,1,-1)
        if (call['active_interval']!=[low,high] or not call['unrestricted_inputs'] or not call['rope_positions_exact'] or
            call['rope_positions_shape']!=[3,1,length] or call['rope_positions_sha256']!=tensor_hash(expected_positions) or call['end_seq_length']!=start+length or
            set(call['projections'])!=set(targets) or (i>0 and call['prefill'])):
            raise ValueError('Policy interval, natural positions or actual projection coverage changed')
        if i>0:
            ids=call['cached_input_ids']
            if len(ids)!=length or data_fingerprint(ids)!=call['input_ids_sha256']:raise ValueError('Cached IDs contradict actual hash')
        greedy_index=i-1-int(bool(gold))
        if greedy_index>=0 and (start!=query['prompt_tokens']+greedy_index or length!=1 or call['logits_to_keep']!=1):
            raise ValueError('Actual greedy forward positions changed')
        for name,p in call['projections'].items():
            if (p['shape']!=[1,length,targets[name]] or p['dtype']!='torch.bfloat16' or
                p['active_rows']!=high-low or p['inactive_rows']!=length-(high-low) or
                not p['inactive_rows_exact'] or not 0<=p['changed_active_rows']<=p['active_rows']):
                raise ValueError('Actual inactive projection rows or dimensions changed')


def verify_prefix(runtime,controller,ledger,query):
    import torch
    hooks=hook_state(runtime.model);controller.set_token_policy('none',query['ledger_tokens'])
    native,first,count=runtime.advance(ledger)
    controller.set_token_policy('question',query['ledger_tokens']);actual,other,actual_count=runtime.advance(ledger)
    assert_complete_cache_equal(native,actual)
    a,b=cache_fingerprint(native),cache_fingerprint(actual)
    if (count!=query['ledger_tokens'] or actual_count!=count or a!=b or
        complete_metadata(native)!=complete_metadata(actual) or not torch.equal(first,other) or
        tensor_hash(first)!=tensor_hash(other) or hook_state(runtime.model)!=hooks):
        raise ValueError('Question-only timing altered complete native ledger prefix')
    return {'source_key':query['source_key'],'ledger_tokens':count,'native_prefix_sha256':a,'actual_prefix_sha256':b,
        'native_prefix_logits_sha256':tensor_hash(first),'actual_prefix_logits_sha256':tensor_hash(other),
        'complete_cache_bookkeeping_and_logits_exact':True,'all48_tensor_fields_exact':True,'hooks_restored':True}


def checked_gate(path,dataset_hash):
    from scripts.analyze_unseen_shared_timing import audit_equivalence
    path=Path(path);report=json.loads(path.read_text())
    actual=audit_equivalence(Path(report['forward_audit']),Path(report['reverse_audit']))
    if (actual!=report or report['dataset_manifest_sha256']!=dataset_hash or
        streaming_hash(path.with_suffix('.source.py'))!=report['analysis_source_sha256']):
        raise ValueError('Both180-readout smoke audits and exact order equivalence required')
    return {'path':str(path),'sha256':streaming_hash(path)}
