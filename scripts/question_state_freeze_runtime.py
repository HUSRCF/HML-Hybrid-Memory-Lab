"""Question-only zero gates with exact matrix, KV and cleanup observations."""
from contextlib import ExitStack
import json
import math

from scripts.oracle_write_hold_runtime import RecurrentHold,GDN_LAYERS,kernel_identity
from scripts.post_encoding_runtime import assert_isolated
from scripts.copy_readout_runtime import SelectionWeights,readout
from scripts.fresh_layer_controls import tensor_hash
from scripts.run_write_locality_load_screen import hook_state
from sas_memory.runtime import cache_tensors
from sas_memory.training_data import data_fingerprint


def canonical(value):
    """JSON representation only; no rounding, tolerance or source mutation."""
    def check(item):
        if isinstance(item,dict):
            keys=set()
            for key,child in item.items():
                if type(key) not in (str,int) or str(key) in keys:
                    raise ValueError('Unsupported or colliding JSON observation keys')
                keys.add(str(key));check(child)
        elif isinstance(item,(tuple,list)):
            for child in item:check(child)
    check(value)
    return json.loads(json.dumps(value,allow_nan=False))


def tensor_record(tensor):
    return {'sha256':tensor_hash(tensor),'shape':list(tensor.shape),'dtype':str(tensor.dtype),
        'stride':list(tensor.stride()),'storage_offset':tensor.storage_offset()}


def full_attention_contract(stats):
    if set(map(int,stats))!={3,7,11,15,19,23}:raise ValueError('All six native Full readers required')
    for s in stats.values():
        if s['prefill_calls']!=0 or s['cached_calls']<2 or s['query_rows']<26:
            raise ValueError('Actual cached question and answer reading required')
        for key in ('max_suppressed_probability','max_suppressed_mass','max_kept_record_mass'):
            if not math.isfinite(s[key]) or s[key]<0 or (key=='max_suppressed_probability' and s[key]>1):
                raise ValueError('Finite native probabilities required')
    # Zero support probability is a possible scientific failure, not an audit gate.


class QuestionHold(RecurrentHold):
    def __init__(self,model,source,known_ids,selected,source_record):
        self.selected=tuple(selected);self.boundary=source.get_seq_length();self.known_ids=list(known_ids)
        if (len(self.known_ids)!=24 or len(set(self.selected))!=len(self.selected) or
            not set(self.selected)<=set(GDN_LAYERS)):
            raise ValueError('Explicit GDN subset and unchanged24known input tokens required')
        self.source=source;self.source_record=source_record;self.question_calls=0;self.final_proof=None
        self.pre_states={};self.saved_hooks=hook_state(model);self.closed=False
        try:
            super().__init__(model,[True]*self.boundary+[False]*24,enabled=bool(self.selected))
            self.handles.append(model.register_forward_hook(self.after,with_kwargs=True))
        except BaseException:
            if hasattr(self,'handles'):self.close()
            raise

    def before(self,module,args,kwargs):
        super().before(module,args,kwargs)
        cache=kwargs.get('past_key_values');ids=kwargs['input_ids']
        if (self.question_calls or cache is None or cache.get_seq_length()!=self.boundary or
            ids.shape!=(1,24) or ids[0].tolist()!=self.known_ids):
            raise ValueError('Exactly one known-question cached forward, no answers or history')
        assert_isolated(self.source,cache)
        import torch
        for index in self.selected:
            actual=cache.layers[index].recurrent_states[0];old=self.source.layers[index].recurrent_states[0]
            if actual.dtype!=old.dtype or not torch.equal(actual,old):raise ValueError('Question starts from a changed recurrent matrix')
            self.pre_states[index]=actual.detach().clone()
        self.question_calls+=1

    def projection(self,index,kind,gdn):
        def hook(module,args,output):
            import torch
            _,length,count,mask=self.current
            if output.ndim!=3 or output.shape[:2]!=(1,length):raise ValueError('Native question gate layout changed')
            native=(output.sigmoid() if kind=='b' else
                -gdn.A_log.float().exp()*torch.nn.functional.softplus(output.float()+gdn.dt_bias))
            active=index in self.selected and count
            self.expected[(index,kind)]=native.masked_fill(mask,0.) if active else native
            return output.masked_fill(mask,float('-inf')) if active else output
        return hook

    def kernel(self,index,name,original):
        def wrapper(*args,**kwargs):
            import torch
            g,b=kwargs['g'],kwargs['beta'];_,length,count,mask=self.current
            self.valid &= (torch.isfinite(g).all() & torch.isfinite(b).all() &
                (g==self.expected.pop((index,'a'))).all() & (b==self.expected.pop((index,'b'))).all())
            held=count if index in self.selected else 0
            if held:self.valid &= ((g.masked_select(mask.expand_as(g))==0).all() &
                (b.masked_select(mask.expand_as(b))==0).all())
            s=self.stats[index];s['chunk_calls' if name.startswith('chunk') else 'recurrent_calls']+=1
            s['rows']+=length;s['held_rows']+=held;s['unheld_rows']+=length-held
            return original(*args,**kwargs)
        return wrapper

    def after(self,module,args,kwargs,output):
        import torch
        actual=output.past_key_values
        if actual.get_seq_length()!=self.boundary+24:raise ValueError('Question did not advance at its original position')
        gates=canonical(super().report());before={};after={}
        for index,saved in self.pre_states.items():
            final=actual.layers[index].recurrent_states[0]
            if final.dtype!=saved.dtype or final.shape!=saved.shape or not torch.equal(final,saved):
                raise ValueError('Selected actual recurrent matrix changed despite zero gates')
            before[str(index)]=tensor_record(saved);after[str(index)]=tensor_record(final)
            if before[str(index)]['sha256']!=after[str(index)]['sha256']:
                raise ValueError('Selected recurrent matrix bytes changed')
        history={};left=dict(cache_tensors(self.source));right=dict(cache_tensors(actual))
        for key,value in left.items():
            if key[1] not in ('keys','values'):continue
            final=right[key][...,:self.boundary,:]
            if value.dtype!=final.dtype or not torch.equal(value,final):raise ValueError('Historical KV changed during question')
            # Equality above proves the actual slice has the source's byte hash.
            history[str(key)]=self.source_record['tensors'][str(key)]['sha256']
        if len(history)!=12:raise ValueError('All12historical KV fields required')
        changed=[i for i in GDN_LAYERS if not torch.equal(self.source.layers[i].conv_states[0],actual.layers[i].conv_states[0])]
        self.close()
        kernels_restored=all(getattr(gdn,name)==original for gdn,name,original in self.originals)
        if hook_state(self.model)!=self.saved_hooks or not kernels_restored:
            raise ValueError('Gate hooks and kernels must be restored before any answer computation')
        self.final_proof={'question_calls':self.question_calls,'boundary':self.boundary,'question_tokens':24,
            'known_ids_sha256':data_fingerprint(self.known_ids),'selected_layers':list(self.selected),
            'gates':gates,'selected_before':before,'selected_after':after,'historical_kv_sha256':history,
            'convolution_changed_layers':changed,'hooks_removed_before_answer':True,
            'kernel_methods_restored_before_answer':True,'source_and_work_storage_isolated':True,
            'no_direct_state_copy_or_rollback':True,'policy_uses_no_answers_or_support_labels':True}

    def close(self):
        if not self.closed:
            super().close();self.closed=True

    def report(self):
        if self.final_proof is None or not self.closed:raise ValueError('Complete question-end proof required')
        return canonical(self.final_proof)


def measure_question(runtime,context,cache,query,layers,source_record):
    if cache.get_seq_length()!=query['ledger_tokens'] or context['id']!=query['context_id']:
        raise ValueError('Registered normal-cache query boundary required')
    boundary=query['ledger_tokens'];active=hook_state(runtime.model)
    keep=[b=='1' for b in query['history_keep_bits']]
    with ExitStack() as stack:
        observer=stack.enter_context(SelectionWeights(runtime.model,{'actual':keep},query['prelude_tokens']))
        hold=stack.enter_context(QuestionHold(runtime.model,cache,query['input_ids'][boundary:],layers,source_record))
        value=readout(runtime.probe(context['ledger'],cache,query['question'],query['answer'],[query['answer']],
            generate=True,prefill='cached',score_batch=False))
        question_proof=hold.report();stats=canonical(observer.stats['actual']);full_attention_contract(stats)
    if hook_state(runtime.model)!=active or value['query_tokens']!=24:raise ValueError('Full reader cleanup or known suffix changed')
    return value,{'reader_calls':None,'attention':stats,'reader_keep_sha256':data_fingerprint(query['history_keep_bits']),
        'question_hold':question_proof}


def freeze_kernel_contract(runtime):
    """Existing all-zero tests plus actual24token chunk before task inference."""
    import torch
    checks=kernel_identity(runtime);gdn=runtime.model.model.layers[0].linear_attn
    generator=torch.Generator(device=runtime.model.device).manual_seed(53129)
    for name,length in (('chunk_gated_delta_rule',24),('recurrent_gated_delta_rule',1)):
        shape=(1,length,gdn.num_v_heads)
        q=torch.randn((*shape,gdn.head_k_dim),dtype=torch.bfloat16,device=runtime.model.device,generator=generator)
        k=torch.randn(q.shape,dtype=q.dtype,device=q.device,generator=generator)
        v=torch.randn((*shape,gdn.head_v_dim),dtype=q.dtype,device=q.device,generator=generator)
        initial=torch.randn((1,gdn.num_v_heads,gdn.head_k_dim,gdn.head_v_dim),device=q.device,generator=generator)
        saved=initial.clone()
        output,final=getattr(gdn,name)(q,k,v,g=torch.zeros(shape,device=q.device),
            beta=torch.zeros(shape,dtype=q.dtype,device=q.device),initial_state=initial,
            output_final_state=True,use_qk_l2norm_in_kernel=True)
        if (not torch.equal(saved,initial) or final.dtype!=initial.dtype or
            not torch.equal(saved,final) or not torch.isfinite(output).all()):
            raise ValueError('Actual native zero-gate state identity failed before tasks')
        checks.append({'kernel':name,'length':length,'initial_state_unmodified':True,
            'final_state_exact':True,'output_finite':True})
    return checks
