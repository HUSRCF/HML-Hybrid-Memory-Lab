"""Gate-only recurrent holds with passive native-kernel and cache checks."""
from contextlib import ExitStack

GDN_LAYERS = tuple(i for i in range(24) if i % 4 != 3)


class RecurrentHold:
    def __init__(self, model, keep, enabled=True):
        import torch
        if not keep or any(type(b) is not bool for b in keep):
            raise ValueError('Boolean original historical mask required')
        self.model,self.keep,self.enabled=model,list(keep),enabled
        self.current=None;self.handles=[];self.originals=[];self.expected={}
        self.valid=torch.ones((),dtype=torch.bool,device=model.device)
        self.stats={i:{'chunk_calls':0,'recurrent_calls':0,'rows':0,'held_rows':0,'unheld_rows':0} for i in GDN_LAYERS}
        self.handles.append(model.register_forward_pre_hook(self.before,with_kwargs=True))
        for i in GDN_LAYERS:
            gdn=model.model.layers[i].linear_attn
            for kind in ('a','b'):
                self.handles.append(getattr(gdn,'in_proj_'+kind).register_forward_hook(self.projection(i,kind,gdn)))
            for name in ('chunk_gated_delta_rule','recurrent_gated_delta_rule'):
                original=getattr(gdn,name);self.originals.append((gdn,name,original))
                setattr(gdn,name,self.kernel(i,name,original))

    def before(self,module,args,kwargs):
        import torch
        if args or kwargs.get('position_ids') is not None:
            raise ValueError('Original unpadded keyword inputs and natural positions required')
        ids=kwargs['input_ids'];cache=kwargs.get('past_key_values')
        past=cache.get_seq_length() if cache is not None else 0
        positions=range(past,past+ids.shape[1])
        held=[self.enabled and p<len(self.keep) and not self.keep[p] for p in positions]
        self.current=(past,ids.shape[1],sum(held),torch.tensor(held,device=ids.device,dtype=torch.bool)[None,:,None])

    def projection(self,index,kind,gdn):
        def hook(module,args,output):
            import torch
            _,length,count,mask=self.current
            if output.ndim!=3 or output.shape[:2]!=(1,length):raise ValueError('Unexpected native gate layout')
            modified=output.masked_fill(mask,float('-inf')) if count else output
            # Native formulas, including native rounding, pinned installed implementation.
            original=(output.sigmoid() if kind=='b' else
                -gdn.A_log.float().exp()*torch.nn.functional.softplus(output.float()+gdn.dt_bias))
            self.expected[(index,kind)]=original.masked_fill(mask,0.) if count else original
            return modified
        return hook

    def kernel(self,index,name,original):
        def wrapper(*args,**kwargs):
            import torch
            g,b=kwargs['g'],kwargs['beta'];_,length,count,mask=self.current
            self.valid &= (torch.isfinite(g).all() & torch.isfinite(b).all() &
                (g==self.expected.pop((index,'a'))).all() & (b==self.expected.pop((index,'b'))).all())
            if count:self.valid &= ((g.masked_select(mask.expand_as(g))==0).all() &
                (b.masked_select(mask.expand_as(b))==0).all())
            stat=self.stats[index];stat['chunk_calls' if name.startswith('chunk') else 'recurrent_calls']+=1
            stat['rows']+=length;stat['held_rows']+=count;stat['unheld_rows']+=length-count
            return original(*args,**kwargs)
        return wrapper

    def close(self):
        for handle in self.handles:handle.remove()
        for gdn,name,original in self.originals:setattr(gdn,name,original)
        self.handles=[]

    def report(self):
        if self.expected or not self.valid.item():raise ValueError('Actual native kernel gates failed exact/finite checks')
        if any(s['rows']==0 for s in self.stats.values()):raise ValueError('Missing actual18layer gate observations')
        return {'enabled':self.enabled,'actual_gates_finite_and_exact':True,'layers':self.stats}

    def __enter__(self):return self
    def __exit__(self,*_):self.close()


def kernel_identity(runtime):
    """Actual installed FLA kernels, before any task outcome is measured."""
    import torch
    gdn=runtime.model.model.layers[0].linear_attn
    generator=torch.Generator(device=runtime.model.device).manual_seed(31297)
    result=[]
    with torch.inference_mode():
        for name,length in [('chunk_gated_delta_rule',1),('chunk_gated_delta_rule',7),
            ('chunk_gated_delta_rule',65),('recurrent_gated_delta_rule',1),('recurrent_gated_delta_rule',7)]:
            shape=(1,length,gdn.num_v_heads)
            q=torch.randn((*shape,gdn.head_k_dim),device=runtime.model.device,dtype=torch.bfloat16,generator=generator)
            k=torch.randn(q.shape,device=q.device,dtype=q.dtype,generator=generator)
            v=torch.randn((*shape,gdn.head_v_dim),device=q.device,dtype=q.dtype,generator=generator)
            initial=torch.randn((1,gdn.num_v_heads,gdn.head_k_dim,gdn.head_v_dim),device=q.device,dtype=torch.float32,generator=generator)
            reference=initial.clone()
            output,final=getattr(gdn,name)(q,k,v,g=torch.zeros(shape,device=q.device),
                beta=torch.zeros(shape,device=q.device,dtype=q.dtype),initial_state=initial,
                output_final_state=True,use_qk_l2norm_in_kernel=True)
            if not torch.equal(final,reference) or not torch.equal(initial,reference) or not torch.isfinite(output).all():
                raise ValueError('Native all-held kernel does not preserve actual FP32 state exactly')
            result.append({'kernel':name,'length':length,'initial_state_unmodified':True,'final_state_exact':True,'output_finite':True})
    return result


def cached_identity(runtime,query,policy):
    """First held token at its original position; native convolution remains active."""
    import torch
    from sas_memory.runtime import fork_cache
    ids=query['prompt_ids'];position=policy['first_held_position']
    keep=[b=='1' for b in policy['history_keep_bits']]
    with torch.inference_mode(),RecurrentHold(runtime.model,keep) as hold:
        prefix=runtime.model(input_ids=runtime.tensor([ids[:position]]),use_cache=True,logits_to_keep=1).past_key_values
        states={i:prefix.layers[i].recurrent_states[0].clone() for i in GDN_LAYERS}
        conv={i:prefix.layers[i].conv_states[0].clone() for i in GDN_LAYERS}
        updated=runtime.model(input_ids=runtime.tensor([[ids[position]]]),past_key_values=fork_cache(prefix),
            use_cache=True,logits_to_keep=1).past_key_values
        if any(not torch.equal(states[i],updated.layers[i].recurrent_states[0]) for i in GDN_LAYERS):
            raise ValueError('Actual cached held token changed recurrent state')
        changed=sum(not torch.equal(conv[i],updated.layers[i].conv_states[0]) for i in GDN_LAYERS)
        kv=[i for i in range(24) if i%4==3]
        if (not changed or prefix.get_seq_length()!=position or updated.get_seq_length()!=position+1 or
            any(updated.layers[i].keys.shape[-2]!=position+1 or
                not torch.equal(prefix.layers[i].keys,updated.layers[i].keys[...,:position,:]) or
                not torch.equal(prefix.layers[i].values,updated.layers[i].values[...,:position,:]) for i in kv)):
            raise ValueError('Convolution/KV did not advance normally')
        report=hold.report()
    return {'id':query['id'],'position':position,'recurrent_states_exact':18,
        'convolution_caches_changed':changed,'kv_layers_advanced':6,'historical_kv_exact':12,'gates':report}


def measure_condition(runtime,ledger,query,policy,regime,observe=False,sham=False):
    from sas_memory.kv_readout import QuestionKVReadout
    from scripts.copy_readout_runtime import SelectionWeights,check_weight_stats
    from scripts.run_oracle_kv_readout import measure
    from scripts.run_write_locality_load_screen import hook_state
    keep=[b=='1' for b in policy['history_keep_bits']]
    active=hook_state(runtime.model);restricted=regime in ('oracle','hold_oracle')
    with ExitStack() as stack:
        hold=stack.enter_context(RecurrentHold(runtime.model,keep,enabled=regime in ('hold','hold_oracle') and not sham))
        reader=stack.enter_context(QuestionKVReadout(runtime.model,keep)) if restricted else None
        observer=stack.enter_context(SelectionWeights(runtime.model,{'actual':keep},query['prelude_tokens'])) if observe else None
        value=measure(runtime,ledger,query);gates=hold.report()
        if observer:check_weight_stats(observer.stats['actual'],restricted)
        stats=observer.stats['actual'] if observer else None
    if hook_state(runtime.model)!=active:raise ValueError('Hold/reader/observer cleanup failed')
    return value,{'gates':gates,'reader_calls':reader.calls if reader else None,'attention':stats}
