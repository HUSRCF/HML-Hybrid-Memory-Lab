"""Separate native log-decay and Delta gates without editing earlier sources."""
from contextlib import ExitStack
from scripts.oracle_write_hold_runtime import RecurrentHold,GDN_LAYERS

VALIDATION_ADDENDUM_SHA256='1eed068b08f6c006b390f553380dfba99843b93d247b276d6101d67444a7b08d'

def validation_addendum():
    """Keep failed ideal checks and pin the explicit native-precision correction."""
    import json
    from pathlib import Path
    from sas_memory.write_locality_data import streaming_hash
    p=Path(__file__).resolve().parents[1]/'configs/gate_component_kernel_reference_addendum.json'
    if streaming_hash(p)!=VALIDATION_ADDENDUM_SHA256:raise ValueError('Validation-only addendum changed')
    d=json.loads(p.read_text());failed=Path(d['failed_run']);probe=Path(d['probe'])
    if streaming_hash(failed/'manifest.json')!=d['failed_manifest_sha256']:raise ValueError('Preserve first failed smoke')
    m=json.loads((failed/'manifest.json').read_text())
    if m['status']!='failed' or m['native_anchors'] or m['models']:raise ValueError('Failure must precede all task inference')
    for n,h in d['failed_sources_sha256'].items():
        if streaming_hash(failed/'sources'/n)!=h:raise ValueError('Preserve every failed-version source byte')
    if streaming_hash(probe/'receipt.json')!=d['probe_receipt_sha256']:raise ValueError('Numerical-only probe changed')
    receipt=json.loads((probe/'receipt.json').read_text())
    if receipt['task_inputs_evaluated']!=0 or not all(c['precision_passed'] for c in receipt['checks']):raise ValueError('Probe mismatch')
    for n,h in receipt['files_sha256'].items():
        if streaming_hash(probe/n)!=h:raise ValueError('Preserve all raw numerical probe tensors')
    for n,h in d['installed_fla_sources_sha256'].items():
        if streaming_hash(Path(n))!=h:raise ValueError('Native FLA implementation changed')
    return d

SETTINGS={'full':('none',False),'hold':('both',False),'oracle':('none',True),'hold_oracle':('both',True),
    'decay_full':('decay',False),'delta_full':('delta',False),'decay_oracle':('decay',True),'delta_oracle':('delta',True)}

class ComponentHold(RecurrentHold):
    def __init__(self,model,keep,mode):
        if mode not in ('none','decay','delta','both'):raise ValueError('Explicit native/decay/Delta/both gate policy required')
        self.mode=mode;super().__init__(model,keep,enabled=mode!='none')
        for s in self.stats.values():s.update(forced_decay_rows=0,forced_delta_rows=0)

    def projection(self,index,kind,gdn):
        def hook(module,args,output):
            import torch
            _,length,count,mask=self.current
            if output.ndim!=3 or output.shape[:2]!=(1,length):raise ValueError('Unexpected native gate layout')
            active=self.mode=='both' or (kind=='a' and self.mode=='decay') or (kind=='b' and self.mode=='delta')
            original=(output.sigmoid() if kind=='b' else
                -gdn.A_log.float().exp()*torch.nn.functional.softplus(output.float()+gdn.dt_bias))
            self.expected[(index,kind)]=original.masked_fill(mask,0.) if count and active else original
            return output.masked_fill(mask,float('-inf')) if count and active else output
        return hook

    def kernel(self,index,name,original):
        def wrapper(*args,**kwargs):
            import torch
            g,b=kwargs['g'],kwargs['beta'];_,length,count,mask=self.current
            self.valid &= (torch.isfinite(g).all() & torch.isfinite(b).all() &
                (g==self.expected.pop((index,'a'))).all() & (b==self.expected.pop((index,'b'))).all())
            stat=self.stats[index]
            if count and self.mode in ('decay','both'):
                self.valid &= (g.masked_select(mask.expand_as(g))==0).all();stat['forced_decay_rows']+=count
            if count and self.mode in ('delta','both'):
                self.valid &= (b.masked_select(mask.expand_as(b))==0).all();stat['forced_delta_rows']+=count
            stat['chunk_calls' if name.startswith('chunk') else 'recurrent_calls']+=1
            stat['rows']+=length;stat['held_rows']+=count;stat['unheld_rows']+=length-count
            return original(*args,**kwargs)
        return wrapper

    def report(self):return {**super().report(),'component_mode':self.mode}

def component_kernel_contract(runtime):
    """Native kernels against explicit pure-decay and one-hot Delta algebra."""
    import torch
    correction=validation_addendum()
    gdn=runtime.model.model.layers[0].linear_attn;device=runtime.model.device
    generator=torch.Generator(device=device).manual_seed(31297);result=[]
    with torch.inference_mode():
        for kernel,length in [('chunk_gated_delta_rule',1),('chunk_gated_delta_rule',7),('chunk_gated_delta_rule',65),
            ('recurrent_gated_delta_rule',1),('recurrent_gated_delta_rule',7)]:
            shape=(1,length,gdn.num_v_heads);fn=getattr(gdn,kernel)
            q=torch.randn((*shape,gdn.head_k_dim),device=device,dtype=torch.bfloat16,generator=generator)
            k=torch.randn(q.shape,device=device,dtype=q.dtype,generator=generator)
            v=torch.randn((*shape,gdn.head_v_dim),device=device,dtype=q.dtype,generator=generator)
            initial=torch.randn((1,gdn.num_v_heads,gdn.head_k_dim,gdn.head_v_dim),device=device,dtype=torch.float32,generator=generator)
            saved=initial.clone()
            for mode in ('pure_decay','pure_delta'):
                if mode=='pure_decay':
                    keys=k;g=torch.full(shape,-.01,device=device);beta=torch.zeros(shape,device=device,dtype=q.dtype)
                    expected=initial*g.sum(1).exp()[...,None,None]
                else:
                    keys=torch.zeros_like(k)
                    for t in range(length):keys[:,t,:,t%gdn.head_k_dim]=1
                    g=torch.zeros(shape,device=device);beta=torch.full(shape,.5,device=device,dtype=q.dtype);expected=initial.clone()
                    for t in range(length):
                        j=t%gdn.head_k_dim;expected[:,:,j,:]=.5*expected[:,:,j,:]+.5*v[:,t].float()
                output,final=fn(q,keys,v,g=g,beta=beta,initial_state=initial,output_final_state=True,use_qk_l2norm_in_kernel=True)
                ideal=expected.clone()
                if mode=='pure_delta' and kernel=='chunk_gated_delta_rule':
                    if length>gdn.head_k_dim:raise ValueError('Distinct one-hot Key directions required')
                    expected=initial.clone()
                    for t in range(length):
                        j=t%gdn.head_k_dim
                        delta=(.5*v[:,t].float()-.5*initial[:,:,j,:].to(v.dtype).float()).to(v.dtype).float()
                        expected[:,:,j,:]=initial[:,:,j,:]+delta
                independent=None
                if mode=='pure_decay':
                    _,other=fn(-q,-keys,-v,g=g,beta=beta,initial_state=initial,output_final_state=True,use_qk_l2norm_in_kernel=True)
                    independent=torch.equal(final,other)
                    if not independent:raise ValueError('Beta0 final state must be exactly independent of Q/K/V')
                if (not torch.equal(saved,initial) or not torch.isfinite(output).all() or
                    not torch.allclose(final,expected,atol=.00005,rtol=.0001)):
                    raise ValueError(f'Native component algebra check failed: {kernel},{length},{mode}')
                result.append({'kernel':kernel,'length':length,'mode':mode,'initial_state_unmodified':True,
                    'output_finite':True,'explicit_state_algebra_passed':True,'atol':.00005,'rtol':.0001,
                    'reference_profile':correction['profile'],'validation_addendum_sha256':VALIDATION_ADDENDUM_SHA256,
                    'original_ideal_algebra_passed':bool(torch.allclose(final,ideal,atol=.00005,rtol=.0001)),
                    'original_ideal_max_absolute_error':(final-ideal).abs().max().item(),
                    'max_absolute_error':(final-expected).abs().max().item(),'independent_of_qkv':independent})
    return result

def measure_condition(runtime,ledger,query,policy,regime,observe=False):
    from sas_memory.kv_readout import QuestionKVReadout
    from scripts.copy_readout_runtime import SelectionWeights,check_weight_stats
    from scripts.run_oracle_kv_readout import measure
    from scripts.run_write_locality_load_screen import hook_state
    mode,restricted=SETTINGS[regime];keep=[b=='1' for b in policy['history_keep_bits']];active=hook_state(runtime.model)
    with ExitStack() as stack:
        hold=stack.enter_context(ComponentHold(runtime.model,keep,mode))
        reader=stack.enter_context(QuestionKVReadout(runtime.model,keep)) if restricted else None
        observer=stack.enter_context(SelectionWeights(runtime.model,{'actual':keep},query['prelude_tokens'])) if observe else None
        value=measure(runtime,ledger,query);gates=hold.report()
        if observer:check_weight_stats(observer.stats['actual'],restricted)
        stats=observer.stats['actual'] if observer else None
    if hook_state(runtime.model)!=active:raise ValueError('Component/reader/observer cleanup failed')
    return value,{'gates':gates,'reader_calls':reader.calls if reader else None,'attention':stats}
