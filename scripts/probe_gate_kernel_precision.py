"""Numerical-only diagnostic after failed strict ideal-algebra check; no task input."""
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.runtime import TextRuntime
from sas_memory.write_locality_data import streaming_hash
def probe(runtime,output_dir):
    """Native kernels against explicit pure-decay and one-hot Delta algebra."""
    import torch
    gdn=runtime.model.model.layers[0].linear_attn;device=runtime.model.device
    generator=torch.Generator(device=device).manual_seed(31297);result=[]
    import numpy as np
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
                independent=None
                if mode=='pure_decay':
                    _,other=fn(-q,-keys,-v,g=g,beta=beta,initial_state=initial,output_final_state=True,use_qk_l2norm_in_kernel=True)
                    independent=torch.equal(final,other)
                    if not independent:raise ValueError('Beta0 final state must be exactly independent of Q/K/V')
                native_expected=expected.clone()
                if mode=='pure_delta' and kernel=='chunk_gated_delta_rule':
                    if length>gdn.head_k_dim:raise ValueError('Distinct one-hot Keys required')
                    native_expected=initial.clone()
                    for t in range(length):
                        j=t%gdn.head_k_dim
                        correction=(.5*v[:,t].float()-.5*initial[:,:,j,:].to(v.dtype).float()).to(v.dtype).float()
                        native_expected[:,:,j,:]=initial[:,:,j,:]+correction
                np.savez_compressed(output_dir/f'{kernel}-{length}-{mode}.npz',**{
                    n:x.float().cpu().numpy() for n,x in {'q':q,'k':keys,'v':v,'g':g,'beta':beta,
                        'initial':saved,'actual':final,'ideal_reference':expected,'precision_reference':native_expected}.items()})
                result.append({'kernel':kernel,'length':length,'mode':mode,'initial_state_unmodified':torch.equal(initial,saved),
                    'output_finite':bool(torch.isfinite(output).all()),'atol':.00005,'rtol':.0001,
                    'ideal_passed':bool(torch.allclose(final,expected,atol=.00005,rtol=.0001)),
                    'precision_passed':bool(torch.allclose(final,native_expected,atol=.00005,rtol=.0001)),
                    'precision_max_error':(final-native_expected).abs().max().item(),
                    'max_absolute_error':(final-expected).abs().max().item(),'independent_of_qkv':independent})
    return result

if __name__=='__main__':
    import torch
    torch.set_num_threads(4)
    output=Path('/media/990Pro/Code/SAS/artifacts/memory_update/gate-components-kernel-precision-probe-20261115-01')
    output.mkdir(exist_ok=False)
    (output/'source.py').write_bytes(Path(__file__).read_bytes())
    runtime=TextRuntime('/media/990Pro/Code/SAS/models/Qwen3.5-2','it')
    checks=probe(runtime,output)
    receipt={'status':'completed_numerical_probe','completed_utc':datetime.now(timezone.utc).isoformat(),
        'task_inputs_evaluated':0,'checks':checks,'source_sha256':streaming_hash(Path(__file__)),
        'files_sha256':{p.name:streaming_hash(p) for p in output.iterdir() if p.is_file()}}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
