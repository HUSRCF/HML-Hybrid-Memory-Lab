import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/home/husrcf/Code/SAS')
import torch
from safetensors.torch import load_file
from sas_memory.runtime import TextRuntime
from sas_memory.controllers import GroupController
from scripts.copy_necessity_data import read_plan
from scripts.run_copy_dependency_transfer import actual_prompt
from scripts.run_oracle_kv_readout import measure

def sha(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
torch.set_num_threads(4);p=read_plan();sp=json.loads(Path(p['source_transfer_plan']).read_text());d=Path(p['source_dataset']);reg=json.loads((d/'registry.json').read_text());cases={c['id']:c for c in json.loads((d/'cases.json').read_text())};q=next(v for v in reg['physical'] if v['id']=='571aa66bb133a769f6f54a82e54ffe78da046507b55347a5946b46be66d593bf');runtime=TextRuntime(p['model_path'],'it');s=sp['models']['attention_only'];c=GroupController(runtime.model,s['group_index'],s['rank'],s['mode'],adapt_gdn_keys=s['adapt_gdn_keys']);c.load_state_dict(load_file(s['checkpoint'],device=str(runtime.model.device)),strict=True);c.eval().requires_grad_(False);c.set_token_policy(s['token_policy'],q['ledger_tokens']);captures={};handles=[]
def obs(k):
 def after(m,args,out):
  if k not in captures:captures[k]={'input':args[0].detach().cpu().contiguous(),'layout':{'input_stride':list(args[0].stride()),'weight_stride':list(m.weight.stride()),'output_stride':list(out.stride()),'input_offset':args[0].storage_offset(),'weight_offset':m.weight.storage_offset(),'output_sha256':sha(out),'input_sha256':sha(args[0])}}
 return after
for k in ['q','k']:handles.append(c.outputs[f'attention_{k}_proj'].register_forward_hook(obs(k)))
with torch.inference_mode():
 result=measure(runtime,actual_prompt(runtime,cases,q),q)
 for h in handles:h.remove()
 report={'readout':result,'layouts':{k:v['layout'] for k,v in captures.items()},'trials':{}}
 for k,cap in captures.items():
  ctrl=cap['input'];module=c.outputs[f'attention_{k}_proj'];w=module.weight;rows=[]
  for input_layout in ['contiguous','column','recorded']:
   if input_layout=='contiguous':x=ctrl.to(runtime.model.device).contiguous()
   elif input_layout=='column':x=ctrl.transpose(-1,-2).contiguous().to(runtime.model.device).transpose(-1,-2)
   else:
    x=torch.empty_strided(ctrl.shape,cap['layout']['input_stride'],device=runtime.model.device);x.copy_(ctrl)
   for weight_layout in ['parameter','row_clone','column_clone']:
    y=w if weight_layout=='parameter' else (w.clone().contiguous() if weight_layout=='row_clone' else w.t().contiguous().t())
    out=torch.nn.functional.linear(x,y);rows.append({'input_layout':input_layout,'weight_layout':weight_layout,'input_stride':list(x.stride()),'weight_stride':list(y.stride()),'sha256':sha(out),'matches_captured':sha(out)==cap['layout']['output_sha256']})
  report['trials'][k]=rows
c.close();out=Path('/media/990Pro/Code/SAS/artifacts/memory_update/attention-projection-layout-20261114-01.json');out.write_text(json.dumps(report,indent=2)+'\n');out.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(report),flush=True)
