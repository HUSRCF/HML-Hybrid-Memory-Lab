import sys,json
from pathlib import Path
sys.path.insert(0,'/home/husrcf/Code/SAS')
import torch
from safetensors.torch import load_file
from sas_memory.runtime import TextRuntime
from sas_memory.controllers import GroupController
from scripts.copy_necessity_data import registered_inputs,read_plan,selected_inputs
from scripts.run_copy_dependency_transfer import actual_prompt
from scripts.diagnose_copy_arm_replay import trace,RECORDS,packet

torch.set_num_threads(4)
r,plan,cases,registry,masks,full=registered_inputs();runtime=TextRuntime(read_plan()['model_path'],'it');spec=plan['models']['attention_only'];c=GroupController(runtime.model,spec['group_index'],spec['rank'],spec['mode'],adapt_gdn_keys=spec['adapt_gdn_keys']);c.load_state_dict(load_file(spec['checkpoint'],device=str(runtime.model.device)),strict=True);c.eval().requires_grad_(False);q=selected_inputs(registry,True)[0][3];ledger=actual_prompt(runtime,{v['id']:v for v in cases},q);c.set_token_policy(spec['token_policy'],q['ledger_tokens']);expected=next(v for v in full['attention_only'] if v['id']==q['id'])
with torch.inference_mode():
 packet(runtime,c,ledger,q,expected,'cold',3)
 trace(runtime,c,ledger,q,'cold_trace')
c.close()
p=Path('/media/990Pro/Code/SAS/artifacts/memory_update/copy-cold-activation-trace-20261114-01.json');p.write_text(json.dumps(RECORDS,indent=2)+'\n');p.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
