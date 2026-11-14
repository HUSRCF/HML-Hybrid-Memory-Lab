import sys,json,gc
from pathlib import Path
sys.path.insert(0,'/home/husrcf/Code/SAS')
import torch
from safetensors.torch import load_file
from sas_memory.runtime import TextRuntime
from sas_memory.controllers import GroupController
from sas_memory.kv_readout import QuestionKVReadout
from scripts.copy_necessity_data import registered_inputs,read_plan,selected_inputs,REGIMES
from scripts.copy_readout_runtime import prove_and_measure,readout,SelectionWeights
from scripts.run_oracle_kv_readout import measure
from scripts.run_copy_dependency_transfer import actual_prompt
from scripts.run_controller_trial import state_fingerprint
from scripts.semantic_order_path_common import cache_fingerprint

torch.set_num_threads(4)
registered,source_plan,cases,registry,masks,full=registered_inputs()
physical,_=selected_inputs(registry,True);cm={c['id']:c for c in cases};name='attention_only';spec=source_plan['models'][name];refs={r['id']:r for r in full[name]}
runtime=TextRuntime(read_plan()['model_path'],'it')
controller=GroupController(runtime.model,spec['group_index'],spec['rank'],spec['mode'],adapt_gdn_keys=spec['adapt_gdn_keys'])
controller.load_state_dict(load_file(spec['checkpoint'],device=str(next(controller.parameters()).device)),strict=True);controller.eval().requires_grad_(False)
q=physical[3];ledger=actual_prompt(runtime,cm,q);expected=readout(refs[q['id']]);results=[]
def probe(stage,repeat):
 controller.set_token_policy(spec['token_policy'],q['ledger_tokens']);actual=measure(runtime,ledger,q)
 row={'stage':stage,'repeat':repeat,'id':q['id'],'expected':expected,'actual':actual,'diff':{k:[expected[k],v] for k,v in actual.items() if v!=expected[k]}}
 results.append(row);print(json.dumps(row),flush=True)
with torch.inference_mode():
 for i in range(3):probe('cold',i)
 initial_sha=state_fingerprint(controller)
 for r in range(2):
  for i,p in enumerate(physical[:3]):
   text=actual_prompt(runtime,cm,p);controller.set_token_policy(spec['token_policy'],p['ledger_tokens']);keeps={k:[x=='1' for x in masks[p['id']]['keep_bits'][k]] for k in REGIMES}
   if i==0:prove_and_measure(runtime,text,p,keeps,refs[p['id']])
   else:
    for k,keep in keeps.items():
     with QuestionKVReadout(runtime.model,keep):measure(runtime,text,p)
  for i in range(3):probe('after_three_masked_queries_'+str(r),i)
 gc.collect();torch.cuda.empty_cache()
 for i in range(3):probe('after_empty_cache',i)
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 for i in range(3):probe('deterministic',i)
 report={'results':results,'adapter_unchanged':initial_sha==state_fingerprint(controller),'float32_matmul_precision':torch.get_float32_matmul_precision(),'allow_tf32':torch.backends.cuda.matmul.allow_tf32}
controller.close()
out=Path('/media/990Pro/Code/SAS/artifacts/memory_update/copy-necessity-sequence-diagnostic-20261114-01.json');out.write_text(json.dumps(report,indent=2)+'\n');out.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({'complete':True,'all_exact':all(not r['diff'] for r in results),'adapter_unchanged':report['adapter_unchanged']}),flush=True)
