import json, sys
from pathlib import Path
sys.path.insert(0, '/home/husrcf/Code/SAS')
import torch
from safetensors.torch import load_file
from sas_memory.controllers import GroupController
from sas_memory.runtime import TextRuntime
from scripts.copy_readout_data import registered_inputs, read_plan, selected_inputs
from scripts.copy_readout_runtime import SelectionWeights, readout
from scripts.run_copy_dependency_transfer import actual_prompt
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_load_screen import hook_state

torch.set_num_threads(4)
registered, plan, cases, registry, masks, full = registered_inputs()
runtime = TextRuntime(read_plan()['model_path'], 'it')
name = 'attention_only'
spec = plan['models'][name]
controller = GroupController(runtime.model, spec['group_index'], spec['rank'], spec['mode'], adapt_gdn_keys=spec['adapt_gdn_keys'])
controller.load_state_dict(load_file(spec['checkpoint'], device=str(next(controller.parameters()).device)), strict=True)
controller.eval().requires_grad_(False)
physical,_ = selected_inputs(registry, True)
case_map = {c['id']:c for c in cases}
rows = {r['id']:r for r in full[name]}
report=[]
with torch.inference_mode():
 for q in physical[:6]:
  ledger = actual_prompt(runtime, case_map, q)
  controller.set_token_policy(spec['token_policy'], q['ledger_tokens'])
  expected=readout(rows[q['id']])
  results=[]
  for repeat in range(3):
   for observed in [False, True]:
    original_hooks=hook_state(runtime.model)
    if observed:
     keeps={r:[x=='1' for x in s] for r,s in masks[q['id']]['keep_bits'].items()}
     with SelectionWeights(runtime.model, keeps, q['prelude_tokens']):
      result=measure(runtime,ledger,q)
    else: result=measure(runtime,ledger,q)
    results.append(dict(repeat=repeat,observed=observed,result=result,diff={k:[expected[k],v] for k,v in result.items() if expected[k]!=v},hooks_restored=hook_state(runtime.model)==original_hooks))
  entry=dict(id=q['id'],expected=expected,replays=results)
  report.append(entry)
  print(json.dumps(entry),flush=True)
controller.close()
out=Path('/media/990Pro/Code/SAS/artifacts/memory_update/copy-readout-replay-diagnostic-20261114-01.json')
out.write_text(json.dumps(report,indent=2)+'\n')
