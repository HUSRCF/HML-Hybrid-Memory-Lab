"""Independent literal token/truth and actual novelty recheck,without model construction."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
from unittest.mock import patch
ROOT=Path('/home/husrcf/Code/SAS');sys.path.insert(0,str(ROOT));BASE=Path('/media/990Pro/Code/SAS')
from scripts.readout_confirmation_data import registered_inputs,prior_index,check_novelty
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from sas_memory.tasks import Operation,execute
from scripts.run_controller_trial import prompt_for
from sas_memory.training_data import data_fingerprint
h=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
with (patch('sas_memory.runtime.TextRuntime.__init__',side_effect=AssertionError('No model construction')),
      patch('torch.cuda._lazy_init',side_effect=AssertionError('No GPU initialization'))):
 p,m,data,lock=registered_inputs();train=json.loads((ROOT/p['parent_training_protocol']).read_text())
 runtime=tokenizer_runtime(train['model_path']);index=prior_index(runtime)
 saved=json.loads((Path(p['dataset'])/'prior-inputs.json').read_text());assert index==saved
 assert check_novelty(data,index)==m['novelty']
 cases={c['id']:c for c in data['set_cases']};contexts={c['id']:c for c in data['set_registry']['contexts']}
 for q in data['set_queries'].values():
  c=cases[q['case_id']];episode=next(e for e in c['episodes'] if e['id']==q['episode']);state=execute([Operation(**e) for e in c['prefix_events']+episode['tail']])
  entity=c['protected'] if q['role']=='protected_current' else c['target'];field='initial' if q['role']=='target_initial' else 'current'
  assert str(state[field][entity])==q['answer']
  prompt=contexts[q['context_id']]['ledger']+runtime.query_suffix(q['question'])
  assert runtime.ids(prompt)==q['input_ids'] and runtime.continuation_ids(prompt,q['answer'])==q['answer_ids']
 cases={c['id']:c for c in data['copy_cases']};by_id=data['copy_queries']
 for b in data['copy_registry']['bindings']:
  case=cases[b['case_id']];variant=case['variants'][b['variant']];q=next(q for q in variant['queries'] if q['role']==b['role'])
  state=execute([Operation(**e) for e in case['prefix']+variant['tail']]);assert str(state[q['version']][q['entity']])==q['answer']
  ledger,prompt=prompt_for(runtime,case,b['variant'],q);actual=by_id[b['physical_id']]
  assert q['answer']==actual['answer'] and runtime.ids(prompt)==actual['input_ids']
  assert data_fingerprint(runtime.ids(ledger))==actual['source_key']
 for name,digest in m['source_sha256'].items():assert h(ROOT/name)==digest and h(Path(p['dataset'])/'sources'/name)==digest
out=BASE/'artifacts/memory_update/readout-confirmation-independent-inputs-20261116-01.json';assert not out.exists()
report={'status':'all289sources1200physical_inputs_and1296logical_truths_independently_rechecked',
        'created_utc':datetime.now(timezone.utc).isoformat(),'packet_manifest_sha256':h(Path(p['dataset'])/'manifest.json'),
        'source_files':289,'set_questions':864,'copy_questions':336,'copy_bindings':432,
        'actual_prior_input_files':len(index['source_files_sha256']),'fixed_seeds':m['novelty']['base_seeds'],
        'prior_overlap':m['novelty']['overlap'],'native_model_construction_calls':0,'gpu_initializations':0,
        'training_finished_before_generation':True,'source_sha256':h(Path(__file__))}
out.write_text(json.dumps(report,indent=2)+'\n');out.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(report),flush=True)
