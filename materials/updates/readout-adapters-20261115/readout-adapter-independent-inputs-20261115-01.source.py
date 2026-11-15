"""Independent literal-fit reconstruction and pre-native registration seal."""
from datetime import datetime,timezone
import hashlib,json,re
from pathlib import Path
from unittest.mock import patch
import sys
ROOT=Path('/home/husrcf/Code/SAS');sys.path.insert(0,str(ROOT))
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from sas_memory.tasks import execute,Operation
from sas_memory.training_data import data_fingerprint
BASE=Path('/media/990Pro/Code/SAS');ART=BASE/'artifacts/memory_update'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((ROOT/'configs/readout_adapter_training_plan.json').read_text());packet=Path(p['registration_destination'])
m=json.loads((packet/'manifest.json').read_text());old=Path(p['source_packet'])
queries=json.loads((old/'queries.json').read_text());cases={c['id']:c for c in json.loads((old/'cases.json').read_text())}
for name,digest in p['source_payload_sha256'].items():
    if sha(old/name)!=digest:raise ValueError('Published source input changed')
registry=json.loads((old/'registry.json').read_text());contexts={c['id']:c for c in registry['contexts']}
actual=json.loads((packet/'examples.json').read_text());expected=[]
if [k for c in registry['contexts'] for k in c['query_ids']]!=list(queries):raise ValueError('Literal cache traversal/order differs')
with (patch('sas_memory.runtime.TextRuntime.__init__',side_effect=AssertionError('No model construction')),
      patch('torch.cuda._lazy_init',side_effect=AssertionError('No GPU initialization'))):
    runtime=tokenizer_runtime(p['model_path'])
    for qid,q in queries.items():
        case=cases[q['case_id']];context=contexts[q['context_id']]
        events=case['prefix_events']+next(e['tail'] for e in case['episodes'] if e['id']==q['episode'])
        state=execute([Operation(**e) for e in events]);entity=case['protected'] if q['role']=='protected_current' else case['target']
        truth=state['initial' if q['role']=='target_initial' else 'current'][entity]
        if str(truth)!=q['answer']:raise ValueError('Independent executable truth differs')
        prompt=context['ledger']+runtime.query_suffix(q['question'])
        full=runtime.ids(prompt+str(truth)+'\n');known=q['input_ids'];answer=full[len(known):]
        if full[:len(known)]!=known or len(answer)!=4:raise ValueError('Teacher bytes or canonical length differ')
        expected.append({'id':qid,'context_id':context['id'],'case_id':case['id'],'ledger_tokens':q['ledger_tokens'],
            'known_ids':known[q['ledger_tokens']:],'answer_ids':answer,
            'teacher_ids':full[q['ledger_tokens']:-1],
            'literal_prompt_ids_sha256':data_fingerprint(known),
            'observed_source_cohort':registry['cohort_by_context'][context['id']]})
if expected!=actual or len(expected)!=1728:raise ValueError('Complete independent fit reconstruction differs')
if len(m['source_sha256'])!=277:raise ValueError('All277source files required')
for name,digest in m['source_sha256'].items():
    if sha(ROOT/name)!=digest or sha(packet/'sources'/name)!=digest:raise ValueError('Sealed source bytes differ')
for name,digest in m['payload_sha256'].items():
    if sha(packet/name)!=digest:raise ValueError('Packet bytes differ')
log=BASE/'logs/memory_update/readout-adapter-full-tests-20261115-02.log'
match=re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$',log.read_text())
if not match or int(match[1])!=360:raise ValueError('Complete unchanged360test regression must pass')
out=ART/'readout-adapter-independent-inputs-20261115-01.json';sealpath=Path(p['registration_seal'])
if out.exists() or sealpath.exists():raise ValueError('Never overwrite independent proof/seal')
receipt={'status':'all277sources_and1728observed_fit_examples_independently_verified_without_inference',
    'created_utc':datetime.now(timezone.utc).isoformat(),'examples':1728,'contexts':288,'case_name_variants':32,
    'base_streams':16,'optimized_cached_input_tokens':sum(len(e['teacher_ids']) for e in actual),
    'answer_plus_newline_tokens':sum(len(e['answer_ids']) for e in actual),'model_construction_calls':0,
    'gpu_initializations':0,'confirmation_cases_constructed':0,'source_sha256':sha(Path(__file__)),
    'packet_manifest_sha256':sha(packet/'manifest.json'),'all_observed_inputs_are_fit_not_confirmation':True}
out.write_text(json.dumps(receipt,indent=2)+'\n');out.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
seal={'status':'sealed_before_learned_readout_native_smoke','created_utc':datetime.now(timezone.utc).isoformat(),
    'packet_manifest_sha256':sha(packet/'manifest.json'),'source_sha256':m['source_sha256'],
    'full_tests_log':str(log),'full_tests_log_sha256':sha(log),'full_tests_passed':int(match[1]),
    'full_tests_seconds':float(match[2]),'independent_inputs':str(out),'independent_inputs_sha256':sha(out),
    'new_tests_log':str(BASE/'logs/memory_update/readout-adapter-new-tests-20261115-02.log'),
    'preserved_full_regression_failure_log':str(BASE/'logs/memory_update/readout-adapter-full-tests-20261115-01.log'),
    'preserved_preseal_failed_test_log':str(BASE/'logs/memory_update/readout-adapter-new-tests-20261115-01.log'),
    'native_inference_started':False,'optimizer_updates_started':False,'confirmation_generated':False}
sealpath.write_text(json.dumps(seal,indent=2)+'\n')
print(json.dumps({'status':receipt['status'],'packet_manifest_sha256':receipt['packet_manifest_sha256'],
    'seal_sha256':sha(sealpath),'full_tests_passed':360,'examples':1728}),flush=True)
