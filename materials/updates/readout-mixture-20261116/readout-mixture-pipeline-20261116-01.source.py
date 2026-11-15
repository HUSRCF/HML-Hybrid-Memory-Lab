"""No-retry regression/seal, observed registration, native safety and paired training pipeline."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, re, subprocess, sys

ROOT=Path('/home/husrcf/Code/SAS')
BASE=Path('/media/990Pro/Code/SAS')
ART=BASE/'artifacts/memory_update'
LOG=BASE/'logs/memory_update'
RUN=BASE/'runs/memory_update'
sys.path.insert(0,str(ROOT))
from scripts.readout_mixture_data import read_plan, sources, PROTOCOL_SHA256

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():
    return datetime.now(timezone.utc).isoformat()
plan=read_plan()
receipt=ART/'readout-mixture-pipeline-20261116-01.json'
seal=Path(plan['source_seal'])
smoke=RUN/'readout-mixture-smoke-20261116-01'
smoke_audit=ART/'readout-mixture-smoke-audit-20261116-01.json'
training=RUN/'readout-mixture-training-20261116-01'
training_audit=ART/'readout-mixture-training-audit-20261116-01.json'
for p in (receipt,seal,Path(plan['dataset']),smoke,smoke_audit,training,training_audit):
    if p.exists():
        raise ValueError('Never overwrite or retry an existing stage: '+str(p))
expected={name:sha(ROOT/name) for name in sorted(sources())}
assert len(expected)==296
proof={'status':'running','started_utc':now(),'source_sha256':sha(Path(__file__)),
       'scientific_sources_at_test_start':expected,'stages':[],
       'accuracy_gate':False,'future_confirmation_generated':False}
receipt.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
def save():
    receipt.write_text(json.dumps(proof,indent=2)+'\n')
def stage(name,args,filename):
    path=LOG/filename
    if path.exists():
        raise ValueError('Never overwrite a stage log')
    item={'stage':name,'started_utc':now(),'command':[sys.executable,'-B',*args],'log':str(path)}
    proof['stages'].append(item)
    save()
    print(json.dumps({'event':'started','stage':name}),flush=True)
    with path.open('x') as f:
        done=subprocess.run(item['command'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    item.update(returncode=done.returncode,completed_utc=now(),log_sha256=sha(path))
    save()
    if done.returncode:
        raise RuntimeError('Stage failed without retry: '+name)
    print(json.dumps({'event':'completed','stage':name}),flush=True)
    return path
try:
    save()
    log=stage('full380_cpu_regression',['-m','unittest','discover','-s','tests','-v'],
              'readout-mixture-full-tests-20261116-01.log')
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$',log.read_text())
    assert match and int(match[1])==380
    assert {name:sha(ROOT/name) for name in expected}==expected
    sealed={'status':'sealed296mixture_sources_before_native_smoke','created_utc':now(),
            'protocol_sha256':PROTOCOL_SHA256,'source_sha256':expected,
            'parent_source_seal_sha256':plan['parent_source_seal_sha256'],
            'full_tests_log':str(log),'full_tests_log_sha256':sha(log),
            'full_tests_passed':380,'full_tests_seconds':float(match[2]),
            'all289predecessor_sources_unchanged':True,
            'future_confirmation_generated':False,'new_native_execution_started':False,
            'pipeline_source_sha256':sha(Path(__file__))}
    seal.write_text(json.dumps(sealed,indent=2)+'\n')
    proof['source_seal_sha256']=sha(seal)
    save()
    stage('register_observed_mixture_inputs',['scripts/prepare_readout_mixture.py'],
          'readout-mixture-registration-20261116-01.log')
    stage('actual_native27_29_30_safety',['scripts/run_readout_mixture_training.py','--smoke','--output',str(smoke)],
          'readout-mixture-smoke-20261116-01.log')
    stage('independent_native_safety_audit',['scripts/analyze_readout_mixture_training.py','--run',str(smoke),
          '--allow-smoke','--output',str(smoke_audit)],'readout-mixture-smoke-audit-20261116-01.log')
    stage('scheduled_paired432_step_training',['scripts/run_readout_mixture_training.py','--output',str(training),
          '--smoke-audit',str(smoke_audit)],'readout-mixture-training-20261116-01.log')
    stage('independent_complete_training_audit',['scripts/analyze_readout_mixture_training.py','--run',str(training),
          '--output',str(training_audit)],'readout-mixture-training-audit-20261116-01.log')
    proof.update(status='completed_paired_training_and_independent_audit_future_cases_not_generated',completed_utc=now())
    save()
    print(json.dumps({'status':proof['status'],'audit':str(training_audit)}),flush=True)
except BaseException as error:
    proof.update(status='failed_no_retry',completed_utc=now(),error=f'{type(error).__name__}: {error}')
    save()
    raise
