"""No-retry operational pipeline: real safety,independent audit,training,full audit."""
from datetime import datetime,timezone
from pathlib import Path
import json,subprocess,sys,hashlib
ROOT=Path('/home/husrcf/Code/SAS');BASE=Path('/media/990Pro/Code/SAS')
ART=BASE/'artifacts/memory_update';LOG=BASE/'logs/memory_update';RUN=BASE/'runs/memory_update'
smoke=RUN/'readout-adapter-smoke-20261115-02';audit=ART/'readout-adapter-smoke-audit-20261115-02.json'
train=RUN/'readout-adapter-training-20261115-01';final=ART/'readout-adapter-training-audit-20261115-01.json'
receipt=ART/'readout-adapter-cache-pipeline-20261115-01.json'
if receipt.exists() or any(p.exists() for p in (smoke,audit,train,final)):raise ValueError('No rerun/overwrite')
entries=[('native_smoke',['scripts/run_readout_adapter_cache_training.py','--smoke','--output',str(smoke)],'readout-adapter-smoke-20261115-02.log'),
         ('independent_smoke_audit',['scripts/analyze_readout_adapter_cache_training.py','--run',str(smoke),'--allow-smoke','--output',str(audit)],'readout-adapter-smoke-audit-20261115-02.log'),
         ('scheduled_three_arm_training',['scripts/run_readout_adapter_cache_training.py','--smoke-audit',str(audit),'--output',str(train)],'readout-adapter-training-20261115-01.log'),
         ('independent_training_audit',['scripts/analyze_readout_adapter_cache_training.py','--run',str(train),'--output',str(final)],'readout-adapter-training-audit-20261115-01.log')]
proof={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'stages':[],'future_confirmation_generated':False}
receipt.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
for name,args,log in entries:
    path=LOG/log
    if path.exists():raise ValueError('Never overwrite pipeline stage log')
    item={'stage':name,'started_utc':datetime.now(timezone.utc).isoformat(),'log':str(path),'command':[sys.executable,'-B',*args]}
    proof['stages'].append(item);receipt.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'event':'started','stage':name}),flush=True)
    with path.open('x') as handle:
        process=subprocess.run(item['command'],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
    item.update(returncode=process.returncode,completed_utc=datetime.now(timezone.utc).isoformat(),log_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if process.returncode:
        proof['status']='failed_no_retry';receipt.write_text(json.dumps(proof,indent=2)+'\n')
        print(json.dumps({'event':'failed','stage':name,'log':str(path)}),flush=True)
        sys.exit(process.returncode)
    print(json.dumps({'event':'completed','stage':name}),flush=True)
    receipt.write_text(json.dumps(proof,indent=2)+'\n')
proof.update(status='completed_all_three_training_arms_and_independent_audit',completed_utc=datetime.now(timezone.utc).isoformat())
receipt.write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'status':proof['status'],'training_audit':str(final)}),flush=True)
