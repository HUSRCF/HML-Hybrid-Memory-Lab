"""No-retry fixed confirmation pipeline,after all training audits and source sealing."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
ROOT=Path('/home/husrcf/Code/SAS');BASE=Path('/media/990Pro/Code/SAS');ART=BASE/'artifacts/memory_update';LOG=BASE/'logs/memory_update';RUN=BASE/'runs/memory_update'
receipt=ART/'readout-confirmation-pipeline-20261116-01.json'
smoke=RUN/'readout-confirmation-smoke-20261116-01';sa=ART/'readout-confirmation-smoke-audit-20261116-01.json'
primary=RUN/'readout-confirmation-primary-20261116-01';pa=ART/'readout-confirmation-primary-audit-20261116-01.json'
if receipt.exists() or any(f.exists() for f in (smoke,sa,primary,pa)):raise ValueError('Never overwrite or retry')
entries=[('prepare_fixed_confirmation',['scripts/prepare_readout_confirmation.py'],'readout-confirmation-registration-20261116-01.log'),
 ('independent_input_reconstruction',['/tmp/check_readout_confirmation_packet.py'],'readout-confirmation-independent-inputs-20261116-01.log'),
 ('replay_observed_fit',['scripts/run_readout_confirmation.py','--smoke','--output',str(smoke)],'readout-confirmation-smoke-20261116-01.log'),
 ('audit_observed_replay',['scripts/analyze_readout_confirmation.py','--run',str(smoke),'--allow-smoke','--output',str(sa)],'readout-confirmation-smoke-audit-20261116-01.log'),
 ('full_frozen_confirmation',['scripts/run_readout_confirmation.py','--smoke-audit',str(sa),'--output',str(primary)],'readout-confirmation-primary-20261116-01.log'),
 ('independent_complete_confirmation_audit',['scripts/analyze_readout_confirmation.py','--run',str(primary),'--output',str(pa)],'readout-confirmation-primary-audit-20261116-01.log')]
proof={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'stages':[],'accuracy_gate':False}
receipt.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
for name,args,log in entries:
 path=LOG/log
 if path.exists():raise ValueError('Never overwrite a stage log')
 item={'stage':name,'started_utc':datetime.now(timezone.utc).isoformat(),'log':str(path),'command':[sys.executable,'-B',*args]};proof['stages'].append(item);receipt.write_text(json.dumps(proof,indent=2)+'\n')
 print(json.dumps({'event':'started','stage':name}),flush=True)
 with path.open('x') as f:process=subprocess.run(item['command'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 item.update(returncode=process.returncode,completed_utc=datetime.now(timezone.utc).isoformat(),log_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 if process.returncode:
  proof['status']='failed_no_retry';receipt.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'event':'failed','stage':name,'log':str(path)}),flush=True);sys.exit(process.returncode)
 receipt.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'event':'completed','stage':name}),flush=True)
proof.update(status='completed_all6000_frozen_confirmation_readouts_and_independent_audit',completed_utc=datetime.now(timezone.utc).isoformat());receipt.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'status':proof['status'],'audit':str(pa)}),flush=True)
