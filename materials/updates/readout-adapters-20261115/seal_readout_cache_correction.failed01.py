"""Independent operational seal after immutable364test correction regression."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,re,sys
from unittest.mock import patch
ROOT=Path('/home/husrcf/Code/SAS');sys.path.insert(0,str(ROOT))
BASE=Path('/media/990Pro/Code/SAS');h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
path=ROOT/'configs/readout_adapter_cache_commit_addendum.json';plan=json.loads(path.read_text())
assert h(path)=='a1ff022ad0d2ed33a6978490aa08a47960398a50758336c3657bd6138dea6d7a'
parent=json.loads(Path(plan['parent_seal']).read_text())
assert h(plan['parent_seal'])==plan['parent_seal_sha256']
for name,digest in parent['source_sha256'].items():assert h(ROOT/name)==digest,name
failed=Path(plan['failed_smoke']);fm=json.loads((failed/'manifest.json').read_text())
assert h(failed/'manifest.json')==plan['failed_manifest_sha256'] and h(plan['failed_log'])==plan['failed_log_sha256']
assert fm['status']=='failed' and 'modified by an inplace operation' in fm['error']
assert not list(failed.glob('*/step-*')) and not list(failed.glob('*/training.jsonl'))
assert not Path(plan['native_smoke_output']).exists() and not Path(plan['training_output']).exists()
log=BASE/'logs/memory_update/readout-adapter-cache-full-tests-20261115-01.log'
match=re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$',log.read_text())
assert match and int(match[1])==364,'Require full364passing tests'
from scripts.readout_adapter_data import registered_inputs,verify_seal
with (patch('sas_memory.runtime.TextRuntime.__init__',side_effect=AssertionError('No model construction')),
      patch('torch.cuda._lazy_init',side_effect=AssertionError('No GPU initialization'))):
    p,packet,*_=registered_inputs()
    assert verify_seal(packet)==parent
names=set(parent['source_sha256'])|set(plan['new_sources']);assert len(names)==282
out=Path(plan['source_seal']);assert not out.exists()
seal={'status':'sealed_cache_commit_correction_before_native_smoke','created_utc':datetime.now(timezone.utc).isoformat(),
      'source_sha256':{name:h(ROOT/name) for name in sorted(names)},'addendum_sha256':h(path),
      'parent_seal_sha256':h(plan['parent_seal']),'full_tests_passed':int(match[1]),'full_tests_seconds':float(match[2]),
      'full_tests_log':str(log),'full_tests_log_sha256':h(log),'failed_smoke_manifest_sha256':h(failed/'manifest.json'),
      'failed_optimizer_updates':0,'literal_fit_examples_reverified':1728,'normal_contexts_reverified':288,
      'source_packet_manifest_sha256':parent['packet_manifest_sha256'],'new_native_smoke_started':False,
      'new_optimizer_updates_started':False,'future_confirmation_generated':False,
      'independent_sealing_source_sha256':h(Path(__file__))}
out.write_text(json.dumps(seal,indent=2)+'\n');out.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'status':seal['status'],'sources':282,'seal_sha256':h(out),'tests':364,'seconds':float(match[2])}),flush=True)
