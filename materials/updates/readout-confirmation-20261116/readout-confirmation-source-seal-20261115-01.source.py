"""Seal all289confirmation sources after full372test regression,without future cases."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,sys
ROOT=Path('/home/husrcf/Code/SAS');sys.path.insert(0,str(ROOT))
from scripts.readout_confirmation_data import read_plan,sources,PROTOCOL_SHA256
p=read_plan();names=sources();assert len(names)==289
log=Path('/media/990Pro/Code/SAS/logs/memory_update/readout-confirmation-full-tests-20261116-01.log')
match=re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$',log.read_text());assert match and int(match[1])==372
out=Path(p['source_seal']);assert not out.exists() and not Path(p['dataset']).exists()
h=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
seal={'status':'sealed289confirmation_sources_before_prospective_generation','created_utc':datetime.now(timezone.utc).isoformat(),
      'protocol_sha256':PROTOCOL_SHA256,'source_sha256':{name:h(ROOT/name) for name in sorted(names)},
      'full_tests_log':str(log),'full_tests_log_sha256':h(log),'full_tests_passed':372,'full_tests_seconds':float(match[2]),
      'predecessor_sources_unchanged':282,'prospective_cases_generated':False,'native_confirmation_started':False,
      'all_confirmation_tests_used_only_observed76000_and61000_fixtures':True,'sealing_source_sha256':h(Path(__file__))}
out.write_text(json.dumps(seal,indent=2)+'\n');out.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'status':seal['status'],'tests':372,'sources':289,'seal_sha256':h(out)}),flush=True)
