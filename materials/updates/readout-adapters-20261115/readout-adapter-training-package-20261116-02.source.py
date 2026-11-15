"""Whitelist-only complete fitted-adapter backup with decompressed byte verification."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tarfile,sys
ROOT=Path('/home/husrcf/Code/SAS');BASE=Path('/media/990Pro/Code/SAS');ART=BASE/'artifacts/memory_update'
sys.path.insert(0,str(ROOT))
from scripts.analyze_readout_adapter_cache_training import audit_run,correction_audit
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
run=BASE/'runs/memory_update/readout-adapter-training-20261115-01';a=ART/'readout-adapter-training-audit-20261115-01.json';audit=json.loads(a.read_text())
assert audit==audit_run(run) and json.loads(a.with_suffix('.correction.json').read_text())==correction_audit(run,audit)
exports=ART/'readout-adapter-fit-exports-20261115-01';er=json.loads((exports/'export-receipt.json').read_text())
for name,digest in er['files_sha256'].items():assert h(exports/name)==digest
out=BASE/'backups/readout-adapter-training-20261116-02';assert not out.exists();out.mkdir(parents=True)
files={}
def add_dir(path,arc):
 for f in sorted(Path(path).rglob('*')):
  if f.is_file() and '__pycache__' not in f.parts:
   assert not f.is_symlink();files[str(Path(arc)/f.relative_to(path))]=f
add_dir(run,'runs/training')
add_dir(BASE/'runs/memory_update/readout-adapter-smoke-20261115-01','runs/failed-native-smoke01')
add_dir(BASE/'runs/memory_update/readout-adapter-smoke-20261115-02','runs/passed-native-smoke02')
add_dir(BASE/'datasets/memory_update/readout-adapter-fit-v1','datasets/observed-fit')
add_dir(exports,'artifacts/fit-exports')
add_dir(ROOT/'materials/updates/readout-adapters-20261115','materials')
for f in sorted(ART.glob('readout-adapter-*')):
 if f.is_file() and f.suffix in ('.json','.py'):files['artifacts/'+f.name]=f
for f in sorted((BASE/'logs/memory_update').glob('readout-adapter-*.log')):
 if not f.name.startswith('readout-adapter-fit-package-'):files['logs/'+f.name]=f
p=json.loads((ROOT/'configs/readout_adapter_training_plan.json').read_text());fixed=Path(p['fixed_gdn']['checkpoint']);assert h(fixed)==p['fixed_gdn']['checkpoint_sha256'];files['fixed-gdn/controller.safetensors']=fixed
for f in (fixed.parent/'checkpoint.json',):
 if f.exists():files['fixed-gdn/'+f.name]=f
seal=json.loads((ART/'readout-adapter-cache-seal-20261115-01.json').read_text())
for name,digest in seal['source_sha256'].items():assert h(ROOT/name)==digest;files['code/'+name]=ROOT/name
manifest={'status':'complete_whitelisted_observed_fit_backup','created_utc':datetime.now(timezone.utc).isoformat(),
          'training_audit_sha256':h(a),'fit_is_not_confirmation':True,'source_sha256':h(Path(__file__)),
          'files':{name:{'sha256':h(f),'bytes':f.stat().st_size} for name,f in sorted(files.items())}}
mp=out/'MANIFEST.files.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
archive=out/'readout-adapter-training-20261116.tar.gz';prefix='readout-adapter-training-20261116/'
with tarfile.open(archive,'w:gz') as tar:
 for name,f in sorted(files.items()):tar.add(f,arcname=prefix+name,recursive=False)
 tar.add(mp,arcname=prefix+'MANIFEST.files.json',recursive=False)
expected={prefix+name:record for name,record in manifest['files'].items()};expected[prefix+'MANIFEST.files.json']={'sha256':h(mp),'bytes':mp.stat().st_size}
with tarfile.open(archive,'r:gz') as tar:
 members=tar.getmembers();assert len(members)==len(expected) and {m.name for m in members}==set(expected)
 for m in members:
  assert m.isfile() and not m.issym() and not m.name.startswith('/') and '..' not in Path(m.name).parts
  data=tar.extractfile(m).read();record=expected[m.name];assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
receipt={'status':'complete_fit_archive_decompressed_bytes_verified','created_utc':datetime.now(timezone.utc).isoformat(),
         'archive':str(archive),'archive_sha256':h(archive),'archive_bytes':archive.stat().st_size,'files':len(files),'archive_members':len(expected),
         'manifest':str(mp),'manifest_sha256':h(mp),'training_audit_sha256':h(a),'new_native_inference':False,
         'pretrained_llm_weights_included':False,'existing_fixed_gdn_and_all_learned_checkpoints_included':True,
         'packaging_source_sha256':h(Path(__file__))}
rp=ART/'readout-adapter-training-package-20261116-02.json';assert not rp.exists();rp.write_text(json.dumps(receipt,indent=2)+'\n');rp.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
(out/'SHA256SUMS.txt').write_text(h(archive)+'  '+archive.name+'\n'+h(mp)+'  '+mp.name+'\n');print(json.dumps(receipt),flush=True)
