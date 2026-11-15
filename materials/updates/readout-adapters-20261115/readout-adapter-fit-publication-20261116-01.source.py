"""Publish exact fitted-adapter backup and verify all remote asset bytes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
ROOT=Path('/home/husrcf/Code/SAS');BASE=Path('/media/990Pro/Code/SAS');ART=BASE/'artifacts/memory_update'
h=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def command(args):return subprocess.run(args,cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
repo='HUSRCF/HML-Hybrid-Memory-Lab';tag='readout-adapter-fit-2026-11-16'
target=command(['git','rev-parse','86e3b1b']).decode().strip()
package=json.loads((ART/'readout-adapter-training-package-20261116-02.json').read_text());archive=Path(package['archive']);manifest=Path(package['manifest']);folder=archive.parent
assert h(archive)==package['archive_sha256'] and h(manifest)==package['manifest_sha256']
payload=json.loads(manifest.read_text());committed={}
for name,value in payload['files'].items():
 if name.startswith('code/'):
  relative=name[len('code/'):]
 elif name.startswith('materials/'):
  relative='materials/updates/readout-adapters-20261115/'+name[len('materials/'):]
 else:continue
 raw=command(['git','show',target+':'+relative]);assert hashlib.sha256(raw).hexdigest()==value['sha256'],relative
 committed[relative]=value['sha256']
notes=folder/'release-notes.md';assert not notes.exists()
notes.write_text('''Complete frozen-GDN readout-control training and observed-fit artifacts.

All three controls add131072parameters,run exactly432steps,and retain the existing frozen2B/GDN weights and normal fact histories. Every1728observed fit question is evaluated at final432;all original baseline errors and new regressions are retained.

| Observed fit | Correct /1728 | Baseline errors repaired | New regressions |
| --- | ---: | ---: | ---: |
| Existing fixed GDN |1633|0|0|
| Query |1720|92|5|
| Output gate |1715|84|2|
| Query + gate |1727|95|1|

These are fit results,not held-out confirmation orarchitecture evidence. All three prior failed scientific criteria remain unchanged. Prospective SET/COPY confirmation is separately registered and does not select arms,checkpoints orseeds from these scores.

The archive contains all final and intermediate readout checkpoints,optimizer/RNG states,the existing fixed-GDN checkpoint,literal fit inputs,all raw fit answers,complete audits,source snapshots and preserved failures. Native2B pretrained weights and the environment are separate dependencies.

Archive SHA256: `'''+package['archive_sha256']+'`\n')
check=subprocess.run(['gh','api',f'repos/{repo}/releases/tags/{tag}'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if check.returncode==0:raise ValueError('Never edit or duplicate an existing release')
if b'404' not in check.stderr and b'Not Found' not in check.stderr:raise RuntimeError(check.stderr.decode())
assets=[archive,manifest,folder/'SHA256SUMS.txt']
command(['gh','release','create',tag,*map(str,assets),'--repo',repo,'--target',target,'--title','Frozen readout controls: complete observed-fit training','--notes-file',str(notes)])
remote=json.loads(command(['gh','api',f'repos/{repo}/releases/tags/{tag}']))
assert not remote['draft'] and remote['target_commitish']==target and {x['name'] for x in remote['assets']}=={f.name for f in assets}
ref=json.loads(command(['gh','api',f'repos/{repo}/git/ref/tags/{tag}']))['object']
while ref['type']=='tag':ref=json.loads(command(['gh','api',f'repos/{repo}/git/tags/'+ref['sha']]))['object']
assert ref['type']=='commit' and ref['sha']==target
download=folder/'remote-download-verified';download.mkdir(exist_ok=False)
command(['gh','release','download',tag,'--repo',repo,'--dir',str(download)])
proof=[]
for f in assets:
 actual=download/f.name;assert h(actual)==h(f) and actual.stat().st_size==f.stat().st_size
 info=next(x for x in remote['assets'] if x['name']==f.name)
 proof.append({'name':f.name,'bytes':f.stat().st_size,'sha256':h(f),'downloaded_bytes_verified':True,'url':info['browser_download_url']})
receipt={'status':'published_fit_backup_remote_assets_roundtrip_byte_verified','completed_utc':datetime.now(timezone.utc).isoformat(),
         'release_url':remote['html_url'],'release_target_commit':target,'archive_sha256':h(archive),
         'archive_bytes':archive.stat().st_size,'archive_members':package['archive_members'],
         'committed_snapshot_files_byte_verified':len(committed),'committed_files_sha256':committed,
         'remote_assets':proof,'fit_not_confirmation':True,'new_native_inference':False,'publication_source_sha256':h(Path(__file__))}
out=ART/'readout-adapter-fit-publication-20261116-01.json';assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n');out.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({k:v for k,v in receipt.items() if k!='committed_files_sha256'}),flush=True)
