"""Publish the fixed committed paired-fit archive and roundtrip every remote asset."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess
ROOT = Path('/home/husrcf/Code/SAS'); BASE = Path('/media/990Pro/Code/SAS')
ART = BASE / 'artifacts/memory_update'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
def command(args):
    return subprocess.run(args, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
repo = 'HUSRCF/HML-Hybrid-Memory-Lab'; tag = 'readout-mixture-fit-2026-11-16'
package = json.loads((ART / 'readout-mixture-fit-package-20261116-01.json').read_text())
archive, manifest = Path(package['archive']), Path(package['manifest']); folder = archive.parent
assert sha(archive) == package['archive_sha256'] and sha(manifest) == package['manifest_sha256']
target = command(['git', 'rev-parse', 'HEAD']).decode().strip()
assert command(['git', 'rev-parse', 'origin/main']).decode().strip() == target
assert not command(['git', 'status', '--porcelain']).strip()
committed = {}
for name, record in json.loads(manifest.read_text())['files'].items():
    if name.startswith('code/'):
        relative = name[len('code/'):]
    elif name.startswith('materials/'):
        relative = 'materials/updates/readout-mixture-20261116/' + name[len('materials/'):]
    else:
        continue
    raw = command(['git', 'show', target + ':' + relative])
    assert hashlib.sha256(raw).hexdigest() == record['sha256'], relative
    committed[relative] = record['sha256']
notes = folder / 'release-notes.md'; assert not notes.exists()
notes.write_text('''Complete paired Gate task-coverage training and all 2400 unique final observed-fit readouts, independently audited.

| Final432regime | SET /864 | COPY physical /672 | COPY logical /864 | COPY six-answer units /96 |
| --- | ---: | ---: | ---: | ---: |
| SET replay | 853 | Not executed in this fit evaluation | — | — |
| SET + COPY | 857 | 619 | 811 | 52 |

The common SET fit has eight repairs and four regressions. Mixed COPY joint success by depth is32/32,15/32,5/32. These are observed fits, not generalization. The separately preregistered seven-model confirmation remains separate; no checkpoint selection, early stopping or extra training is allowed.

Both regimes share Gate initialization, layer15/rank32/131072parameters,432updates,1728exposures and6912supervised answer tokens. Teacher input lengths are27/29/30tokens; exact input-token totals differ46656versus48816,and COPY needs additional prefills,so this is not a FLOPs-matched comparison. All six checkpoints,optimizer/RNG states,training logs,raw answers,literal inputs,source snapshots,safety/independent audits and256full normal-history records are included. Native and copy-on-write training-cache forward values,losses and all48final cache fields matched bitexactly before and after nonzero smoke updates. Smoke optimizer states were discarded.

COPY89000was previously observed and is now explicitly training/fit data. Question/answer adapters leave the complete normal fact histories unchanged. One training initialization and eight SET /sixteen COPY base streams; names,styles,depths and orders are dependent views. This establishes neither shared four-layer controller superiority nor a novel architecture. All earlier failed scientific criteria remain unchanged.

Native2Bweights and the existing Conda environment are external dependencies; linked prior releases provide original readout controls and observed-source dependencies. No pretrained LLM weights are uploaded.

Archive SHA256: `''' + package['archive_sha256'] + '`\n')
check = subprocess.run(['gh', 'api', f'repos/{repo}/releases/tags/{tag}'], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if check.returncode == 0:
    raise ValueError('Never overwrite existing release')
if b'404' not in check.stderr and b'Not Found' not in check.stderr:
    raise RuntimeError(check.stderr.decode())
assets = [archive, manifest, folder / 'SHA256SUMS.txt']
command(['gh', 'release', 'create', tag, *map(str, assets), '--repo', repo, '--target', target,
         '--title', 'Paired Gate SET replay and SET/COPY mixture: complete observed fit', '--notes-file', str(notes)])
remote = json.loads(command(['gh', 'api', f'repos/{repo}/releases/tags/{tag}']))
assert not remote['draft'] and remote['target_commitish'] == target
assert {r['name'] for r in remote['assets']} == {f.name for f in assets}
ref = json.loads(command(['gh', 'api', f'repos/{repo}/git/ref/tags/{tag}']))['object']
while ref['type'] == 'tag':
    ref = json.loads(command(['gh', 'api', f'repos/{repo}/git/tags/' + ref['sha']]))['object']
assert ref['type'] == 'commit' and ref['sha'] == target
download = folder / 'remote-download-verified'; download.mkdir(exist_ok=False)
command(['gh', 'release', 'download', tag, '--repo', repo, '--dir', str(download)])
proof = []
for f in assets:
    downloaded = download / f.name
    assert sha(downloaded) == sha(f) and downloaded.stat().st_size == f.stat().st_size
    info = next(r for r in remote['assets'] if r['name'] == f.name)
    proof.append({'name': f.name, 'sha256': sha(f), 'bytes': f.stat().st_size,
                  'downloaded_bytes_verified': True, 'url': info['browser_download_url']})
receipt = {'status': 'published_paired_fit_backup_remote_assets_roundtrip_byte_verified',
           'completed_utc': datetime.now(timezone.utc).isoformat(), 'release_url': remote['html_url'],
           'release_target_commit': target, 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
           'archive_members': package['archive_members'], 'committed_snapshot_files_byte_verified': len(committed),
           'committed_files_sha256': committed, 'remote_assets': proof, 'physical_fit_answers': 2400,
           'optimizer_updates': 864, 'checkpoints': 6, 'fit_not_confirmation': True,
           'architecture_superiority_claim': False, 'publication_source_sha256': sha(Path(__file__))}
out = ART / 'readout-mixture-fit-publication-20261116-01.json'; assert not out.exists()
out.write_text(json.dumps(receipt, indent=2) + '\n'); out.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({k: v for k, v in receipt.items() if k != 'committed_files_sha256'}), flush=True)
