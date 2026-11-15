"""Publish a complete, committed confirmation snapshot; verify downloaded assets byte for byte."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess

ROOT = Path('/home/husrcf/Code/SAS')
BASE = Path('/media/990Pro/Code/SAS')
ART = BASE / 'artifacts/memory_update'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def command(args):
    return subprocess.run(args, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

repo = 'HUSRCF/HML-Hybrid-Memory-Lab'
tag = 'readout-confirmation-2026-11-16'
package = json.loads((ART / 'readout-confirmation-package-20261116-01.json').read_text())
archive, manifest = Path(package['archive']), Path(package['manifest'])
folder = archive.parent
assert sha(archive) == package['archive_sha256'] and sha(manifest) == package['manifest_sha256']
target = command(['git', 'rev-parse', 'HEAD']).decode().strip()
assert command(['git', 'rev-parse', 'origin/main']).decode().strip() == target
assert not command(['git', 'status', '--porcelain']).strip()
payload = json.loads(manifest.read_text())
committed = {}
for name, record in payload['files'].items():
    if name.startswith('code/'):
        relative = name[len('code/'):]
    elif name.startswith('materials/'):
        relative = 'materials/updates/readout-confirmation-20261116/' + name[len('materials/'):]
    else:
        continue
    raw = command(['git', 'show', target + ':' + relative])
    assert hashlib.sha256(raw).hexdigest() == record['sha256'], relative
    committed[relative] = record['sha256']

report = json.loads((ART / 'readout-confirmation-exports-20261116-01/report.json').read_text())
lines = [
    'Complete prospective SET/COPY evaluation of five frozen models. All 6000 physical readouts and 6480 logical bindings are independently audited and retained.',
    '',
    '| Frozen model | SET /864 | COPY physical /336 | COPY six-answer units /48 |',
    '| --- | ---: | ---: | ---: |']
for model, value in report['summary'].items():
    lines.append('| ' + model + ' | ' + str(value['set']['physical_correct']) + ' | ' + str(value['copy']['physical_correct']) + ' | ' + str(value['copy']['joint6_correct']) + ' |')
lines.extend([
    '',
    'The checkpoints were finalized before constructing the fixed new cohorts. Each readout control adds 131072 parameters and activates only on question/answer tokens, matching its training scope. No Oracle, donor forward, recurrent rollback, sparse reading or additional training is used.',
    '',
    'All 200 normally constructed histories per model are retained as full cache-field/layout/metadata and prefix-logit digests; the three learned readout arms match the frozen GDN histories exactly. Literal inputs, every raw answer, all model pairs and condition tables, source snapshots, training locks, checkpoint and optimizer/RNG dependencies are included.',
    '',
    'There are only eight independent base streams per task and one training initialization. Names, question styles, dependency depths and record orders are paired views, not independent samples. These are simple readout baselines; this release does not establish a new architecture, unique Query-address cause, long-context scaling or superiority of a shared hybrid controller. All earlier failed scientific criteria remain unchanged.',
    '',
    'Native 2B pretrained weights and the Conda environment are external dependencies; normal cache tensor values are regenerated from the literal inputs. Earlier fit and passive-diagnostic dependencies are linked in the archive manifest.',
    '',
    'Archive SHA256: `' + package['archive_sha256'] + '`', ''])
notes = folder / 'release-notes.md'
assert not notes.exists()
notes.write_text('\n'.join(lines))
check = subprocess.run(['gh', 'api', f'repos/{repo}/releases/tags/{tag}'], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if check.returncode == 0:
    raise ValueError('Never edit or duplicate an existing release')
if b'404' not in check.stderr and b'Not Found' not in check.stderr:
    raise RuntimeError(check.stderr.decode())
assets = [archive, manifest, folder / 'SHA256SUMS.txt']
command(['gh', 'release', 'create', tag, *map(str, assets), '--repo', repo, '--target', target,
         '--title', 'Frozen readout controls: complete new-stream SET and COPY confirmation', '--notes-file', str(notes)])
remote = json.loads(command(['gh', 'api', f'repos/{repo}/releases/tags/{tag}']))
assert not remote['draft'] and remote['target_commitish'] == target
assert {x['name'] for x in remote['assets']} == {f.name for f in assets}
ref = json.loads(command(['gh', 'api', f'repos/{repo}/git/ref/tags/{tag}']))['object']
while ref['type'] == 'tag':
    ref = json.loads(command(['gh', 'api', f'repos/{repo}/git/tags/' + ref['sha']]))['object']
assert ref['type'] == 'commit' and ref['sha'] == target
download = folder / 'remote-download-verified'
download.mkdir(exist_ok=False)
command(['gh', 'release', 'download', tag, '--repo', repo, '--dir', str(download)])
proof = []
for f in assets:
    actual = download / f.name
    assert sha(actual) == sha(f) and actual.stat().st_size == f.stat().st_size
    info = next(x for x in remote['assets'] if x['name'] == f.name)
    proof.append({'name': f.name, 'bytes': f.stat().st_size, 'sha256': sha(f),
                  'downloaded_bytes_verified': True, 'url': info['browser_download_url']})
receipt = {
    'status': 'published_confirmation_backup_remote_assets_roundtrip_byte_verified',
    'completed_utc': datetime.now(timezone.utc).isoformat(), 'release_url': remote['html_url'],
    'release_target_commit': target, 'archive_sha256': sha(archive),
    'archive_bytes': archive.stat().st_size, 'archive_members': package['archive_members'],
    'committed_snapshot_files_byte_verified': len(committed), 'committed_files_sha256': committed,
    'remote_assets': proof, 'physical_readouts': 6000, 'logical_bindings': 6480,
    'new_native_inference': False, 'architecture_superiority_claim': False,
    'publication_source_sha256': sha(Path(__file__))}
out = ART / 'readout-confirmation-publication-20261116-01.json'
assert not out.exists()
out.write_text(json.dumps(receipt, indent=2) + '\n')
out.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({k: v for k, v in receipt.items() if k != 'committed_files_sha256'}), flush=True)
