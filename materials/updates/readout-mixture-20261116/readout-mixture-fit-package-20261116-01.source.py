"""Whitelist complete paired training, all checkpoints/RNG and observed fits; verify tar bytes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, tarfile, sys
ROOT = Path('/home/husrcf/Code/SAS'); BASE = Path('/media/990Pro/Code/SAS')
ART = BASE / 'artifacts/memory_update'; sys.path.insert(0, str(ROOT))
from scripts.analyze_readout_mixture_training import audit_run
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
run = BASE / 'runs/memory_update/readout-mixture-training-20261116-01'
audit_path = ART / 'readout-mixture-training-audit-20261116-01.json'
assert json.loads(audit_path.read_text()) == audit_run(run)
exports = ART / 'readout-mixture-fit-exports-20261116-01'
for name, digest in json.loads((exports / 'export-receipt.json').read_text())['files_sha256'].items():
    assert sha(exports / name) == digest
out = BASE / 'backups/readout-mixture-fit-20261116-01'; assert not out.exists(); out.mkdir(parents=True)
files = {}
def directory(path, arc):
    for f in sorted(path.rglob('*')):
        if f.is_file() and '__pycache__' not in f.parts:
            assert not f.is_symlink()
            files[str(Path(arc) / f.relative_to(path))] = f
directory(run, 'runs/training')
directory(BASE / 'runs/memory_update/readout-mixture-smoke-20261116-01', 'runs/passed-native-smoke')
directory(BASE / 'datasets/memory_update/readout-mixture-fit-v1', 'datasets/observed-fit')
directory(exports, 'artifacts/fit-exports')
directory(ROOT / 'materials/updates/readout-mixture-20261116', 'materials')
for f in sorted(ART.glob('readout-mixture-*')):
    if f.is_file() and f.suffix in ('.json', '.py'):
        files['artifacts/' + f.name] = f
for f in sorted((BASE / 'logs/memory_update').glob('readout-mixture-*.log')):
    if not any(w in f.name for w in ('package-', 'publish-', 'document-')):
        files['logs/' + f.name] = f
bp = json.loads((ROOT / 'configs/readout_adapter_training_plan.json').read_text())
fixed = Path(bp['fixed_gdn']['checkpoint'])
assert sha(fixed) == bp['fixed_gdn']['checkpoint_sha256']
files['fixed-gdn/controller.safetensors'] = fixed
if (fixed.parent / 'checkpoint.json').exists():
    files['fixed-gdn/checkpoint.json'] = fixed.parent / 'checkpoint.json'
seal = json.loads((ART / 'readout-mixture-source-seal-20261116-01.json').read_text())
for name, digest in seal['source_sha256'].items():
    assert sha(ROOT / name) == digest
    files['code/' + name] = ROOT / name
manifest = {'status': 'complete_whitelisted_paired_gate_observed_fit_backup',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'training_audit_sha256': sha(audit_path),
            'fit_not_confirmation': True, 'source_sha256': sha(Path(__file__)),
            'external_dependencies': ['Qwen3.5-2B IT native weights and existing Conda base',
                'https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/readout-adapter-fit-2026-11-16',
                'https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/readout-confirmation-2026-11-16'],
            'files': {name: {'sha256': sha(f), 'bytes': f.stat().st_size} for name, f in sorted(files.items())}}
mp = out / 'MANIFEST.files.json'; mp.write_text(json.dumps(manifest, indent=2) + '\n')
archive = out / 'readout-mixture-fit-20261116.tar.gz'; prefix = 'readout-mixture-fit-20261116/'
with tarfile.open(archive, 'w:gz') as tar:
    for name, f in sorted(files.items()):
        tar.add(f, arcname=prefix + name, recursive=False)
    tar.add(mp, arcname=prefix + mp.name, recursive=False)
expected = {prefix + name: record for name, record in manifest['files'].items()}
expected[prefix + mp.name] = {'sha256': sha(mp), 'bytes': mp.stat().st_size}
with tarfile.open(archive, 'r:gz') as tar:
    members = tar.getmembers()
    assert len(members) == len(expected) and {m.name for m in members} == set(expected)
    for member in members:
        assert member.isfile() and not member.issym() and not member.name.startswith('/') and '..' not in Path(member.name).parts
        value = tar.extractfile(member).read(); record = expected[member.name]
        assert len(value) == record['bytes'] and hashlib.sha256(value).hexdigest() == record['sha256']
receipt = {'status': 'complete_paired_fit_archive_decompressed_bytes_verified',
           'created_utc': datetime.now(timezone.utc).isoformat(), 'archive': str(archive),
           'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
           'files': len(files), 'archive_members': len(expected), 'manifest': str(mp), 'manifest_sha256': sha(mp),
           'training_audit_sha256': sha(audit_path), 'new_native_inference': False,
           'pretrained_llm_weights_included': False, 'all_six_checkpoints_optimizer_rng_included': True,
           'packaging_source_sha256': sha(Path(__file__))}
rp = ART / 'readout-mixture-fit-package-20261116-01.json'; assert not rp.exists()
rp.write_text(json.dumps(receipt, indent=2) + '\n'); rp.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
(out / 'SHA256SUMS.txt').write_text(sha(archive) + '  ' + archive.name + '\n' + sha(mp) + '  ' + mp.name + '\n')
print(json.dumps(receipt), flush=True)
