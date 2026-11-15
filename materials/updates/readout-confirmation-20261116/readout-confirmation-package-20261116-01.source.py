"""Archive complete frozen confirmation without active packaging logs or native LLM weights."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys, tarfile
from unittest.mock import patch

ROOT = Path('/home/husrcf/Code/SAS')
BASE = Path('/media/990Pro/Code/SAS')
ART = BASE / 'artifacts/memory_update'
sys.path.insert(0, str(ROOT))
from scripts.analyze_readout_confirmation import audit_run

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

run = BASE / 'runs/memory_update/readout-confirmation-primary-20261116-01'
audit_path = ART / 'readout-confirmation-primary-audit-20261116-01.json'
audit = json.loads(audit_path.read_text())
with (patch('sas_memory.runtime.TextRuntime.__init__', side_effect=AssertionError('No model construction')),
      patch('torch.cuda._lazy_init', side_effect=AssertionError('No GPU initialization'))):
    assert audit == audit_run(run)
assert audit['physical_readouts_audited'] == 6000
pipeline = json.loads((ART / 'readout-confirmation-pipeline-20261116-01.json').read_text())
assert pipeline['status'] == 'completed_all6000_frozen_confirmation_readouts_and_independent_audit'
assert all(s['returncode'] == 0 for s in pipeline['stages'])
exports = ART / 'readout-confirmation-exports-20261116-01'
export = json.loads((exports / 'export-receipt.json').read_text())
assert export['source_audit_sha256'] == sha(audit_path)
for name, digest in export['files_sha256'].items():
    assert sha(exports / name) == digest

files = {}
def add_dir(path, arc):
    for f in sorted(Path(path).rglob('*')):
        if f.is_file() and '__pycache__' not in f.parts:
            assert not f.is_symlink()
            name = str(Path(arc) / f.relative_to(path))
            assert name not in files
            files[name] = f

add_dir(run, 'runs/confirmation-primary')
add_dir(BASE / 'runs/memory_update/readout-confirmation-smoke-20261116-01', 'runs/observed-replay')
add_dir(BASE / 'datasets/memory_update/readout-adapter-confirmation-v1', 'datasets/confirmation')
add_dir(exports, 'artifacts/complete-exports')
add_dir(ROOT / 'materials/updates/readout-confirmation-20261116', 'materials')
# Preserve every saved checkpoint and optimizer/RNG state pinned by the training lock.
add_dir(BASE / 'runs/memory_update/readout-adapter-training-20261115-01', 'dependencies/readout-training')
for name in ('readout-adapter-training-audit-20261115-01.json',
             'readout-adapter-training-audit-20261115-01.correction.json',
             'readout-adapter-cache-seal-20261115-01.json'):
    files['dependencies/training-provenance/' + name] = ART / name
for f in sorted(ART.glob('readout-confirmation-*')):
    if f.is_file() and f.suffix in ('.json', '.py') and not any(x in f.name for x in ('package-', 'publication-')):
        files['artifacts/' + f.name] = f
for f in sorted((BASE / 'logs/memory_update').glob('readout-confirmation-*.log')):
    if not any(x in f.name for x in ('package-', 'publish-')):
        files['logs/' + f.name] = f

training_plan = json.loads((ROOT / 'configs/readout_adapter_training_plan.json').read_text())
fixed = Path(training_plan['fixed_gdn']['checkpoint'])
assert sha(fixed) == training_plan['fixed_gdn']['checkpoint_sha256']
files['dependencies/fixed-gdn/controller.safetensors'] = fixed
if (fixed.parent / 'checkpoint.json').exists():
    files['dependencies/fixed-gdn/checkpoint.json'] = fixed.parent / 'checkpoint.json'
seal = json.loads((ART / 'readout-confirmation-source-seal-20261115-01.json').read_text())
assert len(seal['source_sha256']) == 289
for name, digest in seal['source_sha256'].items():
    assert sha(ROOT / name) == digest
    files['code/' + name] = ROOT / name

out = BASE / 'backups/readout-confirmation-20261116-01'
assert not out.exists()
out.mkdir(parents=True)
manifest = {
    'status': 'complete_whitelisted_frozen_confirmation_backup',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'source_audit_sha256': sha(audit_path),
    'physical_readouts': 6000, 'logical_bindings': 6480,
    'independent_base_streams_per_task': 8,
    'all_fixed_models_and_all_conditions_included': True,
    'pretrained_llm_weights_included': False,
    'normal_cache_tensors_stored': False,
    'normal_cache_digests_and_literal_inputs_stored': True,
    'earlier_dependencies': [
        'https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/readout-adapter-fit-2026-11-16',
        'https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/version-address-observation-2026-11-15'],
    'packaging_source_sha256': sha(Path(__file__)),
    'files': {name: {'sha256': sha(f), 'bytes': f.stat().st_size} for name, f in sorted(files.items())}}
mp = out / 'MANIFEST.files.json'
mp.write_text(json.dumps(manifest, indent=2) + '\n')
archive = out / 'readout-confirmation-20261116.tar.gz'
prefix = 'readout-confirmation-20261116/'
with tarfile.open(archive, 'w:gz') as tar:
    for name, f in sorted(files.items()):
        tar.add(f, arcname=prefix + name, recursive=False)
    tar.add(mp, arcname=prefix + mp.name, recursive=False)
expected = {prefix + name: item for name, item in manifest['files'].items()}
expected[prefix + mp.name] = {'sha256': sha(mp), 'bytes': mp.stat().st_size}
with tarfile.open(archive, 'r:gz') as tar:
    members = tar.getmembers()
    assert len(members) == len(expected) and {m.name for m in members} == set(expected)
    for member in members:
        assert member.isfile() and not member.issym()
        assert not member.name.startswith('/') and '..' not in Path(member.name).parts
        data = tar.extractfile(member).read()
        record = expected[member.name]
        assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
receipt = {
    'status': 'complete_confirmation_archive_decompressed_bytes_verified',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'archive': str(archive), 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
    'files': len(files), 'archive_members': len(expected),
    'manifest': str(mp), 'manifest_sha256': sha(mp),
    'source_audit_sha256': sha(audit_path),
    'packaging_source_sha256': sha(Path(__file__)), 'new_native_inference': False,
    'active_packaging_logs_excluded': True, 'pretrained_llm_weights_included': False}
rp = ART / 'readout-confirmation-package-20261116-01.json'
assert not rp.exists()
rp.write_text(json.dumps(receipt, indent=2) + '\n')
rp.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
(out / 'SHA256SUMS.txt').write_text(sha(archive) + '  ' + archive.name + '\n' + sha(mp) + '  ' + mp.name + '\n')
print(json.dumps(receipt), flush=True)
