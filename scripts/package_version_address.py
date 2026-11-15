"""Byte-verified incremental backup of all registered normal-history version-address observations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_version_address import audit_run

STORAGE = Path('/media/990Pro/Code/SAS')
PHASE = 'version-address'
TAG = 'version-address-observation-2026-11-15'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_audit(report):
    expected = {
        'physical_readouts_audited': 3456,
        'reused_exact_replays_audited': 1728,
        'archived_references_audited': 1728,
        'raw_fp32_files_independently_reaggregated': 576,
        'layer_position_rows_verified': 41472,
        'normal_prefix_caches_audited': 64,
        'normal_after_caches_audited': 512,
        'legacy_operational_readouts': 96,
        'source_files_verified': 268,
    }
    metric_counts = {'cells': 72, 'role_cells': 216,
                     'all_correctness_strata_including_empty': 576,
                     'wording_pairs': 1728, 'same_history_version_pairs': 1152}
    if (report['status'] != 'completed_and_audited_version_address'
        or any(report[key] != value for key, value in expected.items())
        or any(len(report['metrics'][key]) != value for key, value in metric_counts.items())
        or report['smoke'] or report['new_method_or_causal_attention_claim']
        or any(report['prior_failed_criteria_unchanged'].values())):
        raise ValueError('Complete unfiltered observational audit and unchanged failed criteria required')


def package(audit, output):
    report = json.loads(audit.read_text())
    validate_audit(report)
    run = Path(report['source_run'])
    if report != audit_run(run):
        raise ValueError('Audit no longer matches all actual measurement bytes')
    manifest = json.loads((run / 'manifest.json').read_text())
    plan = json.loads((run / 'plan.json').read_text())
    inputs = {}

    def add(path, base, prefix):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Only real regular snapshot files allowed')
        relative = path.relative_to(base)
        if (set(relative.parts) & {'models', '.git', '.cache', '__pycache__', '.venv', '.pytest_cache'}
            or path.suffix in ('.safetensors', '.pt', '.pth')):
            raise ValueError('Exclude pretrained weights, environments and caches')
        key = prefix + '/' + relative.as_posix()
        if key in inputs and inputs[key] != path:
            raise ValueError('Snapshot path collision')
        inputs[key] = path

    def tree(path, base, prefix):
        if path.is_symlink() or not path.is_dir():
            raise ValueError('Explicit real phase directories required')
        for child in sorted(path.rglob('*')):
            if child.is_file():
                add(child, base, prefix)

    for directory in sorted((STORAGE / 'runs/memory_update').glob(PHASE + '*')):
        status = json.loads((directory / 'manifest.json').read_text())['status']
        if status not in ('completed', 'failed'):
            raise ValueError('Every phase job must be terminal before backup')
        tree(directory, STORAGE, 'storage')
    tree(Path(manifest['packet']), STORAGE, 'storage')
    for folder in ('logs/memory_update', 'artifacts/memory_update'):
        for path in sorted((STORAGE / folder).glob(PHASE + '*')):
            if '-packaging-' in path.name or '-publication-' in path.name:
                continue
            if path.is_dir():
                tree(path, STORAGE, 'storage')
            elif path.is_file():
                add(path, STORAGE, 'storage')
    for name, digest in manifest['source_sha256'].items():
        if sha(ROOT / name) != digest or sha(run / 'sources' / name) != digest:
            raise ValueError('All 268 sealed sources must remain identical')
        add(ROOT / name, ROOT, 'code')
    tree(ROOT / 'materials/updates/version-address-20261115', ROOT, 'code')
    for name in ('README.md', 'materials/README.md', '.gitattributes', 'requirements.txt',
                 'configs/experiment_progress.json', 'scripts/run_version_address_pipeline.sh',
                 'scripts/export_version_address.py', 'scripts/version_address_observations.py',
                 'scripts/run_version_address_postprocess.sh',
                 'scripts/package_version_address.py',
                 'scripts/publish_version_address.py', 'scripts/check_version_address_publication_scope.py'):
        add(ROOT / name, ROOT, 'code')
    for name in ('native', 'gdn_all_tokens'):
        state = manifest['models'][name]
        for relative, digest in state['record_mass_files'].items():
            path = run / relative
            if (path.suffix != '.npy' or sha(path) != digest
                or 'storage/' + path.relative_to(STORAGE).as_posix() not in inputs):
                raise ValueError('All original FP32 pooled probability files must be included unchanged')
        for suffix, field in (('.jsonl', 'results_sha256'),
                              ('.references.jsonl', 'references_sha256')):
            path = run / (name + suffix)
            if sha(path) != state[field] or 'storage/' + path.relative_to(STORAGE).as_posix() not in inputs:
                raise ValueError('Every normal readout and archived reference must be complete')

    output.mkdir(parents=True, exist_ok=False)
    records = [{'path': key, 'bytes': path.stat().st_size, 'sha256': sha(path)}
               for key, path in sorted(inputs.items())]
    inventory = {
        'status': 'complete_audited_version_address_snapshot',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'primary_audit_sha256': sha(audit), 'manifest_sha256': report['manifest_sha256'],
        'packet_manifest_sha256': report['packet_manifest_sha256'], 'release_tag': TAG,
        'physical_readouts': 3456, 'old_exact_replays': 1728, 'fresh_readouts': 1728,
        'archived_readouts': 1728, 'normal_prefix_caches': 64, 'normal_after_caches': 512,
        'source_files': 268, 'raw_fp32_record_mass_files': 576, 'layer_position_rows': 41472,
        'model_cohort_style_episode_cells': 72, 'role_cells': 216,
        'all_correctness_strata_including_empty': 576, 'wording_pairs': 1728,
        'same_history_version_pairs': 1152, 'new_method_or_causal_attention_claim': False,
        'predecessor_release': plan['source_release'],
        'training_checkpoint_release': 'https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/matched-shared-training-timing-2026-11-15',
        'restore_scope': 'Incremental snapshot: requires the published predecessor release chain and fixed adapter checkpoint. Local pretrained IT weights and existing Conda base are separate.',
        'excluded': ['pretrained weights', 'environments', 'caches', 'previously published runs and training checkpoints'],
        'files': records,
    }
    inventory_path = output / 'file_manifest.json'
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')
    archive = output / 'version-address-20261115.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
        for record in records:
            path = inputs[record['path']]
            if sha(path) != record['sha256']:
                raise ValueError('Snapshot input changed while packing')
            tar.add(path, arcname=record['path'], recursive=False)
        tar.add(inventory_path, arcname='file_manifest.json', recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        if (len(tar.getmembers()) != len(records) + 1
            or {member.name for member in tar.getmembers()} != {r['path'] for r in records} | {'file_manifest.json'}):
            raise ValueError('Complete archive inventory differs')
        for record in records + [{'path': 'file_manifest.json', 'bytes': inventory_path.stat().st_size,
                                   'sha256': sha(inventory_path)}]:
            member = tar.getmember(record['path'])
            if not member.isfile() or member.size != record['bytes']:
                raise ValueError('Archive member type or size differs')
            digest = hashlib.sha256()
            with tar.extractfile(member) as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != record['sha256']:
                raise ValueError('Decompressed archive bytes differ')
    receipt = {'status': 'all_archive_members_byte_verified', 'files_verified': len(records),
               'archive_entries': len(records) + 1, 'archive': archive.name,
               'bytes': archive.stat().st_size, 'sha256': sha(archive)}
    (output / 'local_verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.audit, args.output)), flush=True)


if __name__ == '__main__':
    main()
