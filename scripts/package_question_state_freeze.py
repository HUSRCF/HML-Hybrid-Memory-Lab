"""Byte-verified incremental backup of every registered question-state control."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_question_state_freeze import audit_run

STORAGE = Path('/media/990Pro/Code/SAS')
PHASE = 'question-state-freeze'
TAG = 'question-state-freeze-2026-11-15'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_audit(report):
    if (report['status'] != 'completed_and_audited_question_state_freeze'
        or report['primary_readouts_audited'] != 6912
        or report['normal_reference_replays_audited'] != 864
        or report['archived_references_audited'] != 864
        or report['normal_prefix_caches_audited'] != 16
        or report['normal_after_caches_audited'] != 128
        or report['source_files_verified'] != 261
        or len(report['metrics']['cells']) != 144
        or len(report['metrics']['normal_to_intervention_pairs']) != 6048
        or len(report['metrics']['wording_pairs']) != 3456
        or report['fresh_confirmation']
        or any(report['prior_failed_criteria_unchanged'].values())):
        raise ValueError('Complete unfiltered primary audit and unchanged failed criteria required')


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
            raise ValueError('All 261 sealed sources must remain identical')
        add(ROOT / name, ROOT, 'code')
    tree(ROOT / 'materials/updates/question-state-freeze-20261115', ROOT, 'code')
    for name in ('README.md', 'materials/README.md', '.gitattributes', 'requirements.txt',
                 'configs/experiment_progress.json', 'scripts/run_question_state_freeze_pipeline.sh',
                 'scripts/export_question_state_freeze.py', 'scripts/question_state_freeze_observations.py',
                 'scripts/package_question_state_freeze.py',
                 'scripts/publish_question_state_freeze.py'):
        add(ROOT / name, ROOT, 'code')
    state = manifest['model_state']
    for name, digest in state['file_sha256'].items():
        path = run / name
        if sha(path) != digest or 'storage/' + path.relative_to(STORAGE).as_posix() not in inputs:
            raise ValueError('Every per-arm and execution file must be complete')
    if sha(run / 'references.jsonl') != state['references_sha256']:
        raise ValueError('Complete archived references must remain unchanged')

    output.mkdir(parents=True, exist_ok=False)
    records = [{'path': key, 'bytes': path.stat().st_size, 'sha256': sha(path)}
               for key, path in sorted(inputs.items())]
    inventory = {
        'status': 'complete_audited_question_state_freeze_snapshot',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'primary_audit_sha256': sha(audit), 'manifest_sha256': report['manifest_sha256'],
        'packet_manifest_sha256': report['packet_manifest_sha256'], 'release_tag': TAG,
        'primary_readouts': 6912, 'normal_physical_replays': 864, 'archived_readouts': 864,
        'normal_prefix_caches': 16, 'normal_after_caches': 128, 'source_files': 261,
        'arm_style_episode_cells': 144, 'normal_intervention_pairs': 6048, 'wording_pairs': 3456,
        'predecessor_release': plan['source_release'],
        'training_checkpoint_release': 'https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/matched-shared-training-timing-2026-11-15',
        'restore_scope': 'Incremental snapshot: requires the published predecessor release chain and fixed adapter checkpoint. Local pretrained IT weights and existing Conda base are separate.',
        'excluded': ['pretrained weights', 'environments', 'caches', 'previously published runs and training checkpoints'],
        'files': records,
    }
    inventory_path = output / 'file_manifest.json'
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')
    archive = output / 'question-state-freeze-20261115.tar.gz'
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
