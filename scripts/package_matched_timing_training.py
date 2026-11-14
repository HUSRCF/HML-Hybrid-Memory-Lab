"""Additive, byte-verified backup of the two audited training schedules."""

import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
STORAGE=Path('/media/990Pro/Code/SAS')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def package(output):
    audit=STORAGE/'artifacts/memory_update/matched-shared-training-timing-audit-20261115-01.json'
    report=json.loads(audit.read_text());run=Path(report['source_run'])
    manifest=json.loads((run/'manifest.json').read_text())
    if (report['status']!='completed_and_audited_matched_shared_training_timing' or not report['operational_controls_passed'] or
        report['optimizer_updates_audited']!=864 or report['manifest_sha256']!=sha(run/'manifest.json') or
        report['source_files_verified']!=155 or sha(audit.with_suffix('.source.py'))!=report['analysis_source_sha256']):
        raise ValueError('Completed independent full training audit required')
    for name,digest in manifest['source_sha256'].items():
        if sha(ROOT/name)!=digest or sha(run/'sources'/name)!=digest:raise ValueError('Sealed training source changed')
    files={}
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for name in tracked:
        p=ROOT/name
        if name and p.is_file() and not p.is_symlink() and not name.startswith('materials/'):
            files['workspace/'+name]=p
    files['workspace/scripts/package_matched_timing_training.py']=Path(__file__)
    selected=[run,Path(manifest['data_source'])] if 'data_source' in manifest else [run]
    selected += [STORAGE/'runs/memory_update/matched-shared-training-timing-smoke-20261115-01']
    selected += [STORAGE/'datasets/memory_update/paired-semantic-order-augmented-v1']
    for folder in selected:
        for p in folder.rglob('*'):
            if p.is_file() and not p.is_symlink():files['storage/'+str(p.relative_to(STORAGE))]=p
    for folder in ['artifacts/memory_update','logs']:
        for p in (STORAGE/folder).glob('matched-shared-training-timing*'):
            if p.is_file():files['storage/'+str(p.relative_to(STORAGE))]=p
    for p in (STORAGE/'logs').glob('trained-shared-timing-*tests*'):
        if p.is_file():files['storage/'+str(p.relative_to(STORAGE))]=p
    records=[{'path':name,'bytes':p.stat().st_size,'sha256':sha(p)} for name,p in sorted(files.items())]
    index={'status':'complete_audited_training_snapshot','created_utc':datetime.now(timezone.utc).isoformat(),
        'audit_sha256':sha(audit),'training_manifest_sha256':report['manifest_sha256'],
        'predecessor_release':'https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/unseen-shared-timing-2026-11-15',
        'excluded':['pretrained backbone weights','environments','compiler caches','unfinished confirmation runs'],
        'files':records}
    output.mkdir(parents=True,exist_ok=False);target=output/'matched-shared-training-timing-20261115.tar.gz'
    raw=(json.dumps(index,indent=2)+'\n').encode()
    with tarfile.open(target,'w:gz',compresslevel=6) as archive:
        for name,p in sorted(files.items()):archive.add(p,arcname=name,recursive=False)
        header=tarfile.TarInfo('file_manifest.json');header.size=len(raw);archive.addfile(header,io.BytesIO(raw))
    with tarfile.open(target,'r:gz') as archive:
        members=archive.getmembers()
        if len(members)!=len(records)+1:raise ValueError('Archive member count differs')
        for record in records:
            actual=archive.extractfile(record['path']).read()
            if len(actual)!=record['bytes'] or hashlib.sha256(actual).hexdigest()!=record['sha256']:
                raise ValueError('Archive bytes differ from registered source file')
        if archive.extractfile('file_manifest.json').read()!=raw:raise ValueError('Embedded manifest differs')
    (output/'file_manifest.json').write_bytes(raw)
    verification={'status':'all_archive_members_byte_verified','files_verified':len(records),
        'archive_entries':len(records)+1,'archive':target.name,'bytes':target.stat().st_size,'sha256':sha(target)}
    (output/'local_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    package(parser.parse_args().output)
