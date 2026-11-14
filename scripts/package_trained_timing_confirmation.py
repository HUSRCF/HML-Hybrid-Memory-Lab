"""Byte-verified incremental backup after the complete16-arm confirmation audit."""

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


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def package(output):
    audit=STORAGE/'artifacts/memory_update/trained-shared-timing-audit-20261115-01.json'
    report=json.loads(audit.read_text());run=Path(report['source_run']);m=json.loads((run/'manifest.json').read_text())
    data=Path(m['dataset']);dataset=json.loads((data/'manifest.json').read_text())
    if (report['status']!='completed_and_audited_trained_shared_timing' or not report['operational_controls_passed'] or
        m['status']!='completed' or m['smoke'] or report['counts']['physical_readouts']!=5376 or
        report['source_files_verified']!=160 or len(m['completed_models'])!=16 or
        report['manifest_sha256']!=sha(run/'manifest.json') or report['dataset_manifest_sha256']!=sha(data/'manifest.json') or
        sha(audit.with_suffix('.source.py'))!=report['analysis_source_sha256']):
        raise ValueError('Complete independently audited5376-readout confirmation required')
    for name,digest in m['source_sha256'].items():
        if sha(ROOT/name)!=digest or sha(run/'sources'/name)!=digest:raise ValueError('Sealed runtime source changed')
    for name,state in m['models'].items():
        if sha(run/f'{name}.jsonl')!=state['results_sha256']:raise ValueError('Complete primary raw results changed')
    for name,digest in dataset['source_sha256'].items():
        if sha(ROOT/name)!=digest or sha(data/'sources'/name)!=digest:raise ValueError('Sealed dataset source changed')
    files={}
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for name in tracked:
        p=ROOT/name
        if name and p.is_file() and not p.is_symlink() and not name.startswith('materials/'):
            files['workspace/'+name]=p
    files['workspace/scripts/package_trained_timing_confirmation.py']=Path(__file__)
    for folder in [data,run,STORAGE/'runs/memory_update/trained-shared-timing-smoke-forward-20261115-01',
                   STORAGE/'runs/memory_update/trained-shared-timing-smoke-reverse-20261115-01']:
        for p in folder.rglob('*'):
            if p.is_file() and not p.is_symlink():files['storage/'+str(p.relative_to(STORAGE))]=p
    for folder in ['artifacts/memory_update','logs']:
        for p in (STORAGE/folder).glob('trained-shared-timing*'):
            if p.is_file() and '-backup-' not in p.name:files['storage/'+str(p.relative_to(STORAGE))]=p
    records=[{'path':name,'bytes':p.stat().st_size,'sha256':sha(p)} for name,p in sorted(files.items())]
    index={'status':'complete_audited_matched_timing_confirmation_snapshot','created_utc':datetime.now(timezone.utc).isoformat(),
        'primary_audit_sha256':sha(audit),'manifest_sha256':report['manifest_sha256'],
        'dataset_manifest_sha256':report['dataset_manifest_sha256'],
        'predecessor_release':'https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/matched-shared-training-timing-2026-11-15',
        'excluded':['pretrained weights','environments','compiler caches','previously published training checkpoint bytes'],
        'files':records}
    output.mkdir(parents=True,exist_ok=False);target=output/'trained-shared-timing-confirmation-20261115.tar.gz'
    raw=(json.dumps(index,indent=2)+'\n').encode()
    with tarfile.open(target,'w:gz',compresslevel=6) as tar:
        for name,p in sorted(files.items()):tar.add(p,arcname=name,recursive=False)
        info=tarfile.TarInfo('file_manifest.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    with tarfile.open(target,'r:gz') as tar:
        if len(tar.getmembers())!=len(records)+1:raise ValueError('Archive member count differs')
        for record in records:
            actual=tar.extractfile(record['path']).read()
            if len(actual)!=record['bytes'] or hashlib.sha256(actual).hexdigest()!=record['sha256']:
                raise ValueError('Archive member bytes differ')
        if tar.extractfile('file_manifest.json').read()!=raw:raise ValueError('Embedded manifest differs')
    (output/'file_manifest.json').write_bytes(raw)
    receipt={'status':'all_archive_members_byte_verified','files_verified':len(records),'archive_entries':len(records)+1,
        'archive':target.name,'bytes':target.stat().st_size,'sha256':sha(target)}
    (output/'local_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    package(parser.parse_args().output)
