"""Build and byte-verify the complete evidence-access phase increment; never upload weights."""

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
STORAGE=Path('/media/990Pro/Code/SAS')
PHASE='trained-timing-evidence-access'
TAG='trained-timing-evidence-access-2026-11-15'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def package(audit_path,output):
    report=json.loads(audit_path.read_text());run=Path(report['source_run']);m=json.loads((run/'manifest.json').read_text())
    if (report['status']!='completed_and_audited_trained_timing_evidence_access' or m['status']!='completed' or m['smoke'] or
        report['new_physical_readouts_audited']!=5040 or report['new_related_bindings_audited']!=6480 or
        report['manifest_sha256']!=sha(run/'manifest.json') or
        any(report['prior_timing_criteria_unchanged'].values())):
        raise ValueError('Complete unchanged primary audit and preserved failed old criteria required')
    inputs={}
    def add(path,base,prefix):
        if path.is_symlink() or not path.is_file():raise ValueError('Real explicit regular files only')
        rel=path.relative_to(base)
        if set(rel.parts)&{'models','.git','.cache','__pycache__','.pytest_cache','.venv'} or path.suffix in {'.safetensors','.pt','.pth'}:
            raise ValueError('Exclude native/training weights, environments and caches')
        key=f'{prefix}/{rel.as_posix()}'
        if key in inputs and inputs[key]!=path:raise ValueError('Archive path collision')
        inputs[key]=path
    def tree(path):
        if path.is_symlink() or not path.is_dir():raise ValueError('Explicit real phase directory required')
        for p in sorted(path.rglob('*')):
            if p.is_file():add(p,STORAGE,'storage')
    dataset=Path(m['dataset'])
    for root in [run,Path(str(run).replace(PHASE+'-20261115-01',PHASE+'-smoke-forward-20261115-01')),
                 Path(str(run).replace(PHASE+'-20261115-01',PHASE+'-smoke-reverse-20261115-01')),dataset,
                 Path(json.loads((run/'plan.json').read_text())['source_dataset']),
                 Path(json.loads((run/'plan.json').read_text())['source_run'])]:tree(root)
    previous=Path(json.loads((run/'plan.json').read_text())['source_audit'])
    for p in (previous,previous.with_suffix('.source.py')):add(p,STORAGE,'storage')
    for folder in ('logs','artifacts/memory_update'):
        for p in sorted((STORAGE/folder).glob(PHASE+'*')):
            if p.is_file():add(p,STORAGE,'storage')
    for name,digest in m['source_sha256'].items():
        p=ROOT/name
        if sha(p)!=digest or sha(run/'sources'/name)!=digest:raise ValueError('Sealed inference source changed')
        add(p,ROOT,'code')
    material=ROOT/'materials/updates'/f'{PHASE}-20261115'
    for p in sorted(material.rglob('*')):
        if p.is_file():add(p,ROOT,'code')
    for name in ['README.md','.gitattributes','requirements.txt','configs/experiment_progress.json',
                 'scripts/export_trained_timing_evidence_access.py','scripts/package_trained_timing_evidence_access.py',
                 'tests/test_trained_timing_evidence_access.py']:
        add(ROOT/name,ROOT,'code')
    # Require every actual primary arm/regime file explicitly, not just a directory count.
    for arm,state in m['models'].items():
        for regime,digest in state['results_sha256'].items():
            path=run/f'{arm}.{regime}.jsonl'
            if sha(path)!=digest or 'storage/'+path.relative_to(STORAGE).as_posix() not in inputs:
                raise ValueError('Every one of the fifteen complete raw result files must be archived')
    output.mkdir(parents=True,exist_ok=False)
    records=[{'path':name,'bytes':path.stat().st_size,'sha256':sha(path)} for name,path in sorted(inputs.items())]
    receipt={'status':'complete_audited_evidence_access_snapshot','created_utc':datetime.now(timezone.utc).isoformat(),
             'primary_audit_sha256':sha(audit_path),'manifest_sha256':report['manifest_sha256'],
             'dataset_manifest_sha256':report['dataset_manifest_sha256'],'mask_registry_sha256':report['mask_registry_sha256'],
             'release_tag':TAG,'predecessor_release':'https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/trained-shared-timing-confirmation-2026-11-15',
             'training_checkpoint_release':'https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/matched-shared-training-timing-2026-11-15',
             'excluded':['pretrained weights','environments','compiler caches','previously published training checkpoint bytes'],
             'files':records}
    mp=output/'file_manifest.json';mp.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    archive=output/f'{PHASE}-20261115.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for rec in records:
            p=inputs[rec['path']]
            if sha(p)!=rec['sha256']:raise ValueError('Input changed during snapshot')
            tar.add(p,arcname=rec['path'],recursive=False)
        tar.add(mp,arcname='file_manifest.json',recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)!=len(records)+1 or {v.name for v in members}!={r['path'] for r in records}|{'file_manifest.json'}:
            raise ValueError('Archive entries differ from the complete manifest')
        for rec in records+[{'path':'file_manifest.json','bytes':mp.stat().st_size,'sha256':sha(mp)}]:
            member=tar.getmember(rec['path'])
            if not member.isfile() or member.size!=rec['bytes']:raise ValueError('Archive member type/size mismatch')
            h=hashlib.sha256()
            with tar.extractfile(member) as file:
                for block in iter(lambda:file.read(4*1024*1024),b''):h.update(block)
            if h.hexdigest()!=rec['sha256']:raise ValueError('Decompressed archive bytes mismatch')
    result={'status':'all_archive_members_byte_verified','files_verified':len(records),'archive_entries':len(records)+1,
            'archive':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)}
    (output/'local_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('audit',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(package(args.audit,args.output)),flush=True)


if __name__=='__main__':main()
