"""Snapshot complete text controls and references; verify every decompressed byte."""

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
STORAGE=Path('/media/990Pro/Code/SAS')
PHASE='text-competence-factorial'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def package(audit,output):
    report=json.loads(audit.read_text());run=Path(report['source_run']);m=json.loads((run/'manifest.json').read_text())
    plan=json.loads((run/'plan.json').read_text())
    if (report['status']!='completed_and_audited_text_competence_factorial' or m['status']!='completed' or m['smoke'] or
        report['new_physical_readouts_audited']!=6720 or report['related_bindings_audited']!=8640 or
        report['manifest_sha256']!=sha(run/'manifest.json') or any(report['prior_timing_criteria_unchanged'].values())):
        raise ValueError('Complete unchanged primary audit and failed old criteria required')
    inputs={}
    def add(p,base,prefix):
        if p.is_symlink() or not p.is_file():raise ValueError('Only explicit real regular files')
        rel=p.relative_to(base)
        if set(rel.parts)&{'models','.git','.cache','__pycache__','.pytest_cache','.venv'} or p.suffix in {'.safetensors','.pt','.pth'}:
            raise ValueError('Exclude weights, environments and caches')
        key=f'{prefix}/{rel.as_posix()}'
        if key in inputs and inputs[key]!=p:raise ValueError('Archive collision')
        inputs[key]=p
    def tree(p):
        if p.is_symlink() or not p.is_dir():raise ValueError('Explicit phase directories only')
        for q in sorted(p.rglob('*')):
            if q.is_file():add(q,STORAGE,'storage')
    predecessor=Path(plan['source_run']);prior=json.loads((predecessor/'plan.json').read_text())
    for p in [run,run.with_name(PHASE+'-smoke-forward-20261115-01'),run.with_name(PHASE+'-smoke-reverse-20261115-01'),
              Path(m['dataset']),predecessor,Path(json.loads((predecessor/'manifest.json').read_text())['dataset']),
              Path(plan['source_dataset']),Path(prior['source_run'])]:tree(p)
    for p in [Path(plan['source_audit']),Path(prior['source_audit'])]:
        add(p,STORAGE,'storage');add(p.with_suffix('.source.py'),STORAGE,'storage')
    for folder in ('logs','artifacts/memory_update'):
        for p in sorted((STORAGE/folder).glob(PHASE+'*')):
            # This log is still being written by the packager; publish its final receipt separately.
            if p.is_file() and '-packaging-' not in p.name:add(p,STORAGE,'storage')
    for name,digest in m['source_sha256'].items():
        if sha(ROOT/name)!=digest or sha(run/'sources'/name)!=digest:raise ValueError('Preserve172 sealed sources')
        add(ROOT/name,ROOT,'code')
    material=ROOT/'materials/updates'/f'{PHASE}-20261115'
    for p in sorted(material.rglob('*')):
        if p.is_file():add(p,ROOT,'code')
    for name in ['README.md','materials/README.md','.gitattributes','requirements.txt','configs/experiment_progress.json',
                 'scripts/export_text_competence_factorial.py','scripts/document_text_competence_factorial.py',
                 'scripts/package_text_competence_factorial.py','tests/test_text_competence_factorial.py']:
        add(ROOT/name,ROOT,'code')
    for arm,state in m['models'].items():
        for regime,digest in state['results_sha256'].items():
            p=run/f'{arm}.{regime}.jsonl'
            if sha(p)!=digest or 'storage/'+p.relative_to(STORAGE).as_posix() not in inputs:raise ValueError('Every20 full raw files required')
    output.mkdir(parents=True,exist_ok=False)
    records=[{'path':key,'bytes':p.stat().st_size,'sha256':sha(p)} for key,p in sorted(inputs.items())]
    manifest={'status':'complete_audited_text_competence_factorial_snapshot','created_utc':datetime.now(timezone.utc).isoformat(),
        'primary_audit_sha256':sha(audit),'manifest_sha256':report['manifest_sha256'],'dataset_manifest_sha256':report['dataset_manifest_sha256'],
        'transformation_registry_sha256':report['transformations_sha256'],'release_tag':'text-competence-factorial-2026-11-15',
        'predecessor_release':'https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/trained-timing-evidence-access-2026-11-15',
        'training_checkpoint_release':'https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/matched-shared-training-timing-2026-11-15',
        'excluded':['pretrained weights','environments','caches','previously published training checkpoints'],'files':records}
    mp=output/'file_manifest.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    archive=output/f'{PHASE}-20261115.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for v in records:
            p=inputs[v['path']]
            if sha(p)!=v['sha256']:raise ValueError('Input changed during snapshot')
            tar.add(p,arcname=v['path'],recursive=False)
        tar.add(mp,arcname='file_manifest.json',recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)!=len(records)+1 or {v.name for v in members}!={r['path'] for r in records}|{'file_manifest.json'}:
            raise ValueError('Complete archive coverage differs')
        for v in records+[{'path':'file_manifest.json','bytes':mp.stat().st_size,'sha256':sha(mp)}]:
            member=tar.getmember(v['path'])
            if not member.isfile() or member.size!=v['bytes']:raise ValueError('Archive type or size mismatch')
            h=hashlib.sha256()
            with tar.extractfile(member) as f:
                for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
            if h.hexdigest()!=v['sha256']:raise ValueError('Decompressed byte hash differs')
    receipt={'status':'all_archive_members_byte_verified','files_verified':len(records),'archive_entries':len(records)+1,
             'archive':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)}
    (output/'local_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(package(a.audit,a.output)),flush=True)


if __name__=='__main__':main()
