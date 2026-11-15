"""Byte-verified snapshot of post-tail cache crossovers and their provenance."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_post_tail_path import audit_run

STORAGE=Path('/media/990Pro/Code/SAS');PHASE='post-tail-path';TAG='post-tail-path-crossover-2026-11-15'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def package(audit,output):
    report=json.loads(audit.read_text());run=Path(report['source_run']);m=json.loads((run/'manifest.json').read_text())
    plan=json.loads((run/'plan.json').read_text())
    if (report['status']!='completed_and_audited_post_tail_path' or report!=audit_run(run) or
        report['new_primary_readouts_audited']!=4608 or report['source_N_H_caches_audited']!=128 or
        report['factorial_caches_audited']!=512 or report['endpoint_operational_replays_audited']!=1536 or
        report['source_files_verified']!=214 or report['metrics']['cell_count']!=32 or any(report['prior_failed_criteria_unchanged'].values())):
        raise ValueError('Complete audited crossover, all32cells and unchanged failed criteria required')
    inputs={}
    def add(path,base,prefix):
        if path.is_symlink() or not path.is_file():raise ValueError('Real regular snapshot files required')
        rel=path.relative_to(base)
        if set(rel.parts)&{'models','.git','.cache','__pycache__','.venv','.pytest_cache'} or path.suffix in ('.safetensors','.pt','.pth'):
            raise ValueError('Exclude pretrained weights, environments and caches')
        key=prefix+'/'+rel.as_posix()
        if key in inputs and inputs[key]!=path:raise ValueError('Snapshot path collision')
        inputs[key]=path
    def tree(path,base,prefix):
        if path.is_symlink() or not path.is_dir():raise ValueError('Explicit real phase directories required')
        for p in sorted(path.rglob('*')):
            if p.is_file():add(p,base,prefix)
    for directory in sorted((STORAGE/'runs/memory_update').glob(PHASE+'*')):
        status=json.loads((directory/'manifest.json').read_text())['status']
        if status not in ('completed','failed'):raise ValueError('Every phase job must be terminal before backup')
        tree(directory,STORAGE,'storage')
    tree(Path(m['packet']),STORAGE,'storage')
    for folder in ('logs/memory_update','artifacts/memory_update'):
        for p in sorted((STORAGE/folder).glob(PHASE+'*')):
            if '-packaging-' in p.name or '-publication-' in p.name:continue
            if p.is_dir():tree(p,STORAGE,'storage')
            elif p.is_file():add(p,STORAGE,'storage')
    for name,digest in m['source_sha256'].items():
        if sha(ROOT/name)!=digest or sha(run/'sources'/name)!=digest:raise ValueError('Preserve all214sealed measurement sources')
        add(ROOT/name,ROOT,'code')
    tree(ROOT/'materials/updates/post-tail-path-20261115',ROOT,'code')
    for name in ('README.md','materials/README.md','.gitattributes','requirements.txt','configs/experiment_progress.json',
        'scripts/run_post_tail_path_pipeline.sh','scripts/export_post_tail_path.py','scripts/package_post_tail_path.py','scripts/publish_post_tail_path.py'):
        add(ROOT/name,ROOT,'code')
    for model,state in m['models'].items():
        for kind in ('primary','endpoints'):
            path=run/f'{model}.{kind}.jsonl'
            if sha(path)!=state['results_sha256'][kind] or 'storage/'+path.relative_to(STORAGE).as_posix() not in inputs:
                raise ValueError('All4complete raw files required')
    output.mkdir(parents=True,exist_ok=False)
    records=[{'path':key,'bytes':p.stat().st_size,'sha256':sha(p)} for key,p in sorted(inputs.items())]
    manifest={'status':'complete_audited_post_tail_path_snapshot','created_utc':datetime.now(timezone.utc).isoformat(),
        'primary_audit_sha256':sha(audit),'manifest_sha256':report['manifest_sha256'],'packet_manifest_sha256':report['packet_manifest_sha256'],
        'new_primary_readouts':4608,'source_N_H_caches':128,'factorial_caches':512,'endpoint_replays':1536,'source_files':214,'release_tag':TAG,
        'predecessor_release':plan['source_release'],
        'training_checkpoint_release':'https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/matched-shared-training-timing-2026-11-15',
        'restore_scope':'This incremental snapshot requires the published predecessor release chain and earlier fixed adapter checkpoints. Local pretrained IT weights and existing Conda base are required separately.',
        'excluded':['pretrained weights','environments','caches','previously published runs and training checkpoints'],'files':records}
    mp=output/'file_manifest.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    archive=output/'post-tail-path-20261115.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for record in records:
            p=inputs[record['path']]
            if sha(p)!=record['sha256']:raise ValueError('Snapshot input changed while packing')
            tar.add(p,arcname=record['path'],recursive=False)
        tar.add(mp,arcname='file_manifest.json',recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        if len(tar.getmembers())!=len(records)+1 or {v.name for v in tar.getmembers()}!={r['path'] for r in records}|{'file_manifest.json'}:
            raise ValueError('Complete archive member inventory differs')
        for record in records+[{'path':'file_manifest.json','bytes':mp.stat().st_size,'sha256':sha(mp)}]:
            member=tar.getmember(record['path'])
            if not member.isfile() or member.size!=record['bytes']:raise ValueError('Archive member type or size differs')
            digest=hashlib.sha256()
            with tar.extractfile(member) as handle:
                for chunk in iter(lambda:handle.read(4*1024*1024),b''):digest.update(chunk)
            if digest.hexdigest()!=record['sha256']:raise ValueError('Decompressed archive bytes differ')
    receipt={'status':'all_archive_members_byte_verified','files_verified':len(records),'archive_entries':len(records)+1,
        'archive':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)}
    (output/'local_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(package(a.audit,a.output)),flush=True)

if __name__=='__main__':main()
