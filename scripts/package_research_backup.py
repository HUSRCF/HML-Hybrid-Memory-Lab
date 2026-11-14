"""Snapshot authorized research materials without models, environments or caches."""

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
STORAGE=Path('/media/990Pro/Code/SAS')
EXCLUDED={'__pycache__','.pytest_cache','.git','.venv','.cache','models'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def code_files():
    files=[ROOT/name for name in ['README.md','.gitignore','requirements.txt']]
    for name in ['sas_memory','scripts','configs','tests','environment','materials']:
        for p in (ROOT/name).rglob('*'):
            parts=p.relative_to(ROOT).parts
            if (p.is_file() and not p.is_symlink() and not set(parts)&EXCLUDED and
                (name!='materials' or p.name=='README.md')):files.append(p)
    return sorted(set(files))


def pack(base,files,target):
    records=[]
    with tarfile.open(target,'w:gz',compresslevel=6) as tar:
        for p in files:
            rel=p.relative_to(base).as_posix()
            if p.is_symlink():raise ValueError('Do not follow external symlinks')
            records.append({'path':rel,'bytes':p.stat().st_size,'sha256':digest(p)})
            tar.add(p,arcname=rel,recursive=False)
    return records


def split_asset(path,limit=32*1024*1024):
    if path.stat().st_size<=limit:return [path]
    pieces=[]
    with path.open('rb') as source:
        index=1
        while block:=source.read(limit):
            part=path.with_name(path.name+f'.part{index:03d}');part.write_bytes(block)
            pieces.append(part);index+=1
    return pieces


def main(output):
    output.mkdir(parents=True,exist_ok=False);material=ROOT/'materials';material.mkdir(exist_ok=True)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'research_goal_status':'paused',
        'release_tag':'research-backup-2026-11-15','repository':'HUSRCF/HML-Hybrid-Memory-Lab',
        'storage_root':str(STORAGE),'excluded':['pretrained model weights','compiler/download caches','Conda/virtual environments'],
        'archives':{},'release_assets':{}}
    for name in ['code','datasets','runs','artifacts','logs']:
        base=ROOT if name=='code' else STORAGE
        files=code_files() if name=='code' else sorted(p for p in (STORAGE/name).rglob('*') if p.is_file())
        target=output/f'{name}.tar.gz';records=pack(base,files,target)
        pieces=split_asset(target)
        manifest['archives'][name]={'filename':target.name,'bytes':target.stat().st_size,'sha256':digest(target),
            'uncompressed_bytes':sum(r['bytes'] for r in records),'files':records,'parts':[p.name for p in pieces]}
        for p in pieces:manifest['release_assets'][p.name]={'bytes':p.stat().st_size,'sha256':digest(p)}
        print(json.dumps({'packed':name,'files':len(records),'bytes':target.stat().st_size,'parts':len(pieces)}),flush=True)
    (material/'archive_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (material/'code_snapshot_manifest.json').write_text(json.dumps(manifest['archives']['code'],indent=2)+'\n')
    (output/'archive_manifest.json').write_bytes((material/'archive_manifest.json').read_bytes())
    for name in ['reports','logs','registrations','experiment_manifests']:(material/name).mkdir(exist_ok=True)
    for name,folder in [('reports','artifacts'),('logs','logs')]:
        for p in (STORAGE/folder).rglob('*'):
            if p.is_file():
                t=material/name/p.relative_to(STORAGE/folder);t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
    for folder in (STORAGE/'datasets').rglob('manifest.json'):
        if 'sources' in folder.parts:continue
        target=material/'registrations'/folder.relative_to(STORAGE/'datasets');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(folder,target)
    for folder in (STORAGE/'runs').rglob('manifest.json'):
        if 'sources' in folder.parts:continue
        target=material/'experiment_manifests'/folder.relative_to(STORAGE/'runs');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(folder,target)
    print(json.dumps({'backup':str(output),'assets':len(manifest['release_assets']),'manifest':str(material/'archive_manifest.json')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
