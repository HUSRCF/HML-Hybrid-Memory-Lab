"""Read-only publication scope verification for the registered synthetic experiment."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT=Path('/home/husrcf/Code/SAS')
STORAGE=Path('/media/990Pro/Code/SAS')
REPO='HUSRCF/HML-Hybrid-Memory-Lab'
OUTPUT=STORAGE/'artifacts/memory_update/question-state-freeze-publication-content-scope-20261115-01.json'
BACKUP=STORAGE/'backups/question-state-freeze-20261115-01'

def cmd(args):return subprocess.check_output(args,cwd=ROOT)
def sha(data):return hashlib.sha256(data).hexdigest()

if OUTPUT.exists():raise ValueError('Never overwrite scope evidence')
repo=json.loads(cmd(['gh','api','repos/'+REPO]))
if repo['private'] or repo['visibility']!='public':raise ValueError('Explicit public repository required')
target=cmd(['git','rev-parse','HEAD']).decode().strip()
if json.loads(cmd(['gh','api','repos/'+REPO+'/commits/'+target]))['sha']!=target:raise ValueError('Exact public target required')
prior=json.loads(cmd(['gh','api','repos/'+REPO+'/releases/tags/version-question-control-2026-11-15']))
if prior['draft']:raise ValueError('Prior same-scope public release required')
objects=set()
for item in cmd(['git','ls-tree','-r','-z',target]).split(b'\0'):
    if not item:continue
    metadata,_=item.split(b'\t',1)
    mode,kind,oid=metadata.split()
    if kind==b'blob':objects.add(oid)
ids=sorted(objects)
batch=subprocess.run(['git','cat-file','--batch'],input=b'\n'.join(ids)+b'\n',cwd=ROOT,stdout=subprocess.PIPE,check=True).stdout
public={};offset=0
for oid in ids:
    end=batch.index(b'\n',offset);header=batch[offset:end].split();size=int(header[2]);start=end+1
    payload=batch[start:start+size]
    if header[0]!=oid or header[1]!=b'blob' or batch[start+size:start+size+1]!=b'\n':raise ValueError('Git blob stream differs')
    public[sha(payload)]=len(payload);offset=start+size+1
if offset!=len(batch):raise ValueError('Trailing git blob bytes')
verification=json.loads((BACKUP/'local_verification.json').read_text())
archive=BACKUP/verification['archive']
if sha(archive.read_bytes())!=verification['sha256']:raise ValueError('Archive changed')
patterns={
    'private_key_header':rb'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----',
    'github_access_token':rb'\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b',
    'huggingface_access_token':rb'\bhf_[A-Za-z0-9]{20,}\b',
    'aws_access_key':rb'\bAKIA[A-Z0-9]{16}\b',
}
matching=0;unmatched=[];hits=[];total=0
with tarfile.open(archive,'r:gz') as tar:
    for member in tar.getmembers():
        if not member.isfile():raise ValueError('Unexpected nonregular archive entry')
        path=Path(member.name)
        if set(path.parts)&{'models','.ssh','.aws','.git','.cache','.venv'} or path.suffix in ('.safetensors','.pt','.pth','.pem','.key') or path.name.startswith('.env'):
            raise ValueError('Excluded material in archive')
        data=tar.extractfile(member).read();total+=1
        digest=sha(data)
        for name,pattern in patterns.items():
            if re.search(pattern,data):hits.append({'path':member.name,'pattern':name})
        if digest in public and public[digest]==len(data):matching+=1
        else:
            allowed=member.name=='file_manifest.json' or any(member.name.startswith(prefix) for prefix in (
                'storage/runs/memory_update/question-state-freeze',
                'storage/datasets/memory_update/question-state-freeze-v1/',
                'storage/artifacts/memory_update/question-state-freeze',
                'storage/logs/memory_update/question-state-freeze'))
            if not allowed:raise ValueError('Unclassified unpublished snapshot path: '+member.name)
            unmatched.append({'path':member.name,'bytes':len(data),'sha256':digest})
if hits:raise ValueError('Credential-pattern hits; public release not attempted')
receipt={'status':'public_commit_and_registered_synthetic_material_scope_verified',
    'created_utc':datetime.now(timezone.utc).isoformat(),'repository':REPO,'repository_visibility':repo['visibility'],
    'target_public_commit':target,'prior_public_release':prior['html_url'],
    'archive_sha256':verification['sha256'],'archive_members':total,'members_identical_to_public_git_blob':matching,
    'remaining_registered_synthetic_or_runtime_members':unmatched,'credential_pattern_hits':hits,
    'excluded':['pretrained weights','environments','caches','SSH/AWS material','credential files'],
    'source_sha256':sha(Path(__file__).read_bytes()),
    'scope':'User requested repository backup including materials and continued experiments. Remaining files are synthetic SET histories/questions, generated diagnostic outputs and runtime provenance under the registered phase paths. This is the same material scope as the preceding public release.'}
OUTPUT.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ('status','target_public_commit','archive_members','members_identical_to_public_git_blob')}))
