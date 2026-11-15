"""Publish a complete verified snapshot; roundtrip all assets before publication."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REPO='HUSRCF/HML-Hybrid-Memory-Lab'
TAG='natural-update-load-2026-11-15'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args,check=True):
    result=subprocess.run(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(f'{args[0]} failed ({result.returncode}): {result.stderr.decode(errors="replace")}')
    return result


def api(path):return json.loads(command(['gh','api',path]).stdout)


def find_release():
    result=command(['gh','api',f'repos/{REPO}/releases/tags/{TAG}'],check=False)
    if not result.returncode:return json.loads(result.stdout)
    if '404' not in result.stderr.decode(errors='replace'):
        raise RuntimeError('Cannot inspect release state; no mutation attempted')
    # Unpublished drafts need not resolve through the public tag endpoint.
    page=1
    while True:
        releases=api(f'repos/{REPO}/releases?per_page=100&page={page}')
        matches=[r for r in releases if r['tag_name']==TAG]
        if len(matches)>1:raise ValueError('Multiple matching drafts require explicit reconciliation')
        if matches:return matches[0]
        if len(releases)<100:return None
        page+=1


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def publish(args):
    if args.receipt.exists():raise ValueError('Never overwrite a publication receipt')
    audit=json.loads(args.audit.read_text());inventory=json.loads((args.backup/'file_manifest.json').read_text())
    verification=json.loads((args.backup/'local_verification.json').read_text())
    archive=args.backup/verification['archive']
    if (audit['status']!='completed_and_audited_natural_load' or audit['primary_readouts_audited']!=3456 or
        audit['after_caches_audited']!=256 or audit['prefix_caches_audited']!=32 or audit['source_files_verified']!=240 or
        len(audit['metrics']['cells'])!=36 or inventory['primary_audit_sha256']!=sha(args.audit) or
        inventory['manifest_sha256']!=audit['manifest_sha256'] or inventory['release_tag']!=TAG or
        verification['status']!='all_archive_members_byte_verified' or verification['sha256']!=sha(archive) or
        verification['bytes']!=archive.stat().st_size or verification['files_verified']!=len(inventory['files']) or
        verification['archive_entries']!=len(inventory['files'])+1 or any(audit['prior_failed_criteria_unchanged'].values())):
        raise ValueError('Complete untouched audit and byte-verified archive required')
    target=command(['git','rev-parse',args.target+'^{commit}']).stdout.decode().strip()
    if command(['git','rev-parse','HEAD']).stdout.decode().strip()!=target:
        raise ValueError('Publish the explicitly reviewed current commit')
    code_files=0
    for record in inventory['files']:
        if not record['path'].startswith('code/'):continue
        relative=record['path'][5:]
        blob=command(['git','show',f'{target}:{relative}']).stdout
        if hashlib.sha256(blob).hexdigest()!=record['sha256'] or len(blob)!=record['bytes']:
            raise ValueError(f'Committed snapshot bytes differ: {relative}')
        code_files+=1
    if api(f'repos/{REPO}/commits/{target}')['sha']!=target:
        raise ValueError('Exact snapshot commit must already be pushed')
    files=[archive,args.backup/'file_manifest.json',args.backup/'local_verification.json']
    expected={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
    receipt={'status':'publishing','started_utc':datetime.now(timezone.utc).isoformat(),'repository':REPO,
        'tag':TAG,'target_commit':target,'audit_sha256':sha(args.audit),'archive_local_verification':verification,
        'committed_snapshot_files_byte_verified':code_files,'expected_assets':expected,
        'publication_source_sha256':sha(Path(__file__)),'release_notes_sha256':sha(args.notes)}
    if args.prior_receipt:
        previous=json.loads(args.prior_receipt.read_text())
        if previous['status']!='failed' or previous['target_commit']!=target or previous['audit_sha256']!=sha(args.audit):
            raise ValueError('Retain the exact prior failed publication attempt')
        receipt['prior_failed_receipt']={'path':str(args.prior_receipt),'sha256':sha(args.prior_receipt),
            'source_sha256':previous['publication_source_sha256']}
    write(args.receipt,receipt)
    try:
        release=find_release()
        if release is None:
            command(['gh','release','create',TAG,*map(str,files),'--repo',REPO,'--target',target,
                '--draft','--title','Normal-update load diagnostic: complete frozen-model evidence',
                '--notes-file',str(args.notes)])
            release=find_release()
        if release is None:raise ValueError('Created draft is not visible to its owner')
        route=f'repos/{REPO}/releases/{release["id"]}'
        receipt['release_id']=release['id']
        if release['target_commitish']!=target:raise ValueError('Preserve a pre-existing release with a different target')
        assets={a['name']:a for a in release['assets']}
        if set(assets)<set(expected) and release['draft']:
            for name,a in assets.items():
                want=expected[name]
                if a['size']!=want['bytes'] or a.get('digest')!='sha256:'+want['sha256']:
                    raise ValueError('Preserve a pre-existing draft asset with unexpected bytes')
            missing=[str(p) for p in files if p.name not in assets]
            command(['gh','release','upload',TAG,*missing,'--repo',REPO])
            release=api(route);assets={a['name']:a for a in release['assets']}
        if set(assets)!=set(expected):raise ValueError('Release asset inventory mismatch; preserve existing remote state')
        for name,want in expected.items():
            a=assets[name]
            if a['size']!=want['bytes'] or a.get('digest')!='sha256:'+want['sha256']:
                raise ValueError('Server-reported asset size/digest mismatch')
        # Asset IDs also work while the tag has not been published.
        for name,want in expected.items():
            payload=command(['gh','api','-H','Accept: application/octet-stream',
                f'repos/{REPO}/releases/assets/{assets[name]["id"]}']).stdout
            if len(payload)!=want['bytes'] or hashlib.sha256(payload).hexdigest()!=want['sha256']:
                raise ValueError('Downloaded remote bytes differ from verified local snapshot')
        if release['draft']:
            command(['gh','api','--method','PATCH',route,'-F','draft=false','-f','make_latest=false'])
        published=api(route)
        if (published['draft'] or published['target_commitish']!=target or
            api(f'repos/{REPO}/commits/{TAG}')['sha']!=target):raise ValueError('Final public release/tag verification failed')
        refreshed={a['name']:a for a in published['assets']}
        if set(refreshed)!=set(expected) or any(refreshed[n]['size']!=v['bytes'] or
            refreshed[n].get('digest')!='sha256:'+v['sha256'] for n,v in expected.items()):
            raise ValueError('Published asset metadata drift')
        receipt.update(status='published_remote_assets_roundtrip_byte_verified',completed_utc=datetime.now(timezone.utc).isoformat(),
            release_url=published['html_url'],release_id=published['id'],published_at=published['published_at'],
            remote_assets=[{'name':n,'id':refreshed[n]['id'],**v,'downloaded_bytes_verified':True} for n,v in expected.items()])
    except BaseException as e:receipt.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:
        write(args.receipt,receipt);args.receipt.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--backup',type=Path,required=True);p.add_argument('--notes',type=Path,required=True)
    p.add_argument('--target',required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--prior-receipt',type=Path)
    result=publish(p.parse_args());print(json.dumps({k:result[k] for k in ('status','release_url','target_commit','committed_snapshot_files_byte_verified')}),flush=True)

if __name__=='__main__':main()
