"""Snapshot update-budget measurements, archived references and complete verification evidence."""

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
STORAGE=Path('/media/990Pro/Code/SAS')
PHASE='update-budget'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def package(audit,output):
    report=json.loads(audit.read_text());run=Path(report['source_run']);m=json.loads((run/'manifest.json').read_text())
    plan=json.loads((run/'plan.json').read_text())
    if (report['status']!='completed_and_audited_update_budget' or m['status']!='completed' or m['smoke'] or
        report['new_physical_readouts_audited']!=13440 or report['related_bindings_audited']!=17280 or
        report['manifest_sha256']!=sha(run/'manifest.json') or any(report['prior_timing_criteria_unchanged'].values())):
        raise ValueError('Complete unchanged primary audit and failed old criteria required')
    auxpath=STORAGE/'artifacts/memory_update/update-budget-prefix-verification-20261115-01.json'
    aux=json.loads(auxpath.read_text());checks=aux['checks']
    if (aux['status']!='completed' or len(checks)!=240 or aux['weights_before']!=aux['weights_after'] or
        not aux['hooks_restored'] or aux['primary_source_sha256']!=m['source_sha256'] or
        aux['source_sha256']!=sha(ROOT/'scripts/verify_update_budget_prefix.py') or
        sha(auxpath.with_suffix('.source.py'))!=aux['source_sha256'] or
        aux['protocol_sha256']!=sha(ROOT/'configs/update_budget_controls_plan.json') or
        any(not c['complete_prefix_memory_exact'] or not c['prefix_logits_exact'] or
            c['prefix_sha256']!=c['masked_prefix_sha256'] or c['historical_kv_fields_exact']!=12 for c in checks)):
        raise ValueError('All240 complete-cache/logits/KV preservation checks required')
    controls=m['control_query_ids'];names=plan['model_order']
    if [(c['model'],c['id'],c['write_family']) for c in checks]!=[(n,k,g) for n in names for k in controls for g in ('random0','random1','early','recent')]:
        raise ValueError('All5models x12controls x4write policies required without duplicates')
    for c in checks:
        state=m['models'][c['model']];family=c['write_family'];regime=family+'_full';restricted=family+'_oracle'
        reference={v['id']:v['readout'] for v in state['condition_controls'][regime]}
        masked={v['id']:v['readout'] for v in state['condition_controls'][restricted]}
        if c['regime']!=regime or c['readout']!=reference[c['id']] or c['restricted_readout']!=masked[c['id']]:
            raise ValueError('All auxiliary five-field readouts must match the complete primary controls')
        gates=c['actual_native_gates'];stats=list(gates['layers'].values())
        if (not gates['enabled'] or not gates['actual_gates_finite_and_exact'] or gates['component_mode']!='delta' or
            set(gates['layers'])!={str(i) for i in range(24) if i%4!=3} or
            any(s!=stats[0] or s['held_rows']<=0 or s['rows']!=s['held_rows']+s['unheld_rows'] or
                s['forced_decay_rows']!=0 or
                s['forced_delta_rows']!=s['held_rows'] for s in stats)):
            raise ValueError('Auxiliary actual18layer component gate coverage changed')
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
    predecessor=Path(plan['source_run'])
    for p in sorted((STORAGE/'runs/memory_update').glob(PHASE+'*')):
        if not p.is_dir():continue
        state=json.loads((p/'manifest.json').read_text())
        if state['status'] not in ('completed','failed'):raise ValueError('All phase jobs must be terminal before backup')
        tree(p)
    for p in [Path(m['dataset']),predecessor,Path(plan['source_dataset'])]:tree(p)
    for p in [Path(plan['source_audit'])]:
        add(p,STORAGE,'storage');add(p.with_suffix('.source.py'),STORAGE,'storage')
    for folder in ('logs','logs/memory_update'):
        for p in sorted((STORAGE/folder).glob(PHASE+'*')):
            if p.is_file() and '-packaging-' not in p.name:add(p,STORAGE,'storage')
    for folder in ('logs/memory_update','artifacts/memory_update'):
        for p in sorted((STORAGE/folder).glob(PHASE+'*')):
            # This log is still being written by the packager; publish its final receipt separately.
            if p.is_file() and '-packaging-' not in p.name:add(p,STORAGE,'storage')
            elif p.is_dir():tree(p)
    for name,digest in m['source_sha256'].items():
        if sha(ROOT/name)!=digest or sha(run/'sources'/name)!=digest:raise ValueError('Preserve197 sealed primary sources')
        add(ROOT/name,ROOT,'code')
    material=ROOT/'materials/updates'/f'{PHASE}-20261115'
    for p in sorted(material.rglob('*')):
        if p.is_file():add(p,ROOT,'code')
    for name in ['README.md','materials/README.md','.gitattributes','requirements.txt','configs/experiment_progress.json',
                 'scripts/export_update_budget.py','scripts/verify_update_budget_prefix.py',
                 'scripts/package_update_budget.py','scripts/publish_update_budget.py',
                 'configs/gate_component_kernel_reference_addendum.json',
                 'scripts/run_update_budget_pipeline.sh','scripts/analyze_update_budget_mask_structure.py',
                 'scripts/report_update_budget.py']:
        add(ROOT/name,ROOT,'code')
    for arm,state in m['models'].items():
        for regime,digest in state['results_sha256'].items():
            p=run/f'{arm}.{regime}.jsonl'
            if sha(p)!=digest or 'storage/'+p.relative_to(STORAGE).as_posix() not in inputs:raise ValueError('Every40 full raw files required')
    output.mkdir(parents=True,exist_ok=False)
    records=[{'path':key,'bytes':p.stat().st_size,'sha256':sha(p)} for key,p in sorted(inputs.items())]
    manifest={'status':'complete_audited_update_budget_snapshot','created_utc':datetime.now(timezone.utc).isoformat(),
        'primary_audit_sha256':sha(audit),'manifest_sha256':report['manifest_sha256'],'dataset_manifest_sha256':report['dataset_manifest_sha256'],
        'policy_registry_sha256':m['policies_sha256'],'new_primary_readouts':13440,'archived_reference_readouts':6720,'release_tag':'update-budget-controls-2026-11-15',
        'predecessor_release':'https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/gate-components-factorial-2026-11-15',
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
