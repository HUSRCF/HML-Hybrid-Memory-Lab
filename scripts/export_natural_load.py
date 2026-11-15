"""Export every independently audited normal-update cell without filtering."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_natural_load import audit_run, aggregate
from scripts.natural_load_data import registered_inputs
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def write_csv(path, rows):
    if not rows:
        raise ValueError('No empty exported table')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    with path.open(newline='') as f:
        if len(list(csv.DictReader(f))) != len(rows):
            raise ValueError('CSV roundtrip row count differs')


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--materials',type=Path)
    args = p.parse_args()
    if args.output.exists() or (args.materials and args.materials.exists()):
        raise ValueError('Never overwrite exports')
    audit = json.loads(args.audit.read_text()); actual = audit_run(Path(audit['source_run']))
    if actual != audit or audit['primary_readouts_audited'] != 3456 or audit['source_files_verified'] != 240:
        raise ValueError('Complete unchanged independent audit required')
    _,_,registry,queries,_ = registered_inputs();rows=[];readers=[];episodes=defaultdict(list)
    for model in ('native','gdn_all_tokens'):
        for line in (Path(audit['source_run'])/f'{model}.jsonl').read_text().splitlines():
            value=json.loads(line);q=queries[value['id']]
            row={k:q[k] for k in ('id','seed','naming','episode','kind','tail_records','active_entities','role','question_style',
                'prefix_evidence','support_event_index','ledger_tokens','question')}
            row.update({k:value[k] for k in ('model','reader','answer','generated','greedy_exact','answer_nll','query_tokens')})
            rows.append(row);episodes[(model,value['reader'],q['case_id'],q['episode'])].append(row)
            for layer,stat in value['proof']['attention'].items():
                readers.append({'id':q['id'],'model':model,'reader':value['reader'],'layer':layer,**stat})
    metrics=audit['metrics'];tables={'primary':rows,'reader-probabilities':readers,'cells':metrics['cells'],
        'strata':metrics['strata'],'contrasts':metrics['paired_contrasts']}
    # Flatten nested aggregate fields explicitly, retaining all per-stream pairs.
    tables['per-stream-contrasts']=[{**{k:v for k,v in c.items() if k not in ('left','right')},
        **{f'left_{k}':v for k,v in c['left'].items()},**{f'right_{k}':v for k,v in c['right'].items()}}
        for c in metrics['per_stream_paired_contrasts']]
    tables['episodes']=[{'model':m,'reader':r,'case_id':c,'episode':e,**aggregate(v)} for (m,r,c,e),v in episodes.items()]
    if len(rows)!=3456 or len(tables['cells'])!=36 or len(tables['episodes'])!=576:
        raise ValueError('All registered rows/cells/episode cells required')
    args.output.mkdir(parents=True,exist_ok=False)
    for name,values in tables.items():write_csv(args.output/(name+'.csv'),values)
    write_json(args.output/'metrics.json',metrics)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=['before']+[f'{kind}-{count}-load{load}' for count in (64,512) for load in (2,16) for kind in ('unrelated','related')]
    roles=['target_current','target_initial','protected_current'];fig,axes=plt.subplots(4,3,figsize=(18,14),sharey=True)
    for i,(model,reader) in enumerate(( (m,r) for m in ('native','gdn_all_tokens') for r in ('full','oracle') )):
        for j,role in enumerate(roles):
            ax=axes[i,j];groups=defaultdict(list)
            for row in rows:
                if (row['model'],row['reader'],row['role'])==(model,reader,role):groups[row['episode']].append(row)
            totals=[len(groups[e]) for e in order];correct=[sum(r['greedy_exact'] for r in groups[e]) for e in order]
            if set(totals)!={32}:raise ValueError('All32paired repetitions per role/cell required')
            bars=ax.bar(range(9),correct,color=['#666666']+['#3267a8' if 'unrelated' in e else '#b76a30' for e in order[1:]])
            ax.bar_label(bars,labels=[f'{v}/32' for v in correct],fontsize=7,padding=2)
            ax.set_xticks(range(9),['before']+[e.replace('unrelated','U').replace('related','R').replace('load','L') for e in order[1:]],rotation=40,ha='right',fontsize=8)
            ax.set_ylim(0,37);ax.set_title(f'{model} / {reader} / {role}',fontsize=10);ax.set_ylabel('Correct answers')
    fig.suptitle('Normal updates: every registered condition, 8 streams x 2 names x 2 question styles\nOracle is privileged; repetitions are not independent streams.',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(args.output/'all-conditions.png',dpi=150);fig.savefig(args.output/'all-conditions.svg');plt.close(fig)
    report=['# Complete normal-update load diagnostic','',
        'All 3,456 readouts, 36 cells and 576 episode cells are retained; eight independent streams. Full and privileged Oracle are separate. No new training.','',
        '| Model | Reader | Condition | Correct | Mean answer NLL |','|---|---|---|---:|---:|']
    report.extend(f"| {c['model']} | {c['reader']} | {c['episode']} | {c['correct']}/{c['questions']} | {c['answer_nll_mean']:.6f} |" for c in metrics['cells'])
    report.extend(['','Length and record positions match across load and naming arms; entity recurrence and text still change. Accuracy alone does not identify GDN interference. Oracle is not a deployable method. All earlier failed scientific criteria remain false.','',
        'See all role/style/name strata and paired per-stream gains/losses in the adjacent CSV files. These are descriptive pilot results; there is no significance, architecture, long-context scaleup or novelty claim.'])
    (args.output/'report.md').write_text('\n'.join(report)+'\n');(args.output/'export-source.py').write_bytes(Path(__file__).read_bytes())
    files={f.name:{'bytes':f.stat().st_size,'sha256':streaming_hash(f)} for f in sorted(args.output.iterdir())}
    receipt={'status':'all_complete_grid_exports_written_and_roundtrip_verified','audit_sha256':streaming_hash(args.audit),
        'source_manifest_sha256':audit['manifest_sha256'],'rows':{n:len(v) for n,v in tables.items()},'files':files,
        'export_source_sha256':streaming_hash(Path(__file__))}
    write_json(args.output/'export-receipt.json',receipt)
    if args.materials:
        shutil.copytree(args.output,args.materials)
        for f in args.output.iterdir():
            if f.read_bytes()!=(args.materials/f.name).read_bytes():raise ValueError('Materials copy bytes differ')
    print(json.dumps({'status':receipt['status'],'rows':receipt['rows']}),flush=True)


if __name__=='__main__':main()
