"""Export the complete eight-arm diagnostic without filtering baseline errors."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_question_state_freeze import audit_run
from scripts.question_state_freeze_data import ARMS
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def write_csv(path,rows):
    if not rows:raise ValueError('No empty export')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    with path.open(newline='') as f:
        if len(list(csv.DictReader(f)))!=len(rows):raise ValueError('CSV roundtrip count differs')


def figures(output,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=['before']+[f'{kind}-{n}-load{load}' for n in (64,512) for load in (2,16) for kind in ('unrelated','related')]
    role={(r['arm'],r['question_style'],r['episode'],r['role']):r for r in metrics['role_cells']}
    joint=defaultdict(list)
    for r in metrics['episode_cells']:joint[(r['arm'],r['question_style'],r['episode'])].append(r['correct']==3)
    for style in ('state_term','event_term'):
        fig,axes=plt.subplots(1,4,figsize=(26,6),sharey=True,constrained_layout=True)
        for j,kind in enumerate(('target_current','target_initial','protected_current','all_three_correct')):
            values=[]
            for arm in ARMS:
                row=[]
                for e in order:
                    if kind=='all_three_correct':
                        v=joint[(arm,style,e)]
                        if len(v)!=16:raise ValueError('All16same-cache joint cells required')
                        row.append(sum(v))
                    else:
                        r=role[(arm,style,e,kind)]
                        if r['questions']!=16:raise ValueError('Every role cell denominator must be16')
                        row.append(r['correct'])
                values.append(row)
            ax=axes[j];im=ax.imshow(values,vmin=0,vmax=16,cmap='viridis',aspect='auto')
            for i,row in enumerate(values):
                for k,v in enumerate(row):ax.text(k,i,f'{v}/16',ha='center',va='center',fontsize=8,color='black' if v>=9 else 'white')
            ax.set_title(kind);ax.set_xticks(range(9),['before']+[e.replace('unrelated','U').replace('related','R').replace('load','L') for e in order[1:]],rotation=45,ha='right')
            ax.set_yticks(range(8),[a if not layers else a+' ('+','.join(map(str,layers))+')' for a,layers in ARMS.items()])
        fig.colorbar(im,ax=list(axes),label='Correct of16paired case/name repetitions',shrink=.8)
        fig.suptitle(f'All question-only GDN controls / {style}\n8 reused streams x2names; U=unrelated,R=related,L=active accounts; all history conditions retained.',fontsize=13)
        fig.savefig(output/(style+'-all-conditions.png'),dpi=150);fig.savefig(output/(style+'-all-conditions.svg'));plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--materials',type=Path)
    p.add_argument('--observations',type=Path,required=True);args=p.parse_args()
    if args.output.exists() or (args.materials and args.materials.exists()):raise ValueError('Never overwrite exports')
    audit=json.loads(args.audit.read_text());actual=audit_run(Path(audit['source_run']))
    if actual!=audit or audit['primary_readouts_audited']!=6912 or audit['source_files_verified']!=261:
        raise ValueError('Complete unchanged261source primary audit required')
    observations=json.loads(args.observations.read_text())
    if (observations['status']!='all6912raw_answers_and6048pairs_independently_regraded' or
        observations['audit_sha256']!=streaming_hash(args.audit) or observations['manifest_sha256']!=audit['manifest_sha256'] or
        observations['readouts']!=6912 or observations['pairs']!=6048):raise ValueError('Complete independent grading required')
    run=Path(audit['source_run']);m=json.loads((run/'manifest.json').read_text());packet=Path(m['packet'])
    pm=json.loads((packet/'manifest.json').read_text());qp=packet/'queries.json'
    if streaming_hash(qp)!=pm['payload_sha256']['queries.json']:raise ValueError('Question bytes changed')
    queries=json.loads(qp.read_text());rows=[];weights=[];gates=[];matrices=[]
    for arm in ARMS:
        for line in (run/f'{arm}.jsonl').read_text().splitlines():
            raw=json.loads(line);q=queries[raw['id']]
            row={k:q[k] for k in ('id','seed','naming','episode','role','question_style','question','tail_records',
                'active_entities','ledger_tokens','support_event_index','source_key')}
            row.update({k:raw[k] for k in ('arm','model','reader','answer','generated','greedy_exact','answer_nll','query_tokens')});rows.append(row)
            for layer,s in raw['proof']['attention'].items():weights.append({'arm':arm,'id':q['id'],'layer':layer,**s})
            hold=raw['proof']['question_hold']
            for layer,s in hold['gates']['layers'].items():gates.append({'arm':arm,'id':q['id'],'layer':layer,'selected':int(layer) in ARMS[arm],**s})
            for layer,before in hold['selected_before'].items():
                matrices.append({'arm':arm,'id':q['id'],'layer':layer,'before_sha256':before['sha256'],
                    'after_sha256':hold['selected_after'][layer]['sha256'],'dtype':before['dtype'],'shape':before['shape'],
                    'hooks_removed_before_answer':hold['hooks_removed_before_answer']})
    references=[json.loads(l) for l in (run/'references.jsonl').read_text().splitlines()]
    metrics=audit['metrics'];timings=[{'arm':arm,'readouts':m['model_state']['completed_by_arm'][arm],
        'wall_seconds_with_observers':m['model_state']['readout_wall_seconds_with_observers'][arm],
        'seconds_per_readout_with_observers':m['model_state']['readout_wall_seconds_with_observers'][arm]/864,
        'selected_gdn_layers':ARMS[arm],'zero_gate_layer_rows_per_question':24*len(ARMS[arm]),
        'interpretation':'Includes source forks,finite/gate/matrix/KV proofs,passive attention observation,NLL and generation; not deployment decode latency.'} for arm in ARMS]
    tables={'primary':rows,'archived-references':[{k:r[k] for k in ('id','answer','generated','greedy_exact','answer_nll','archived_reference','source_run','source_manifest_sha256')} for r in references],
        'reader-probabilities':weights,'actual-gate-rows':gates,'selected-matrix-equality':matrices,
        'cells':metrics['cells'],'role-cells':metrics['role_cells'],'episode-cells':metrics['episode_cells'],
        'normal-to-intervention-pairs':metrics['normal_to_intervention_pairs'],'wording-pairs':metrics['wording_pairs'],
        'paired-contrasts':metrics['paired_contrasts'],'per-seed-contrasts':metrics['per_seed_contrasts'],
        'same-cache-joint3':metrics['same_cache_joint3'],'same-cache-joint6':metrics['same_cache_joint6'],'observed-walltime':timings,
        'value-attribution':observations['value_attribution_all_readouts'],
        'all-original-errors':observations['all_original_errors_in_every_arm'],
        'original17paired-failures-descriptive-only':observations['original17paired_failures_descriptive_only'],
        'error-category-counts':observations['error_category_counts'],
        'style-totals-independent':observations['style_totals'],'role-totals-independent':observations['role_totals']}
    expected={'primary':6912,'archived-references':864,'reader-probabilities':41472,'actual-gate-rows':124416,
        'selected-matrix-equality':31104,'cells':144,'role-cells':432,'episode-cells':2304,
        'normal-to-intervention-pairs':6048,'wording-pairs':3456,'paired-contrasts':378,'per-seed-contrasts':3024,
        'same-cache-joint3':16,'same-cache-joint6':8,'observed-walltime':8,
        'value-attribution':6912,'all-original-errors':400,'original17paired-failures-descriptive-only':136,
        'error-category-counts':8,'style-totals-independent':16,'role-totals-independent':48}
    if {k:len(v) for k,v in tables.items()}!=expected:raise ValueError('Complete eight-arm export counts required')
    args.output.mkdir(parents=True,exist_ok=False)
    for name,values in tables.items():write_csv(args.output/(name+'.csv'),values)
    write_json(args.output/'metrics.json',metrics);figures(args.output,metrics)
    totals=defaultdict(lambda:[0,0,0,0])
    for r in metrics['cells']:totals[(r['arm'],r['question_style'])][0]+=r['correct'];totals[(r['arm'],r['question_style'])][1]+=r['questions']
    for r in metrics['paired_contrasts']:totals[(r['arm'],r['question_style'])][2]+=r['gains'];totals[(r['arm'],r['question_style'])][3]+=r['losses']
    for r in observations['style_totals']:
        if totals[(r['arm'],r['style'])]!=[r['correct'],r['questions'],r['gains'],r['losses']]:raise ValueError('Independent totals differ')
    report=['# Complete question-phase local state-freeze control','','All6912newreadouts,864actual normal replays and864labelled archived references retained. Each style/role/condition has16case/name repetitions from8reused underlying streams.','',
        '| Arm | Wording | Correct | Repairs versus normal | Regressions versus normal |','|---|---|---:|---:|---:|']
    for arm in ARMS:
        for style in ('state_term','event_term'):
            c,n,g,l=totals[(arm,style)];report.append(f'| {arm} | {style} | {c}/{n} | {g} | {l} |')
    report+=['','All144condition cells,432role cells,6048normal/intervention pairs,3456wording pairs and all baseline failures are adjacent. All31104selected matrix before/after SHA pairs and124416actual layer-gate reports are retained.','',
        'History sources remain normal. Interventions apply only to24known question/formatting tokens; all hooks and kernel wrappers are removed before answer scoring/generation. Full attention is unrestricted. No Oracle,donor,new parameters or training. Zero gates change immediate readouts and future question-state trajectories; no unique fault source or architecture effectiveness follows.','',
        'Walltime includes diagnostic observers,hashes,source forks,NLL and generation; it is not deployment decode latency. All old scientific criteria remain false.']
    (args.output/'report.md').write_text('\n'.join(report)+'\n');(args.output/'export-source.py').write_bytes(Path(__file__).read_bytes())
    files={f.name:{'bytes':f.stat().st_size,'sha256':streaming_hash(f)} for f in sorted(args.output.iterdir())}
    receipt={'status':'complete_eight_arm_exports_roundtrip_verified','audit_sha256':streaming_hash(args.audit),
        'manifest_sha256':audit['manifest_sha256'],'observations_sha256':streaming_hash(args.observations),
        'rows':expected,'files':files,'export_source_sha256':streaming_hash(Path(__file__))}
    write_json(args.output/'export-receipt.json',receipt)
    if args.materials:
        shutil.copytree(args.output,args.materials)
        for f in args.output.iterdir():
            if f.read_bytes()!=(args.materials/f.name).read_bytes():raise ValueError('Material copy bytes differ')
    print(json.dumps({'status':receipt['status'],'rows':expected}),flush=True)


if __name__=='__main__':main()
