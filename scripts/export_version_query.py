"""Export every new/archived version-question cell and exact input pair."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_version_query_corrected import audit_run
from scripts.version_query_data import registered_inputs,read_plan
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def write_csv(path,rows):
    if not rows:raise ValueError('No empty export table')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    with path.open(newline='') as f:
        if len(list(csv.DictReader(f)))!=len(rows):raise ValueError('CSV roundtrip row count differs')


def figures(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=['before']+[f'{kind}-{n}-load{load}' for n in (64,512) for load in (2,16) for kind in ('unrelated','related')]
    for origin,styles in (('new',('state_term','event_term')),('archived',('original','entity_first'))):
        fig,axes=plt.subplots(4,3,figsize=(18,14),sharey=True)
        for i,(model,style) in enumerate(( (m,s) for m in ('native','gdn_all_tokens') for s in styles )):
            for j,role in enumerate(('target_current','target_initial','protected_current')):
                grouped=defaultdict(list)
                for r in rows:
                    if (r['origin'],r['model'],r['question_style'],r['role'])==(origin,model,style,role):grouped[r['episode']].append(r)
                if {len(grouped[e]) for e in order}!={16}:raise ValueError('Every role/style/episode must contain16paired case/name repetitions')
                correct=[sum(r['greedy_exact'] for r in grouped[e]) for e in order];ax=axes[i,j]
                bars=ax.bar(range(9),correct,color=['#666666']+['#3267a8' if 'unrelated' in e else '#b76a30' for e in order[1:]])
                ax.bar_label(bars,labels=[f'{v}/16' for v in correct],fontsize=7,padding=2)
                ax.set_xticks(range(9),['before']+[e.replace('unrelated','U').replace('related','R').replace('load','L') for e in order[1:]],rotation=40,ha='right',fontsize=8)
                ax.set_ylim(0,19);ax.set_title(f'{model} / {style} / {role}',fontsize=10);ax.set_ylabel('Correct answers')
        label='NEW single-token pairs' if origin=='new' else 'ARCHIVED predecessor Full references'
        fig.suptitle(label+' — every condition, 8 reused streams x 2 paired names\nEach wording is separate; old/new sentence syntax is not token matched.',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(output/(origin+'-all-conditions.png'),dpi=150)
        fig.savefig(output/(origin+'-all-conditions.svg'));plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--materials',type=Path);args=p.parse_args()
    if args.output.exists() or (args.materials and args.materials.exists()):raise ValueError('Never overwrite exports')
    audit=json.loads(args.audit.read_text());actual=audit_run(Path(audit['source_run']))
    if actual!=audit or audit['primary_readouts_audited']!=1728 or audit['archived_full_references_audited']!=1728:
        raise ValueError('Complete unchanged246source primary audit required')
    _,_,registry,queries,refs=registered_inputs();plan=read_plan();source=Path(plan['source_packet'])
    old_manifest=json.loads((source/'manifest.json').read_text())
    if streaming_hash(source/'queries.json')!=old_manifest['payload_sha256']['queries.json']:raise ValueError('Archived question bytes changed')
    old_queries=json.loads((source/'queries.json').read_text());rows=[];reader_rows=[]
    run=Path(audit['source_run'])
    for model in ('native','gdn_all_tokens'):
        for origin,filename,qmap in (('new',f'{model}.jsonl',queries),('archived',f'{model}.references.jsonl',old_queries)):
            for line in (run/filename).read_text().splitlines():
                raw=json.loads(line);q=qmap[raw['id']]
                row={k:q[k] for k in ('id','seed','naming','episode','role','question_style','question','tail_records',
                    'active_entities','ledger_tokens','prefix_evidence','support_event_index','source_key')}
                row.update(origin=origin,**{k:raw[k] for k in ('model','reader','answer','generated','greedy_exact','answer_nll','query_tokens')})
                row.update(control_term=q.get('control_term'),paired_query_id=q.get('paired_query_id'),
                    different_question_token_index=q.get('different_question_token_index'));rows.append(row)
                for layer,stat in raw['proof']['attention'].items():reader_rows.append({'origin':origin,'model':model,'id':q['id'],'layer':layer,**stat})
    token_pairs=[]
    for q in queries.values():
        if q['question_style']!='state_term':continue
        other=queries[q['paired_query_id']];index=q['different_token_index']
        token_pairs.append({'state_id':q['id'],'event_id':other['id'],'seed':q['seed'],'naming':q['naming'],'episode':q['episode'],'role':q['role'],
            'state_question':q['question'],'event_question':other['question'],'state_word':q['control_term'],'event_word':other['control_term'],
            'question_tokens':24,'different_question_token_index':q['different_question_token_index'],'different_full_prompt_index':index,
            'state_token_id':q['input_ids'][index],'event_token_id':other['input_ids'][index],'source_key':q['source_key']})
    metrics=audit['metrics'];pairs=[{**{k:v for k,v in r.items() if k not in ('left','right')},
        **{f'state_{k}':v for k,v in r['left'].items()},**{f'event_{k}':v for k,v in r['right'].items()}}
        for r in metrics['per_stream_state_to_event_pairs']]
    tables={'primary':[r for r in rows if r['origin']=='new'],'archived-references':[r for r in rows if r['origin']=='archived'],
        'reader-probabilities':reader_rows,'token-pairs':token_pairs,'cells':metrics['cells'],'role-cells':metrics['role_cells'],
        'episodes':metrics['episodes'],'state-to-event-pairs':pairs,'paired-contrasts':metrics['paired_contrasts'],
        'per-seed-contrasts':metrics['per_seed_paired_contrasts'],'value-attribution':metrics['value_attribution_all_readouts']}
    expected={'primary':1728,'archived-references':1728,'reader-probabilities':20736,'token-pairs':432,'cells':72,'role-cells':216,
        'episodes':1152,'state-to-event-pairs':864,'paired-contrasts':54,'per-seed-contrasts':432,'value-attribution':3456}
    if {k:len(v) for k,v in tables.items()}!=expected:raise ValueError('Complete new and archived export counts required')
    args.output.mkdir(parents=True,exist_ok=False)
    for name,values in tables.items():write_csv(args.output/(name+'.csv'),values)
    write_json(args.output/'metrics.json',metrics);figures(args.output,rows)
    report=['# Complete single-token version-question control','','All 1,728 new Full readouts and 1,728 explicitly archived references are retained. New matched comparison: state_term -> event_term.','',
        '| Origin | Model | Question style | Condition | Correct | Mean answer NLL |','|---|---|---|---|---:|---:|']
    report.extend(f"| {r['origin']} | {r['model']} | {r['question_style']} | {r['episode']} | {r['correct']}/{r['questions']} | {r['answer_nll_mean']:.6f} |" for r in metrics['cells'])
    report.extend(['','Each new style/role/episode/model has16case/name repetitions, with only8underlying streams. Every known question has24tokens and each pair changes only its sixth token. Historical source caches match the predecessor.','',
        'Old versus new sentence syntax is not token matched; old rows are archived evidence, not fresh inference. Token-matched paraphrase effects do not identify a unique memory mechanism, prove an update method, or establish architecture effectiveness. All prior failed criteria remain false.','',
        'Plots use U/R/L for unrelated/related/active accounts;64/512are tail record counts. Full role tables, all864paired gains/losses, per-stream counts and every generated answer value attribution are adjacent.'])
    (args.output/'report.md').write_text('\n'.join(report)+'\n');(args.output/'export-source.py').write_bytes(Path(__file__).read_bytes())
    files={f.name:{'bytes':f.stat().st_size,'sha256':streaming_hash(f)} for f in sorted(args.output.iterdir())}
    receipt={'status':'complete_new_and_archived_exports_roundtrip_verified','audit_sha256':streaming_hash(args.audit),
        'manifest_sha256':audit['manifest_sha256'],'rows':expected,'files':files,'export_source_sha256':streaming_hash(Path(__file__))}
    write_json(args.output/'export-receipt.json',receipt)
    if args.materials:
        shutil.copytree(args.output,args.materials)
        for f in args.output.iterdir():
            if f.read_bytes()!=(args.materials/f.name).read_bytes():raise ValueError('Materials export copy differs')
    print(json.dumps({'status':receipt['status'],'rows':expected}),flush=True)


if __name__=='__main__':main()
