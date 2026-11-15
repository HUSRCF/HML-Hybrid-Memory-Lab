"""Export every normal readout and pooled observation after the complete independent audit."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_version_address import audit_run
from scripts.version_address_data import MODELS,LAYERS,POSITIONS
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def write_csv(path,rows):
    if not rows:raise ValueError('Nonempty complete table required')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    with path.open(newline='') as f:
        if len(list(csv.DictReader(f)))!=len(rows):raise ValueError('CSV roundtrip count differs')


def figures(output,rows,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Layer15/boundary23 was fixed in the protocol. All matched numeric errors are shown.
    points={};largest=0.
    for model in MODELS:
        for cohort in ('reused','fresh'):
            values=[r for r in rows if r['model']==model and r['cohort']==cohort and not r['greedy_exact']]
            observed=[]
            for row in values:
                obs=next(v for v in row['address_rows'] if v['layer']==15 and v['suffix_position']==23)
                if obs['emitted_record_mass'] is not None:
                    observed.append(obs);largest=max(largest,obs['support_mass'],obs['emitted_record_mass'])
            points[(model,cohort)]=(observed,len(values)-len(observed))
    bound=max(.001,largest*1.05)
    fig,axes=plt.subplots(2,2,figsize=(11,9),sharex=True,sharey=True,constrained_layout=True)
    colors={'same_account_other_version_value':'tab:orange','other_account_recorded_value':'tab:red',
            'noncanonical_correct_integer':'tab:blue','correct':'tab:gray'}
    for i,model in enumerate(MODELS):
        for j,cohort in enumerate(('reused','fresh')):
            ax=axes[i,j];values,excluded=points[(model,cohort)]
            for category in sorted({r['output_category'] for r in values}):
                v=[r for r in values if r['output_category']==category]
                ax.scatter([r['support_mass'] for r in v],[r['emitted_record_mass'] for r in v],
                    label=category,s=16,alpha=.55,color=colors.get(category,'tab:gray'))
            ax.plot([0,bound],[0,bound],color='black',linewidth=.7);ax.set_xlim(0,bound);ax.set_ylim(0,bound)
            ax.set_title(f'{model} / {cohort}\n{len(values)} matched-value errors; {excluded} other errors')
            ax.set_xlabel('Pooled attention mass on true support record');ax.set_ylabel('Mass on emitted-value record')
            if values:ax.legend(fontsize=6,loc='best')
    fig.suptitle('Normal failures / layer15 / known suffix position23\nAll matched-value errors retained. Eight streams per cohort; paired names/history/questions. Weights are not causal contributions.')
    for ext in ('png','svg'):fig.savefig(output/('normal-errors-primary-location.'+ext),dpi=160)
    plt.close(fig)
    values=np.empty((2,2,2,6),dtype=np.float64)
    pairs=metrics['same_history_version_pairs']
    for mi,model in enumerate(MODELS):
        for ci,cohort in enumerate(('reused','fresh')):
            group=[p for p in pairs if p['model']==model and p['cohort']==cohort]
            if len(group)!=288:raise ValueError('Every current/initial paired view required')
            for pi,pos in enumerate(POSITIONS):
                for li,layer in enumerate(LAYERS):
                    cell=[v['preference_shift_current_minus_initial'] for p in group for v in p['pooled_version_comparisons']
                        if v['layer']==layer and v['suffix_position']==pos]
                    if len(cell)!=288:raise ValueError('All layer/position contrasts required')
                    values[mi,ci,pi,li]=sum(cell)/288
    limit=max(1e-8,float(np.abs(values).max()))
    fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    for mi,model in enumerate(MODELS):
        for ci,cohort in enumerate(('reused','fresh')):
            ax=axes[mi,ci];im=ax.imshow(values[mi,ci],cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
            ax.set_title(f'{model} / {cohort} / all288 paired views')
            ax.set_xticks(range(6),LAYERS);ax.set_xlabel('Attention layer (zero based)')
            ax.set_yticks(range(2),['question end (11)','answer boundary (23)'])
            for pi in range(2):
                for li in range(6):ax.text(li,pi,f'{values[mi,ci,pi,li]:.4f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=list(axes.flat),label='Mean current-vs-initial shift in (last minus first record mass)',shrink=.8)
    fig.suptitle('Same-history, same-entity version-word contrast / all layers and positions\nEight underlying streams per cohort, no significance or causal attention claim.')
    for ext in ('png','svg'):fig.savefig(output/('pooled-version-word-contrasts.'+ext),dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--materials',type=Path);args=p.parse_args()
    if args.output.exists() or args.materials and args.materials.exists():raise ValueError('Never overwrite exports')
    audit=json.loads(args.audit.read_text());run=Path(audit['source_run'])
    if (audit['status']!='completed_and_audited_version_address' or audit['physical_readouts_audited']!=3456 or
        audit['source_files_verified']!=268 or audit!=audit_run(run)):raise ValueError('Complete unchanged independent audit required')
    m=json.loads((run/'manifest.json').read_text());packet=Path(m['packet']);pm=json.loads((packet/'manifest.json').read_text())
    qp=packet/'queries.json'
    if streaming_hash(qp)!=pm['payload_sha256']['queries.json']:raise ValueError('Questions changed')
    queries=json.loads(qp.read_text());rows=[];primary=[];observations=[];references=[];arrays=[]
    for model in MODELS:
        state=m['models'][model]
        for line in (run/(model+'.jsonl')).read_text().splitlines():
            row=json.loads(line);q=queries[row['id']];rows.append(row)
            identity={k:q[k] for k in ('id','seed','naming','episode','question_style','role','question','tail_records','active_entities','ledger_tokens')}
            identity.update(model=model,cohort=row['cohort'])
            primary.append({**identity,**{k:row[k] for k in ('answer','generated','greedy_exact','answer_nll','record_mass_file','record_mass_index','record_mass_file_sha256')}})
            for obs in row['address_rows']:observations.append({**identity,**obs,'answer_correct':row['greedy_exact']})
        references.extend({'model':model,**r} for r in map(json.loads,(run/(model+'.references.jsonl')).read_text().splitlines()))
        for relative,digest in state['record_mass_files'].items():
            array=np.load(run/relative,allow_pickle=False)
            if streaming_hash(run/relative)!=digest or array.dtype!=np.float32:raise ValueError('Exact raw arrays changed')
            arrays.append({'model':model,'relative_path':relative,'sha256':digest,'bytes':(run/relative).stat().st_size,'dtype':str(array.dtype),'shape':list(array.shape)})
    metrics=audit['metrics'];tables={'primary':primary,'archived-references':references,'pooled-record-observations':observations,
        'raw-fp32-array-inventory':arrays,'cells':metrics['cells'],'role-cells':metrics['role_cells'],
        'all-correctness-strata-including-empty':metrics['all_correctness_strata_including_empty'],
        'wording-pairs':metrics['wording_pairs'],'same-history-version-pairs':metrics['same_history_version_pairs']}
    expected={'primary':3456,'archived-references':1728,'pooled-record-observations':41472,'raw-fp32-array-inventory':576,
        'cells':72,'role-cells':216,'all-correctness-strata-including-empty':576,'wording-pairs':1728,'same-history-version-pairs':1152}
    if {k:len(v) for k,v in tables.items()}!=expected:raise ValueError('Entire observational grid required')
    args.output.mkdir(parents=True,exist_ok=False)
    for name,values in tables.items():write_csv(args.output/(name+'.csv'),values)
    write_json(args.output/'metrics.json',metrics);figures(args.output,rows,metrics)
    totals=[]
    for model in MODELS:
        for cohort in ('reused','fresh'):
            for style in ('state_term','event_term'):
                v=[r for r in rows if r['model']==model and r['cohort']==cohort and queries[r['id']]['question_style']==style]
                if len(v)!=432:raise ValueError('All style/cohort questions required')
                totals.append({'model':model,'cohort':cohort,'style':style,'correct':sum(r['greedy_exact'] for r in v),'questions':432})
    report=['# Complete normal-history pooled record observations','','All3456physical normal readouts,1728exact old replays and1728labelled archived references retained. All576original FP32 files are in the complete backup; inventory provides exact paths and hashes.','',
        '| Model | Cohort | Wording | Correct /432 |','|---|---|---|---:|']
    for r in totals:report.append(f"| {r['model']} | {r['cohort']} | {r['style']} | {r['correct']} |")
    report+=['','All layers,positions,roles,wordings and baseline errors remain in adjacent tables. Correctness strata include empty groups explicitly. Head means can hide specialization,and attention probability is not causal answer contribution.','',
        'Normal recurrent updates and unrestricted full attention; observation hooks removed before answer scoring/generation. No Oracle,donor,new parameters,training,environment or scaleup. Fresh means new synthetic streams,not a new task. All old failed scientific criteria remain false.']
    (args.output/'report.md').write_text('\n'.join(report)+'\n');(args.output/'export-source.py').write_bytes(Path(__file__).read_bytes())
    files={f.name:{'bytes':f.stat().st_size,'sha256':streaming_hash(f)} for f in sorted(args.output.iterdir())}
    receipt={'status':'complete_version_address_exports_roundtrip_verified','audit_sha256':streaming_hash(args.audit),
        'manifest_sha256':audit['manifest_sha256'],'rows':expected,'style_totals':totals,'files':files}
    write_json(args.output/'export-receipt.json',receipt)
    if args.materials:
        shutil.copytree(args.output,args.materials)
        for f in args.output.iterdir():
            if f.read_bytes()!=(args.materials/f.name).read_bytes():raise ValueError('Materials copy differs')
    print(json.dumps({'status':receipt['status'],'rows':expected,'style_totals':totals}),flush=True)


if __name__=='__main__':main()
