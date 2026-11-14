"""Export all frozen-prefix evidence-access results and independently count raw rows."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json
from scripts.trained_timing_evidence_access_data import MODELS,REGIMES,registered_inputs,read_plan,selected_inputs
from scripts.trained_timing_evidence_access_metrics import summarize


def independent_counts(rows,cases,registry,metrics):
    """Count physical answers, all role bindings and joint cases without metric helpers."""
    result={}
    for name in MODELS:
        result[name]={}
        for regime in REGIMES:
            indexed={r['id']:r for r in rows[name][regime]}
            bound={(b['case_id'],b['variant'],b['role']):indexed[b['physical_id']] for b in registry['bindings']}
            stats=metrics['models'][name][regime]
            roles={role:{v:sum(bound[c['id'],v,role]['greedy_exact'] for c in cases)
                         for v in ('before','control','treatment')}
                   for role in ('target_current','target_initial','protected_current')}
            joint={c['id']:all(bound[c['id'],v,r]['greedy_exact'] for v in ('control','treatment') for r in roles)
                   for c in cases}
            actual={'physical_correct':sum(r['greedy_exact'] for r in indexed.values()),
                    'joint_six_after':sum(joint.values()),'role_variant_counts':roles,
                    'joint_by_depth':{str(d):sum(joint[c['id']] for c in cases if c['dependency_depth']==d) for d in (1,2,3)}}
            if (len(indexed)!=336 or actual['physical_correct']!=stats['physical_correct'] or
                actual['joint_six_after']!=stats['joint_six_after'] or any(
                    count!=stats['role_variant_counts'][r][v]['correct'] for r,values in roles.items() for v,count in values.items()) or
                any(joint[c['case_id']]!=c['joint_after'] for c in stats['case_outcomes'])):
                raise ValueError('Independent all-row role/joint/physical counts disagree with the audit')
            result[name][regime]=actual
        for reference in ('full','nonslice'):
            oracle={r['id']:r for r in rows[name]['oracle']};other={r['id']:r for r in rows[name][reference]}
            counts={k:0 for k in ('preserved','regression','repair','both_wrong')}
            for key,a in oracle.items():
                b=other[key]
                label=('preserved' if b['greedy_exact'] else 'repair') if a['greedy_exact'] else ('regression' if b['greedy_exact'] else 'both_wrong')
                counts[label]+=1
            if any(counts[k]!=metrics['oracle_paired_contrasts'][name][reference]['physical'][k] for k in counts):
                raise ValueError('Independent paired physical counts disagree with the audit')
            result[name]['oracle_minus_'+reference]=counts
    return result


def export(audit_path,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    report=json.loads(audit_path.read_text());run=Path(report['source_run'])
    manifest=json.loads((run/'manifest.json').read_text())
    registered,_,cases,registry,_,_=registered_inputs()
    physical,bindings=selected_inputs(registry)
    if (report['status']!='completed_and_audited_trained_timing_evidence_access' or manifest['smoke'] or
        report['new_physical_readouts_audited']!=5040 or report['new_related_bindings_audited']!=6480 or
        report['manifest_sha256']!=streaming_hash(run/'manifest.json') or
        streaming_hash(ROOT/'scripts/analyze_trained_timing_evidence_access.py')!=report['analysis_source_sha256']):
        raise ValueError('Unchanged complete independently audited primary results required')
    all_rows={}
    for name in MODELS:
        all_rows[name]={}
        for regime in REGIMES:
            path=run/f'{name}.{regime}.jsonl'
            if streaming_hash(path)!=manifest['models'][name]['results_sha256'][regime]:raise ValueError('Audited raw bytes changed')
            all_rows[name][regime]=[json.loads(line) for line in path.read_text().splitlines()]
    metrics=summarize(all_rows,cases,registry,read_plan())
    if metrics!=report['metrics']:raise ValueError('Complete raw-data metrics do not reproduce the audit')
    crosscheck=independent_counts(all_rows,cases,registry,metrics)
    suffixes=('.png','.pdf','.csv','.pairs.csv','.data.json','.layout.json','.source.py','.crosscheck.json','.provenance.json')
    outputs=[prefix.with_suffix(s) for s in suffixes]
    if any(p.exists() for p in outputs):raise ValueError('Preserve earlier exports')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    names=['Native','Shared: train all / deploy all','Shared: train question / deploy all',
           'Shared: train question / deploy question','GDN: train all / deploy all']
    items=[(n,r) for n in MODELS for r in REGIMES]
    labels=[f'{label} | {r}' for label in names for r in REGIMES]
    colors={'full':'#6b8499','oracle':'#2e8b70','nonslice':'#c88645'}
    joint=[metrics['models'][n][r]['joint_six_after'] for n,r in items]
    current=[sum(metrics['models'][n][r]['role_variant_counts']['target_current'][v]['correct']
                 for v in ('control','treatment')) for n,r in items]
    physical_correct=[metrics['models'][n][r]['physical_correct'] for n,r in items]
    fig,axes=plt.subplots(1,3,figsize=(15,9),gridspec_kw={'width_ratios':[1,1,1]})
    for ax,values,limit,label,title in zip(axes,[joint,current,physical_correct],[48,96,336],
            ['Joint cases correct / 48','Current bindings correct / 96','Physical questions correct / 336'],
            ['A. All six post-update answers','B. Current target after update','C. All roles and time points']):
        ax.barh(np.arange(15),values,color=[colors[r] for _,r in items],height=.68)
        ax.set(yticks=np.arange(15),yticklabels=labels if ax is axes[0] else [],xlim=(0,limit*1.12),xlabel=label,title=title)
        for i,value in enumerate(values):ax.text(value+limit*.015,i,str(value),va='center',fontsize=8)
        for pos in (2.5,5.5,8.5,11.5):ax.axhline(pos,color='#d8d8d8',lw=.7)
        ax.invert_yaxis();ax.set_axisbelow(True);ax.grid(axis='x',alpha=.2)
        for side in ('top','right'):ax.spines[side].set_visible(False)
    fig.suptitle('Frozen-prefix evidence access — all five checkpoints and three regimes',fontsize=14)
    fig.text(.02,.06,'Reused cohort: 8 source streams, 48 related cases. All 5040 primary readouts shown; related bindings are correlated.',fontsize=9)
    fig.text(.02,.027,'Program oracle masks query-time history access only. Dense eager kernels; no speed claim. Oracle failure alone does not identify a composition failure.',fontsize=9)
    fig.tight_layout(rect=[0,.1,1,.94]);fig.savefig(prefix.with_suffix('.png'),dpi=180);fig.savefig(prefix.with_suffix('.pdf'));plt.close(fig)
    bound={q['id']:[b for b in bindings if b['physical_id']==q['id']] for q in physical}
    fields=['model','token_policy','regime','physical_id','related_binding_count','roles',
            'answer','generated','greedy_exact','answer_nll','query_tokens']
    with prefix.with_suffix('.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for name,regime in items:
            for row in all_rows[name][regime]:
                writer.writerow({'model':name,'token_policy':row['token_policy'],'regime':regime,'physical_id':row['id'],
                    'related_binding_count':len(bound[row['id']]),'roles':';'.join(sorted({b['role'] for b in bound[row['id']]})),
                    **{k:row[k] for k in ['answer','generated','greedy_exact','answer_nll','query_tokens']}})
    with prefix.with_suffix('.pairs.csv').open('w',newline='') as file:
        fields=['model','reference','case_id','physical_id','base_case_id','dependency_depth','question_style',
                'distractor_updates','variant','role','candidate_correct','reference_correct','gain','loss','nll_candidate_minus_reference']
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for name in MODELS:
            for ref in ('full','nonslice'):
                for row in metrics['oracle_paired_contrasts'][name][ref]['current_target']['individual_pairs']:
                    writer.writerow({'model':name,'reference':ref,**{k:row[k] for k in fields[2:]}})
    # CSV values are independently decoded and compared to the actual stored raw rows.
    with prefix.with_suffix('.csv').open(newline='') as file:
        exported=list(csv.DictReader(file))
    raw_order=[r for n,regime in items for r in all_rows[n][regime]]
    for out,row in zip(exported,raw_order,strict=True):
        if (out['model']!=row['model'] or out['regime']!=row['regime'] or out['physical_id']!=row['id'] or
            out['answer']!=row['answer'] or out['generated']!=row['generated'] or
            (out['greedy_exact']=='True')!=row['greedy_exact'] or float(out['answer_nll'])!=row['answer_nll'] or
            int(out['query_tokens'])!=row['query_tokens']):raise ValueError('Exported CSV fails exact raw-record equality')
    if len(exported)!=5040:raise ValueError('Missing complete primary export')
    write_json(prefix.with_suffix('.data.json'),metrics)
    write_json(prefix.with_suffix('.layout.json'),{'model_order':list(MODELS),'regimes':list(REGIMES),'labels':labels,
        'joint_counts':joint,'current_counts':current,'physical_counts':physical_correct,'figure_inches':[15,9],'dpi':180,
        'intervals':'Exact fixed-cohort counts; no binomial intervals on correlated bindings.'})
    prefix.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    write_json(prefix.with_suffix('.crosscheck.json'),{'status':'all5040_csv_rows_exact_and_independent_counts_match',
        'counts':crosscheck,'raw_csv_rows_verified':len(exported),'old_gates_unchanged':report['prior_timing_criteria_unchanged']})
    write_json(prefix.with_suffix('.provenance.json'),{'status':'all_five_models_three_regimes_exported_from_actual_audited5040readouts',
        'audit_path':str(audit_path),'audit_sha256':streaming_hash(audit_path),'source_run':str(run),
        'manifest_sha256':report['manifest_sha256'],'mask_registry_sha256':report['mask_registry_sha256'],
        'dataset_manifest_sha256':report['dataset_manifest_sha256'],'export_source_sha256':streaming_hash(Path(__file__)),
        'output_sha256':{p.name:streaming_hash(p) for p in outputs[:-1]}})
    return {'artifacts':len(outputs),'physical_rows':len(exported),'prefix':str(prefix)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('audit',type=Path);parser.add_argument('--prefix',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(export(args.audit,args.prefix)),flush=True)


if __name__=='__main__':main()
