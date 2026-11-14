"""Export every factorial condition, loss contrast and original task without selection."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json
from scripts.text_competence_factorial_data import MODELS,REGIMES,DESTINATION,read_plan,registered_inputs
from scripts.text_competence_factorial_metrics import summarize


def independent_counts(rows,cases,registry,metrics):
    result={}
    for name in MODELS:
        result[name]={}
        for regime in REGIMES:
            indexed={r['id']:r for r in rows[name][regime]}
            bound={(b['case_id'],b['variant'],b['role']):indexed[b['physical_id']] for b in registry['bindings']}
            roles={role:{v:sum(bound[c['id'],v,role]['greedy_exact'] for c in cases) for v in ('before','control','treatment')}
                   for role in ('target_current','target_initial','protected_current')}
            joint={c['id']:all(bound[c['id'],v,r]['greedy_exact'] for v in ('control','treatment') for r in roles) for c in cases}
            unique={v['actual_prompt_sha256']:v['greedy_exact'] for v in indexed.values()}
            actual={'original_tasks_correct':sum(r['greedy_exact'] for r in indexed.values()),'joint':sum(joint.values()),'roles':roles,
                'joint_by_depth':{str(d):sum(joint[c['id']] for c in cases if c['dependency_depth']==d) for d in (1,2,3)},
                'unique_actual_inputs':len(unique),'unique_actual_inputs_correct':sum(unique.values())}
            stat=metrics['models'][name][regime];dedup=metrics['deduplicated_actual_input_counts'][name][regime]
            if (len(indexed)!=336 or actual['original_tasks_correct']!=stat['physical_correct'] or actual['joint']!=stat['joint_six_after'] or
                any(value!=stat['role_variant_counts'][r][v]['correct'] for r,values in roles.items() for v,value in values.items()) or
                any(joint[c['case_id']]!=c['joint_after'] for c in stat['case_outcomes']) or
                actual['unique_actual_inputs']!=dedup['unique_inputs'] or actual['unique_actual_inputs_correct']!=dedup['correct']):
                raise ValueError('Independent complete role/joint/original-task/unique-input counts disagree')
            result[name][regime]=actual
    return result


def export(audit,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    report=json.loads(audit.read_text());run=Path(report['source_run']);manifest=json.loads((run/'manifest.json').read_text())
    _,cases,registry,_,_=registered_inputs();plan=read_plan()
    if (report['status']!='completed_and_audited_text_competence_factorial' or manifest['smoke'] or
        report['new_physical_readouts_audited']!=6720 or report['related_bindings_audited']!=8640 or
        report['manifest_sha256']!=streaming_hash(run/'manifest.json') or
        report['analysis_source_sha256']!=streaming_hash(ROOT/'scripts/analyze_text_competence_factorial.py')):
        raise ValueError('Unchanged independently audited complete primary required')
    rows={}
    for name in MODELS:
        rows[name]={}
        for regime in REGIMES:
            p=run/f'{name}.{regime}.jsonl'
            if streaming_hash(p)!=manifest['models'][name]['results_sha256'][regime]:raise ValueError('Actual raw result bytes changed')
            rows[name][regime]=[json.loads(s) for s in p.read_text().splitlines()]
    collisions=json.loads((DESTINATION/'collisions.json').read_text());metrics=summarize(rows,cases,registry,plan,collisions)
    if metrics!=report['metrics']:raise ValueError('Complete raw-data metrics differ from independent audit')
    counts=independent_counts(rows,cases,registry,metrics)
    suffixes=('.png','.pdf','.csv','.pairs.csv','.factorial.csv','.data.json','.layout.json','.crosscheck.json','.source.py','.provenance.json')
    outputs=[prefix.with_suffix(s) for s in suffixes]
    if any(p.exists() for p in outputs):raise ValueError('Preserve earlier exports')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    names=['Native','Shared train all / deploy all','Shared train question / deploy all',
           'Shared train question / deploy question','GDN train all / deploy all']
    cells=[(n,r) for n in MODELS for r in REGIMES];labels=[f'{n} | {r}' for n in names for r in REGIMES]
    colors=dict(zip(REGIMES,['#778da0','#2b8a73','#d39337','#8266ad']))
    joint=[counts[n][r]['joint'] for n,r in cells]
    current=[sum(counts[n][r]['roles']['target_current'][v] for v in ('control','treatment')) for n,r in cells]
    total=[counts[n][r]['original_tasks_correct'] for n,r in cells]
    fig,axes=plt.subplots(1,3,figsize=(16,11))
    for ax,values,limit,label,title in zip(axes,[joint,current,total],[48,96,336],
            ['Joint cases correct / 48','Current bindings correct / 96','Original task IDs correct / 336'],
            ['A. All six post-update answers','B. Current target after update','C. All original tasks and roles']):
        ax.barh(np.arange(20),values,color=[colors[r] for _,r in cells],height=.7)
        ax.set(yticks=np.arange(20),yticklabels=labels if ax is axes[0] else [],xlim=(0,limit*1.12),xlabel=label,title=title)
        for i,v in enumerate(values):ax.text(v+limit*.015,i,str(v),va='center',fontsize=8)
        for pos in (3.5,7.5,11.5,15.5):ax.axhline(pos,color='#ddd',lw=.7)
        ax.invert_yaxis();ax.set_axisbelow(True);ax.grid(axis='x',alpha=.2)
        for side in ('top','right'):ax.spines[side].set_visible(False)
    fig.suptitle('Text competence controls — all five registered configurations and four conditions',fontsize=14)
    fig.text(.015,.055,'Reused 8 source streams; all 6720 new readouts. Original task IDs retain pairing; transformed-input collisions are disclosed and are not independent evidence.',fontsize=9)
    fig.text(.015,.025,'Program-selected evidence / executed COPY are privileged controls. Text, positions and prefix computation change; no unique causal or architecture claim.',fontsize=9)
    fig.tight_layout(rect=[0,.085,1,.94]);fig.savefig(prefix.with_suffix('.png'),dpi=180);fig.savefig(prefix.with_suffix('.pdf'));plt.close(fig)
    bound={q['id']:[b for b in registry['bindings'] if b['physical_id']==q['id']] for q in registry['physical']}
    fields=['model','regime','token_policy','physical_id','actual_prompt_sha256','roles','related_binding_count',
            'answer','generated','greedy_exact','answer_nll','query_tokens']
    with prefix.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,regime in cells:
            for row in rows[name][regime]:
                writer.writerow({'model':name,'regime':regime,'physical_id':row['id'],
                    'roles':';'.join(sorted({b['role'] for b in bound[row['id']]})),'related_binding_count':len(bound[row['id']]),
                    **{k:row[k] for k in ['token_policy','actual_prompt_sha256','answer','generated','greedy_exact','answer_nll','query_tokens']}})
    with prefix.with_suffix('.pairs.csv').open('w',newline='') as f:
        fields=['model','candidate','reference','case_id','physical_id','base_case_id','dependency_depth','question_style',
            'distractor_updates','variant','role','candidate_correct','reference_correct','gain','loss','nll_candidate_minus_reference']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name in MODELS:
            for c in metrics['paired_contrasts'][name]:
                for pair in c['current_target']['individual_pairs']:
                    writer.writerow({'model':name,'candidate':c['candidate'],'reference':c['reference'],**{k:pair[k] for k in fields[3:]}})
    with prefix.with_suffix('.factorial.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['model','physical_id','nll_interaction']);writer.writeheader()
        for name in MODELS:
            for v in metrics['factorial_nll_interactions'][name]['all_physical_interactions']:
                writer.writerow({'model':name,'physical_id':v['id'],'nll_interaction':v['nll_interaction']})
    with prefix.with_suffix('.csv').open(newline='') as f:csvrows=list(csv.DictReader(f))
    for out,row in zip(csvrows,[v for n,r in cells for v in rows[n][r]],strict=True):
        if (out['model']!=row['model'] or out['regime']!=row['regime'] or out['physical_id']!=row['id'] or
            out['actual_prompt_sha256']!=row['actual_prompt_sha256'] or out['answer']!=row['answer'] or out['generated']!=row['generated'] or
            out['token_policy']!=row['token_policy'] or int(out['related_binding_count'])!=len(bound[row['id']]) or
            out['roles']!=';'.join(sorted({b['role'] for b in bound[row['id']]})) or
            (out['greedy_exact']=='True')!=row['greedy_exact'] or float(out['answer_nll'])!=row['answer_nll'] or int(out['query_tokens'])!=row['query_tokens']):
            raise ValueError('CSV differs from actual original raw readouts')
    if len(csvrows)!=6720:raise ValueError('Missing complete exported task coverage')
    # Decode the two derived CSVs and independently recompute from raw NLLs,
    # original bindings and the registered contrast order, without metric helpers.
    indexed={n:{r:{v['id']:v for v in rows[n][r]} for r in REGIMES} for n in MODELS}
    current=[b for b in registry['bindings'] if b['role']=='target_current' and b['variant']!='before']
    expected_pairs=[(n,c,r,b) for n in MODELS for c,r in plan['analysis']['contrasts'] for b in current]
    with prefix.with_suffix('.pairs.csv').open(newline='') as f:pairrows=list(csv.DictReader(f))
    for out,(n,c,r,b) in zip(pairrows,expected_pairs,strict=True):
        candidate=indexed[n][c][b['physical_id']];reference=indexed[n][r][b['physical_id']]
        flags={'candidate_correct':candidate['greedy_exact'],'reference_correct':reference['greedy_exact'],
               'gain':candidate['greedy_exact'] and not reference['greedy_exact'],
               'loss':reference['greedy_exact'] and not candidate['greedy_exact']}
        if (out['model']!=n or out['candidate']!=c or out['reference']!=r or any(
                out[k]!=str(b[k]) for k in ('case_id','physical_id','base_case_id','dependency_depth','question_style',
                                            'distractor_updates','variant','role')) or
            any(out[k]!=str(v) for k,v in flags.items()) or
            float(out['nll_candidate_minus_reference'])!=candidate['answer_nll']-reference['answer_nll']):
            raise ValueError('Paired CSV differs from raw answers/NLLs or original task bindings')
    with prefix.with_suffix('.factorial.csv').open(newline='') as f:factorrows=list(csv.DictReader(f))
    expected_factor=[(n,q['id']) for n in MODELS for q in registry['physical']]
    for out,(n,key) in zip(factorrows,expected_factor,strict=True):
        value=indexed[n]['compiled_support'][key]['answer_nll']-indexed[n]['support_text'][key]['answer_nll']-\
              indexed[n]['compiled_full'][key]['answer_nll']+indexed[n]['full'][key]['answer_nll']
        if out['model']!=n or out['physical_id']!=key or float(out['nll_interaction'])!=value:
            raise ValueError('Factorial CSV differs from its four actual raw NLLs')
    if len(pairrows)!=1920 or len(factorrows)!=1680:raise ValueError('Missing complete paired/factorial CSV coverage')
    write_json(prefix.with_suffix('.data.json'),metrics)
    write_json(prefix.with_suffix('.layout.json'),{'labels':labels,'joint_counts':joint,'current_counts':current,'original_task_counts':total,
        'model_order':list(MODELS),'regimes':list(REGIMES),'figure_inches':[16,11],'dpi':180,'input_collisions':collisions['by_regime'],
        'intervals':'Exact fixed-cohort counts; related task variants and identical transformed inputs are not independent.'})
    write_json(prefix.with_suffix('.crosscheck.json'),{'status':'all6720_csv_rows_exact_and_independent_task_and_unique_input_counts_match',
        'counts':counts,'raw_csv_rows_verified':6720,'current_target_paired_rows':1920,'factorial_physical_rows':1680,
        'current_target_paired_rows_independently_verified':len(pairrows),
        'factorial_physical_rows_independently_verified':len(factorrows),
        'old_gates_unchanged':report['prior_timing_criteria_unchanged']})
    prefix.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    write_json(prefix.with_suffix('.provenance.json'),{'status':'all_twenty_cells_reproduced_from_actual_audited6720readouts',
        'audit_path':str(audit),'audit_sha256':streaming_hash(audit),'source_run':str(run),'manifest_sha256':report['manifest_sha256'],
        'dataset_manifest_sha256':report['dataset_manifest_sha256'],'export_source_sha256':streaming_hash(Path(__file__)),
        'output_sha256':{p.name:streaming_hash(p) for p in outputs[:-1]}})
    return {'artifacts':len(outputs),'original_task_readouts':6720,'prefix':str(prefix)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path);p.add_argument('--prefix',type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.audit,a.prefix)),flush=True)


if __name__=='__main__':main()
