"""Export every registered arm and physical answer from an audited timing run."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json
from scripts.unseen_shared_timing_data import ORDER,registered_inputs,selected_inputs
from scripts.unseen_shared_timing_metrics import summarize


def export(audit_path,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    report=json.loads(audit_path.read_text());run=Path(report['source_run'])
    manifest=json.loads((run/'manifest.json').read_text())
    registered,plan,cases,registry=registered_inputs();physical,bindings=selected_inputs(registry)
    if (report['status']!='completed_and_audited_unseen_shared_timing' or not report['operational_controls_passed'] or
        report['manifest_sha256']!=streaming_hash(run/'manifest.json') or
        streaming_hash(ROOT/'scripts/analyze_unseen_shared_timing.py')!=report['analysis_source_sha256']):
        raise ValueError('An unchanged successful independent primary audit is required')
    all_rows={}
    for name in ORDER:
        path=run/f'{name}.jsonl'
        if streaming_hash(path)!=manifest['models'][name]['results_sha256']:raise ValueError('Audited complete readouts changed')
        all_rows[name]=[json.loads(line) for line in path.read_text().splitlines()]
    metrics=summarize(all_rows,cases,registry,registered)
    if metrics!=report['metrics']:raise ValueError('Full raw-data metrics do not reproduce the audit')
    outputs=[prefix.with_suffix(s) for s in ('.png','.pdf','.csv','.data.json','.layout.json','.source.py','.provenance.json')]
    if any(p.exists() for p in outputs):raise ValueError('Preserve earlier exports')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    labels=['Native','Shared / all tokens','Shared / question + answer','Shared / ledger only',
        'Independent','Local / all tokens','Local / question + answer','Attention only','GDN / all tokens','GDN / question + answer']
    colors=['#8a929c','#2166ac','#d78a00','#b44949',*['#6b8d8f']*6]
    y=np.arange(len(ORDER));fig,axes=plt.subplots(1,2,figsize=(12.5,6.6),gridspec_kw={'width_ratios':[1,1.5]})
    joint=[metrics['models'][n]['joint_six_after'] for n in ORDER]
    axes[0].barh(y,joint,color=colors,height=.65)
    axes[0].set(yticks=y,yticklabels=labels,xlim=(0,52),xlabel='Joint cases correct / 48',title='A. All six post-update answers correct')
    for j,value in enumerate(joint):axes[0].text(value+.5,j,str(value),va='center',fontsize=9)
    roles=['target_current','target_initial','protected_current'];role_colors=['#3878b5','#6d9e6f','#af76a4']
    role_data={}
    for i,(role,color,label) in enumerate(zip(roles,role_colors,['Current','Historical','Unrelated'])):
        values=[sum(metrics['models'][n]['role_variant_counts'][role][v]['correct'] for v in ['control','treatment']) for n in ORDER]
        role_data[role]=values
        axes[1].barh(y+(i-1)*.22,values,height=.2,label=label,color=color)
    axes[1].set(yticks=y,yticklabels=[],xlim=(0,100),xlabel='Post-update related bindings correct / 96',title='B. Current, historical and unrelated answers')
    handles,legend_labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='center left',bbox_to_anchor=(.875,.5),fontsize=9,frameon=False)
    for ax in axes:
        ax.invert_yaxis();ax.set_axisbelow(True);ax.grid(axis='x',alpha=.22)
        for spine in ['top','right']:ax.spines[spine].set_visible(False)
    p=metrics['within_checkpoint_question_functional_preservation']
    fig.suptitle('Unseen-stream shared-controller timing — all ten registered arms',fontsize=14)
    fig.text(.02,.065,f"Question-only vs shared/all: {p['preserved']} preserved successes; {p['regression']} regressions; {p['repair']} repairs; {p['both_wrong']} both wrong (336 physical questions).",fontsize=9)
    fig.text(.02,.025,'One frozen IT checkpoint; 8 source streams. Related bindings are correlated. Schedule ablations do not establish a semantic-write mechanism.',fontsize=8)
    fig.tight_layout(rect=[0,.105,.875,.94]);fig.savefig(prefix.with_suffix('.png'),dpi=180);fig.savefig(prefix.with_suffix('.pdf'));plt.close(fig)
    bound={q['id']:[b for b in bindings if b['physical_id']==q['id']] for q in physical}
    fields=['arm','token_policy','physical_id','related_binding_count','roles','answer','generated','greedy_exact','answer_nll','query_tokens']
    with prefix.with_suffix('.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for name in ORDER:
            for row in all_rows[name]:
                writer.writerow({'arm':name,'token_policy':row['token_policy'],'physical_id':row['id'],
                    'related_binding_count':len(bound[row['id']]),'roles':';'.join(sorted({b['role'] for b in bound[row['id']]})),
                    **{k:row[k] for k in ['answer','generated','greedy_exact','answer_nll','query_tokens']}})
    write_json(prefix.with_suffix('.data.json'),metrics)
    write_json(prefix.with_suffix('.layout.json'),{'arm_order':list(ORDER),'labels':labels,'joint_counts':joint,'role_counts':role_data,
        'joint_denominator':48,'role_denominator':96,'physical_questions':336,'base_streams':8,'figure_inches':[12.5,6.6],'dpi':180,
        'intervals':'No binomial intervals on correlated bindings; exact registered fixed-cohort counts.'})
    prefix.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    write_json(prefix.with_suffix('.provenance.json'),{'status':'all_ten_arms_reproduced_from_actual_audited3360readouts',
        'audit_path':str(audit_path),'audit_sha256':streaming_hash(audit_path),'source_run':str(run),
        'manifest_sha256':report['manifest_sha256'],'dataset_manifest_sha256':report['dataset_manifest_sha256'],
        'export_source_sha256':streaming_hash(Path(__file__)),
        'output_sha256':{p.name:streaming_hash(p) for p in outputs[:-1]}})
    return {'artifacts':len(outputs),'physical_rows':sum(map(len,all_rows.values())),'prefix':str(prefix)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('audit',type=Path);parser.add_argument('--prefix',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(export(args.audit,args.prefix)),flush=True)


if __name__=='__main__':main()
