"""Compare identical GDN projection allocations under all-token/question timing."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.calibration import file_hash
from sas_memory.training_data import data_fingerprint
from scripts.compare_paired_semantic_controllers import functional_units, paired_counts
from scripts.compare_semantic_single_path_controls import evaluated_rows
from scripts.run_controller_trial import write_json

MODELS = ["gdn_all","gdn_question"]
CONDITIONS = ["canonical","permuted"]


def teacher_cells(run,arm,model_audit,data):
    evaluation=run/arm/'step-000432/evaluation'
    summary=json.loads((evaluation/'summary.json').read_text())
    if (file_hash(evaluation/'summary.json')!=model_audit['canonical']['evaluation_summary_sha256'] or
        file_hash(evaluation/'teacher_train.jsonl')!=summary['file_sha256']['teacher_train.jsonl']):
        raise ValueError('Audited teacher-fit observations changed')
    rows=[json.loads(line) for line in (evaluation/'teacher_train.jsonl').read_text().splitlines()]
    if len(rows)!=len(data['train']):raise ValueError('Incomplete teacher-fit cohort')
    cells=defaultdict(list)
    for number,(row,example) in enumerate(zip(rows,data['train'])):
        if (row['training_index']!=number or row['case_id']!=example['case']['id'] or row['variant']!=example['variant'] or row['role']!=example['query']['role']):
            raise ValueError('Teacher-fit example identity/order changed')
        cells[example['case']['family'],row['role']].append(row)
    result=[{'family':family,'role':role,'examples':len(members),
             'numeric_tokens_correct':sum(sum(left==right for left,right in zip(row['answer_ids'][:3],row['teacher_prediction_ids'][:3])) for row in members),
             'numeric_tokens':3*len(members),'newline_correct':sum(row['answer_ids'][3]==row['teacher_prediction_ids'][3] for row in members),
             'teacher_loss_including_newline':sum(row['teacher_loss_including_newline'] for row in members)/len(members)}
            for (family,role),members in sorted(cells.items())]
    fit=model_audit['teacher_train_fit']
    if (sum(cell['numeric_tokens_correct'] for cell in result)!=fit['teacher_numeric_tokens_correct'] or
        sum(cell['numeric_tokens'] for cell in result)!=fit['teacher_numeric_tokens'] or
        sum(cell['newline_correct'] for cell in result)!=fit['teacher_newline_correct']):
        raise ValueError('Family/role teacher-fit counts disagree with the independent audit')
    return result


def compare(audit):
    if (audit["status"] != "completed_and_audited" or audit["validation_records_audited"] != 864 or
        audit["teacher_train_records_audited"] != 864 or not audit["paired_initialization_verified"]):
        raise ValueError("Require full independently audited GDN timing training/evaluation")
    run = Path(audit["source_run"])
    if file_hash(run/'manifest.json') != audit["manifest_sha256"]:
        raise ValueError("Timing trial manifest changed after audit")
    plan = json.loads((run/'plan.json').read_text())
    reference = Path(plan["component_reference_run"])
    baseline = audit["component_reference_audit"]
    original = baseline['arms']['gdn_only']
    if (baseline["status"] != "completed_and_audited" or file_hash(reference/'manifest.json') != baseline["manifest_sha256"] or
        audit["initial_state_sha256"] != original["initial_state_sha256"] or audit["parameters"] != original["parameters"]):
        raise ValueError("All-token reference or paired allocation/initialization changed")
    data = json.loads((Path(plan['data_source'])/'data.json').read_text())
    if data_fingerprint(data) != plan['data_sha256']:
        raise ValueError("Audited timing dataset changed")
    expected = {(e['case']['id'],e['variant'],e['query']['role']):e['query']['answer'] for e in data['validation']}
    sources = {'gdn_all':(reference,'gdn_only',original),'gdn_question':(run,'gdn_question',audit)}
    rows = {condition:{name:evaluated_rows(path,arm,condition,model,expected) for name,(path,arm,model) in sources.items()} for condition in CONDITIONS}
    units = {condition:{name:functional_units(members) for name,members in models.items()} for condition,models in rows.items()}
    counts = {condition:{name:{metric:{'correct':sum(values.values()),'units':len(values)} for metric,values in metrics.items()}
                         for name,metrics in models.items()} for condition,models in units.items()}
    pairs = {condition:{metric:paired_counts(models['gdn_all'][metric],models['gdn_question'][metric]) for metric in models['gdn_all']}
             for condition,models in units.items()}
    roles = {condition:{name:{role:{'correct':sum(row['greedy_exact'] for row in members if row['role']==role),
                                   'examples':sum(row['role']==role for row in members)}
                              for role in ('target_current','protected_current','target_initial')}
                        for name,members in models.items()} for condition,models in rows.items()}
    updates = {condition:{name:{family:{'correct':sum(row['greedy_exact'] for row in members if row['family']==family and row['role']=='target_current' and row['variant']!='before'),
                                        'examples':sum(row['family']==family and row['role']=='target_current' and row['variant']!='before' for row in members)}
                               for family in sorted({row['family'] for row in members})} for name,members in models.items()} for condition,models in rows.items()}
    return {'status':'completed_paired_gdn_timing_comparison','source_run':str(run),'manifest_sha256':audit['manifest_sha256'],
        'all_token_reference_run':str(reference),'reference_manifest_sha256':baseline['manifest_sha256'],
        'parameters_each':audit['parameters'],'paired_initial_state_sha256':audit['initial_state_sha256'],
        'counts':counts,'all_to_question':pairs,'role_counts':roles,'target_updates_by_family':updates,
        'canonical_to_permuted':{name:{metric:paired_counts(units['canonical'][name][metric],units['permuted'][name][metric]) for metric in units['canonical'][name]} for name in MODELS},
        'teacher_train_fit':{'gdn_all':original['teacher_train_fit'],'gdn_question':audit['teacher_train_fit']},
        'teacher_fit_by_family_role':{name:teacher_cells(path,arm,model,data) for name,(path,arm,model) in sources.items()},
        'question_native_prefix_checks':audit['native_prefix_checks'],
        'interpretation':plan['interpretation']+' Both arms use the same twelve underlying development streams and one paired initialization. Equal totals are not equivalence, and fit differences can confound a timing gap. The restricted arm includes question and answer computation, with all native historical Attention available.'}


def plot(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    metrics=[('individual','Individual answers /432'),('stable_unchanged_case_role','Stable unchanged facts /96'),
             ('updated_target_except_add','Updated target excluding ADD /80')]
    figure,axes=plt.subplots(1,3,figsize=(12,4.8))
    for axis,(metric,title) in zip(axes,metrics):
        for offset,(name,label,color) in enumerate((('gdn_all','Adapt all tokens','#2563eb'),('gdn_question','Adapt question / answer','#059669'))):
            cells=[report['counts'][condition][name][metric] for condition in CONDITIONS]
            bars=axis.bar(np.arange(2)+(offset-.5)*.3,[100*cell['correct']/cell['units'] for cell in cells],.3,label=label,color=color)
            for bar,cell in zip(bars,cells):
                axis.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,str(cell['correct']),ha='center',fontsize=9)
        axis.set_xticks(range(2),['Canonical','Permuted'])
        axis.set_ylim(0,110)
        axis.set_title(title,fontsize=10)
        axis.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('All required answers correct (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    figure.legend(handles,labels,frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,.07),fontsize=10)
    figure.suptitle('Frozen Qwen3.5-2B IT: same GDN adapters, different adaptation timing',fontsize=13)
    figure.text(.5,.025,'Paired initial weights, 696,192 parameters, 432 updates each. Twelve reused development streams; no equivalence claim.',ha='center',fontsize=8,color='#475569')
    figure.tight_layout(rect=(0,.17,1,.91))
    output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(output,dpi=200,bbox_inches='tight')
    figure.savefig(output.with_suffix('.pdf'),bbox_inches='tight')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plot',type=Path)
    args=parser.parse_args()
    report=compare(json.loads(args.audit.read_text()))
    report.update(audit_report_sha256=file_hash(args.audit),analysis_source_sha256=file_hash(Path(__file__)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    if args.plot:plot(report,args.plot)
    print(json.dumps({condition:{name:metrics['individual'] for name,metrics in models.items()} for condition,models in report['counts'].items()}))


if __name__=='__main__':
    main()
