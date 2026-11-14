"""Export audited native locality counts, paired effects and a research figure."""

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import CELLS, ROLES, streaming_hash
from sas_memory.write_locality_metrics import summarize
from sas_memory.tasks import Operation, execute
from sas_memory.training_data import data_fingerprint
from scripts.run_controller_trial import write_json


def export(audit_path):
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'completed_and_audited' or audit['records_audited'] != 576:
        raise ValueError('A complete independently audited native locality screen is required')
    run = Path(audit['source_run'])
    manifest = json.loads((run/'manifest.json').read_text())
    raw = run/'validation.native.jsonl'
    if streaming_hash(run/'manifest.json') != audit['manifest_sha256'] or streaming_hash(raw) != manifest['result_sha256']:
        raise ValueError('Audited manifest or raw observations changed')
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    metrics = summarize(rows)
    if metrics != audit['metrics']:
        raise ValueError('Reconstructed counts/contrasts differ from the independent audit')
    data = json.loads((Path(manifest['dataset'])/'data.json').read_text())
    if data_fingerprint(data) != audit['data_sha256']:
        raise ValueError('Audited semantic events changed before error inspection')
    examples = {(e['case']['id'],e['variant'],e['query']['role']):e for e in data['validation']}
    errors = []
    for row in rows:
        if row['greedy_exact']:
            continue
        example = examples[row['case_id'],row['variant'],row['role']]
        case = example['case']
        events = case['prefix']+case['variants'][row['variant']]['tail']
        truth = execute([Operation(**event) for event in events])
        old_a = str(case['prefix'][0]['value'])
        old_c = str(case['prefix'][2]['value'])
        current_a = str(truth['current'][case['target_entity']])
        current_c = str(truth['current'][case['other_entity']])
        generated = row['generated'].strip()
        if row['role']=='target_initial' and generated==current_a:
            kind = 'current_A_instead_of_initial_A'
        elif row['role']=='protected_current' and generated in (old_a,current_a):
            kind = 'A_value_instead_of_B'
        elif generated in (old_c,current_c):
            kind = 'C_value_instead_of_requested_value'
        elif generated in {str(event['value']) for event in case['prefix'][3:]}:
            kind = 'another_record_literal'
        else:
            kind = 'other_output'
        errors.append({**{field:row[field] for field in ('case_id','variant','role','seed','active_accounts','value_version','answer','generated')},
                       'category':kind,'initial_A':old_a,'current_A':current_a,'initial_C':old_c,'current_C':current_c,
                       'query_entity':example['query']['entity'],
                       'matches_initial_A':generated==old_a,'matches_current_A':generated==current_a,
                       'matches_initial_C':generated==old_c,'matches_current_C':generated==current_c})
    error_summary = {'errors':len(errors), 'category_counts':dict(Counter(row['category'] for row in errors)),
                     'role_counts':dict(Counter(row['role'] for row in errors)), 'rows':errors,
                     'interpretation':'Exact output-value matches only. Matching another entity/version value does not identify the responsible representation, erase operator or Attention address.'}
    return {'status':'completed_native_locality_diagnostics', 'source_run':str(run),
            'audit_report_sha256':streaming_hash(audit_path), 'manifest_sha256':audit['manifest_sha256'],
            'protocol_sha256':audit['protocol_sha256'], 'dataset_manifest_sha256':audit['dataset_manifest_sha256'],
            'analysis_source_sha256':streaming_hash(Path(__file__)), 'metrics':metrics,'error_value_matches':error_summary,
            'interpretation':'Native full-history developmental screen at fixed record count and aligned token positions. Four base streams with related value siblings, loads and queries. Functional regressions do not identify a damaged pathway or demonstrate a need for joint architecture changes. NLL-only shifts are descriptive.'}


def plot(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels = ('Repeat A, C', 'Update A', 'Update C', 'Update A and C')
    colors = ('#64748b','#2563eb','#d97706','#059669')
    titles = ('Current value of A', 'Protected current value of B', 'Initial value of A')
    lookup = {(row['active_accounts'],row['cell'],row['role']):row for row in report['metrics']['cell_role_counts']}
    figure, axes = plt.subplots(1,3,figsize=(13,5))
    for axis, role, title in zip(axes,ROLES,titles):
        for index,(cell,label,color) in enumerate(zip(CELLS,labels,colors)):
            cells = [lookup[load,cell,role] for load in (8,32,128)]
            bars = axis.bar(np.arange(3)+(index-1.5)*.19,[100*row['correct']/row['examples'] for row in cells],.19,color=color,label=label)
            for bar,row in zip(bars,cells):
                axis.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,str(row['correct']),ha='center',fontsize=8)
        axis.set_xticks(range(3),['8','32','128'])
        axis.set_xlabel('Active accounts')
        axis.set_ylim(0,112)
        axis.set_title(title,fontsize=11)
        axis.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Greedy exact accuracy (%)')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles,legend_labels,ncol=4,frameon=False,loc='lower center',bbox_to_anchor=(.5,.065),fontsize=9)
    figure.suptitle('Native Qwen3.5-2B IT: fixed-length semantic write controls',fontsize=13)
    figure.text(.5,.025,'Counts above bars are /16 related readouts. Four base streams; 2,865-token prompts; no training or significance claim.',ha='center',fontsize=8,color='#475569')
    figure.tight_layout(rect=(0,.20,1,.91))
    path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(path,dpi=200,bbox_inches='tight')
    figure.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plot',type=Path)
    args = parser.parse_args()
    report = export(args.audit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    with args.output.with_suffix('.csv').open('w') as output:
        rows = report['metrics']['cell_role_counts']
        writer = csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.plot:
        plot(report,args.plot)
    print(json.dumps({'correct':report['metrics']['correct'],'examples':576,
                      'screen_signal':report['metrics']['screen_signal']['present'],
                      'by_load':[row['individual'] for row in report['metrics']['functional_units_by_load']]}))


if __name__ == '__main__':
    main()
