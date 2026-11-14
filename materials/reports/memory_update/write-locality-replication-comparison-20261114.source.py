"""Compare complete discovery/replica cohorts without pairing different seeds."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.locality_replication import discovery_reference, primary_recurrence, read_plan, summarize
from sas_memory.tasks import Operation, execute
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from sas_memory.write_locality_metrics import paired
from scripts.run_controller_trial import write_json


def error_matches(rows, dataset, expected_hash):
    data = json.loads((dataset/'data.json').read_text())
    if data_fingerprint(data) != expected_hash:
        raise ValueError('Audited entity/version facts changed')
    examples = {(e['case']['id'],e['variant'],e['query']['role']):e for e in data['validation']}
    errors = []
    for row in rows:
        if row['greedy_exact']:
            continue
        example = examples[row['case_id'],row['variant'],row['role']]
        case = example['case']
        truth = execute([Operation(**event) for event in case['prefix']+case['variants'][row['variant']]['tail']])
        old_a,old_c = str(case['prefix'][0]['value']),str(case['prefix'][2]['value'])
        current_a,current_c = str(truth['current'][case['target_entity']]),str(truth['current'][case['other_entity']])
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
        errors.append({**{key:row[key] for key in ('case_id','variant','role','seed','active_accounts','value_version','answer','generated')},
                       'category':kind,'initial_A':old_a,'current_A':current_a,'initial_C':old_c,'current_C':current_c})
    return {'errors':len(errors),'category_counts':dict(Counter(row['category'] for row in errors)),
            'role_counts':dict(Counter(row['role'] for row in errors)),'rows':errors,
            'interpretation':'Exact output-value matches describe entity/version behavior; they do not identify the causal storage or addressing pathway.'}


def export(path):
    audit = json.loads(path.read_text())
    plan = read_plan()
    discovery = discovery_reference(plan)
    if audit['status']!='completed_and_audited' or audit['records_audited']!=576 or audit['weights_sha256']!=discovery['weights_sha256']:
        raise ValueError('Complete independently audited replica and identical native model required')
    cohorts = {}
    for label,report in (('discovery',discovery),('replica',audit)):
        run = Path(report['source_run'])
        manifest = json.loads((run/'manifest.json').read_text())
        raw = run/'validation.native.jsonl'
        if streaming_hash(run/'manifest.json')!=report['manifest_sha256'] or streaming_hash(raw)!=manifest['result_sha256']:
            raise ValueError('Audited cohort manifest/observations changed')
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        cohort_plan = json.loads((run/'plan.json').read_text())
        if summarize(rows,cohort_plan)!=report['metrics']:
            raise ValueError('Complete cohort statistics changed')
        clusters = []
        for spec in plan['primary_contrasts']:
            for seed in cohort_plan['development_grid']['base_seeds']:
                members = [row for row in rows if row['active_accounts']==spec['active_accounts'] and
                           row['role']==spec['role'] and row['seed']==seed]
                result = paired(members,spec['left_cell'],spec['right_cell'])
                if result['units']!=4:
                    raise ValueError('Primary base-stream cluster is incomplete')
                clusters.append({'primary':spec['id'],'seed':seed,**result})
        cohorts[label] = {'source_run':str(run),'manifest_sha256':report['manifest_sha256'],
                          'seeds':cohort_plan['development_grid']['base_seeds'],'metrics':report['metrics'],
                          'primary_by_base_stream':clusters,
                          'error_value_matches':error_matches(rows,Path(manifest['dataset']),report['data_sha256'])}
    primary = primary_recurrence(audit['metrics'],plan)
    if primary!=audit['primary_recurrence']:
        raise ValueError('Replica primary recurrence changed after audit')
    comparisons = []
    for spec,replica_result in zip(plan['primary_contrasts'],primary['contrasts']):
        original = next(row for row in discovery['metrics']['preservation_contrasts'] if
                        all(row[key]==spec[key] for key in ('active_accounts','axis','fixed_other_bit','role')))
        comparisons.append({'id':spec['id'],'active_accounts':spec['active_accounts'],'role':spec['role'],
                            'discovery':original,'replica':replica_result})
    return {'status':'completed_native_locality_replication_comparison','audit_report_sha256':streaming_hash(path),
            'analysis_source_sha256':streaming_hash(Path(__file__)),'native_weights_sha256':audit['weights_sha256'],
            'cohorts':cohorts,'primary_comparison':comparisons,'event_path_gate_passed':primary['event_path_gate_passed'],
            'interpretation':'Independent-seed replication under the same generator, four base streams per cohort. All tail pairs are evaluated within their own cohort. No case pairing across seeds, independent-n576 confidence, architecture or recurrent-destruction claim. Initial-A also differs in the availability of redundant recent old-value evidence.'}


def plot(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    figure,axes = plt.subplots(1,2,figsize=(10,4.8))
    titles = ('Protected B, 32 accounts','Initial A, 128 accounts')
    for axis,comparison,title in zip(axes,report['primary_comparison'],titles):
        for offset,(key,label,color) in enumerate((('reference_correct','Repeat old A','#64748b'),('candidate_correct','Write new A','#059669'))):
            counts = [comparison[cohort][key] for cohort in ('discovery','replica')]
            bars = axis.bar(np.arange(2)+(offset-.5)*.3,[100*value/16 for value in counts],.3,label=label,color=color)
            for bar,value in zip(bars,counts):
                axis.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,str(value),ha='center',fontsize=10)
        axis.set_xticks(range(2),['Discovery','New-seed replica'])
        axis.set_ylim(0,112)
        axis.set_title(title,fontsize=11)
        axis.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Greedy exact accuracy (%)')
    handles,labels = axes[0].get_legend_handles_labels()
    figure.legend(handles,labels,ncol=2,frameon=False,loc='lower center',bbox_to_anchor=(.5,.075))
    figure.suptitle('Native fixed-length update controls: two registered primaries',fontsize=13)
    figure.text(.5,.025,'C is updated in both conditions. Counts /16 related readouts; four base streams per cohort; no training.',ha='center',fontsize=8,color='#475569')
    figure.tight_layout(rect=(0,.20,1,.90))
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
        rows = [{'cohort':label,**row} for label,cohort in report['cohorts'].items() for row in cohort['metrics']['cell_role_counts']]
        writer = csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.plot:plot(report,args.plot)
    print(json.dumps({'cohorts':{label:cohort['metrics']['correct'] for label,cohort in report['cohorts'].items()},
                      'primary':{row['id']:{'reference':row['replica']['reference_correct'],'candidate':row['replica']['candidate_correct'],
                                           'regressed':row['replica']['regressed'],'recurs':row['replica']['recurs']} for row in report['primary_comparison']},
                      'event_path_gate_passed':report['event_path_gate_passed']}))


if __name__=='__main__':main()
