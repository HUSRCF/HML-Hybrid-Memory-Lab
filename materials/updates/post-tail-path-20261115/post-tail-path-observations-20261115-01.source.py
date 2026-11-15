"""Posthoc descriptive extraction; no additional selection, gate or inference."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def extract(audit_path):
    audit=json.loads(audit_path.read_text());m=audit['metrics']
    if audit['status']!='completed_and_audited_post_tail_path' or m['cell_count']!=32:raise ValueError('Complete audited grid required')
    grid=[];held_kv={'correct':0,'readouts':0,'new_primary_readouts':0,'archived_endpoint_references':0}
    for model,cells in m['cells'].items():
        for cell,values in cells.items():
            condition,reader=cell.split('.');strata=m['strata'][model][cell]
            row={'model':model,'condition':condition,'reader':reader,'origin':values['measurement_origin'],
                'new_tail':strata['evidence']['False'],'fixed_prefix':strata['evidence']['True'],
                'all':values,'by_role':strata['role']}
            grid.append(row)
            if condition[-1]=='H':
                held_kv['correct']+=row['new_tail']['greedy_exact'];held_kv['readouts']+=row['new_tail']['readouts']
                held_kv['new_primary_readouts' if row['origin']=='new_primary' else 'archived_endpoint_references']+=row['new_tail']['readouts']
    run=Path(audit['source_run']);manifest=json.loads((run/'manifest.json').read_text())
    component_counts={model:{kind:sorted({len(v['component_differences'][kind]['different_keys']) for v in state['sources'].values()})
        for kind in ('recurrent_states','conv_states','kv')} for model,state in manifest['models'].items()}
    return {'status':'posthoc_descriptive_extraction_of_complete_factorial','audit_sha256':streaming_hash(audit_path),
        'source_sha256':streaming_hash(Path(__file__)),'all32_new_and_prefix_evidence_cells':grid,
        'held_tail_KV_new_value_controls':held_kv,'component_difference_counts':component_counts,
        'all_paired_contrasts':m['paired_contrasts'],
        'nll_interaction_means':{name:{reader:v['means'] for reader,v in groups.items()} for name,groups in m['nll_factorial_interactions'].items()},
        'independent_streams':8,'reused_cohort':True,'fresh_confirmation':False,'new_scientific_gate':False,
        'architecture_effectiveness_claim':False,'previous_failed_criteria_unchanged':audit['prior_failed_criteria_unchanged'],
        'interpretation_limits':['Crossed caches can be off distribution.','A failed answer does not prove semantic information is absent from a tensor.',
            'Old recurrent state can change subsequent query and answer computation.','Oracle support is privileged, not an implemented router.',
            'Repeated readouts and archived references are not independent streams.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite an extraction')
    result=extract(a.audit);write_json(a.output,result);a.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:result[k] for k in ('status','audit_sha256','held_tail_KV_new_value_controls','component_difference_counts')}),flush=True)


if __name__=='__main__':main()
