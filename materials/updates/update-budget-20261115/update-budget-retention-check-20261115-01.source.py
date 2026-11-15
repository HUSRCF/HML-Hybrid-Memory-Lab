"""Post-hoc descriptive retention equivalence on every completed control readout."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.training_data import data_fingerprint


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists() or args.output.with_suffix('.source.py').exists():raise ValueError('Preserve prior post-hoc receipts')
    a=json.loads(args.audit.read_text());run=Path(a['source_run']);m=json.loads((run/'manifest.json').read_text())
    if a['status']!='completed_and_audited_update_budget' or a['manifest_sha256']!=sha(run/'manifest.json'):
        raise ValueError('Complete unchanged audit required')
    policy_path=Path(m['dataset'])/'policies.json'
    policies=json.loads(policy_path.read_text())
    structure=json.loads((args.audit.parent/'update-budget-mask-structure-20261115-01.json').read_text())
    if (data_fingerprint(policies)!=m['policies_sha256'] or structure['policy_sha256']!=sha(policy_path) or
        structure['dataset_manifest_sha256']!=a['dataset_manifest_sha256']):raise ValueError('Locked policy semantics or original file bytes changed')
    cells=[]
    for name in m['model_order']:
        for family in ('random0','random1','early','recent'):
            for reader in ('full','oracle'):
                regime=family+'_'+reader;path=run/f'{name}.{regime}.jsonl'
                if sha(path)!=m['models'][name]['results_sha256'][regime]:raise ValueError('Raw readouts changed')
                rows=[json.loads(s) for s in path.read_text().splitlines()]
                if [r['id'] for r in rows]!=m['selected_physical_ids'] or len(rows)!=336:raise ValueError('Full physical registry required')
                matched=[r for r in rows if policies[r['id']]['support_overlap'][family]['exact_oracle_write_mask']]
                unmatched=[r for r in rows if not policies[r['id']]['support_overlap'][family]['exact_oracle_write_mask']]
                cells.append({'model':name,'family':family,'reader':reader,'matched':len(matched),
                    'matched_correct':sum(r['greedy_exact'] for r in matched),'unmatched':len(unmatched),
                    'unmatched_correct':sum(r['greedy_exact'] for r in unmatched)})
    oracle=[r for r in cells if r['reader']=='oracle'];full=[r for r in cells if r['reader']=='full']
    if len(cells)!=40 or any(r['unmatched_correct'] for r in cells) or any(r['matched_correct']!=r['matched'] for r in oracle):
        raise ValueError('Descriptive retention predicate does not hold on every row')
    out={'status':'all13440control_readouts_retention_checked','created_utc':datetime.now(timezone.utc).isoformat(),
        'posthoc_descriptive_not_preregistered_success_gate':True,'model_inference':False,'primary_audit_sha256':sha(args.audit),
        'source_sha256':sha(Path(__file__)),'cells':cells,'all_readouts':13440,
        'oracle_reader':{'rows':6720,'matched_correct':sum(r['matched_correct'] for r in oracle),
            'unmatched_rows':sum(r['unmatched'] for r in oracle),'unmatched_correct':0,
            'correct_iff_exact_support_write_mask_on_this_cohort':True},
        'full_reader':{'rows':6720,'matched_correct':sum(r['matched_correct'] for r in full),
            'unmatched_rows':sum(r['unmatched'] for r in full),'unmatched_correct':0},
        'interpretation':['Matching-mask successes must replay old semantic-Delta endpoints by the registered contract; not independent discoveries.',
            'Controls suppress the relevant SET during its own encoding when they select another SET. They do not isolate damage by subsequent unrelated updates.',
            'Descriptive on8reused streams. Oracle type budgets already determine COPY selection; no query-independent architecture or new scientific success claim.']}
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in out.items() if k not in ('cells','interpretation')}),flush=True)


if __name__=='__main__':main()
