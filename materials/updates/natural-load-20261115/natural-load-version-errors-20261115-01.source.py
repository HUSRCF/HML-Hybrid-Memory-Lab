"""Posthoc attribution of EVERY generated answer to executable ledger records."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.natural_load_data import registered_inputs, ROLES
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def extract(audit_path):
    a=json.loads(audit_path.read_text());run=Path(a['source_run']);m=json.loads((run/'manifest.json').read_text())
    registered,cases,registry,queries,_=registered_inputs()
    if (a['status']!='completed_and_audited_natural_load' or a['primary_readouts_audited']!=3456 or
        a['source_files_verified']!=240 or streaming_hash(run/'manifest.json')!=a['manifest_sha256'] or
        m['source_sha256']!=registered['source_sha256'] or m['status']!='completed' or any(a['prior_failed_criteria_unchanged'].values())):
        raise ValueError('Complete immutable independently audited grid required')
    case_by_id={c['id']:c for c in cases};records={}
    for c in registry['contexts']:
        case=case_by_id[c['case_id']];episode=next(e for e in case['episodes'] if e['id']==c['episode'])
        events=case['prefix_events']+episode['tail']
        if len({e['value'] for e in events})!=len(events):raise ValueError('Unique recorded values required for exact attribution')
        records[c['id']]=(case,events,{e['value']:i for i,e in enumerate(events)})
    rows=[];summaries=defaultdict(Counter);seen=set()
    for model in ('native','gdn_all_tokens'):
        raw=run/f'{model}.jsonl'
        if streaming_hash(raw)!=m['models'][model]['results_sha256']:raise ValueError('Raw readout bytes changed')
        for line in raw.read_text().splitlines():
            row=json.loads(line);q=queries[row['id']];key=(model,row['reader'],q['id'])
            if key in seen:raise ValueError('Duplicate physical readout')
            seen.add(key);case,events,by_value=records[q['context_id']]
            match=re.fullmatch(r'\s*(-?\d+)\s*',row['generated']);number=int(match.group(1)) if match else None
            event_index=by_value.get(number);event=events[event_index] if event_index is not None else None
            entity=case['protected'] if q['role']=='protected_current' else case['target']
            if row['greedy_exact']:category='correct'
            elif match is None:category='non_integer_output'
            elif number==int(q['answer']):category='same_integer_noncanonical_text'
            elif event is None:category='unrecorded_integer'
            elif event['target']==entity:category='queried_account_other_version'
            else:category='other_account_recorded_value'
            if row['greedy_exact']!=(bool(match) and match.group(1)==q['answer']):
                raise ValueError('Unconstrained grading differs from archived readout')
            source=None
            if event:
                if event['target']==case['target']:
                    source=('target_initial' if event_index==next(i for i,e in enumerate(events) if e['target']==case['target'])
                        else 'target_pre_tail_current' if event_index==18 else 'target_new_current')
                elif event['target']==case['protected']:source='protected_record'
                else:source='other_prefix_record' if event_index<19 else 'tail_distractor_record'
            result={k:q[k] for k in ('id','seed','naming','episode','role','question_style','answer','prefix_evidence','support_event_index')}
            result.update(model=model,reader=row['reader'],generated=row['generated'],greedy_exact=row['greedy_exact'],
                category=category,recorded_value_source=source,matched_event_index=event_index,
                matched_account=event['target'] if event else None,queried_account=entity)
            rows.append(result);summaries[(model,row['reader'],q['episode'],q['role'])][category]+=1
    if len(rows)!=3456 or len(seen)!=3456 or len(summaries)!=108:
        raise ValueError('All3456answers and108role cells required; no error-only selection')
    totals=defaultdict(Counter)
    for row in rows:totals[(row['model'],row['reader'])][row['category']]+=1
    return {'status':'posthoc_exact_ledger_value_attribution_all_readouts','audit_sha256':streaming_hash(audit_path),
        'manifest_sha256':a['manifest_sha256'],'readouts':3456,'role_cells':108,'independent_streams':8,
        'totals':[{'model':model,'reader':reader,'counts':dict(counts)} for (model,reader),counts in sorted(totals.items())],
        'cells':[{'model':model,'reader':reader,'episode':episode,'role':role,'counts':dict(counts)}
            for (model,reader,episode,role),counts in sorted(summaries.items())],
        'rows':rows,'analysis_source_sha256':streaming_hash(Path(__file__)),
        'limits':'Posthoc after observing the complete grid. Exact numeric attribution identifies which recorded value was emitted; it does not establish attention to that record, a causal version-address mechanism, or recurrent-state loss. No data, model, gate or tolerance changes.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Never overwrite attribution')
    value=extract(args.audit);write_json(args.output,value);args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status':value['status'],'readouts':value['readouts'],'totals':value['totals']}),flush=True)


if __name__=='__main__':main()
