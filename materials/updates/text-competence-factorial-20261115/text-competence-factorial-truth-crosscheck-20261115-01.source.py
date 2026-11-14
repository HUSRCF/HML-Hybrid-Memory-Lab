"""Check saved transformations and all alias bindings with an independent SET/COPY interpreter."""

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def interpret(events):
    current={};initial={};snapshots=[]
    for event in events:
        if event['kind']=='SET':
            if type(event['value']) is not int or event['source'] is not None:raise ValueError('Malformed SET')
            value=event['value']
        elif event['kind']=='COPY':
            if event['value'] is not None or event['source'] not in current:raise ValueError('Malformed or uninitialized COPY')
            value=current[event['source']]
        else:raise ValueError('Only registered SET/COPY programs allowed')
        current[event['target']]=value
        if event['target'] not in initial:initial[event['target']]=value
        snapshots.append({'current':dict(current),'initial':dict(initial)})
    if not snapshots:raise ValueError('No events')
    return snapshots


def verify(dataset,output):
    manifest=json.loads((dataset/'manifest.json').read_text())
    cases=json.loads((dataset/'cases.json').read_text());registry=json.loads((dataset/'registry.json').read_text())
    transforms=json.loads((dataset/'transformations.json').read_text())
    if (manifest['status']!='registered_before_inference' or manifest['counts']['physical_queries']!=336 or
        len(cases)!=48 or len(registry['physical'])!=336 or len(registry['bindings'])!=432 or len(transforms)!=336):
        raise ValueError('Complete fixed-cohort registrations required')
    case_map={c['id']:c for c in cases};original={q['id']:q for q in registry['physical']}
    regimes=('full','support_text','compiled_full','compiled_support');snapshot_pairs=0
    for key,entry in transforms.items():
        if list(entry)!=list(regimes):raise ValueError('All four cells required')
        full=entry['full']['events'];indices=entry['support_text']['original_event_indices']
        if indices!=sorted(set(indices)) or any(not 0<=i<len(full) for i in indices):raise ValueError('Support indices must preserve event order')
        if entry['support_text']['events']!=[full[i] for i in indices]:raise ValueError('Support text is not the prescribed complete-event subsequence')
        for source,compiled in [('full','compiled_full'),('support_text','compiled_support')]:
            left=entry[source]['events'];right=entry[compiled]['events']
            if len(left)!=len(right):raise ValueError('Compilation changed event count')
            states=interpret(left);others=interpret(right)
            if states!=others:raise ValueError('At least one entity current/initial snapshot changed')
            for before,after,expected in zip(left,right,states,strict=True):
                if before['kind']=='SET':
                    if after!=before:raise ValueError('Original SET changed during compilation')
                elif after!={'kind':'SET','target':before['target'],'value':expected['current'][before['target']],'source':None}:
                    raise ValueError('COPY was not compiled to its actual online snapshot value')
            snapshot_pairs+=len(states)
        for regime,q in entry.items():
            if any(q[field]!=original[key][field] for field in ('id','question','question_ids','answer','answer_ids')):
                raise ValueError('Original question or label changed')
            if q['compiled']!=(regime in ('compiled_full','compiled_support')):raise ValueError('Compiled label drift')
    binding_checks=0
    for binding in registry['bindings']:
        case=case_map[binding['case_id']];world=case['prefix']+case['variants'][binding['variant']]['tail']
        query=next(q for q in case['variants'][binding['variant']]['queries'] if q['role']==binding['role'])
        expected=interpret(world)[-1][query['version']][query['entity']]
        entry=transforms[binding['physical_id']]
        if entry['full']['events']!=world or str(expected)!=query['answer']:raise ValueError('Alias binding changed original world/truth')
        for regime in regimes:
            q=entry[regime];actual=interpret(q['events'])[-1][query['version']][query['entity']]
            if str(actual)!=str(expected) or q['answer']!=str(expected) or q['question']!=query['question']:
                raise ValueError('Saved transformed program changed an alias binding answer')
            binding_checks+=1
    if binding_checks!=1728:raise ValueError('All432 bindings under all4conditions required')
    report={'status':'all_saved_program_snapshots_and_1728_alias_bindings_independently_verified',
        'verified_utc':datetime.now(timezone.utc).isoformat(),'dataset':str(dataset),'physical_registrations':1344,
        'related_binding_checks':binding_checks,'intermediate_current_initial_snapshot_pairs':snapshot_pairs,
        'source_sha256':digest(Path(__file__)),'files_sha256':{n:digest(dataset/n) for n in
            ('manifest.json','cases.json','registry.json','transformations.json')},
        'new_model_inference':False,'new_scientific_success_gate':False,'changes_previous_gates':False,
        'note':'Independent operational crosscheck only; related bindings and repeated transformed inputs are correlated.'}
    with output.open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    target=output.with_suffix('.source.py')
    with target.open('xb') as f:f.write(Path(__file__).read_bytes())
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=Path('/media/990Pro/Code/SAS/datasets/memory_update/text-competence-factorial-v1'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=verify(a.dataset,a.output)
    print(json.dumps({k:r[k] for k in ('status','physical_registrations','related_binding_checks','intermediate_current_initial_snapshot_pairs')}),flush=True)


if __name__=='__main__':main()
