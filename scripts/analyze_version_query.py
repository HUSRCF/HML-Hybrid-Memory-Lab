"""Independent246source audit of single-token version-question controls."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.version_query_data import (read_plan, registered_inputs, sources, selected_contexts,
    context_queries, DESTINATION, PROTOCOL, MODELS, READERS, STYLES, ROLES)
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json
from scripts.run_version_query import RUNNER_REUSE, verify_model_sources, primary_subset_replay
from scripts.analyze_post_encoding import audit_cache, audit_reader
from scripts.post_encoding_runtime import check_gate_stats
from scripts.analyze_trained_timing_evidence_access import audit_readout
from scripts.analyze_gate_components import audit_kernel_checks
from scripts.canonical_necessity_common import audit_affines
from scripts.gate_component_runtime import validation_addendum, VALIDATION_ADDENDUM_SHA256


def aggregate(rows):
    return {'questions':len(rows), 'correct':sum(r['greedy_exact'] for r in rows),
        'answer_nll_mean':sum(r['answer_nll'] for r in rows)/len(rows)}


def paired(left, right):
    pairs = list(zip(left, right))
    if not pairs or len(left) != len(right):
        raise ValueError('Complete paired rows required')
    return {'pairs':len(pairs), 'left':aggregate(left), 'right':aggregate(right),
        'gains':sum(b['greedy_exact'] and not a['greedy_exact'] for a,b in pairs),
        'losses':sum(a['greedy_exact'] and not b['greedy_exact'] for a,b in pairs),
        'generated_changed':sum(a['generated'] != b['generated'] for a,b in pairs),
        'answer_nll_delta_right_minus_left':sum(b['answer_nll']-a['answer_nll'] for a,b in pairs)/len(pairs)}


def attribute(rows,query_map,cases,origin):
    records={}
    for case in cases:
        for episode in case['episodes']:
            events=case['prefix_events']+episode['tail']
            if len({e['value'] for e in events})!=len(events):raise ValueError('Unique recorded values required')
            records[(case['id'],episode['id'])]=(case,events,{e['value']:i for i,e in enumerate(events)})
    result=[]
    for model in sorted(rows):
        values=rows[model]
        for row in values:
            q=query_map[row['id']];case,events,index=records[(q['case_id'],q['episode'])]
            match=re.fullmatch(r'\s*(-?\d+)\s*',row['generated']);number=int(match.group(1)) if match else None
            hit=index.get(number);event=None if hit is None else events[hit]
            entity=case['protected'] if q['role']=='protected_current' else case['target']
            if row['greedy_exact']:category='correct'
            elif match is None:category='non_integer_output'
            elif number==int(q['answer']):category='same_integer_noncanonical_text'
            elif event is None:category='unrecorded_integer'
            elif event['target']==entity:category='queried_account_other_version'
            else:category='other_account_recorded_value'
            source=None
            if event:
                if event['target']==case['target']:
                    first=next(i for i,e in enumerate(events) if e['target']==case['target'])
                    source='target_initial' if hit==first else 'target_pre_tail_current' if hit==18 else 'target_new_current'
                elif event['target']==case['protected']:source='protected_record'
                else:source='other_prefix_record' if hit<19 else 'tail_distractor_record'
            result.append({'id':q['id'],'origin':origin,'model':model,'question_style':q['question_style'],
                'seed':q['seed'],'naming':q['naming'],'episode':q['episode'],'role':q['role'],
                'greedy_exact':row['greedy_exact'],'generated':row['generated'],'category':category,
                'recorded_value_source':source,'matched_event_index':hit,
                'matched_account':event['target'] if event else None,'queried_account':entity})
    return result


def summarize(all_rows,queries,refs,cases):
    cells=defaultdict(list);roles=defaultdict(list);episodes=defaultdict(list);pairs=[]
    archived=refs['archived_full'];old_queries=refs['archived_queries']
    for origin,rows,qmap in (('new',all_rows,queries),('archived',archived,old_queries)):
        for model in sorted(rows):
            values=rows[model]
            for row in values:
                q=qmap[row['id']];key=(origin,model,q['question_style'],q['episode'])
                cells[key].append(row);roles[(*key,q['role'])].append(row)
                episodes[(*key,q['seed'],q['naming'])].append(row)
    for model in sorted(all_rows):
        values=all_rows[model]
        indexed={r['id']:r for r in values}
        for left in values:
            q=queries[left['id']]
            if q['question_style']!='state_term':continue
            right=indexed[q['paired_query_id']];other=queries[right['id']]
            if other['question_style']!='event_term' or other['answer']!=q['answer']:
                raise ValueError('Only fixed-truth state-to-event pairs permitted')
            pairs.append({'model':model,'seed':q['seed'],'naming':q['naming'],'episode':q['episode'],'role':q['role'],
                'state_id':left['id'],'event_id':right['id'],'different_question_token_index':q['different_question_token_index'],
                **paired([left],[right])})
    combined=defaultdict(list);by_seed=defaultdict(list)
    for item in pairs:
        key=(item['model'],item['episode'],item['role']);combined[key].append(item);by_seed[(*key,item['seed'])].append(item)
    def reductions(groups,fields):
        return [{**dict(zip(fields,key)),'pairs':len(values),'state_correct':sum(r['left']['correct'] for r in values),
            'event_correct':sum(r['right']['correct'] for r in values),'gains':sum(r['gains'] for r in values),
            'losses':sum(r['losses'] for r in values),'generated_changed':sum(r['generated_changed'] for r in values),
            'answer_nll_delta_event_minus_state':sum(r['answer_nll_delta_right_minus_left'] for r in values)/len(values)}
            for key,values in sorted(groups.items())]
    attribution=attribute(all_rows,queries,cases,'new')+attribute(archived,old_queries,cases,'archived')
    original=json.loads(Path(read_plan()['source_version_attribution']).read_text())
    expected={(r['model'],r['id']):r for r in original['rows'] if r['reader']=='full'}
    for row in attribution:
        if row['origin']!='archived':continue
        old=expected[(row['model'],row['id'])]
        if any(row[k]!=old[k] for k in ('category','recorded_value_source','matched_event_index','matched_account','queried_account','generated','greedy_exact')):
            raise ValueError('Preserve independent attribution of every archived baseline error')
    categories=defaultdict(lambda:defaultdict(int))
    for r in attribution:categories[(r['origin'],r['model'],r['question_style'])][r['category']]+=1
    return {'cells':[{'origin':o,'model':m,'question_style':s,'episode':e,**aggregate(v)} for (o,m,s,e),v in sorted(cells.items())],
        'role_cells':[{'origin':o,'model':m,'question_style':s,'episode':e,'role':r,**aggregate(v)} for (o,m,s,e,r),v in sorted(roles.items())],
        'episodes':[{'origin':o,'model':m,'question_style':s,'episode':e,'seed':seed,'naming':name,**aggregate(v)}
            for (o,m,s,e,seed,name),v in sorted(episodes.items())],
        'per_stream_state_to_event_pairs':pairs,
        'paired_contrasts':reductions(combined,('model','episode','role')),
        'per_seed_paired_contrasts':reductions(by_seed,('model','episode','role','seed')),
        'value_attribution_all_readouts':attribution,
        'value_attribution_totals':[{'origin':o,'model':m,'question_style':s,'counts':dict(v)} for (o,m,s),v in sorted(categories.items())],
        'independent_streams_new':len({queries[r['id']]['seed'] for v in all_rows.values() for r in v}),
        'independent_streams_archived':8,
        'interpretation':'Complete reused-cohort descriptive token-matched question control.16case/name pairs per new role/episode/model in primary; names are repeated streams. Old versus new syntax is not token matched. No new update method,unique retrieval mechanism,significance or architecture claim.'}


def audit_run(path, allow_smoke=False):
    path = Path(path); m = json.loads((path/'manifest.json').read_text()); plan = read_plan()
    registered, cases, registry, queries, refs = registered_inputs()
    order = list(MODELS[::-1] if m['reverse_model_order'] else MODELS)
    weights = m.get('weights_after', {})
    if (m['status'] != 'completed' or (m['smoke'] and not allow_smoke) or m['training'] or m['new_adapters'] or m['fresh_streams'] or not m['reused_cohort'] or
            m['runner_reuse'] != RUNNER_REUSE or
            (m['reverse_model_order'] and not m['smoke']) or m['model_order'] != order or m['completed_models'] != order or
            list(m['models']) != order or m['packet'] != str(DESTINATION) or
            m['packet_manifest_sha256'] != streaming_hash(DESTINATION/'manifest.json') or
            m['protocol_sha256'] != streaming_hash(PROTOCOL) or m['source_sha256'] != registered['source_sha256'] or
            set(m['source_sha256']) != sources() or len(sources()) != 246 or
            m['model_path'] != plan['model_path'] or m['packages'] != plan['packages'] or
            m['actual_eos_token_id'] != refs['runtime']['actual_eos_token_id'] or
            m['generation_config_sha256'] != refs['runtime']['generation_config_sha256'] or
            m['generation_config_sha256'] != {p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')} or
            m['generation'] != plan['generation'] or m['arithmetic_profile'] != plan['arithmetic_profile'] or
            m['precision_addendum_sha256'] != streaming_hash(Path(plan['precision_addendum'])) or
            m['validation_addendum_sha256'] != VALIDATION_ADDENDUM_SHA256 or
            json.loads((path/'plan.json').read_text()) != plan or
            json.loads((path/'precision_addendum.json').read_text()) != json.loads(Path(plan['precision_addendum']).read_text()) or
            json.loads((path/'kernel_validation_addendum.json').read_text()) != validation_addendum() or
            streaming_hash(path/'installed_modeling_source.py') != plan['installed_modeling_source_sha256'] or
            streaming_hash(Path(plan['installed_modeling_source'])) != plan['installed_modeling_source_sha256'] or
            weights != m['weights_before'] or weights['sha256'] != plan['native_loaded_weights_sha256'] or
            not weights['all_parameters_frozen'] or not weights['no_parameter_gradients'] or not m['hooks_restored'] or
            not datetime.fromisoformat(plan['registered_utc']) < datetime.fromisoformat(registered['created_utc']) <
                datetime.fromisoformat(m['started_utc']) < datetime.fromisoformat(m['completed_utc'])):
        raise ValueError('Completed frozen246source runtime and chronological gates required')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name) != digest or streaming_hash(path/'sources'/name) != digest:
            raise ValueError('Measured source bytes changed')
    audit_kernel_checks(m['native_component_kernel_checks'])
    if m['smoke']:
        if m['smoke_equivalence_gate'] is not None or m['primary_same_subset_replay'] is not None:
            raise ValueError('Smoke cannot have a primary gate')
    else:
        from scripts.run_version_query import checked_gate
        gate = m['smoke_equivalence_gate']
        if checked_gate(Path(gate['path']),m['packet_manifest_sha256']) != gate:
            raise ValueError('Complete opposite-order smoke gate changed')
        eq = json.loads(Path(gate['path']).read_text())
        if eq['source_sha256'] != m['source_sha256'] or any(datetime.fromisoformat(t) >= datetime.fromisoformat(m['started_utc']) for t in eq['smoke_completed_utc']):
            raise ValueError('Exact same-source smoke equivalence must precede primary inference')
        if m['primary_same_subset_replay'] != primary_subset_replay(path,m,gate):
            raise ValueError('Actual primary/smoke subset must replay exactly')
    controls = refs['legacy_ids']; anchors = {v['id']:v['readout'] for v in m['native_anchors']}
    if [v['id'] for v in m['native_anchors']] != controls or anchors != refs['legacy_full']['native']:
        raise ValueError('All12unchanged native anchors required')
    contexts = selected_contexts(registry,m['smoke'])
    all_rows = {}; prefix_count = 0; after_count = 0
    for name in order:
        state = m['models'][name]; spec = plan['models'][name]
        if (state['token_policy'] != ('all' if spec else 'none') or not state['hooks_restored'] or
            state['runner_bindings'] != RUNNER_REUSE or state['predecessor_sources'] != verify_model_sources(state,refs['source_caches'][name])):
            raise ValueError('Frozen token policy or restored hooks changed')
        for label in ('native_before_attach_checks','attached_none_checks','legacy_full_checks','native_restore_checks'):
            ids = [] if label == 'attached_none_checks' and spec is None else controls
            reference = refs['legacy_full'][name] if label == 'legacy_full_checks' else anchors
            if [v['id'] for v in state[label]] != ids or any(v['readout'] != reference[v['id']] for v in state[label]):
                raise ValueError('All96legacy operational replays must be exact')
        if spec:
            if (state['description'] != refs['model_states'][name]['description'] or state['adapter_parameters'] != spec['parameters'] or
                    state['checkpoint_sha256'] != spec['checkpoint_sha256'] or streaming_hash(Path(spec['checkpoint'])) != spec['checkpoint_sha256'] or
                    state['adapter_state_sha256_before'] != refs['model_states'][name]['adapter_state_sha256_before'] or
                    state['adapter_state_sha256_after'] != state['adapter_state_sha256_before'] or
                    not state['all_adapter_parameters_frozen'] or not state['arithmetic_hooks_removed']):
                raise ValueError('Frozen adapter bytes or arithmetic hooks changed')
            audit_affines(state,plan['arithmetic_profile'])
        before = [c for c in contexts if c['episode'] == 'before']; after = [c for c in contexts if c['episode'] != 'before']
        if list(state['prefixes']) != [c['prefix_key'] for c in before] or list(state['sources']) != [c['id'] for c in after]:
            raise ValueError('Every normal common prefix and tail cache must be retained')
        for c in before:
            audit_cache(state['prefixes'][c['prefix_key']],c['prefix_tokens'],64); prefix_count += 1
        for c in after:
            source = state['sources'][c['id']]; parent = state['prefixes'][c['prefix_key']]
            if not source['all_sources_built_before_any_question'] or not source['pairwise_storage_isolated']:
                raise ValueError('Answer-independent isolated normal sources required')
            audit_cache(source['cache'],c['ledger_tokens'],8)
            proof = source['tail_proof']; check_gate_stats(proof['gates'],'none',c['tail_tokens'])
            kv = {k:t['sha256'] for k,t in parent['tensors'].items() if "'keys'" in k or "'values'" in k}
            if (proof['tail_tokens'] != c['tail_tokens'] or not proof['isolated_from_prefix'] or
                    proof['historical_kv_sha256'] != kv or proof['both_mode_exact_recurrent_keys'] != [] or
                    proof['write_keep_sha256'] != data_fingerprint('1'*c['prefix_tokens']+'0'*c['tail_tokens'])):
                raise ValueError('Normal unmodified tail and exact historical12KV bytes required')
            after_count += 1
        reference_path = path/f'{name}.references.jsonl'
        if state['archived_references'] != 864 or streaming_hash(reference_path) != state['references_sha256']:
            raise ValueError('All864archived Full references per model required')
        archived = [json.loads(l) for l in reference_path.read_text().splitlines()]
        if len(archived) != 864:
            raise ValueError('Retain every predecessor Full baseline error')
        for row,old in zip(archived,refs['archived_full'][name]):
            annotated = {'archived_reference':True,'predecessor_source_run':plan['source_run'],
                'predecessor_manifest_sha256':plan['source_manifest_sha256']}
            if row != {**old,**annotated}:
                raise ValueError('Explicit archived provenance or original readout changed')
            audit_readout(row,refs['archived_queries'][row['id']])
        raw = path/f'{name}.jsonl'
        if streaming_hash(raw) != state['results_sha256']:
            raise ValueError('Raw result bytes changed')
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        expected = [(q['id'],reader) for c in contexts for q in context_queries(c,queries,m['smoke']) for reader in READERS]
        if [(r['id'],r['reader']) for r in rows] != expected or state['completed_primary'] != len(expected):
            raise ValueError('Complete unfiltered ordered query/reader grid required')
        for row in rows:
            q = queries[row['id']]
            audit_readout(row,{**q,'question_tokens':len(q['input_ids'])-q['ledger_tokens']})
            if (row['model'] != name or row['registration_sha256'] != data_fingerprint(q) or
                    row['source_key'] != q['source_key'] or row['prompt_ids_sha256'] != q['prompt_ids_sha256'] or
                    row['prefill'] != 'cached' or row['measurement_origin'] != 'normally_encoded_without_intervention'):
                raise ValueError('Actual registered normal readout identity changed')
            audit_reader(row['proof'],q,row['reader'])
        all_rows[name] = rows
    total = sum(map(len,all_rows.values()))
    if total != (108 if m['smoke'] else 1728) or prefix_count != (2 if m['smoke'] else 32) or after_count != (16 if m['smoke'] else 256):
        raise ValueError('Complete registered readout/cache count required')
    metrics=summarize(all_rows,queries,refs,cases)
    if (len(metrics['cells'])!=72 or len(metrics['role_cells'])!=216 or
        len(metrics['per_stream_state_to_event_pairs'])!=(54 if m['smoke'] else 864) or
        len(metrics['value_attribution_all_readouts'])!=total+1728 or
        any(c['questions']!=(3 if m['smoke'] else 48) for c in metrics['cells'] if c['origin']=='new') or
        any(c['questions']!=48 for c in metrics['cells'] if c['origin']=='archived')):
        raise ValueError('All labelled72cells,216role cells and state-to-event pairs required')
    return {'status':'audited_version_query_smoke' if m['smoke'] else 'completed_and_audited_version_query',
        'source_run':str(path),'manifest_sha256':streaming_hash(path/'manifest.json'),
        'packet_manifest_sha256':m['packet_manifest_sha256'],'source_files_verified':246,
        'primary_readouts_audited':total,'archived_full_references_audited':1728,'prefix_caches_audited':prefix_count,'after_caches_audited':after_count,
        'legacy_readouts_audited':96,'smoke':m['smoke'],'prior_failed_criteria_unchanged':refs['prior_failed_criteria'],
        'fresh_confirmation':False,'metrics':metrics,
        'analysis_source_sha256':streaming_hash(Path(__file__))}


def audit_equivalence(forward, reverse):
    f = audit_run(forward,allow_smoke=True); r = audit_run(reverse,allow_smoke=True)
    fm = json.loads((forward/'manifest.json').read_text()); rm = json.loads((reverse/'manifest.json').read_text())
    if (not f['smoke'] or not r['smoke'] or fm['model_order'] != list(MODELS) or rm['model_order'] != list(MODELS[::-1]) or
            fm['packet_manifest_sha256'] != rm['packet_manifest_sha256'] or fm['source_sha256'] != rm['source_sha256'] or
            fm['weights_before'] != rm['weights_before'] or fm['native_anchors'] != rm['native_anchors'] or
            fm['native_component_kernel_checks'] != rm['native_component_kernel_checks'] or f['metrics'] != r['metrics']):
        raise ValueError('Opposite-order operational inputs and complete outcomes must match exactly')
    for name in MODELS:
        if fm['models'][name] != rm['models'][name] or (forward/f'{name}.jsonl').read_bytes() != (reverse/f'{name}.jsonl').read_bytes():
            raise ValueError('Every raw readout,48cache fields,logits,tail proof,legacy/adapter state must match exactly')
        if (forward/f'{name}.references.jsonl').read_bytes() != (reverse/f'{name}.references.jsonl').read_bytes():
            raise ValueError('All labelled archived references must match exactly')
    return {'status':'audited_version_query_smoke_equivalence','forward_run':str(forward),'reverse_run':str(reverse),
        'forward_manifest_sha256':f['manifest_sha256'],'reverse_manifest_sha256':r['manifest_sha256'],
        'packet_manifest_sha256':fm['packet_manifest_sha256'],'source_sha256':fm['source_sha256'],
        'smoke_completed_utc':[fm['completed_utc'],rm['completed_utc']],
        'readouts_per_smoke':108,'raw_files_exact':4,'accuracy_gate':False,
        'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path)
    p.add_argument('--allow-smoke',action='store_true'); p.add_argument('--forward',type=Path); p.add_argument('--reverse',type=Path)
    p.add_argument('--output',type=Path,required=True); args = p.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite an independent audit')
    if args.run is not None and args.forward is None and args.reverse is None:
        value = audit_run(args.run,args.allow_smoke)
    elif args.run is None and args.forward is not None and args.reverse is not None:
        value = audit_equivalence(args.forward,args.reverse)
    else:
        raise ValueError('Choose one complete-run audit or opposite-order equivalence')
    write_json(args.output,value); args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:value[k] for k in ('status','primary_readouts_audited','source_files_verified') if k in value}),flush=True)


if __name__ == '__main__':
    main()
