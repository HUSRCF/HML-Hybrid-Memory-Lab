"""Independent exact coverage, matrix/gate, baseline and equal-group audit."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.question_state_freeze_data import (PROTOCOL,DESTINATION,MODEL,ARMS,read_plan,
    registered_inputs,sources,selected_contexts,context_queries,verified_seal)
from scripts.question_state_freeze_runtime import canonical,full_attention_contract
from scripts.run_question_state_freeze import source_replay,primary_subset_replay
from scripts.analyze_trained_timing_evidence_access import audit_readout
from scripts.analyze_post_encoding import audit_cache
from scripts.post_encoding_runtime import check_gate_stats
from scripts.analyze_gate_components import audit_kernel_checks
from scripts.canonical_necessity_common import audit_affines
from scripts.gate_component_runtime import validation_addendum,VALIDATION_ADDENDUM_SHA256,GDN_LAYERS
from scripts.copy_readout_runtime import readout
from scripts.run_controller_trial import write_json
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash


def audit_question_proof(proof,q,arm,source):
    full_attention_contract(proof['attention'])
    if proof['reader_calls'] is not None or proof['reader_keep_sha256']!=data_fingerprint(q['history_keep_bits']):
        raise ValueError('Every arm must use unrestricted Full reading')
    p=proof['question_hold'];selected=ARMS[arm]
    if (p['question_calls']!=1 or p['boundary']!=q['ledger_tokens'] or p['question_tokens']!=24 or
        p['known_ids_sha256']!=data_fingerprint(q['input_ids'][q['ledger_tokens']:]) or
        p['selected_layers']!=selected or any(not p[k] for k in ('hooks_removed_before_answer',
            'kernel_methods_restored_before_answer','source_and_work_storage_isolated',
            'no_direct_state_copy_or_rollback','policy_uses_no_answers_or_support_labels')) or
        p['gates']['enabled']!=bool(selected) or not p['gates']['actual_gates_finite_and_exact'] or
        set(p['gates']['layers'])!={str(i) for i in GDN_LAYERS}):
        raise ValueError('Known-question-only gate scope or cleanup proof changed')
    for key,stat in p['gates']['layers'].items():
        held=24 if int(key) in selected else 0
        if stat!={'chunk_calls':1,'recurrent_calls':0,'rows':24,'held_rows':held,'unheld_rows':24-held}:
            raise ValueError('All18actual gates must match exact selected question rows')
    if set(p['selected_before'])!={str(i) for i in selected} or set(p['selected_after'])!=set(p['selected_before']):
        raise ValueError('Every selected recurrent matrix requires before/after bytes')
    for index in selected:
        key=str(index);original=source['tensors'][str((index,'recurrent_states',0))]
        before=p['selected_before'][key];after=p['selected_after'][key]
        if any(before[k]!=original[k] or after[k]!=before[k] for k in ('sha256','shape','dtype')):
            raise ValueError('Selected actual question matrix differs from its own normal history')
        for record in (before,after):
            if (len(record['stride'])!=len(record['shape']) or record['storage_offset']<0 or
                any(type(v) is not int or v<0 for v in record['stride'])):
                raise ValueError('Actual selected tensor layout missing')
    kv={k:v['sha256'] for k,v in source['tensors'].items() if "'keys'" in k or "'values'" in k}
    if p['historical_kv_sha256']!=kv or len(kv)!=12:
        raise ValueError('Exact actual12historical KV slices required')
    changed=p['convolution_changed_layers']
    if len(set(changed))!=len(changed) or not set(changed)<=set(GDN_LAYERS):
        raise ValueError('Native convolution observation inventory changed')


def aggregate(rows):
    return {'questions':len(rows),'correct':sum(r['greedy_exact'] for r in rows),
        'answer_nll_mean':sum(r['answer_nll'] for r in rows)/len(rows)}


def summarize(rows,queries):
    cells=defaultdict(list);roles=defaultdict(list);episodes=defaultdict(list);contrasts=[];wording=[]
    for arm in ARMS:
        for row in rows[arm]:
            q=queries[row['id']];base=(arm,q['question_style'],q['episode'])
            cells[base].append(row);roles[(*base,q['role'])].append(row)
            episodes[(*base,q['seed'],q['naming'])].append(row)
    indexed={arm:{r['id']:r for r in values} for arm,values in rows.items()}
    for row in rows['normal']:
        q=queries[row['id']]
        for arm in list(ARMS)[1:]:
            other=indexed[arm][q['id']]
            contrasts.append({'arm':arm,'id':q['id'],**{k:q[k] for k in ('seed','naming','question_style','episode','role')},
                'normal_correct':row['greedy_exact'],'intervention_correct':other['greedy_exact'],
                'gain':int(not row['greedy_exact'] and other['greedy_exact']),
                'loss':int(row['greedy_exact'] and not other['greedy_exact']),
                'generated_changed':row['generated']!=other['generated'],
                'normal_generated':row['generated'],'intervention_generated':other['generated'],
                'answer_nll_delta_intervention_minus_normal':other['answer_nll']-row['answer_nll']})
    for arm in ARMS:
        for row in rows[arm]:
            q=queries[row['id']]
            if q['question_style']!='state_term':continue
            other=indexed[arm][q['paired_query_id']]
            wording.append({'arm':arm,'state_id':q['id'],'event_id':q['paired_query_id'],
                **{k:q[k] for k in ('seed','naming','episode','role')},'state_correct':row['greedy_exact'],
                'event_correct':other['greedy_exact'],'gain':int(not row['greedy_exact'] and other['greedy_exact']),
                'loss':int(row['greedy_exact'] and not other['greedy_exact'])})
    groups=defaultdict(list);by_seed=defaultdict(list)
    for row in contrasts:
        key=tuple(row[k] for k in ('arm','question_style','episode','role'))
        groups[key].append(row);by_seed[(*key,row['seed'])].append(row)
    def reduce(group,fields):
        return [{**dict(zip(fields,key)),'pairs':len(values),
            'normal_correct':sum(r['normal_correct'] for r in values),'intervention_correct':sum(r['intervention_correct'] for r in values),
            'gains':sum(r['gain'] for r in values),'losses':sum(r['loss'] for r in values),
            'generated_changed':sum(r['generated_changed'] for r in values),
            'answer_nll_delta_intervention_minus_normal':sum(r['answer_nll_delta_intervention_minus_normal'] for r in values)/len(values)}
            for key,values in sorted(group.items())]
    joint3=defaultdict(lambda:[0,0]);both=defaultdict(list)
    for (arm,style,episode,seed,name),values in episodes.items():
        if len(values)!=3:raise ValueError('Every same-cache cell must retain all three roles')
        joint3[(arm,style)][0]+=all(v['greedy_exact'] for v in values);joint3[(arm,style)][1]+=1
        both[(arm,episode,seed,name)].extend(values)
    joint6=defaultdict(lambda:[0,0])
    for (arm,_,_,_),values in both.items():
        if len(values)!=6:raise ValueError('Every same-cache condition requires both full wordings')
        joint6[arm][0]+=all(v['greedy_exact'] for v in values);joint6[arm][1]+=1
    return {'cells':[{'arm':a,'question_style':s,'episode':e,**aggregate(v)} for (a,s,e),v in sorted(cells.items())],
        'role_cells':[{'arm':a,'question_style':s,'episode':e,'role':r,**aggregate(v)} for (a,s,e,r),v in sorted(roles.items())],
        'episode_cells':[{'arm':a,'question_style':s,'episode':e,'seed':seed,'naming':name,**aggregate(v)}
            for (a,s,e,seed,name),v in sorted(episodes.items())],
        'normal_to_intervention_pairs':contrasts,'wording_pairs':wording,
        'paired_contrasts':reduce(groups,('arm','question_style','episode','role')),
        'per_seed_contrasts':reduce(by_seed,('arm','question_style','episode','role','seed')),
        'same_cache_joint3':[{'arm':a,'question_style':s,'correct':v[0],'contexts':v[1]} for (a,s),v in sorted(joint3.items())],
        'same_cache_joint6':[{'arm':a,'correct':v[0],'contexts':v[1]} for a,v in sorted(joint6.items())],
        'interpretation':'Exploratory reused-cohort control; all eight arms, both wordings and baseline failures retained.8underlying streams, paired names/history conditions. No architecture,unique source or significance claim.'}


def audit_freeze_contract(checks):
    expected=[('chunk_gated_delta_rule',1),('chunk_gated_delta_rule',7),('chunk_gated_delta_rule',65),
        ('recurrent_gated_delta_rule',1),('recurrent_gated_delta_rule',7),
        ('chunk_gated_delta_rule',24),('recurrent_gated_delta_rule',1)]
    if [(c['kernel'],c['length']) for c in checks]!=expected or any(
        set(c)!={'kernel','length','initial_state_unmodified','final_state_exact','output_finite'} or
        not all(c[k] for k in ('initial_state_unmodified','final_state_exact','output_finite')) for c in checks):
        raise ValueError('Existing and actual24token zero-gate kernel identity required')


def audit_run(path,allow_smoke=False):
    path=Path(path);m=json.loads((path/'manifest.json').read_text());p=read_plan()
    registered,cases,registry,queries,refs=registered_inputs();state=m['model_state']
    seal=verified_seal(registered,m['packet_manifest_sha256']);weights=m['weights_after']
    expected_order=list(ARMS)[::-1] if m['reverse_arm_order'] else list(ARMS)
    if (m['status']!='completed' or (m['smoke'] and not allow_smoke) or
        any(m[k] for k in ('training','new_adapters','fresh_confirmation','oracle_evidence','donor_forward')) or
        (m['reverse_arm_order'] and not m['smoke']) or m['arm_order']!=expected_order or m['model']!=MODEL or
        m['packet']!=str(DESTINATION) or m['packet_manifest_sha256']!=streaming_hash(DESTINATION/'manifest.json') or
        m['protocol_sha256']!=streaming_hash(PROTOCOL) or m['source_sha256']!=registered['source_sha256'] or
        set(m['source_sha256'])!=sources() or len(sources())!=261 or m['registration_seal']!=seal or
        m['model_path']!=p['model_path'] or m['packages']!=p['packages'] or m['generation']!=p['generation'] or
        m['actual_eos_token_id']!=refs['runtime']['actual_eos_token_id'] or
        m['generation_config_sha256']!=refs['runtime']['generation_config_sha256'] or
        m['generation_config_sha256']!={f.name:streaming_hash(f) for f in Path(p['model_path']).glob('generation_config.json')} or
        m['arithmetic_profile']!=p['arithmetic_profile'] or
        m['precision_addendum_sha256']!=streaming_hash(Path(p['precision_addendum'])) or
        m['validation_addendum_sha256']!=VALIDATION_ADDENDUM_SHA256 or
        json.loads((path/'plan.json').read_text())!=p or
        json.loads((path/'precision_addendum.json').read_text())!=json.loads(Path(p['precision_addendum']).read_text()) or
        json.loads((path/'kernel_validation_addendum.json').read_text())!=validation_addendum() or
        streaming_hash(path/'installed_modeling_source.py')!=p['installed_modeling_source_sha256'] or
        streaming_hash(Path(p['installed_modeling_source']))!=p['installed_modeling_source_sha256'] or
        weights!=m['weights_before'] or weights['sha256']!=p['native_loaded_weights_sha256'] or
        not weights['all_parameters_frozen'] or not weights['no_parameter_gradients'] or not m['hooks_restored'] or
        not datetime.fromisoformat(p['registered_utc'])<datetime.fromisoformat(registered['created_utc'])<
            datetime.fromisoformat(seal['created_utc'])<datetime.fromisoformat(m['started_utc'])<datetime.fromisoformat(m['completed_utc'])):
        raise ValueError('Completed261source frozen runtime and exact preregistration required')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Measured source changed')
    audit_kernel_checks(m['native_component_kernel_checks']);audit_freeze_contract(m['native_freeze_kernel_checks'])
    if m['smoke']:
        if m['smoke_equivalence_gate'] is not None or m['primary_same_subset_replay'] is not None:raise ValueError('Unexpected smoke gate')
    else:
        from scripts.run_question_state_freeze import checked_gate
        gate=m['smoke_equivalence_gate']
        if checked_gate(Path(gate['path']),m['packet_manifest_sha256'])!=gate:raise ValueError('Exact smoke gate changed')
        eq=json.loads(Path(gate['path']).read_text())
        if eq['source_sha256']!=m['source_sha256'] or any(datetime.fromisoformat(t)>=datetime.fromisoformat(m['started_utc']) for t in eq['smoke_completed_utc']):
            raise ValueError('Exact same-source smoke equivalence must precede primary')
        if m['primary_same_subset_replay']!=primary_subset_replay(path,m,gate):raise ValueError('Shared actual432readouts differ')
    controls=refs['legacy_ids'];anchors={r['id']:r['readout'] for r in m['native_anchors']}
    if [r['id'] for r in m['native_anchors']]!=controls or anchors!=refs['legacy_full']['native']:raise ValueError('All12native anchors required')
    for label in ('native_before_attach_checks','attached_none_checks','legacy_full_checks','native_restore_checks'):
        expected=refs['legacy_full'][MODEL] if label=='legacy_full_checks' else anchors
        if [r['id'] for r in state[label]]!=controls or any(r['readout']!=expected[r['id']] for r in state[label]):
            raise ValueError('All48native/adapter operational replays required')
    spec=p['model_spec']
    if (state['token_policy']!='all' or state['description']!=refs['model_state']['description'] or
        state['adapter_parameters']!=spec['parameters'] or state['checkpoint_sha256']!=spec['checkpoint_sha256'] or
        streaming_hash(Path(spec['checkpoint']))!=spec['checkpoint_sha256'] or
        state['adapter_state_sha256_before']!=refs['model_state']['adapter_state_sha256_before'] or
        state['adapter_state_sha256_after']!=state['adapter_state_sha256_before'] or
        not state['all_adapter_parameters_frozen'] or not state['arithmetic_hooks_removed'] or not state['hooks_restored'] or
        state['predecessor_sources']!=source_replay(state,refs['source_caches'])):
        raise ValueError('Frozen existing adapter and normal source bytes required')
    audit_affines(state,p['arithmetic_profile'])
    contexts=selected_contexts(registry,m['smoke']);before=[c for c in contexts if c['episode']=='before'];after=[c for c in contexts if c['episode']!='before']
    if list(state['prefixes'])!=[c['prefix_key'] for c in before] or list(state['sources'])!=[c['id'] for c in after]:raise ValueError('Complete source histories required')
    for c in before:audit_cache(state['prefixes'][c['prefix_key']],c['prefix_tokens'],442)
    for c in after:
        s=state['sources'][c['id']];parent=state['prefixes'][c['prefix_key']]
        audit_cache(s['cache'],c['ledger_tokens'],50);check_gate_stats(s['tail_proof']['gates'],'none',c['tail_tokens'])
        kv={k:v['sha256'] for k,v in parent['tensors'].items() if "'keys'" in k or "'values'" in k}
        if (not s['all_sources_built_before_any_question'] or not s['pairwise_storage_isolated'] or
            s['tail_proof']['historical_kv_sha256']!=kv or not s['tail_proof']['isolated_from_prefix'] or
            s['tail_proof']['tail_tokens']!=c['tail_tokens'] or s['tail_proof']['both_mode_exact_recurrent_keys']!=[] or
            s['tail_proof']['write_keep_sha256']!=data_fingerprint('1'*c['prefix_tokens']+'0'*c['tail_tokens'])):
            raise ValueError('No query-dependent write or historical KV modification allowed')
    reference=path/'references.jsonl';archived=[json.loads(l) for l in reference.read_text().splitlines()]
    if state['archived_references']!=864 or len(archived)!=864 or streaming_hash(reference)!=state['references_sha256']:
        raise ValueError('All864archived predecessor answers and failures required')
    for row,old in zip(archived,refs['normal_full']):
        if row!={**old,'archived_reference':True,'source_run':p['source_run'],'source_manifest_sha256':p['source_manifest_sha256']}:
            raise ValueError('Archived answer or provenance changed')
    rows={};expected_ids=[q['id'] for c in contexts for q in context_queries(c,queries,m['smoke'])]
    normal={r['id']:r for r in refs['normal_full']}
    for arm in ARMS:
        raw=path/f'{arm}.jsonl'
        if streaming_hash(raw)!=state['file_sha256'][raw.name]:raise ValueError('Per-arm raw bytes changed')
        rows[arm]=[json.loads(l) for l in raw.read_text().splitlines()]
        if [r['id'] for r in rows[arm]]!=expected_ids or state['completed_by_arm'][arm]!=len(expected_ids):raise ValueError('Complete unfiltered per-arm grid required')
        if not math.isfinite(state['readout_wall_seconds_with_observers'][arm]) or state['readout_wall_seconds_with_observers'][arm]<0:raise ValueError('Measured readout walltime required')
        for row in rows[arm]:
            q=queries[row['id']];c=next(c for c in contexts if c['id']==q['context_id'])
            source=state['prefixes'][c['prefix_key']] if c['episode']=='before' else state['sources'][c['id']]['cache']
            if (row['model']!=MODEL or row['reader']!='full' or row['arm']!=arm or row['prefill']!='cached' or
                row['measurement_origin']!='normally_encoded_history_question_only_gate_policy' or
                row['registration_sha256']!=data_fingerprint(q) or row['source_key']!=q['source_key'] or
                row['prompt_ids_sha256']!=q['prompt_ids_sha256']):raise ValueError('Registered normal history or arm identity changed')
            audit_readout(row,{**q,'question_tokens':24});audit_question_proof(row['proof'],q,arm,source)
            if arm=='normal' and (readout(row)!=readout(normal[row['id']]) or
                {k:row['proof'][k] for k in ('reader_calls','attention','reader_keep_sha256')}!=normal[row['id']]['proof']):
                raise ValueError('Actual normal baseline must exactly replay all old successes and failures')
    execution=path/'execution.jsonl'
    if streaming_hash(execution)!=state['file_sha256'][execution.name]:raise ValueError('Execution log bytes changed')
    actual=[json.loads(l) for l in execution.read_text().splitlines()]
    indexed={arm:{r['id']:r for r in values} for arm,values in rows.items()}
    expected=[indexed[arm][key] for key in expected_ids for arm in expected_order]
    if actual!=expected or state['completed_primary']!=len(expected) or state['normal_reference_replays']!=len(expected_ids):
        raise ValueError('All arms must execute in registered order with exact normal replay')
    count=432 if m['smoke'] else 6912
    metrics=summarize(rows,queries)
    if (len(expected)!=count or len(metrics['cells'])!=144 or len(metrics['role_cells'])!=432 or
        len(metrics['episode_cells'])!=(144 if m['smoke'] else 2304) or
        len(metrics['normal_to_intervention_pairs'])!=(378 if m['smoke'] else 6048) or
        len(metrics['wording_pairs'])!=(216 if m['smoke'] else 3456) or
        any(c['questions']!=(3 if m['smoke'] else 48) for c in metrics['cells'])):
        raise ValueError('Every registered arm/style/role/episode and paired contrast required')
    return {'status':'audited_question_state_freeze_smoke' if m['smoke'] else 'completed_and_audited_question_state_freeze',
        'source_run':str(path),'manifest_sha256':streaming_hash(path/'manifest.json'),'packet_manifest_sha256':m['packet_manifest_sha256'],
        'primary_readouts_audited':count,'normal_reference_replays_audited':len(expected_ids),'archived_references_audited':864,
        'normal_prefix_caches_audited':len(before),'normal_after_caches_audited':len(after),'legacy_operational_readouts_audited':60,
        'source_files_verified':261,'smoke':m['smoke'],'metrics':metrics,
        'prior_failed_criteria_unchanged':refs['prior_failed_criteria'],'fresh_confirmation':False,
        'analysis_source_sha256':streaming_hash(Path(__file__))}


def audit_equivalence(forward,reverse):
    forward=Path(forward);reverse=Path(reverse);f=audit_run(forward,True);r=audit_run(reverse,True)
    fm=json.loads((forward/'manifest.json').read_text());rm=json.loads((reverse/'manifest.json').read_text())
    if (not f['smoke'] or not r['smoke'] or fm['arm_order']!=list(ARMS) or rm['arm_order']!=list(ARMS)[::-1] or
        fm['source_sha256']!=rm['source_sha256'] or f['metrics']!=r['metrics'] or
        any(fm[k]!=rm[k] for k in ('packet_manifest_sha256','weights_before','weights_after','native_anchors',
            'native_component_kernel_checks','native_freeze_kernel_checks','registration_seal'))):
        raise ValueError('Exact opposite-arm-order inputs and outcomes required')
    # Execution-line order and observed walltimes are intentionally different;
    # the same question order within each arm gives byte-identical per-arm files.
    for arm in ARMS:
        if (forward/f'{arm}.jsonl').read_bytes()!=(reverse/f'{arm}.jsonl').read_bytes():raise ValueError('Per-arm raw outcome/proof bytes depend on arm execution order')
    if (forward/'references.jsonl').read_bytes()!=(reverse/'references.jsonl').read_bytes():raise ValueError('Archived reference bytes differ')
    fs=canonical({k:v for k,v in fm['model_state'].items() if k not in ('file_sha256','readout_wall_seconds_with_observers')})
    rs=canonical({k:v for k,v in rm['model_state'].items() if k not in ('file_sha256','readout_wall_seconds_with_observers')})
    if fs!=rs:raise ValueError('Actual normal caches,gate/adapter controls or bookkeeping depend on arm order')
    return {'status':'audited_question_state_freeze_smoke_equivalence','forward_run':str(forward),'reverse_run':str(reverse),
        'forward_manifest_sha256':f['manifest_sha256'],'reverse_manifest_sha256':r['manifest_sha256'],
        'packet_manifest_sha256':fm['packet_manifest_sha256'],'source_sha256':fm['source_sha256'],
        'smoke_completed_utc':[fm['completed_utc'],rm['completed_utc']],'readouts_per_smoke':432,
        'raw_arm_files_exact':8,'archived_reference_files_exact':1,'accuracy_gate':False,
        'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);p.add_argument('--allow-smoke',action='store_true')
    p.add_argument('--forward',type=Path);p.add_argument('--reverse',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Never overwrite independent audit')
    if args.run and not args.forward and not args.reverse:value=audit_run(args.run,args.allow_smoke)
    elif not args.run and args.forward and args.reverse:value=audit_equivalence(args.forward,args.reverse)
    else:raise ValueError('Choose one run or opposite-order equivalence')
    write_json(args.output,value);args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:value[k] for k in ('status','primary_readouts_audited','source_files_verified') if k in value}),flush=True)


if __name__=='__main__':main()
