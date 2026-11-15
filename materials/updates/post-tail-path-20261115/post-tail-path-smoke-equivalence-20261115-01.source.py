"""Independently verify the complete post-tail memory factorial and replay gates."""
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from sas_memory.training_data import data_fingerprint
from scripts.run_controller_trial import write_json
from scripts.post_tail_path_data import (read_plan,registered_inputs,sources,selected_contexts,context_queries,
    DESTINATION,PROTOCOL,MODELS,CONDITIONS,PRIMARY,ENDPOINTS,READERS)
from scripts.post_tail_path_runtime import memory_identity,expected_tensors,component,COMPONENTS
from scripts.post_tail_path_metrics import summarize
from scripts.analyze_post_encoding import core_cache,audit_cache,audit_reader
from scripts.analyze_trained_timing_evidence_access import audit_readout
from scripts.canonical_necessity_common import audit_affines
from scripts.gate_component_runtime import validation_addendum,VALIDATION_ADDENDUM_SHA256
from scripts.analyze_gate_components import audit_kernel_checks
from scripts.copy_readout_runtime import readout


def audit_composition(record,source_records,label,tokens,checks):
    if (record['logits_sha256'] is not None or record['source_guard_checks']!=checks or len(record['sha256'])!=64 or
        record['metadata']!=source_records['N']['metadata'] or record['metadata']['original_metadata']['sequence_length']!=tokens or
        record['tensors']!=expected_tensors(source_records,label) or set(record['layouts'])!=set(record['tensors'])):
        raise ValueError('Composed cache must have exactly the selected tensor origins, genuine bookkeeping and no borrowed logits')
    for key,layout in record['layouts'].items():
        if (len(layout['stride'])!=len(record['tensors'][key]['shape']) or
            any(type(v) is not int or v<0 for v in [layout['storage_offset'],*layout['stride']])):
            raise ValueError('Complete actual composed tensor layouts required')
    if label in ENDPOINTS and record['sha256']!=source_records[label[0]]['sha256']:raise ValueError('Natural endpoint composition changed cache content')


def component_differences(source_records):
    result={name:{'equal_keys':[],'different_keys':[]} for name in COMPONENTS}
    for key,left in source_records['N']['tensors'].items():
        kind=COMPONENTS[component(ast.literal_eval(key))]
        result[kind]['equal_keys' if left==source_records['H']['tensors'][key] else 'different_keys'].append(key)
    return result


def audit_run(path,allow_smoke=False):
    path=Path(path);m=json.loads((path/'manifest.json').read_text());plan=read_plan()
    registered,cases,registry,queries,endpoints,refs=registered_inputs();correction=validation_addendum()
    order=list(MODELS[::-1] if m['reverse_model_order'] else MODELS);weights=m['weights_after'];profile=json.loads(Path(plan['precision_addendum']).read_text())
    if (m['status']!='completed' or m['training'] or m['new_adapters'] or m['fresh_confirmation'] or
        (m['smoke'] and not allow_smoke) or (m['reverse_model_order'] and not m['smoke']) or
        m['model_order']!=order or m['completed_models']!=order or list(m['models'])!=order or
        m['packet']!=str(DESTINATION) or m['packet_manifest_sha256']!=streaming_hash(DESTINATION/'manifest.json') or
        m['protocol_sha256']!=streaming_hash(PROTOCOL) or m['source_sha256']!=registered['source_sha256'] or set(m['source_sha256'])!=sources() or
        m['model_path']!=plan['model_path'] or m['generation']!=plan['generation'] or m['packages']!=plan['packages'] or
        m['arithmetic_profile']!=plan['arithmetic_profile'] or m['arithmetic_profile']!=profile['profile'] or
        m['actual_eos_token_id']!=refs['runtime']['actual_eos_token_id'] or m['generation_config_sha256']!=refs['runtime']['generation_config_sha256'] or
        m['generation_config_sha256']!={p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')} or
        m['precision_addendum_sha256']!=streaming_hash(Path(plan['precision_addendum'])) or
        m['validation_addendum_sha256']!=VALIDATION_ADDENDUM_SHA256 or json.loads((path/'plan.json').read_text())!=plan or
        json.loads((path/'precision_addendum.json').read_text())!=profile or json.loads((path/'kernel_validation_addendum.json').read_text())!=correction or
        streaming_hash(path/'installed_modeling_source.py')!=refs['runtime']['installed_modeling_source_sha256'] or
        streaming_hash(Path(refs['runtime']['installed_modeling_source']))!=refs['runtime']['installed_modeling_source_sha256'] or
        m['weights_before']!=weights or weights['sha256']!=plan['native_loaded_weights_sha256'] or not weights['all_parameters_frozen'] or
        not weights['no_parameter_gradients'] or not m['hooks_restored'] or
        not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(registered['created_utc'])<
            datetime.fromisoformat(m['started_utc'])<datetime.fromisoformat(m['completed_utc'])):
        raise ValueError('Frozen214source scope, parameters, runtime or chronology changed')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Measured source bytes changed')
    audit_kernel_checks(m['native_component_kernel_checks'])
    if m['smoke']:
        if m['smoke_equivalence_gate'] is not None:raise ValueError('Unexpected smoke gate')
    else:
        from scripts.run_post_tail_path import checked_gate
        gate=m['smoke_equivalence_gate'];eq=json.loads(Path(gate['path']).read_text())
        if (checked_gate(Path(gate['path']),m['packet_manifest_sha256'])!=gate or eq['runtime_source_sha256']!=m['source_sha256'] or
            any(datetime.fromisoformat(t)>=datetime.fromisoformat(m['started_utc']) for t in eq['smoke_completed_utc'])):
            raise ValueError('Both exact operational smokes must precede primary inference')
    controls=refs['legacy_ids'];anchors={v['id']:v['readout'] for v in m['native_anchors']}
    if [v['id'] for v in m['native_anchors']]!=controls or anchors!=refs['legacy_full']['native']:raise ValueError('All original native anchors required')
    contexts=selected_contexts(registry,m['smoke']);per_source=6 if m['smoke'] else 12;all_rows={};source_count=0;memory_count=0;replay_count=0
    for name in order:
        state=m['models'][name];spec=plan['models'][name];old=refs['model_states'][name]
        if state['token_policy']!=('all' if spec else 'none') or not state['hooks_restored']:raise ValueError('Frozen model policy/restoration changed')
        for label in ('native_before_attach_checks','attached_none_checks','legacy_full_checks','native_restore_checks'):
            values=state[label];ids=[] if label=='attached_none_checks' and spec is None else controls
            reference=refs['legacy_full'][name] if label=='legacy_full_checks' else anchors
            if [v['id'] for v in values]!=ids or any(v['readout']!=reference[v['id']] for v in values):raise ValueError('Legacy/native exact replay failed')
        if spec:
            if (state['description']!=old['description'] or state['adapter_parameters']!=spec['parameters'] or
                state['checkpoint_sha256']!=spec['checkpoint_sha256'] or streaming_hash(Path(spec['checkpoint']))!=spec['checkpoint_sha256'] or
                state['adapter_state_sha256_before']!=old['adapter_state_sha256_before'] or
                state['adapter_state_sha256_after']!=state['adapter_state_sha256_before'] or not state['all_adapter_parameters_frozen'] or
                not state['arithmetic_hooks_removed']):raise ValueError('Fixed adapter changed')
            audit_affines(state,plan['arithmetic_profile'])
        expected={kind:[] for kind in ('primary','endpoints')};expected_memories=[];expected_prefixes=list(dict.fromkeys(c['prefix_key'] for c in contexts))
        if list(state['prefixes'])!=expected_prefixes or list(state['sources'])!=[c['id'] for c in contexts]:raise ValueError('All prefixes/N/H source pairs required')
        for context in contexts:
            parent=state['prefixes'][context['prefix_key']]
            audit_cache(parent,context['prefix_tokens'],2+4*(10+8*per_source))
            if core_cache(parent)!=core_cache(refs['prefix_caches'][name][context['prefix_key']]):raise ValueError('Original normal prefix/logits changed')
            sources_record=state['sources'][context['id']];source_records={s:v['cache'] for s,v in sources_record['caches'].items()}
            if set(source_records)!=set('NH') or not sources_record['all_conditions_built_before_queries'] or not sources_record['pairwise_storage_isolated']:
                raise ValueError('Both isolated sources and all question-independent compositions required')
            for label in ENDPOINTS:
                actual=sources_record['caches'][label[0]];previous=refs['source_caches'][name][context['id']][label]
                audit_cache(actual['cache'],context['ledger_tokens'],10+8*per_source)
                if core_cache(actual['cache'])!=core_cache(previous['cache']) or actual['tail_proof']!=previous['tail_proof']:raise ValueError('Published N/H cache, logits or actual gates changed')
                source_count+=1
            if source_records['N']['metadata']!=source_records['H']['metadata'] or sources_record['component_differences']!=component_differences(source_records):
                raise ValueError('Complete source bookkeeping or per-component differences changed')
            prefix_kv={k:v['sha256'] for k,v in parent['tensors'].items() if "'keys'" in k or "'values'" in k}
            for condition in CONDITIONS:
                mid=context['id']+'.'+condition;expected_memories.append(mid);memory=state['memories'][mid]
                audit_composition(memory['cache'],source_records,condition,context['ledger_tokens'],per_source+2)
                if (memory['context_id']!=context['id'] or memory['condition']!=condition or memory['source_key']!=context['source_key'] or
                    memory['memory_identity']!=memory_identity(memory['cache']) or memory['historical_kv_sha256']!=prefix_kv):
                    raise ValueError('Original-prefix KV, complete component identity or context changed')
                for q in context_queries(context,queries,m['smoke']):
                    for reader in READERS:expected['primary' if condition in PRIMARY else 'endpoints'].append((q['id'],condition,reader,mid))
                memory_count+=1
        if list(state['memories'])!=expected_memories:raise ValueError('Every registered composed memory/order required')
        endpoint_index={label:{(v['id'],v['reader']):readout(v) for v in values} for label,values in endpoints[name].items()}
        indexed=[];raws={}
        for kind in ('primary','endpoints'):
            rp=path/f'{name}.{kind}.jsonl'
            if streaming_hash(rp)!=state['results_sha256'][kind]:raise ValueError('Raw bytes changed')
            rows=[json.loads(line) for line in rp.read_text().splitlines()];raws[kind]=rows
            if [(r['id'],r['condition'],r['reader'],r['memory_id']) for r in rows]!=expected[kind]:raise ValueError('Full unfiltered raw coverage/order required')
            for row in rows:
                q=queries[row['id']];audit_readout(readout(row),{**q,'question_tokens':len(q['input_ids'])-q['ledger_tokens']});audit_reader(row['proof'],q,row['reader'])
                if (row['model']!=name or row['prefill']!='cached' or row['registration_sha256']!=data_fingerprint(q) or
                    row['source_key']!=q['source_key'] or row['prompt_ids_sha256']!=q['prompt_ids_sha256'] or
                    row['memory_identity']!=state['memories'][row['memory_id']]['memory_identity'] or
                    row['measurement_origin']!=('new_primary' if kind=='primary' else 'endpoint_operational_replay')):
                    raise ValueError('Question tokens, origin or original source changed')
                if kind=='endpoints' and readout(row)!=endpoint_index[row['condition']][(row['id'],row['reader'])]:raise ValueError('Published endpoint replay drift')
                indexed.append(row)
            replay_count+=len(rows) if kind=='endpoints' else 0
        if state['completed_primary']!=len(raws['primary']) or state['completed_endpoint_replays']!=len(raws['endpoints']):raise ValueError('Terminal counts changed')
        # Recreate execution order across the two raw streams before checking duplicates.
        execution={(r['id'],r['condition'],r['reader']):r for r in indexed};seen={}
        for context in contexts:
            for label in CONDITIONS:
                for q in context_queries(context,queries,m['smoke']):
                    for reader in READERS:
                        row=execution[(q['id'],label,reader)];key=(row['memory_identity'],q['id'],reader)
                        previous=seen.setdefault(key,row)
                        if readout(row)!=readout(previous) or row['equivalent_condition']!=previous['condition']:
                            raise ValueError('Equivalent combinations must be retained and exactly replayed')
        all_rows[name]=raws['primary']
    return {'status':'audited_post_tail_path_smoke' if m['smoke'] else 'completed_and_audited_post_tail_path',
        'source_run':str(path),'manifest_sha256':streaming_hash(path/'manifest.json'),'packet_manifest_sha256':m['packet_manifest_sha256'],
        'new_primary_readouts_audited':sum(map(len,all_rows.values())),'endpoint_operational_replays_audited':replay_count,
        'source_N_H_caches_audited':source_count,'factorial_caches_audited':memory_count,'legacy_operational_readouts':96,
        'source_files_verified':len(sources()),'baseline_failures_excluded':0,'fresh_confirmation':False,
        'prior_failed_criteria_unchanged':refs['prior_failed_criteria'],'original_ideal_kernel_failure_preserved':True,
        'architecture_effectiveness_claim':False,'metrics':None if m['smoke'] else summarize(all_rows,endpoints,queries,plan),
        'analysis_source_sha256':streaming_hash(Path(__file__))}


def audit_equivalence(forward_path,reverse_path):
    paths=[Path(forward_path),Path(reverse_path)];reports=[];manifests=[]
    for p in paths:
        r=json.loads(p.read_text())
        if (r['status']!='audited_post_tail_path_smoke' or r['new_primary_readouts_audited']!=288 or
            r['endpoint_operational_replays_audited']!=96 or audit_run(Path(r['source_run']),True)!=r or
            streaming_hash(p.with_suffix('.source.py'))!=r['analysis_source_sha256']):raise ValueError('Complete independently audited288smokes required')
        reports.append(r);manifests.append(json.loads((Path(r['source_run'])/'manifest.json').read_text()))
    fm,rm=manifests
    if (fm['model_order']!=list(MODELS) or rm['model_order']!=list(MODELS[::-1]) or
        any(fm[k]!=rm[k] for k in ('native_anchors','native_component_kernel_checks','source_sha256','packet_manifest_sha256','weights_before','weights_after'))):
        raise ValueError('Opposite model order or native runtime differs')
    for name in MODELS:
        if fm['models'][name]!=rm['models'][name]:raise ValueError('Opposite-order sources, compositions, parameters or restores differ')
        for kind in ('primary','endpoints'):
            if (Path(reports[0]['source_run'])/f'{name}.{kind}.jsonl').read_bytes()!=(Path(reports[1]['source_run'])/f'{name}.{kind}.jsonl').read_bytes():
                raise ValueError('Opposite-order complete raw bytes differ')
    return {'status':'audited_post_tail_path_smoke_equivalence','forward_audit':str(paths[0]),'reverse_audit':str(paths[1]),
        'forward_audit_sha256':streaming_hash(paths[0]),'reverse_audit_sha256':streaming_hash(paths[1]),
        'packet_manifest_sha256':reports[0]['packet_manifest_sha256'],'runtime_source_sha256':fm['source_sha256'],
        'smoke_completed_utc':[m['completed_utc'] for m in manifests],'exact_new_primary_readouts':288,'exact_endpoint_replays':96,
        'accuracy_gate':False,'primary_evaluated':False,'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',nargs='?',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-smoke',action='store_true');p.add_argument('--equivalence',nargs=2,type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite an audit receipt')
    report=audit_equivalence(*a.equivalence) if a.equivalence else audit_run(a.run,a.allow_smoke)
    write_json(a.output,report);a.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','runtime_source_sha256')}),flush=True)

if __name__=='__main__':main()
