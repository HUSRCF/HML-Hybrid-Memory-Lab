"""Independent reaggregation of all raw record masses and normal reference checks."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
import numpy as np
from pathlib import Path
import re
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.version_address_data import (PROTOCOL,DESTINATION,MODELS,LAYERS,POSITIONS,read_plan,
    registered_inputs,sources,selected_contexts,verified_seal)
from scripts.question_state_freeze_runtime import canonical,full_attention_contract
from scripts.run_version_address import replay_sources,subset_replay
from scripts.analyze_post_encoding import audit_cache
from scripts.analyze_trained_timing_evidence_access import audit_readout
from scripts.analyze_gate_components import audit_kernel_checks
from scripts.canonical_necessity_common import audit_affines
from scripts.gate_component_runtime import validation_addendum,VALIDATION_ADDENDUM_SHA256
from scripts.post_encoding_runtime import check_gate_stats
from scripts.copy_readout_runtime import readout
from scripts.fresh_layer_controls import tensor_hash
from scripts.run_controller_trial import write_json
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from sas_memory.tasks import Operation,execute


def verify_observations(array,row,q,context,case):
    """No call to the runner's summary function; recompute from saved raw FP32 arrays."""
    n=len(context['records']);capture=row['capture']
    if (array.shape!=(6,2,n) or array.dtype!=np.float32 or not np.isfinite(array).all() or (array<0).any() or
        capture['question_calls']!=1 or capture['question_tokens']!=24 or capture['positions']!=list(POSITIONS) or
        capture['known_ids_sha256']!=data_fingerprint(q['input_ids'][q['ledger_tokens']:]) or
        any(not capture[k] for k in ('all_own_hooks_removed_before_answers','native_attention_return_not_replaced',
            'no_roles_answers_or_support_in_capture_interface')) or set(capture['layers'])!={str(i) for i in LAYERS}):
        raise ValueError('Complete immutable normal-question observations required')
    events=case['prefix_events']+next(e['tail'] for e in case['episodes'] if e['id']==context['episode'])
    state=execute([Operation(**e) for e in events]);entity=case['protected'] if q['role']=='protected_current' else case['target']
    truth=state['initial' if q['role']=='target_initial' else 'current'][entity]
    indices=[i for i,e in enumerate(events) if e['target']==entity]
    support=min(indices) if q['role']=='target_initial' else max(indices)
    if str(truth)!=q['answer'] or support!=q['support_event_index'] or len({e['value'] for e in events})!=n:
        raise ValueError('Independent executable truth differs')
    parsed=re.fullmatch(r'\s*(-?\d+)\s*',row['generated']);emitted=None
    if parsed:
        emitted=next((i for i,e in enumerate(events) if e['value']==int(parsed[1])),None)
    category=('correct' if parsed and parsed[1]==q['answer'] else 'non_integer' if parsed is None else
        'noncanonical_correct_integer' if int(parsed[1])==int(q['answer']) else
        'same_account_other_version_value' if emitted in indices else
        'other_account_recorded_value' if emitted is not None else 'unrecorded_integer')
    expected=[]
    for li,layer in enumerate(LAYERS):
        aux=capture['layers'][str(layer)]
        if (aux['native_weights_dtype']!='torch.bfloat16' or aux['native_weights_shape']!=[1,8,24,q['ledger_tokens']+24] or
            aux['pooled_record_mass_sha256']!=tensor_hash(torch.from_numpy(array[li]))):
            raise ValueError('Actual complete FP32 observation differs from its native capture')
        for pi,pos in enumerate(POSITIONS):
            probs=array[li,pi];float_values=[float(x) for x in probs];peers=[i for i in indices if i!=support]
            total=float(probs.sum(dtype=np.float32));pre=aux['prelude_mass'][pi];question=aux['question_mass'][pi];whole=aux['total_mass'][pi]
            if any(not math.isfinite(x) or x<0 for x in (total,pre,question,whole)):raise ValueError('Invalid pooled probabilities')
            greater=lambda i:sum(x>float_values[i] for x in float_values)
            best=[i for i,_ in sorted(enumerate(float_values),key=lambda pair:(-pair[1],pair[0]))[:3]]
            expected.append({'layer':layer,'suffix_position':pos,'record_count':n,'entity':entity,
                'entity_record_indices':indices,'support_record_index':support,'support_mass':float_values[support],
                'support_rank_competition':1+greater(support),'support_equal_mass_ties':sum(x==float_values[support] for x in float_values),
                'entity_version_masses':[float_values[i] for i in indices],'entity_mass':float(probs[indices].sum(dtype=np.float32)),
                'other_version_mass':float(probs[peers].sum(dtype=np.float32)) if peers else 0.,
                'top_record_indices':best,'top_record_masses':[float_values[i] for i in best],
                'emitted_record_index':emitted,'emitted_record_mass':None if emitted is None else float_values[emitted],
                'emitted_record_rank_competition':None if emitted is None else 1+greater(emitted),
                'record_mass':total,'prelude_mass':pre,'question_mass':question,'total_mass':whole,
                'fp32_partition_rounding_residual':total+pre+question-whole,'output_category':category,
                'raw_capture_sha256':aux['pooled_record_mass_sha256']})
    if row['address_rows']!=expected:raise ValueError('Saved labels/ranks/record masses differ from independent raw-array reconstruction')


def summarize(rows,queries,registry):
    cells=defaultdict(list);roles=defaultdict(list);groups=defaultdict(list);wording=[];versions=[]
    index={}
    for model,values in rows.items():
        for r in values:
            q=queries[r['id']];base=(model,r['cohort'],q['question_style'],q['episode'])
            cells[base].append(r);roles[(*base,q['role'])].append(r);index[(model,q['context_id'],q['question_style'],q['role'])]=r
            for obs in r['address_rows']:
                groups[(model,r['cohort'],q['question_style'],q['role'],obs['layer'],obs['suffix_position'],r['greedy_exact'])].append(obs)
    for model,values in rows.items():
        by_id={r['id']:r for r in values}
        for row in values:
            q=queries[row['id']]
            if q['question_style']=='state_term':
                other=by_id[q['paired_query_id']]
                wording.append({'model':model,'cohort':row['cohort'],'state_id':q['id'],'event_id':other['id'],
                    'state_correct':row['greedy_exact'],'event_correct':other['greedy_exact'],
                    'gain':int(not row['greedy_exact'] and other['greedy_exact']),
                    'loss':int(row['greedy_exact'] and not other['greedy_exact'])})
            if q['role']=='target_current':
                other=index[(model,q['context_id'],q['question_style'],'target_initial')];iq=queries[other['id']]
                differences=[i for i,(a,b) in enumerate(zip(q['input_ids'],iq['input_ids'])) if a!=b]
                if differences!=[q['ledger_tokens']+5] or len(q['input_ids'])!=len(iq['input_ids']):
                    raise ValueError('Current/initial comparisons must differ in exactly one known semantic token')
                comparison=[]
                for cur,initial in zip(row['address_rows'],other['address_rows']):
                    if cur['entity_record_indices']!=initial['entity_record_indices']:raise ValueError('Same history/entity required')
                    cp=cur['entity_version_masses'][-1]-cur['entity_version_masses'][0]
                    ip=initial['entity_version_masses'][-1]-initial['entity_version_masses'][0]
                    comparison.append({'layer':cur['layer'],'suffix_position':cur['suffix_position'],
                        'current_last_minus_first_mass':cp,'initial_last_minus_first_mass':ip,
                        'preference_shift_current_minus_initial':cp-ip,
                        'pooled_preference_matches_both':cp>0 and ip<0})
                versions.append({'model':model,'cohort':row['cohort'],'current_id':row['id'],'initial_id':other['id'],
                    'seed':q['seed'],'naming':q['naming'],'episode':q['episode'],'question_style':q['question_style'],
                    'current_correct':row['greedy_exact'],'initial_correct':other['greedy_exact'],'pooled_version_comparisons':comparison})
    agg=lambda v:{'questions':len(v),'correct':sum(r['greedy_exact'] for r in v),
        'answer_nll_mean':sum(r['answer_nll'] for r in v)/len(v)}
    stats=[]
    for model in MODELS:
        for cohort in ('reused','fresh'):
            for style in ('state_term','event_term'):
                for role in ('target_current','target_initial','protected_current'):
                    for layer in LAYERS:
                        for pos in POSITIONS:
                            for correct in (False,True):
                                v=groups[(model,cohort,style,role,layer,pos,correct)]
                                stats.append({'model':model,'cohort':cohort,'question_style':style,'role':role,
                                    'layer':layer,'suffix_position':pos,'answer_correct':correct,'observations':len(v),
                                    'support_mass_mean':sum(x['support_mass'] for x in v)/len(v) if v else None,
                                    'entity_mass_mean':sum(x['entity_mass'] for x in v)/len(v) if v else None,
                                    'support_top_rank_count':sum(x['support_rank_competition']==1 for x in v),
                                    'emitted_record_above_support_count':sum(x['emitted_record_mass'] is not None and x['emitted_record_mass']>x['support_mass'] for x in v)})
    return {'cells':[{'model':m,'cohort':co,'question_style':s,'episode':e,**agg(v)} for (m,co,s,e),v in sorted(cells.items())],
        'role_cells':[{'model':m,'cohort':co,'question_style':s,'episode':e,'role':r,**agg(v)} for (m,co,s,e,r),v in sorted(roles.items())],
        'all_correctness_strata_including_empty':stats,'wording_pairs':wording,'same_history_version_pairs':versions,
        'limits':'Pooled probabilities are observations,not causal answer contributions. All heads are averaged; all layers and fixed positions retained. Fresh refers to new generated streams,not new operations/tasks. No new method or architecture gain; all old failed criteria unchanged.'}


def audit_run(path,allow_smoke=False):
    path=Path(path);m=json.loads((path/'manifest.json').read_text());p=read_plan();registered,cases,registry,queries,refs=registered_inputs()
    seal=verified_seal(registered,m['packet_manifest_sha256']);w=m['weights_after'];order=list(MODELS)[::-1] if m['reverse_model_order'] else list(MODELS)
    if (m['status']!='completed' or m['smoke'] and not allow_smoke or m['model_order']!=order or
        m['reverse_model_order'] and not m['smoke'] or any(m[k] for k in ('training','new_adapters','oracle_reading','donor_forward','changes_recurrent_updates')) or
        m['packet']!=str(DESTINATION) or m['packet_manifest_sha256']!=streaming_hash(DESTINATION/'manifest.json') or
        m['protocol_sha256']!=streaming_hash(PROTOCOL) or m['source_sha256']!=registered['source_sha256'] or
        set(m['source_sha256'])!=sources() or len(sources())!=268 or m['registration_seal']!=seal or
        m['model_path']!=p['model_path'] or m['packages']!=p['packages'] or m['generation']!=p['generation'] or
        m['arithmetic_profile']!=p['arithmetic_profile'] or m['serialization_numpy_version']!=np.__version__ or
        m['actual_eos_token_id']!=refs['runtime']['actual_eos_token_id'] or m['generation_config_sha256']!=refs['runtime']['generation_config_sha256'] or
        m['generation_config_sha256']!={f.name:streaming_hash(f) for f in Path(p['model_path']).glob('generation_config.json')} or
        m['precision_addendum_sha256']!=streaming_hash(Path(p['precision_addendum'])) or m['validation_addendum_sha256']!=VALIDATION_ADDENDUM_SHA256 or
        json.loads((path/'plan.json').read_text())!=p or json.loads((path/'precision_addendum.json').read_text())!=json.loads(Path(p['precision_addendum']).read_text()) or
        json.loads((path/'kernel_validation_addendum.json').read_text())!=validation_addendum() or
        streaming_hash(path/'installed_modeling_source.py')!=p['installed_modeling_source_sha256'] or streaming_hash(Path(p['installed_modeling_source']))!=p['installed_modeling_source_sha256'] or
        w!=m['weights_before'] or w['sha256']!=p['native_loaded_weights_sha256'] or not w['all_parameters_frozen'] or not w['no_parameter_gradients'] or not m['hooks_restored'] or
        not datetime.fromisoformat(p['registered_utc'])<datetime.fromisoformat(registered['created_utc'])<datetime.fromisoformat(seal['created_utc'])<datetime.fromisoformat(m['started_utc'])<datetime.fromisoformat(m['completed_utc'])):
        raise ValueError('Complete268source normal observational runtime required')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Measured source changed')
    audit_kernel_checks(m['native_component_kernel_checks'])
    if m['smoke']:
        if m['smoke_equivalence_gate'] is not None or m['primary_same_subset_replay'] is not None:raise ValueError('Unexpected smoke gate')
    else:
        from scripts.run_version_address import checked_gate
        gate=m['smoke_equivalence_gate']
        if gate!=checked_gate(Path(gate['path']),m['packet_manifest_sha256']) or m['primary_same_subset_replay']!=subset_replay(path,m,gate):raise ValueError('Exact smoke gate or shared subset changed')
        eq=json.loads(Path(gate['path']).read_text())
        if eq['source_sha256']!=m['source_sha256'] or any(datetime.fromisoformat(t)>=datetime.fromisoformat(m['started_utc']) for t in eq['smoke_completed_utc']):raise ValueError('Smoke must precede primary')
    controls=refs['legacy_ids'];anchors={r['id']:r['readout'] for r in m['native_anchors']}
    if [r['id'] for r in m['native_anchors']]!=controls or anchors!=refs['legacy_full']['native']:raise ValueError('All12native anchors required')
    contexts=selected_contexts(registry,m['smoke']);cmap={c['id']:c for c in contexts};case_map={c['id']:c for c in cases}
    before=[c for c in contexts if c['episode']=='before'];after=[c for c in contexts if c['episode']!='before'];all_rows={};massfiles=0;replays=0
    for name in MODELS:
        s=m['models'][name];spec=p['models'][name]
        if s['token_policy']!=('all' if spec else 'none'):raise ValueError('Original all-token frozen adapter policy required')
        for field in ('native_before_attach_checks','attached_none_checks','legacy_full_checks','native_restore_checks'):
            expected=[] if field=='attached_none_checks' and spec is None else controls
            values=refs['legacy_full'][name] if field=='legacy_full_checks' else anchors
            if [r['id'] for r in s[field]]!=expected or any(r['readout']!=values[r['id']] for r in s[field]):raise ValueError('All96native/adapter control replays required')
        if not s['hooks_restored'] or s['predecessor_sources']!=replay_sources(s,refs['source_caches'][name]):raise ValueError('Reused normal memory changed')
        if spec:
            if (s['description']!=refs['model_states'][name]['description'] or s['adapter_parameters']!=spec['parameters'] or
                s['checkpoint_sha256']!=spec['checkpoint_sha256'] or streaming_hash(Path(spec['checkpoint']))!=spec['checkpoint_sha256'] or
                s['adapter_state_sha256_before']!=refs['model_states'][name]['adapter_state_sha256_before'] or s['adapter_state_sha256_after']!=s['adapter_state_sha256_before'] or
                not s['all_adapter_parameters_frozen'] or not s['arithmetic_hooks_removed']):raise ValueError('Existing GDN adapter changed')
            audit_affines(s,p['arithmetic_profile'])
        if list(s['prefixes'])!=[c['prefix_key'] for c in before] or list(s['sources'])!=[c['id'] for c in after]:raise ValueError('All normal caches required')
        for c in before:audit_cache(s['prefixes'][c['prefix_key']],c['prefix_tokens'],64)
        for c in after:
            v=s['sources'][c['id']];audit_cache(v['cache'],c['ledger_tokens'],8);proof=v['tail_proof'];check_gate_stats(proof['gates'],'none',c['tail_tokens'])
            kv={k:t['sha256'] for k,t in s['prefixes'][c['prefix_key']]['tensors'].items() if "'keys'" in k or "'values'" in k}
            if (not v['all_sources_built_before_any_question'] or not v['pairwise_storage_isolated'] or proof['historical_kv_sha256']!=kv or
                not proof['isolated_from_prefix'] or proof['tail_tokens']!=c['tail_tokens'] or proof['both_mode_exact_recurrent_keys']!=[] or
                proof['write_keep_sha256']!=data_fingerprint('1'*c['prefix_tokens']+'0'*c['tail_tokens'])):raise ValueError('Normal tail writes and full historical KV required')
        raw=path/(name+'.jsonl');rows=[json.loads(l) for l in raw.read_text().splitlines()]
        expected=[key for c in contexts for key in c['query_ids']]
        if streaming_hash(raw)!=s['results_sha256'] or [r['id'] for r in rows]!=expected or s['completed_primary']!=len(expected):raise ValueError('Complete ordered normal query grid required')
        refs_by_id={r['id']:r for r in refs['normal_full'][name]};actualfiles={};cachefiles={};replayed=0
        for row in rows:
            q=queries[row['id']];c=cmap[q['context_id']];audit_readout(row,{**q,'question_tokens':24});full_attention_contract(row['proof']['attention'])
            if (row['model']!=name or row['reader']!='full' or row['cohort']!=registry['cohort_by_context'][c['id']] or
                row['registration_sha256']!=data_fingerprint(q) or row['source_key']!=q['source_key'] or row['prompt_ids_sha256']!=q['prompt_ids_sha256'] or
                row['prefill']!='cached' or row['measurement_origin']!='normal_history_read_only_record_observation' or
                row['proof']['reader_calls'] is not None or row['proof']['reader_keep_sha256']!=data_fingerprint(q['history_keep_bits'])):raise ValueError('Normal readout identity changed')
            relative=name+'.record-masses/'+data_fingerprint(c['id'])+'.npy';f=path/relative
            if row['record_mass_file']!=relative or row['record_mass_index']!=c['query_ids'].index(row['id']):raise ValueError('Raw array identity differs')
            if relative not in cachefiles:
                if streaming_hash(f)!=s['record_mass_files'][relative]:raise ValueError('Raw FP32 file changed')
                block=np.load(f,allow_pickle=False)
                if block.dtype!=np.float32 or block.shape!=(6,6,2,len(c['records'])):raise ValueError('Complete raw FP32 array shape required')
                cachefiles[relative]=block;actualfiles[relative]=streaming_hash(f)
            if row['record_mass_file_sha256']!=actualfiles[relative]:raise ValueError('Raw file hash differs from row')
            verify_observations(cachefiles[relative][row['record_mass_index']],row,q,c,case_map[q['case_id']])
            if row['cohort']=='reused':
                if readout(row)!=readout(refs_by_id[row['id']]) or row['proof']!=refs_by_id[row['id']]['proof']:raise ValueError('Normal prior answer/proof changed')
                replayed+=1
        if actualfiles!=s['record_mass_files'] or replayed!=s['reused_reference_replays']:raise ValueError('Complete raw files and old replays required')
        archive=path/(name+'.references.jsonl');oldrows=[json.loads(l) for l in archive.read_text().splitlines()]
        if (len(oldrows)!=864 or s['archived_references']!=864 or streaming_hash(archive)!=s['references_sha256'] or
            oldrows!=[{**r,'archived_reference':True,'source_run':p['wording_run'],'source_manifest_sha256':p['wording_manifest_sha256']} for r in refs['normal_full'][name]]):raise ValueError('All old answers and errors must remain archived')
        all_rows[name]=rows;replays+=replayed;massfiles+=len(actualfiles)
    total=sum(len(v) for v in all_rows.values());metrics=summarize(all_rows,queries,registry)
    if (total!=(216 if m['smoke'] else 3456) or replays!=(108 if m['smoke'] else 1728) or massfiles!=(36 if m['smoke'] else 576) or
        len(metrics['cells'])!=72 or len(metrics['role_cells'])!=216 or len(metrics['wording_pairs'])!=total//2 or len(metrics['same_history_version_pairs'])!=total//3):raise ValueError('Full cohort/style/role/layer/position coverage required')
    return {'status':'audited_version_address_smoke' if m['smoke'] else 'completed_and_audited_version_address',
        'source_run':str(path),'manifest_sha256':streaming_hash(path/'manifest.json'),'packet_manifest_sha256':m['packet_manifest_sha256'],
        'physical_readouts_audited':total,'reused_exact_replays_audited':replays,'archived_references_audited':1728,
        'raw_fp32_files_independently_reaggregated':massfiles,'layer_position_rows_verified':total*12,
        'normal_prefix_caches_audited':len(before)*2,'normal_after_caches_audited':len(after)*2,'legacy_operational_readouts':96,
        'source_files_verified':268,'smoke':m['smoke'],'prior_failed_criteria_unchanged':refs['prior_failed_criteria'],
        'new_method_or_causal_attention_claim':False,'metrics':metrics,'analysis_source_sha256':streaming_hash(Path(__file__))}


def audit_equivalence(forward,reverse):
    forward=Path(forward);reverse=Path(reverse);f=audit_run(forward,True);r=audit_run(reverse,True)
    fm=json.loads((forward/'manifest.json').read_text());rm=json.loads((reverse/'manifest.json').read_text())
    if (not f['smoke'] or not r['smoke'] or fm['model_order']!=list(MODELS) or rm['model_order']!=list(MODELS)[::-1] or
        f['metrics']!=r['metrics'] or any(fm[k]!=rm[k] for k in ('packet_manifest_sha256','source_sha256',
            'weights_before','weights_after','native_anchors','native_component_kernel_checks','registration_seal','serialization_numpy_version'))):
        raise ValueError('Exact opposite-order inputs,controls and outcomes required')
    for name in MODELS:
        for suffix in ('.jsonl','.references.jsonl'):
            if (forward/(name+suffix)).read_bytes()!=(reverse/(name+suffix)).read_bytes():raise ValueError('Raw outcome/proof bytes depend on model order')
        for relative,digest in fm['models'][name]['record_mass_files'].items():
            if streaming_hash(forward/relative)!=digest or streaming_hash(reverse/relative)!=digest:raise ValueError('Raw FP32 observations depend on model order')
        fs={k:v for k,v in fm['models'][name].items() if k!='readout_wall_seconds_with_observers'}
        rs={k:v for k,v in rm['models'][name].items() if k!='readout_wall_seconds_with_observers'}
        if canonical(fs)!=canonical(rs):raise ValueError('Normal caches/controls/bookkeeping depend on model order')
    return {'status':'audited_version_address_smoke_equivalence','forward_run':str(forward),'reverse_run':str(reverse),
        'forward_manifest_sha256':f['manifest_sha256'],'reverse_manifest_sha256':r['manifest_sha256'],
        'packet_manifest_sha256':fm['packet_manifest_sha256'],'source_sha256':fm['source_sha256'],
        'smoke_completed_utc':[fm['completed_utc'],rm['completed_utc']],'readouts_per_smoke':216,
        'raw_result_files_exact':2,'archived_reference_files_exact':2,'raw_fp32_files_exact':36,'accuracy_gate':False,
        'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);p.add_argument('--allow-smoke',action='store_true')
    p.add_argument('--forward',type=Path);p.add_argument('--reverse',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite audit')
    if a.run and not a.forward and not a.reverse:value=audit_run(a.run,a.allow_smoke)
    elif not a.run and a.forward and a.reverse:value=audit_equivalence(a.forward,a.reverse)
    else:raise ValueError('Choose one complete run or exact opposite-order equivalence')
    write_json(a.output,value);a.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:value[k] for k in ('status','physical_readouts_audited','source_files_verified') if k in value}),flush=True)


if __name__=='__main__':main()
