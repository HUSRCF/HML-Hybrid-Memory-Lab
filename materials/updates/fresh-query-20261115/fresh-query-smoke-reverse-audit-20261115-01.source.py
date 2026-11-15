"""Independently audit fresh Query donors, exact controls and all3264readouts."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from sas_memory.training_data import data_fingerprint
from scripts.run_controller_trial import write_json
from scripts.fresh_query_data import (read_plan,registered_inputs,sources,selected_contexts,context_queries,
    DESTINATION,PROTOCOL,MODELS,READERS,ORIGINS,CONDITIONS)
from scripts.fresh_query_metrics import summarize
from scripts.post_tail_path_runtime import memory_identity
from scripts.post_encoding_runtime import ATTENTION_LAYERS,GDN_LAYERS,check_gate_stats
from scripts.analyze_post_encoding import audit_cache,audit_reader
from scripts.analyze_trained_timing_evidence_access import audit_readout
from scripts.canonical_necessity_common import audit_affines
from scripts.gate_component_runtime import validation_addendum,VALIDATION_ADDENDUM_SHA256
from scripts.analyze_gate_components import audit_kernel_checks
from scripts.copy_readout_runtime import readout


def core_bank(record):return {k:v for k,v in record.items() if k!='guard_checks'}


def audit_tensor(record,rows):
    if (len(record['sha256'])!=64 or len(record['shape'])!=4 or record['shape'][0]!=1 or record['shape'][-2]!=rows or
        not all(type(v) is int and v>0 for v in record['shape']) or not record['finite'] or
        not record['dtype'] or not record['device'] or len(record['stride'])!=4 or
        any(type(v) is not int or v<0 for v in [record['storage_offset'],*record['stride']])):
        raise ValueError('Complete finite actual four-dimensional Query descriptor required')


def audit_exchange(proof,q,donor,mode,reader_proof):
    count=len(q['input_ids'])-q['ledger_tokens'];layers={str(i) for i in ATTENTION_LAYERS}
    if (proof['mode']!=mode or proof['boundary']!=q['ledger_tokens'] or proof['question_tokens']!=count or
        proof['question_ids_sha256']!=data_fingerprint(q['input_ids'][q['ledger_tokens']:]) or
        not proof['non_query_arguments_preserved'] or not proof['decode_queries_unmodified'] or
        set(proof['layers'])!=layers or (mode=='capture' and proof['model_calls']!=1)):
        raise ValueError('Question-only known-input exchange scope changed')
    for layer,stat in proof['layers'].items():
        if (stat['question_calls']!=1 or stat['question_rows']!=count or stat['eager_calls']!=proof['model_calls'] or
            stat['eager_calls']!=reader_proof['attention'][layer]['cached_calls'] or
            stat['passthrough_calls']!=stat['eager_calls']-1 or stat['non_query_tensor_checks']<2*stat['eager_calls'] or
            stat['native_query_checks']!=stat['eager_calls']):raise ValueError('Actual native layer/call/argument preservation coverage changed')
        audit_tensor(stat['native_query'],count)
        if mode=='capture':
            if stat['native_query']!=donor['tensors'][layer] or stat.get('actual_replacement') or 'used_query' in stat:
                raise ValueError('Capture must preserve and record the actual native Query')
        else:
            audit_tensor(stat['used_query'],count)
            if stat['used_query']!=donor['tensors'][layer] or not stat['actual_replacement']:
                raise ValueError('Replacement must use exactly the reader-specific actual donor tensor')


def audit_restoration(held,normal,prefix,tokens,checks):
    expected={key:(prefix if "'recurrent_states'" in key else normal)['tensors'][key] for key in normal['tensors']}
    if (held['logits_sha256'] is not None or held['source_guard_checks']!=checks or len(held['sha256'])!=64 or
        held['metadata']!=normal['metadata'] or held['metadata']['original_metadata']['sequence_length']!=tokens or
        held['tensors']!=expected or set(held['layouts'])!=set(expected)):
        raise ValueError('Restore only18recurrent matrices; preserve normal after-tail bookkeeping and no borrowed logits')
    for key,layout in held['layouts'].items():
        if len(layout['stride'])!=len(held['tensors'][key]['shape']) or any(type(v) is not int or v<0 for v in [layout['storage_offset'],*layout['stride']]):
            raise ValueError('Actual restored-cache tensor layout missing')
        if "'recurrent_states'" not in key and layout!=normal['layouts'][key]:raise ValueError('Normal non-recurrent layout changed')


def audit_run(path,allow_smoke=False):
    path=Path(path);m=json.loads((path/'manifest.json').read_text());plan=read_plan()
    registered,cases,registry,queries,refs=registered_inputs();correction=validation_addendum()
    order=list(MODELS[::-1] if m['reverse_model_order'] else MODELS);weights=m['weights_after'];profile=json.loads(Path(plan['precision_addendum']).read_text())
    if (m['status']!='completed' or m['training'] or m['new_adapters'] or not m['fresh_streams'] or
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
        raise ValueError('Frozen222source scope, parameters, runtime or chronology changed')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Measured source bytes changed')
    audit_kernel_checks(m['native_component_kernel_checks'])
    if m['smoke']:
        if m['smoke_equivalence_gate'] is not None:raise ValueError('Unexpected smoke gate')
    else:
        from scripts.run_fresh_query import checked_gate
        gate=m['smoke_equivalence_gate'];eq=json.loads(Path(gate['path']).read_text())
        if (checked_gate(Path(gate['path']),m['packet_manifest_sha256'])!=gate or eq['runtime_source_sha256']!=m['source_sha256'] or
            any(datetime.fromisoformat(t)>=datetime.fromisoformat(m['started_utc']) for t in eq['smoke_completed_utc'])):
            raise ValueError('Both exact operational smokes must precede primary inference')
    controls=refs['legacy_ids'];anchors={v['id']:v['readout'] for v in m['native_anchors']}
    if [v['id'] for v in m['native_anchors']]!=controls or anchors!=refs['legacy_full']['native']:raise ValueError('All original native anchors required')
    contexts=selected_contexts(registry,m['smoke']);p=6 if m['smoke'] else 12
    all_rows={};source_count=0;prefix_count=0;capture_count=0;replay_count=0;query_tensors=0
    for name in order:
        state=m['models'][name];spec=plan['models'][name];old=refs['model_states'][name]
        if state['token_policy']!=('all' if spec else 'none') or not state['hooks_restored']:raise ValueError('Frozen policy/restoration changed')
        for label in ('native_before_attach_checks','attached_none_checks','legacy_full_checks','native_restore_checks'):
            values=state[label];ids=[] if label=='attached_none_checks' and spec is None else controls
            reference=refs['legacy_full'][name] if label=='legacy_full_checks' else anchors
            if [v['id'] for v in values]!=ids or any(v['readout']!=reference[v['id']] for v in values):raise ValueError('Legacy exact replay failed')
        if spec:
            if (state['description']!=old['description'] or state['adapter_parameters']!=spec['parameters'] or
                state['checkpoint_sha256']!=spec['checkpoint_sha256'] or streaming_hash(Path(spec['checkpoint']))!=spec['checkpoint_sha256'] or
                state['adapter_state_sha256_before']!=old['adapter_state_sha256_before'] or
                state['adapter_state_sha256_after']!=state['adapter_state_sha256_before'] or not state['all_adapter_parameters_frozen'] or
                not state['arithmetic_hooks_removed']):raise ValueError('Fixed adapter changed')
            audit_affines(state,plan['arithmetic_profile'])
        before=[c for c in contexts if c['episode']=='before'];after=[c for c in contexts if c['episode']!='before']
        if list(state['prefixes'])!=[c['prefix_key'] for c in before] or list(state['sources'])!=[c['id'] for c in after]:
            raise ValueError('All fresh prefixes and tail sources required')
        for context in before:
            audit_cache(state['prefixes'][context['prefix_key']],context['prefix_tokens'],10+33*p);prefix_count+=1
        for context in after:
            source=state['sources'][context['id']];normal=source['caches']['N'];held=source['caches']['H'];parent=state['prefixes'][context['prefix_key']]
            if set(source['caches'])!=set(ORIGINS) or not source['all_stream_caches_and_donors_built_before_any_fresh_answer'] or not source['pairwise_storage_isolated']:
                raise ValueError('Question-independent isolated sources and donors required')
            audit_cache(normal,context['ledger_tokens'],3+9*p);audit_restoration(held,normal,parent,context['ledger_tokens'],2+9*p)
            prefix_kv={k:v['sha256'] for k,v in parent['tensors'].items() if "'keys'" in k or "'values'" in k}
            tail=source['proofs']['normal_tail'];restore=source['proofs']['restoration']
            check_gate_stats(tail['gates'],'none',context['tail_tokens'])
            if (tail['tail_tokens']!=context['tail_tokens'] or tail['both_mode_exact_recurrent_keys'] or not tail['isolated_from_prefix'] or
                tail['write_keep_sha256']!=data_fingerprint('1'*context['prefix_tokens']+'0'*context['tail_tokens']) or
                tail['historical_kv_sha256']!=prefix_kv or restore['historical_kv_sha256']!=prefix_kv or
                restore['restored_recurrent_keys']!=[k for k in parent['tensors'] if "'recurrent_states'" in k] or
                restore['normal_non_recurrent_fields_exact']!=30 or not restore['normal_bookkeeping_exact'] or
                source['identities']!={o:memory_identity(source['caches'][o]) for o in ORIGINS}):
                raise ValueError('Only recurrent restoration with normal KV/conv/positions permitted')
            source_count+=2
        raw={}
        for kind in ('primary','same-source','donors'):
            rp=path/f'{name}.{kind}.jsonl'
            if streaming_hash(rp)!=state['results_sha256'][kind]:raise ValueError('Frozen raw bytes changed')
            raw[kind]=[json.loads(line) for line in rp.read_text().splitlines()]
        expected={k:[] for k in raw}
        for seed in dict.fromkeys(c['seed'] for c in contexts):
            group=[c for c in contexts if c['seed']==seed];bc=next(c for c in group if c['episode']=='before')
            for q in context_queries(bc,queries,m['smoke']):
                for reader in READERS:expected['primary'].append((q['id'],'before',reader))
            for context in group:
                if context['episode']=='before':continue
                for q in context_queries(context,queries,m['smoke']):
                    for reader in READERS:
                        expected['primary'].extend((q['id'],c,reader) for c in ('NN','HH','NH','HN'))
                        expected['same-source'].extend((q['id'],o*2,reader) for o in ORIGINS)
                        expected['donors'].extend((q['id'],o,reader) for o in ORIGINS)
        for kind,rows in raw.items():
            if [(r['id'],r['origin'] if kind=='donors' else r['condition'],r['reader']) for r in rows]!=expected[kind]:
                raise ValueError('Every registered role/style/stream must remain in raw order')
        donors={}
        for row in raw['donors']:
            q=queries[row['id']];bank=row['bank'];origin=row['origin'];proof=row['capture_proof'];reader=row['reader']
            if (row['model']!=name or row['registration_sha256']!=data_fingerprint(q) or row['source_key']!=q['source_key'] or
                row['source_cache_identity']!=state['sources'][q['context_id']]['identities'][origin] or
                not row['isolated_from_all_cache_and_donor_storage'] or bank['guard_checks']!=(10 if origin=='N' else 9) or
                core_bank(bank)!=core_bank(proof['donor']) or proof['donor']['guard_checks']!=1 or bank['reader']!=reader or
                bank['boundary']!=q['ledger_tokens'] or bank['prompt_ids_sha256']!=q['prompt_ids_sha256'] or
                bank['question_tokens']!=len(q['input_ids'])-q['ledger_tokens'] or
                bank['question_ids_sha256']!=data_fingerprint(q['input_ids'][q['ledger_tokens']:]) or
                set(bank['tensors'])!={str(i) for i in ATTENTION_LAYERS}):raise ValueError('Answer-free donor provenance or immutable bank changed')
            audit_reader(proof,q,reader);audit_exchange(proof['query_exchange'],q,bank,'capture',proof)
            if reader=='oracle' and proof['reader_calls']!={'prefill':0,'cached':1}:raise ValueError('Oracle donor must contain only one known-question forward')
            donors[(q['id'],reader,origin)]=bank;capture_count+=1;query_tensors+=6
        primary_index={(r['id'],r['condition'],r['reader']):r for r in raw['primary']}
        for kind in ('primary','same-source'):
            for row in raw[kind]:
                q=queries[row['id']];condition=row['condition'];reader=row['reader'];proof=row['proof']
                audit_readout(readout(row),{**q,'question_tokens':len(q['input_ids'])-q['ledger_tokens']});audit_reader(proof,q,reader)
                origin=('same_source_operational_replay' if kind=='same-source' else 'before_baseline' if condition=='before' else
                    'unpatched_after_baseline' if condition[0]==condition[1] else 'cross_source_query')
                if (row['model']!=name or row['prefill']!='cached' or row['registration_sha256']!=data_fingerprint(q) or
                    row['source_key']!=q['source_key'] or row['prompt_ids_sha256']!=q['prompt_ids_sha256'] or row['measurement_origin']!=origin):
                    raise ValueError('Native tokens, grading or measurement origin changed')
                patched=kind=='same-source' or condition in ('NH','HN')
                if patched:
                    bank=donors[(q['id'],reader,condition[1])];audit_exchange(proof['query_exchange'],q,bank,'replace',proof)
                elif proof['query_exchange'] is not None:raise ValueError('Baseline must run entirely without Query replacement')
                if kind=='same-source':
                    baseline=primary_index[(q['id'],condition,reader)]
                    if readout(row)!=readout(baseline) or any(proof[k]!=baseline['proof'][k] for k in ('reader_calls','attention','reader_keep_sha256')):
                        raise ValueError('Same-source replacement must exactly replay all five outcomes and reader probabilities')
                    if any(v['native_query']!=bank['tensors'][i] for i,v in proof['query_exchange']['layers'].items()):
                        raise ValueError('Self replacement original Query must equal its actual donor')
        for context in after:
            for q in context_queries(context,queries,m['smoke']):
                for reader in READERS:
                    if donors[(q['id'],reader,'N')]['tensors']==donors[(q['id'],reader,'H')]['tensors']:
                        for candidate,base in (('NH','NN'),('HN','HH')):
                            if readout(primary_index[(q['id'],candidate,reader)])!=readout(primary_index[(q['id'],base,reader)]):
                                raise ValueError('Equal donor tensors must give equal recipient outputs')
        if (state['completed_primary']!=len(raw['primary']) or state['completed_same_source_replays']!=len(raw['same-source']) or
            state['completed_captures']!=len(raw['donors'])):raise ValueError('Terminal complete counts differ')
        all_rows[name]=raw['primary'];replay_count+=len(raw['same-source'])
    return {'status':'audited_fresh_query_smoke' if m['smoke'] else 'completed_and_audited_fresh_query',
        'source_run':str(path),'manifest_sha256':streaming_hash(path/'manifest.json'),'packet_manifest_sha256':m['packet_manifest_sha256'],
        'primary_readouts_audited':sum(map(len,all_rows.values())),'same_source_replays_audited':replay_count,
        'question_only_captures_audited':capture_count,'actual_donor_query_tensors_audited':query_tensors,
        'normal_prefix_caches_audited':prefix_count,'after_source_caches_audited':source_count,'legacy_operational_readouts':96,
        'source_files_verified':len(sources()),'baseline_failures_excluded':0,'fresh_streams':True,
        'prior_failed_criteria_unchanged':refs['prior_failed_criteria'],'original_ideal_kernel_failure_preserved':True,
        'architecture_effectiveness_claim':False,'metrics':None if m['smoke'] else summarize(all_rows,queries,{**plan,'seeds':registered['seeds']}),
        'analysis_source_sha256':streaming_hash(Path(__file__))}


def audit_equivalence(forward_path,reverse_path):
    paths=[Path(forward_path),Path(reverse_path)];reports=[];manifests=[]
    for path in paths:
        report=json.loads(path.read_text())
        if (report['status']!='audited_fresh_query_smoke' or report['primary_readouts_audited']!=204 or
            report['same_source_replays_audited']!=96 or report['question_only_captures_audited']!=96 or
            report['actual_donor_query_tensors_audited']!=576 or audit_run(Path(report['source_run']),True)!=report or
            streaming_hash(path.with_suffix('.source.py'))!=report['analysis_source_sha256']):raise ValueError('Complete independently audited204smokes required')
        reports.append(report);manifests.append(json.loads((Path(report['source_run'])/'manifest.json').read_text()))
    fm,rm=manifests
    if fm['model_order']!=list(MODELS) or rm['model_order']!=list(MODELS[::-1]) or any(fm[k]!=rm[k] for k in
        ('native_anchors','native_component_kernel_checks','source_sha256','packet_manifest_sha256','weights_before','weights_after')):
        raise ValueError('Opposite model order and exact native runtime required')
    for name in MODELS:
        if fm['models'][name]!=rm['models'][name]:raise ValueError('Actual source caches, fixed parameters or restores differ')
        for kind in ('primary','same-source','donors'):
            if (Path(reports[0]['source_run'])/f'{name}.{kind}.jsonl').read_bytes()!=(Path(reports[1]['source_run'])/f'{name}.{kind}.jsonl').read_bytes():
                raise ValueError('Opposite-order complete raw bytes differ')
    return {'status':'audited_fresh_query_smoke_equivalence','forward_audit':str(paths[0]),'reverse_audit':str(paths[1]),
        'forward_audit_sha256':streaming_hash(paths[0]),'reverse_audit_sha256':streaming_hash(paths[1]),
        'packet_manifest_sha256':reports[0]['packet_manifest_sha256'],'runtime_source_sha256':fm['source_sha256'],
        'smoke_completed_utc':[v['completed_utc'] for v in manifests],'exact_primary_readouts':204,'exact_same_source_replays':96,
        'exact_question_only_captures':96,'accuracy_gate':False,'primary_evaluated':False,'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',nargs='?',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-smoke',action='store_true');parser.add_argument('--equivalence',nargs=2,type=Path);args=parser.parse_args()
    if args.output.exists():raise ValueError('Never overwrite an audit receipt')
    report=audit_equivalence(*args.equivalence) if args.equivalence else audit_run(args.run,args.allow_smoke)
    write_json(args.output,report);args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','runtime_source_sha256')}),flush=True)


if __name__=='__main__':main()
