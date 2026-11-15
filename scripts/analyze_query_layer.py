"""Independently audit every single-layer call, source and registered question."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from sas_memory.training_data import data_fingerprint
from scripts.run_controller_trial import write_json
from scripts.query_layer_data import (read_plan,registered_inputs,sources,selected_contexts,context_queries,
    DESTINATION,PROTOCOL,MODELS,LAYERS,DIRECTIONS,ORIGINS,REFERENCES,self_control_ids)
from scripts.query_layer_metrics import summarize
from scripts.analyze_fresh_query import core_bank,audit_tensor,audit_exchange,audit_restoration
from scripts.analyze_post_encoding import core_cache
from scripts.post_tail_path_runtime import memory_identity
from scripts.post_encoding_runtime import ATTENTION_LAYERS,GDN_LAYERS,check_gate_stats
from scripts.analyze_post_encoding import audit_cache,audit_reader
from scripts.analyze_trained_timing_evidence_access import audit_readout
from scripts.canonical_necessity_common import audit_affines
from scripts.gate_component_runtime import validation_addendum,VALIDATION_ADDENDUM_SHA256
from scripts.analyze_gate_components import audit_kernel_checks
from scripts.copy_readout_runtime import readout


def audit_single_exchange(proof,q,donor,recipient,layer,reader_proof):
    count=len(q['input_ids'])-q['ledger_tokens']
    if (proof['mode']!='replace' or proof['selected_layer']!=layer or layer not in LAYERS or
        proof['boundary']!=q['ledger_tokens'] or proof['question_tokens']!=count or
        proof['question_ids_sha256']!=data_fingerprint(q['input_ids'][q['ledger_tokens']:]) or
        not proof['non_query_arguments_preserved'] or not proof['decode_queries_unmodified'] or
        not proof['unselected_question_queries_native'] or set(proof['layers'])!={str(i) for i in LAYERS}):
        raise ValueError('Exactly one registered known-question Query layer required')
    replaced=[]
    for key,stat in proof['layers'].items():
        index=int(key);selected=index==layer
        if (stat['question_calls']!=1 or stat['question_rows']!=count or stat['eager_calls']!=proof['model_calls'] or
            stat['eager_calls']!=reader_proof['attention'][key]['cached_calls'] or
            stat['passthrough_calls']!=stat['eager_calls']-1 or stat['non_query_tensor_checks']<2*stat['eager_calls'] or
            stat['native_query_checks']!=stat['eager_calls'] or stat['actual_replacement']!=selected or
            stat['original_query_object_preserved']==selected):
            raise ValueError('Single-layer call, object identity or non-Query preservation coverage changed')
        for field in ('native_query','used_query'):audit_tensor(stat[field],count)
        if index<=layer and stat['native_query']!=recipient['tensors'][key]:
            raise ValueError('No Query upstream of selected layer may be clamped or changed')
        if selected:
            replaced.append(index)
            if stat['used_query']!=donor['tensors'][key]:raise ValueError('Exact Full donor required in selected layer')
            for attr in ('shape','dtype','device','stride','storage_offset'):
                if stat['used_query'][attr]!=stat['native_query'][attr]:raise ValueError('Selected Query layout changed')
        elif stat['used_query']!=stat['native_query']:
            raise ValueError('Unselected layer must use actual native Query, including causal downstream changes')
    if replaced!=[layer]:raise ValueError('Exactly one actual Query copy required')


def audit_run(path,allow_smoke=False):
    path=Path(path);m=json.loads((path/'manifest.json').read_text());plan=read_plan()
    registered,cases,registry,queries,refs=registered_inputs();correction=validation_addendum()
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
        raise ValueError('Frozen230source scope, parameters, runtime or chronology changed')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Measured source bytes changed')
    audit_kernel_checks(m['native_component_kernel_checks'])
    if m['smoke']:
        if m['smoke_equivalence_gate'] is not None:raise ValueError('Unexpected smoke gate')
    else:
        from scripts.run_query_layer import checked_gate
        gate=m['smoke_equivalence_gate'];eq=json.loads(Path(gate['path']).read_text())
        if (checked_gate(Path(gate['path']),m['packet_manifest_sha256'])!=gate or eq['runtime_source_sha256']!=m['source_sha256'] or
            any(datetime.fromisoformat(t)>=datetime.fromisoformat(m['started_utc']) for t in eq['smoke_completed_utc'])):
            raise ValueError('Both exact operational smokes must precede primary inference')
    controls=refs['legacy_ids'];anchors={v['id']:v['readout'] for v in m['native_anchors']}
    if [v['id'] for v in m['native_anchors']]!=controls or anchors!=refs['legacy_full']['native']:raise ValueError('All original native anchors required')
    contexts=selected_contexts(registry,m['smoke']);control_ids=self_control_ids(contexts,queries,m['smoke'])
    n=3 if m['smoke'] else 6
    all_rows={};source_count=0;prefix_count=0;capture_count=0;self_count=0;reference_count=0;query_tensors=0
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
        previous_sources=refs['source_caches'][name]
        old_rows={(r['id'],r['condition']):r for r in refs['full_readouts'][name]}
        old_donors={(r['id'],r['origin']):r for r in refs['full_donors'][name]}
        if list(state['prefixes'])!=[c['prefix_key'] for c in before] or list(state['sources'])!=[c['id'] for c in after]:
            raise ValueError('All reused prefixes and tail sources required')
        for context in before:
            record=state['prefixes'][context['prefix_key']]
            audit_cache(record,context['prefix_tokens'],34+72*n)
            if core_cache(record)!=core_cache(previous_sources['prefixes'][context['prefix_key']]):raise ValueError('Exact predecessor prefix required')
            prefix_count+=1
        for context in after:
            cid=context['id'];source=state['sources'][cid];normal=source['caches']['N'];held=source['caches']['H'];parent=state['prefixes'][context['prefix_key']]
            controls=sum(q['id'] in control_ids for q in context_queries(context,queries,m['smoke']))
            if set(source['caches'])!=set(ORIGINS) or not source['all_stream_caches_and_donors_built_before_any_fresh_answer'] or not source['pairwise_storage_isolated']:
                raise ValueError('Question-independent isolated sources and donors required')
            audit_cache(normal,context['ledger_tokens'],3+18*n+12*controls)
            audit_restoration(held,normal,parent,context['ledger_tokens'],2+18*n+12*controls)
            if (source['proofs']!=previous_sources['sources'][cid]['proofs'] or
                source['identities']!={o:memory_identity(source['caches'][o]) for o in ORIGINS} or
                any(core_cache(source['caches'][o])!=core_cache(previous_sources['sources'][cid]['caches'][o]) for o in ORIGINS)):
                raise ValueError('Exact normal-tail source and old-matrix restoration required')
            source_count+=2
        raw={}
        for kind in ('primary','same-source','donors','references'):
            rp=path/f'{name}.{kind}.jsonl'
            if streaming_hash(rp)!=state['results_sha256'][kind]:raise ValueError('Frozen raw bytes changed')
            raw[kind]=[json.loads(line) for line in rp.read_text().splitlines()]
        expected={k:[] for k in raw}
        for context in after:
            for q in context_queries(context,queries,m['smoke']):
                expected['primary'].extend((q['id'],d,layer) for d in DIRECTIONS for layer in LAYERS)
                expected['references'].extend((q['id'],c,None) for c in REFERENCES)
                if q['id'] in control_ids:expected['same-source'].extend((q['id'],o*2,layer) for o in ORIGINS for layer in LAYERS)
                expected['donors'].extend((q['id'],o,None) for o in ORIGINS)
        for kind,rows in raw.items():
            if [(r['id'],r['origin'] if kind=='donors' else r['condition'],r.get('layer')) for r in rows]!=expected[kind]:
                raise ValueError('Complete unfiltered layer/direction/question coverage and raw order required')
        donors={}
        for row in raw['donors']:
            q=queries[row['id']];bank=row['bank'];origin=row['origin'];proof=row['capture_proof'];prior=old_donors[(q['id'],origin)]
            checks=(20 if origin=='N' else 19)+(12 if q['id'] in control_ids else 0)
            if (row['model']!=name or row['reader']!='full' or row['registration_sha256']!=data_fingerprint(q) or row['source_key']!=q['source_key'] or
                row['source_cache_identity']!=state['sources'][q['context_id']]['identities'][origin] or
                not row['isolated_from_all_cache_and_donor_storage'] or bank['guard_checks']!=checks or
                core_bank(bank)!=core_bank(prior['bank']) or proof!=prior['capture_proof']):
                raise ValueError('Exact immutable Full-only predecessor donor and capture required')
            audit_reader(proof,q,'full');audit_exchange(proof['query_exchange'],q,bank,'capture',proof)
            donors[(q['id'],origin)]=bank;capture_count+=1;query_tensors+=6
        for kind in ('primary','same-source','references'):
            for row in raw[kind]:
                q=queries[row['id']];condition=row['condition'];layer=row['layer'];proof=row['proof']
                audit_readout(readout(row),{**q,'question_tokens':len(q['input_ids'])-q['ledger_tokens']});audit_reader(proof,q,'full')
                origin={'primary':'single_layer_cross_source_query','same-source':'single_layer_same_source_operational_replay',
                    'references':'predecessor_full_reference_operational_replay'}[kind]
                if (row['model']!=name or row['reader']!='full' or row['prefill']!='cached' or
                    row['registration_sha256']!=data_fingerprint(q) or row['source_key']!=q['source_key'] or
                    row['prompt_ids_sha256']!=q['prompt_ids_sha256'] or row['measurement_origin']!=origin):
                    raise ValueError('Full-reader native tokens, grading or origin changed')
                if kind=='references':
                    prior=old_rows[(q['id'],condition)]
                    if readout(row)!=readout(prior) or proof!=prior['proof'] or layer is not None:raise ValueError('Exact predecessor Full replay required')
                    if condition[0]!=condition[1]:audit_exchange(proof['query_exchange'],q,donors[(q['id'],condition[1])],'replace',proof)
                    elif proof['query_exchange'] is not None:raise ValueError('Natural reference must not patch Queries')
                else:
                    donor=donors[(q['id'],condition[1])];recipient=donors[(q['id'],condition[0])]
                    audit_single_exchange(proof['query_exchange'],q,donor,recipient,layer,proof)
                    if kind=='same-source':
                        prior=old_rows[(q['id'],condition)]
                        if (readout(row)!=readout(prior) or any(proof[k]!=prior['proof'][k] for k in ('reader_calls','attention','reader_keep_sha256')) or
                            any(v['native_query']!=recipient['tensors'][i] for i,v in proof['query_exchange']['layers'].items())):
                            raise ValueError('Actual single-layer self copy must exactly replay native outputs and probabilities')
                    elif donor['tensors'][str(layer)]==recipient['tensors'][str(layer)] and readout(row)!=readout(old_rows[(q['id'],condition[0]*2)]):
                        raise ValueError('Equal selected Query must preserve recipient output')
        if (state['completed_primary']!=len(raw['primary']) or state['completed_same_source_replays']!=len(raw['same-source']) or
            state['completed_captures']!=len(raw['donors']) or state['completed_reference_replays']!=len(raw['references'])):
            raise ValueError('All terminal raw counts required')
        all_rows[name]=raw['primary'];self_count+=len(raw['same-source']);reference_count+=len(raw['references'])
    return {'status':'audited_query_layer_smoke' if m['smoke'] else 'completed_and_audited_query_layer',
        'source_run':str(path),'manifest_sha256':streaming_hash(path/'manifest.json'),'packet_manifest_sha256':m['packet_manifest_sha256'],
        'new_primary_readouts_audited':sum(map(len,all_rows.values())),'reference_operational_replays_audited':reference_count,
        'single_layer_same_source_replays_audited':self_count,'question_only_captures_audited':capture_count,'actual_donor_query_tensors_audited':query_tensors,
        'normal_prefix_caches_audited':prefix_count,'after_source_caches_audited':source_count,'legacy_operational_readouts':96,
        'source_files_verified':len(sources()),'baseline_failures_excluded':0,'fresh_confirmation':False,
        'prior_failed_criteria_unchanged':refs['prior_failed_criteria'],'original_ideal_kernel_failure_preserved':True,
        'architecture_effectiveness_claim':False,'metrics':None if m['smoke'] else summarize(all_rows,queries,refs),
        'analysis_source_sha256':streaming_hash(Path(__file__))}


def audit_equivalence(forward_path,reverse_path):
    reports=[];manifests=[]
    for path in (Path(forward_path),Path(reverse_path)):
        report=json.loads(path.read_text())
        if (report['status']!='audited_query_layer_smoke' or report['new_primary_readouts_audited']!=288 or
            report['reference_operational_replays_audited']!=96 or report['single_layer_same_source_replays_audited']!=48 or
            report['question_only_captures_audited']!=48 or report['actual_donor_query_tensors_audited']!=288 or
            report!=audit_run(Path(report['source_run']),True) or
            streaming_hash(path.with_suffix('.source.py'))!=report['analysis_source_sha256']):
            raise ValueError('Complete independently audited288readout smokes required')
        reports.append(report);manifests.append(json.loads((Path(report['source_run'])/'manifest.json').read_text()))
    fm,rm=manifests
    if not rm['reverse_model_order'] or fm['reverse_model_order'] or any(fm[k]!=rm[k] for k in
        ('native_anchors','native_component_kernel_checks','source_sha256','packet_manifest_sha256','weights_before','weights_after')):
        raise ValueError('Opposite orders and exact frozen runtime required')
    for name in MODELS:
        if fm['models'][name]!=rm['models'][name]:raise ValueError('Model state or actual cache/query proof depends on model order')
        for kind in ('primary','same-source','references','donors'):
            if (Path(reports[0]['source_run'])/f'{name}.{kind}.jsonl').read_bytes()!=(Path(reports[1]['source_run'])/f'{name}.{kind}.jsonl').read_bytes():
                raise ValueError('Every raw output and proof must be byte-identical in opposite model orders')
    return {'status':'audited_query_layer_smoke_equivalence','forward_audit':str(forward_path),'reverse_audit':str(reverse_path),
        'packet_manifest_sha256':reports[0]['packet_manifest_sha256'],'runtime_source_sha256':fm['source_sha256'],
        'smoke_completed_utc':[m['completed_utc'] for m in manifests],'exact_new_primary_readouts':288,
        'exact_reference_replays':96,'exact_single_layer_self_replays':48,'exact_question_only_captures':48,
        'accuracy_gate':False,'primary_evaluated':False,'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',nargs='?',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-smoke',action='store_true');parser.add_argument('--equivalence',nargs=2,type=Path);args=parser.parse_args()
    if args.output.exists():raise ValueError('Never overwrite an audit receipt')
    report=audit_equivalence(*args.equivalence) if args.equivalence else audit_run(args.run,args.allow_smoke)
    write_json(args.output,report);args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','runtime_source_sha256')}),flush=True)


if __name__=='__main__':main()
