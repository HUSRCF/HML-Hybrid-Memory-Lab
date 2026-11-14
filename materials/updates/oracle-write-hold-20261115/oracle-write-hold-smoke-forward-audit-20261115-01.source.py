"""Independently audit full fixed-input hold execution and paired outcomes."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json
from scripts.copy_readout_runtime import readout,check_weight_stats
from scripts.analyze_trained_timing_evidence_access import audit_readout
from scripts.analyze_unseen_shared_timing import audit_prefix
from scripts.canonical_necessity_common import audit_affines
from scripts.oracle_write_hold_data import (MODELS,REGIMES,PROTOCOL,DESTINATION,read_plan,registered_inputs,
    sources,control_ids,selected_inputs,unique_prefixes)
from scripts.oracle_write_hold_runtime import GDN_LAYERS
from scripts.oracle_write_hold_metrics import summarize


def audit_gates(proof,held,enabled,cached_probe=False):
    if (proof['enabled']!=enabled or not proof['actual_gates_finite_and_exact'] or
        set(map(str,proof['layers']))!={str(i) for i in GDN_LAYERS}):raise ValueError('All18actual native gate observations required')
    canonical=None
    for s in proof['layers'].values():
        if (s['chunk_calls']<1 or s['recurrent_calls']<1 or s['held_rows']!=held or
            s['rows']!=s['held_rows']+s['unheld_rows'] or s['unheld_rows']<1 or
            (cached_probe and (s['chunk_calls']!=1 or s['recurrent_calls']!=1))):raise ValueError('Actual hold coverage/counts changed')
        if canonical is not None and s!=canonical:raise ValueError('Different per-layer update schedules')
        canonical=s


def audit_proof(proof,policy,regime,observed=False,sham=False):
    held=regime in ('hold','hold_oracle') and not sham
    audit_gates(proof['gates'],policy['held_token_count'] if held else 0,held)
    restricted=regime in ('oracle','hold_oracle')
    if restricted:
        if proof['reader_calls']['prefill']!=1 or proof['reader_calls']['cached']<1:raise ValueError('Reader coverage changed')
    elif proof['reader_calls'] is not None:raise ValueError('Unexpected reader restriction')
    if observed:check_weight_stats(proof['attention'],restricted)
    elif proof['attention'] is not None:raise ValueError('Unexpected extra attention control')


def audit_run(path,allow_smoke=False):
    path=Path(path);m=json.loads((path/'manifest.json').read_text());plan=read_plan()
    registered,cases,registry,queries,policies,references=registered_inputs()
    physical,bindings=selected_inputs(registry,m['smoke']);ids=[q['id'] for q in physical];controls=control_ids(cases,registry)
    order=list(MODELS[::-1] if m['reverse_model_order'] else MODELS);refs=registered['references'];weights=m['weights_after']
    if (m['status']!='completed' or (m['smoke'] and not allow_smoke) or m['training'] or m['new_adapters'] or
        (m['reverse_model_order'] and not m['smoke']) or m['dataset']!=str(DESTINATION) or
        m['dataset_manifest_sha256']!=streaming_hash(DESTINATION/'manifest.json') or m['protocol_sha256']!=streaming_hash(PROTOCOL) or
        m['policies_sha256']!=registered['policies_sha256'] or m['queries_sha256']!=registered['queries_sha256'] or m['references']!=refs or
        m['model_path']!=plan['model_path'] or m['packages']!=plan['packages'] or
        m['actual_eos_token_id']!=refs['actual_eos_token_id'] or m['generation_config_sha256']!=refs['generation_config_sha256'] or
        m['generation_config_sha256']!={p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')} or
        m['arithmetic_profile']!=plan['arithmetic_profile'] or m['precision_addendum_sha256']!=plan['precision_addendum_sha256'] or
        m['generation']!=plan['generation'] or m['selected_physical_ids']!=ids or m['bindings_per_model_regime']!=len(bindings) or
        m['control_query_ids']!=controls or m['model_order']!=order or m['completed_models']!=order or list(m['models'])!=order or
        set(m['source_sha256'])!=sources() or m['weights_before']!=weights or weights['sha256']!=plan['native_loaded_weights_sha256'] or
        not weights['all_parameters_frozen'] or not weights['no_parameter_gradients'] or not m['hooks_restored'] or
        json.loads((path/'plan.json').read_text())!=plan or
        json.loads((path/'precision_addendum.json').read_text())!=json.loads(Path(plan['precision_addendum']).read_text()) or
        streaming_hash(path/'installed_modeling_source.py')!=refs['installed_modeling_source_sha256'] or
        streaming_hash(Path(refs['installed_modeling_source']))!=refs['installed_modeling_source_sha256'] or
        not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(registered['registered_utc'])<
            datetime.fromisoformat(m['started_utc'])<datetime.fromisoformat(m['completed_utc'])):
        raise ValueError('Complete scope, immutable runtime, input, native weights or chronology changed')
    for n,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/n)!=digest or streaming_hash(path/'sources'/n)!=digest:raise ValueError('Measured source bytes changed')
    expected_kernel=[('chunk_gated_delta_rule',1),('chunk_gated_delta_rule',7),('chunk_gated_delta_rule',65),
        ('recurrent_gated_delta_rule',1),('recurrent_gated_delta_rule',7)]
    if [(v['kernel'],v['length']) for v in m['native_kernel_identity_checks']]!=expected_kernel or any(
        not all(v[k] for k in ('initial_state_unmodified','final_state_exact','output_finite')) for v in m['native_kernel_identity_checks']):
        raise ValueError('Actual all-held native kernel identity tests failed')
    if m['smoke']:
        if m['smoke_equivalence_gate'] is not None:raise ValueError('Unexpected smoke gate')
    else:
        from scripts.run_oracle_write_hold import checked_gate
        gate=m['smoke_equivalence_gate'];eq=json.loads(Path(gate['path']).read_text())
        if (checked_gate(Path(gate['path']),m['dataset_manifest_sha256'])!=gate or eq['runtime_source_sha256']!=m['source_sha256'] or
            any(datetime.fromisoformat(t)>=datetime.fromisoformat(m['started_utc']) for t in eq['smoke_completed_utc'])):
            raise ValueError('Exact operational smoke gate absent or invalid')
    ref={n:{r:{v['id']:readout(v) for v in rows} for r,rows in cell.items()} for n,cell in references.items()}
    if [v['id'] for v in m['native_anchors']]!=controls:raise ValueError('All12native anchors required')
    anchors={v['id']:v['readout'] for v in m['native_anchors']}
    if any(v!=ref['native']['full'][key] for key,v in anchors.items()):raise ValueError('Native anchors differ')
    all_rows={};prefix_count=0
    previous=json.loads((Path(plan['source_run'])/'manifest.json').read_text())
    old_prefix={v['source_key']:v for v in previous['models']['shared_question_trained_question']['prefix_checks']['full']}
    for name in order:
        state=m['models'][name];spec=plan['models'][name];policy=spec['token_policy'] if spec else 'none'
        if (state['token_policy']!=policy or state['completed_queries_per_regime']!={r:len(physical) for r in REGIMES} or
            list(state['results_sha256'])!=list(REGIMES) or not state['hooks_restored']):raise ValueError('Model allocation/coverage changed')
        for label in ('native_before_attach_checks','attached_none_checks','native_restore_checks'):
            values=state[label];expected=[] if label=='attached_none_checks' and not spec else controls
            if [v['id'] for v in values]!=expected or any(v['readout']!=anchors[v['id']] for v in values):
                raise ValueError('Native before/none/detached exact replay failed')
        if spec:
            training=json.loads((Path(spec['training_run'])/'manifest.json').read_text())
            if (state['description']!=training['arms'][spec['training_arm']]['description'] or state['adapter_parameters']!=spec['parameters'] or
                state['checkpoint_sha256']!=spec['checkpoint_sha256'] or streaming_hash(Path(spec['checkpoint']))!=spec['checkpoint_sha256'] or
                state['adapter_state_sha256_before']!=refs['training'][name]['checkpoint_state_sha256'] or
                state['adapter_state_sha256_after']!=state['adapter_state_sha256_before'] or
                not state['all_adapter_parameters_frozen'] or not state['arithmetic_hooks_removed']):raise ValueError('Frozen adapter changed')
            audit_affines(state,plan['arithmetic_profile'])
        prefix_queries=unique_prefixes([q for q in queries.values() if q['id'] in set(ids+controls)]) if policy=='question' else []
        audit_prefix(state['prefix_checks'],prefix_queries);prefix_count+=len(prefix_queries)
        if any(v!=old_prefix[v['source_key']] for v in state['prefix_checks']):raise ValueError('Original native prefix proof drift')
        if [v['id'] for v in state['cached_state_identity']]!=controls:raise ValueError('All12cached state checks required')
        for v in state['cached_state_identity']:
            if (v['position']!=policies[v['id']]['first_held_position'] or v['recurrent_states_exact']!=18 or
                not 0<v['convolution_caches_changed']<=18 or v['kv_layers_advanced']!=6 or v['historical_kv_exact']!=12):
                raise ValueError('Actual cached state/convolution/KV identity check failed')
            audit_gates(v['gates'],1,True,cached_probe=True)
        if [(v['id'],v['regime']) for v in state['noop_wrapper_controls']]!=[(key,r) for key in controls for r in ('full','oracle')]:
            raise ValueError('All24no-op wrapper controls required')
        for v in state['noop_wrapper_controls']:
            if v['readout']!=ref[name][v['regime']][v['id']]:raise ValueError('No-op wrapper changed old readout')
            audit_proof(v['proof'],policies[v['id']],v['regime'],True,True)
        all_rows[name]={}
        for regime in REGIMES:
            rp=path/f'{name}.{regime}.jsonl'
            if streaming_hash(rp)!=state['results_sha256'][regime]:raise ValueError('Raw readout bytes changed')
            rows=[json.loads(line) for line in rp.read_text().splitlines()]
            if [v['id'] for v in rows]!=ids:raise ValueError('Every original query/order required')
            observed=state['condition_controls'][regime]
            if [v['id'] for v in observed]!=controls:raise ValueError('All12prescribed controls required in each condition')
            controls_by_id={v['id']:v for v in observed}
            for v in observed:
                audit_readout(v['readout'],queries[v['id']]);audit_proof(v['proof'],policies[v['id']],regime,True)
                if regime in ('full','oracle') and v['readout']!=ref[name][regime][v['id']]:raise ValueError('Control baseline drift')
            for row in rows:
                q=queries[row['id']];audit_readout(readout(row),q)
                if (row['model']!=name or row['regime']!=regime or row['prefill']!='full' or row['token_policy']!=policy or
                    row['reader_policy']!=('oracle' if regime in ('oracle','hold_oracle') else 'full') or
                    row['arithmetic_profile']!=plan['arithmetic_profile'] or row['registration_sha256']!=data_fingerprint(q) or
                    row['policy_sha256']!=data_fingerprint(policies[q['id']]) or row['prompt_ids_sha256']!=q['prompt_ids_sha256']):
                    raise ValueError('Original tokens, policy or execution labels changed')
                audit_proof(row['proof'],policies[q['id']],regime,q['id'] in controls)
                if q['id'] in controls and (row['proof']!=controls_by_id[q['id']]['proof'] or readout(row)!=controls_by_id[q['id']]['readout']):
                    raise ValueError('Control differs from stored raw outcome')
                if regime in ('full','oracle') and readout(row)!=ref[name][regime][q['id']]:raise ValueError('Original full/oracle five-field drift')
            all_rows[name][regime]=rows
    all_rows={n:all_rows[n] for n in MODELS}
    return {'status':'audited_oracle_write_hold_smoke' if m['smoke'] else 'completed_and_audited_oracle_write_hold',
        'source_run':str(path),'manifest_sha256':streaming_hash(path/'manifest.json'),'dataset_manifest_sha256':m['dataset_manifest_sha256'],
        'new_physical_readouts_audited':len(physical)*20,'exact_previous_readout_replays':len(physical)*10,
        'related_bindings_audited':len(bindings)*20,'question_only_native_prefix_checks':prefix_count,
        'cached_state_identity_controls':60,'actual_attention_controls':240,'noop_wrapper_controls':120,
        'source_files_verified':len(sources()),'prior_timing_criteria_unchanged':refs['prior_timing_criteria_unchanged'],
        'metrics':None if m['smoke'] else summarize(all_rows,cases,registry,plan),
        'interpretation':plan['interpretation_limits'],'analysis_source_sha256':streaming_hash(Path(__file__))}


def audit_equivalence(forward_path,reverse_path):
    paths=[Path(forward_path),Path(reverse_path)];reports=[];manifests=[]
    for p in paths:
        r=json.loads(p.read_text())
        if (r['status']!='audited_oracle_write_hold_smoke' or r['new_physical_readouts_audited']!=360 or
            audit_run(Path(r['source_run']),True)!=r or streaming_hash(p.with_suffix('.source.py'))!=r['analysis_source_sha256']):
            raise ValueError('Two complete independent operational smoke audits required')
        reports.append(r);manifests.append(json.loads((Path(r['source_run'])/'manifest.json').read_text()))
    fm,rm=manifests
    if (fm['model_order']!=list(MODELS) or rm['model_order']!=list(MODELS[::-1]) or
        any(fm[k]!=rm[k] for k in ('native_anchors','native_kernel_identity_checks','source_sha256','references',
            'dataset_manifest_sha256','weights_before','weights_after'))):raise ValueError('Opposite-order runtime/control mismatch')
    for n in MODELS:
        if fm['models'][n]!=rm['models'][n]:raise ValueError('Opposite-order actual control or parameter mismatch')
        for r in REGIMES:
            if ((Path(reports[0]['source_run'])/f'{n}.{r}.jsonl').read_bytes()!=
                (Path(reports[1]['source_run'])/f'{n}.{r}.jsonl').read_bytes()):raise ValueError('Opposite-order complete raw bytes differ')
    return {'status':'audited_oracle_write_hold_smoke_equivalence','forward_audit':str(paths[0]),'reverse_audit':str(paths[1]),
        'forward_audit_sha256':streaming_hash(paths[0]),'reverse_audit_sha256':streaming_hash(paths[1]),
        'dataset_manifest_sha256':reports[0]['dataset_manifest_sha256'],'runtime_source_sha256':fm['source_sha256'],
        'smoke_completed_utc':[m['completed_utc'] for m in manifests],'exact_readouts':360,'accuracy_gate':False,
        'primary_evaluated':False,'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',nargs='?',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-smoke',action='store_true');p.add_argument('--equivalence',nargs=2,type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite audit receipt')
    value=audit_equivalence(*a.equivalence) if a.equivalence else audit_run(a.run,a.allow_smoke)
    write_json(a.output,value);a.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in value.items() if k not in ('metrics','runtime_source_sha256')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
