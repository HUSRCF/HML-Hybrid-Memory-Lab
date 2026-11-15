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
from scripts.gate_component_data import (MODELS,REGIMES,PROTOCOL,DESTINATION,read_plan,registered_inputs,
    sources,control_ids,selected_inputs,unique_prefixes,ARCHIVED)
from scripts.gate_component_runtime import GDN_LAYERS,SETTINGS,validation_addendum,VALIDATION_ADDENDUM_SHA256
from scripts.gate_component_metrics import summarize


def audit_gates(proof,policy,mode):
    held=0 if mode=='none' else policy['held_token_count']
    if (proof['component_mode']!=mode or proof['enabled']!=(mode!='none') or not proof['actual_gates_finite_and_exact'] or
        set(map(str,proof['layers']))!={str(i) for i in GDN_LAYERS}):raise ValueError('All18actual component gate observations required')
    canonical=None
    for stat in proof['layers'].values():
        if (stat['chunk_calls']<1 or stat['recurrent_calls']<1 or stat['held_rows']!=held or
            stat['rows']!=stat['held_rows']+stat['unheld_rows'] or stat['unheld_rows']<1 or
            stat['forced_decay_rows']!=(held if mode in ('decay','both') else 0) or
            stat['forced_delta_rows']!=(held if mode in ('delta','both') else 0)):
            raise ValueError('Exact single-gate/untouched-gate coverage failed')
        if canonical is not None and stat!=canonical:raise ValueError('Different per-layer gate schedules')
        canonical=stat


def audit_proof(proof,policy,regime,observed=False):
    mode,restricted=SETTINGS[regime];audit_gates(proof['gates'],policy,mode)
    if restricted:
        if proof['reader_calls']['prefill']!=1 or proof['reader_calls']['cached']<1:raise ValueError('Reader coverage changed')
    elif proof['reader_calls'] is not None:raise ValueError('Unexpected reader restriction')
    if observed:check_weight_stats(proof['attention'],restricted)
    elif proof['attention'] is not None:raise ValueError('Unexpected extra attention observation')


def audit_kernel_checks(checks):
    import math
    expected=[(k,t,mode) for k,t in [('chunk_gated_delta_rule',1),('chunk_gated_delta_rule',7),('chunk_gated_delta_rule',65),
        ('recurrent_gated_delta_rule',1),('recurrent_gated_delta_rule',7)] for mode in ('pure_decay','pure_delta')]
    if [(v['kernel'],v['length'],v['mode']) for v in checks]!=expected:raise ValueError('All10actual native component checks required')
    for v in checks:
        if (not v['initial_state_unmodified'] or not v['output_finite'] or not v['explicit_state_algebra_passed'] or
            v['reference_profile']!='native_bf16_chunk_fp32_recurrent_reference_v1' or
            v['validation_addendum_sha256']!=VALIDATION_ADDENDUM_SHA256 or
            not isinstance(v['original_ideal_algebra_passed'],bool) or
            not math.isfinite(v['original_ideal_max_absolute_error']) or v['original_ideal_max_absolute_error']<0 or
            v['atol']!=.00005 or v['rtol']!=.0001 or not math.isfinite(v['max_absolute_error']) or v['max_absolute_error']<0 or
            v['independent_of_qkv']!=(True if v['mode']=='pure_decay' else None)):
            raise ValueError('Native pure-decay/one-hot Delta algebra check failed')


def audit_run(path,allow_smoke=False):
    path=Path(path);m=json.loads((path/'manifest.json').read_text());plan=read_plan()
    registered,cases,registry,queries,policies,references=registered_inputs()
    correction=validation_addendum()
    if (m['validation_addendum_sha256']!=VALIDATION_ADDENDUM_SHA256 or
        json.loads((path/'kernel_validation_addendum.json').read_text())!=correction or
        not datetime.fromisoformat(correction['registered_utc'])<datetime.fromisoformat(m['started_utc'])):
        raise ValueError('Native precision correction must be pinned before any task inference')
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
    audit_kernel_checks(m['native_component_kernel_checks'])
    if m['smoke']:
        if m['smoke_equivalence_gate'] is not None:raise ValueError('Unexpected smoke gate')
    else:
        from scripts.run_gate_components import checked_gate
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
    old_prefix={v['source_key']:v for v in previous['models']['shared_question_trained_question']['prefix_checks']}
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
        if [(v['id'],v['regime']) for v in state['endpoint_controls']]!=[(key,r) for key in controls for r in ARCHIVED]:
            raise ValueError('All48old endpoint controls required per model')
        for v in state['endpoint_controls']:
            if v['readout']!=ref[name][v['regime']][v['id']]:raise ValueError('New wrapper changed a frozen old endpoint')
            audit_proof(v['proof'],policies[v['id']],v['regime'],True)
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
            for row in rows:
                q=queries[row['id']];audit_readout(readout(row),q)
                if (row['model']!=name or row['regime']!=regime or row['prefill']!='full' or row['token_policy']!=policy or
                    row['reader_policy']!=('oracle' if SETTINGS[regime][1] else 'full') or
                    row['gate_component']!=SETTINGS[regime][0] or row['arithmetic_profile']!=plan['arithmetic_profile'] or row['registration_sha256']!=data_fingerprint(q) or
                    row['policy_sha256']!=data_fingerprint(policies[q['id']]) or row['prompt_ids_sha256']!=q['prompt_ids_sha256']):
                    raise ValueError('Original tokens, policy or execution labels changed')
                audit_proof(row['proof'],policies[q['id']],regime,q['id'] in controls)
                if q['id'] in controls and (row['proof']!=controls_by_id[q['id']]['proof'] or readout(row)!=controls_by_id[q['id']]['readout']):
                    raise ValueError('Control differs from stored raw outcome')
            all_rows[name][regime]=rows
    all_rows={n:all_rows[n] for n in MODELS}
    return {'status':'audited_gate_component_smoke' if m['smoke'] else 'completed_and_audited_gate_components',
        'source_run':str(path),'manifest_sha256':streaming_hash(path/'manifest.json'),'dataset_manifest_sha256':m['dataset_manifest_sha256'],
        'new_physical_readouts_audited':len(physical)*20,'archived_reference_readouts_verified':6720,
        'related_bindings_audited':len(bindings)*20,'question_only_native_prefix_checks':prefix_count,
        'new_condition_attention_controls':240,'archived_endpoint_exact_controls':240,'all_layer_attention_observations':480,
        'source_files_verified':len(sources()),'extra_validation_registration_files_verified':1,
        'original_ideal_kernel_contract_failed_and_preserved':True,
        'validation_addendum_sha256':VALIDATION_ADDENDUM_SHA256,
        'prior_timing_criteria_unchanged':refs['prior_timing_criteria_unchanged'],
        'metrics':None if m['smoke'] else summarize(all_rows,references,cases,registry,plan),
        'interpretation':plan['interpretation_limits'],'analysis_source_sha256':streaming_hash(Path(__file__))}


def audit_equivalence(forward_path,reverse_path):
    paths=[Path(forward_path),Path(reverse_path)];reports=[];manifests=[]
    for p in paths:
        r=json.loads(p.read_text())
        if (r['status']!='audited_gate_component_smoke' or r['new_physical_readouts_audited']!=360 or
            audit_run(Path(r['source_run']),True)!=r or streaming_hash(p.with_suffix('.source.py'))!=r['analysis_source_sha256']):
            raise ValueError('Two complete independent operational smoke audits required')
        reports.append(r);manifests.append(json.loads((Path(r['source_run'])/'manifest.json').read_text()))
    fm,rm=manifests
    if (fm['model_order']!=list(MODELS) or rm['model_order']!=list(MODELS[::-1]) or
        any(fm[k]!=rm[k] for k in ('native_anchors','native_component_kernel_checks','source_sha256','references',
            'dataset_manifest_sha256','weights_before','weights_after'))):raise ValueError('Opposite-order runtime/control mismatch')
    for n in MODELS:
        if fm['models'][n]!=rm['models'][n]:raise ValueError('Opposite-order actual control or parameter mismatch')
        for r in REGIMES:
            if ((Path(reports[0]['source_run'])/f'{n}.{r}.jsonl').read_bytes()!=
                (Path(reports[1]['source_run'])/f'{n}.{r}.jsonl').read_bytes()):raise ValueError('Opposite-order complete raw bytes differ')
    return {'status':'audited_gate_component_smoke_equivalence','forward_audit':str(paths[0]),'reverse_audit':str(paths[1]),
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
