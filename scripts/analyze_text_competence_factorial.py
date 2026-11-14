"""Independently audit all text transformations, frozen execution and factorial outcomes."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json
from scripts.copy_readout_runtime import readout
from scripts.analyze_trained_timing_evidence_access import audit_readout
from scripts.analyze_unseen_shared_timing import audit_prefix
from scripts.canonical_necessity_common import audit_affines
from scripts.text_competence_factorial_data import (MODELS,REGIMES,PROTOCOL,DESTINATION,read_plan,registered_inputs,
    queries_for,unique_prefixes,selected_inputs,control_ids,sources,require_exact_full)
from scripts.text_competence_factorial_metrics import summarize
from scripts.run_text_competence_factorial import checked_gate


def audit_attention(stats):
    if set(stats)!={str(i) for i in (3,7,11,15,19,23)}:raise ValueError('Every actual softmax layer required')
    for s in stats.values():
        if (s['prefill_calls']!=1 or s['cached_calls']<1 or s['query_rows']<1 or
            not 0<s['max_record_probability']<=1 or not 0<s['max_record_mass']<=1.01):
            raise ValueError('Actual full-history attention is vacuous or incomplete')


def audit_identical_inputs(all_rows,transforms):
    seen={}
    for regime in REGIMES:
        for row in all_rows[regime]:
            q=transforms[row['id']][regime];key=(q['prompt_ids_sha256'],q['ledger_tokens'])
            if key in seen and readout(row)!=seen[key]:raise ValueError('Identical transformed tokens/boundary produced inconsistent complete readouts')
            seen[key]=readout(row)


def audit_run(path,allow_smoke=False):
    path=Path(path);m=json.loads((path/'manifest.json').read_text());plan=read_plan()
    if m['status']!='completed' or (m['smoke'] and not allow_smoke):raise ValueError('Completed primary or explicitly operational smoke required')
    registered,cases,registry,transforms,references=registered_inputs()
    physical,bindings=selected_inputs(registry,m['smoke']);ids=[q['id'] for q in physical];controls=control_ids(cases,registry)
    order=list(MODELS[::-1] if m['reverse_model_order'] else MODELS)
    profile=json.loads(Path(plan['precision_addendum']).read_text());weights=m['weights_after'];refs=registered['references']
    if (m['training'] or m['new_adapters'] or (m['reverse_model_order'] and not m['smoke']) or
        m['dataset']!=str(DESTINATION) or m['dataset_manifest_sha256']!=streaming_hash(DESTINATION/'manifest.json') or
        m['protocol_sha256']!=streaming_hash(PROTOCOL) or m['transformations_sha256']!=registered['transformations_sha256'] or
        m['collisions_sha256']!=registered['collisions_sha256'] or m['references']!=refs or
        m['model_path']!=plan['model_path'] or m['packages']!=plan['packages'] or
        m['actual_eos_token_id']!=refs['actual_eos_token_id'] or m['generation_config_sha256']!=refs['generation_config_sha256'] or
        m['generation_config_sha256']!={p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')} or
        m['arithmetic_profile']!=plan['arithmetic_profile'] or m['precision_addendum_sha256']!=plan['precision_addendum_sha256'] or
        m['generation']!=plan['generation'] or m['selected_physical_ids']!=ids or m['bindings_per_model_regime']!=len(bindings) or
        m['control_query_ids']!=controls or m['model_order']!=order or m['completed_models']!=order or list(m['models'])!=order or
        set(m['source_sha256'])!=sources() or m['weights_before']!=weights or weights['sha256']!=plan['native_loaded_weights_sha256'] or
        not weights['all_parameters_frozen'] or not weights['no_parameter_gradients'] or not m['hooks_restored'] or
        json.loads((path/'plan.json').read_text())!=plan or json.loads((path/'precision_addendum.json').read_text())!=profile or
        streaming_hash(path/'installed_modeling_source.py')!=refs['installed_modeling_source_sha256'] or
        streaming_hash(Path(refs['installed_modeling_source']))!=refs['installed_modeling_source_sha256'] or
        not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(registered['registered_utc'])<
            datetime.fromisoformat(m['started_utc'])<datetime.fromisoformat(m['completed_utc'])):
        raise ValueError('Actual scope, complete sources, native runtime, frozen parameters or chronology changed')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Preserve all sealed live/archive sources')
    if m['smoke']:
        if m['smoke_equivalence_gate'] is not None:raise ValueError('Operational smoke has an unexpected primary gate')
    else:
        gate=m['smoke_equivalence_gate'];eq=json.loads(Path(gate['path']).read_text())
        if (checked_gate(Path(gate['path']),m['dataset_manifest_sha256'])!=gate or eq['runtime_source_sha256']!=m['source_sha256'] or
            any(datetime.fromisoformat(t)>=datetime.fromisoformat(m['started_utc']) for t in eq['smoke_completed_utc'])):
            raise ValueError('Primary preceded exact smoke equivalence or changed its sources')
    reference={n:{r['id']:r for r in rows} for n,rows in references.items()}
    if list(m['native_anchors'])!=list(REGIMES):raise ValueError('All four native anchor conditions required')
    anchors={}
    for regime in REGIMES:
        values=m['native_anchors'][regime]
        if [v['id'] for v in values]!=controls:raise ValueError('All12 independent native anchor queries required')
        anchors[regime]={v['id']:v['readout'] for v in values}
        for key,value in anchors[regime].items():
            audit_readout(value,transforms[key][regime])
            if regime=='full':require_exact_full(value,reference['native'][key],'native_anchor',key)
    all_rows={};q_prefix=0
    previous=json.loads((Path(plan['source_run'])/'manifest.json').read_text())
    previous_prefix={p['source_key']:p for p in previous['models']['shared_question_trained_question']['prefix_checks']}
    for name in order:
        state=m['models'][name];spec=plan['models'][name];policy=spec['token_policy'] if spec else 'none'
        if (state['token_policy']!=policy or state['completed_queries_per_regime']!={r:len(physical) for r in REGIMES} or
            list(state['results_sha256'])!=list(REGIMES) or not state['hooks_restored']):raise ValueError('Fixed deployment or complete arm coverage changed')
        if spec:
            training=json.loads((Path(spec['training_run'])/'manifest.json').read_text())
            if (state['description']!=training['arms'][spec['training_arm']]['description'] or state['adapter_parameters']!=spec['parameters'] or
                state['checkpoint_sha256']!=spec['checkpoint_sha256'] or streaming_hash(Path(spec['checkpoint']))!=spec['checkpoint_sha256'] or
                state['adapter_state_sha256_before']!=refs['training'][name]['checkpoint_state_sha256'] or
                state['adapter_state_sha256_after']!=state['adapter_state_sha256_before'] or
                not state['all_adapter_parameters_frozen'] or not state['arithmetic_hooks_removed']):raise ValueError('Actual adapter bytes or scope changed')
            audit_affines(state,plan['arithmetic_profile'])
        all_rows[name]={}
        for regime in REGIMES:
            for label in ('native_before_attach_checks','native_restore_checks','attached_none_checks'):
                values=state[label][regime];expected=[] if label=='attached_none_checks' and not spec else controls
                if [v['id'] for v in values]!=expected or any(v['readout']!=anchors[regime][v['id']] for v in values):
                    raise ValueError('Actual before/none/detach native replay failed')
            prefix_queries=unique_prefixes(queries_for(transforms,cases,registry,regime,m['smoke'],True)) if policy=='question' else []
            audit_prefix(state['prefix_checks'][regime],prefix_queries);q_prefix+=len(prefix_queries)
            if regime=='full':
                for p in state['prefix_checks'][regime]:
                    if p!=previous_prefix[p['source_key']]:raise ValueError('Original full native-prefix proof differs from the completed reference')
            result_path=path/f'{name}.{regime}.jsonl'
            if streaming_hash(result_path)!=state['results_sha256'][regime]:raise ValueError('Actual raw result bytes changed')
            rows=[json.loads(line) for line in result_path.read_text().splitlines()]
            if [v['id'] for v in rows]!=ids:raise ValueError('All original paired physical IDs and order required')
            for row in rows:
                q=transforms[row['id']][regime];audit_readout(readout(row),q)
                if (row['model']!=name or row['regime']!=regime or row['prefill']!='full' or row['reader_policy']!='full' or
                    row['token_policy']!=policy or row['arithmetic_profile']!=plan['arithmetic_profile'] or
                    row['registration_sha256']!=data_fingerprint(q) or row['actual_prompt_sha256']!=q['prompt_ids_sha256'] or
                    row['ledger_source_key']!=q['source_key']):raise ValueError('Actual transformed input or execution labels changed')
                if regime=='full':require_exact_full(row,reference[name][row['id']],name,row['id'])
            observed=state['attention_controls'][regime]
            if [v['id'] for v in observed]!=controls:raise ValueError('All12 prescribed all-depth attention controls required')
            indexed={v['id']:v for v in rows}
            for v in observed:
                q=transforms[v['id']][regime];audit_readout(v['readout'],q);audit_attention(v['stats'])
                if v['id'] in indexed and v['readout']!=readout(indexed[v['id']]):raise ValueError('Actual observed control differs from recorded primary readout')
                if regime=='full':require_exact_full(v['readout'],reference[name][v['id']],name,v['id'])
                if name=='native' and v['readout']!=anchors[regime][v['id']]:raise ValueError('Native arm control differs from independent anchor')
            all_rows[name][regime]=rows
        audit_identical_inputs(all_rows[name],transforms)
    ordered={name:all_rows[name] for name in MODELS}
    collisions=json.loads((DESTINATION/'collisions.json').read_text())
    metrics=None if m['smoke'] else summarize(ordered,cases,registry,plan,collisions)
    return {'status':'audited_text_competence_factorial_smoke' if m['smoke'] else 'completed_and_audited_text_competence_factorial',
        'source_run':str(path),'manifest_sha256':streaming_hash(path/'manifest.json'),
        'dataset_manifest_sha256':m['dataset_manifest_sha256'],'transformations_sha256':registered['transformations_sha256'],
        'collisions_sha256':registered['collisions_sha256'],'weights_sha256':weights['sha256'],'arithmetic_profile':plan['arithmetic_profile'],
        'model_order':order,'new_physical_readouts_audited':len(physical)*20,'fresh_full_reference_readouts_audited':len(physical)*5,
        'transformed_text_readouts_audited':len(physical)*15,'related_bindings_audited':len(bindings)*20,
        'native_anchors':48,'native_before_attach_exact_replays':240,'attached_none_identities':192,'detached_native_exact_replays':240,
        'all_layer_full_history_attention_controls':240,'question_only_native_prefix_checks':q_prefix,
        'source_files_verified':len(sources()),'all_original_full_reference_fields_exact':True,'identical_actual_input_replays_exact':True,
        'prior_timing_criteria_unchanged':refs['prior_timing_criteria_unchanged'],'metrics':metrics,
        'interpretation':plan['interpretation_limits'],'analysis_source_sha256':streaming_hash(Path(__file__))}


def compare_smokes(fm,rm,forward,reverse):
    if (fm['model_order']!=list(MODELS) or rm['model_order']!=list(MODELS[::-1]) or
        fm['native_anchors']!=rm['native_anchors'] or fm['source_sha256']!=rm['source_sha256'] or
        fm['references']!=rm['references'] or fm['dataset_manifest_sha256']!=rm['dataset_manifest_sha256'] or
        fm['weights_before']!=rm['weights_before'] or fm['weights_after']!=rm['weights_after']):
        raise ValueError('Opposite-order native anchors, scopes or source bytes differ')
    for name in MODELS:
        if fm['models'][name]!=rm['models'][name]:raise ValueError('Opposite-order control/readout/prefix/arithmetic proofs differ')
        for regime in REGIMES:
            if forward[name][regime]!=reverse[name][regime]:raise ValueError('Opposite-order actual complete raw rows differ')


def audit_equivalence(forward_path,reverse_path):
    paths=[Path(forward_path),Path(reverse_path)];reports=[json.loads(p.read_text()) for p in paths];manifests=[];data=[]
    for p,r in zip(paths,reports):
        if (r['status']!='audited_text_competence_factorial_smoke' or r['new_physical_readouts_audited']!=360 or
            audit_run(Path(r['source_run']),True)!=r or streaming_hash(p.with_suffix('.source.py'))!=r['analysis_source_sha256']):
            raise ValueError('Complete independent360-readout operational audit required')
        run=Path(r['source_run']);manifests.append(json.loads((run/'manifest.json').read_text()))
        data.append({n:{regime:[json.loads(s) for s in (run/f'{n}.{regime}.jsonl').read_text().splitlines()] for regime in REGIMES} for n in MODELS})
    compare_smokes(*manifests,*data)
    return {'status':'audited_text_competence_factorial_smoke_equivalence','forward_audit':str(paths[0]),'reverse_audit':str(paths[1]),
        'forward_audit_sha256':streaming_hash(paths[0]),'reverse_audit_sha256':streaming_hash(paths[1]),
        'dataset_manifest_sha256':reports[0]['dataset_manifest_sha256'],'runtime_source_sha256':manifests[0]['source_sha256'],
        'smoke_completed_utc':[m['completed_utc'] for m in manifests],'exact_readouts':360,'exact_attention_controls':240,
        'exact_question_only_prefix_checks':reports[0]['question_only_native_prefix_checks'],'accuracy_gate':False,'primary_evaluated':False,
        'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',nargs='?',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-smoke',action='store_true');p.add_argument('--equivalence',nargs=2,type=Path)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous audit')
    report=audit_equivalence(*a.equivalence) if a.equivalence else audit_run(a.run,a.allow_smoke)
    write_json(a.output,report);a.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:report[k] for k in ('status','new_physical_readouts_audited','exact_readouts','source_files_verified') if k in report}),flush=True)


if __name__=='__main__':main()
