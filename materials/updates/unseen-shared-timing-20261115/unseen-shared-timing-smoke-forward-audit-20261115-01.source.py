"""Independent complete-answer, actual timing and smoke-order audit."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_fresh_copy_confirmation import audit_readout
from scripts.analyze_fresh_layer_localization import sha
from scripts.canonical_necessity_common import audit_affines
from scripts.copy_readout_runtime import readout
from scripts.run_controller_trial import write_json
from scripts.unseen_shared_timing_data import (DESTINATION,PROTOCOL,ORDER,registered_inputs,
    selected_inputs,sample_ids,safety_samples,unique_prefix_queries,recheck_prior)
from scripts.unseen_shared_timing_metrics import summarize
from scripts.unseen_shared_timing_runtime import audit_policy,sources,checked_gate


def projection_targets(description,config):
    result={}
    for name,shape in description['output_shapes'].items():
        if name.endswith('_key'):
            result[name]=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
        elif name=='attention_q_proj':result[name]=2*shape[0]
        else:result[name]=shape[0]
    return result


def audit_prefix(checks,queries):
    if [p['source_key'] for p in checks]!=[q['source_key'] for q in queries]:raise ValueError('Actual native-prefix coverage incomplete')
    for p,q in zip(checks,queries):
        if (p['ledger_tokens']!=q['ledger_tokens'] or not p['complete_cache_bookkeeping_and_logits_exact'] or
            not p['all48_tensor_fields_exact'] or not p['hooks_restored'] or
            p['native_prefix_sha256']!=p['actual_prefix_sha256'] or p['native_prefix_logits_sha256']!=p['actual_prefix_logits_sha256']):
            raise ValueError('Question-only timing changed native complete prefix or logits')
        sha(p['native_prefix_sha256']);sha(p['native_prefix_logits_sha256'])


def audit_anchors(checks,samples,anchors):
    if [r['id'] for r in checks]!=samples or any(not r['exact'] or r['readout']!=anchors[r['id']] for r in checks):
        raise ValueError('Actual before-attach/detached native anchors differ')


def audit_run(path,allow_smoke=False):
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    if m['status']!='completed' or (m['smoke'] and not allow_smoke):raise ValueError('Completed run required; operational smoke is not primary evidence')
    registered,plan,cases,registry=registered_inputs()
    profile=json.loads((ROOT/plan['precision_addendum']).read_text());refs=registered['references']
    physical,bindings=selected_inputs(registry,m['smoke']);queries={q['id']:q for q in physical}
    samples=sample_ids(cases,registry);safety=safety_samples(registry)
    order=list(ORDER[::-1] if m['reverse_order'] else ORDER)
    config=json.loads((Path(plan['model_path'])/'config.json').read_text());config=config.get('text_config',config)
    if (m['training'] or m['new_adapters'] or (m['reverse_order'] and not m['smoke']) or
        m['dataset']!=str(DESTINATION) or m['dataset_manifest_sha256']!=streaming_hash(DESTINATION/'manifest.json') or
        m['protocol_sha256']!=streaming_hash(PROTOCOL) or m['precision_addendum_sha256']!=plan['precision_addendum_sha256'] or
        m['arithmetic_profile']!=profile['profile'] or m['references']!=refs or m['independent_novelty_recheck']!=recheck_prior(DESTINATION,registered) or
        json.loads((path/'plan.json').read_text())!=plan or json.loads((path/'precision_addendum.json').read_text())!=profile or
        m['model_path']!=plan['model_path'] or m['packages']!=plan['packages'] or
        m['generation']!={'max_tokens':12,'prefill':'full','score_batch':False} or
        m['selected_physical_ids']!=list(queries) or m['related_bindings_per_arm']!=len(bindings) or
        m['sample_query_ids']!=samples or m['model_order']!=order or m['completed_models']!=order or list(m['models'])!=order or
        set(m['source_sha256'])!=sources() or m['weights_before']!=m['weights_after'] or
        m['weights_after']['sha256']!=plan['native_loaded_weights_sha256'] or
        not m['weights_after']['all_parameters_frozen'] or not m['weights_after']['no_parameter_gradients'] or not m['hooks_restored'] or
        m['actual_eos_token_id']!=refs['actual_eos_token_id'] or m['installed_modeling_source_sha256']!=refs['installed_modeling_source_sha256'] or
        streaming_hash(path/'installed_modeling_source.py')!=refs['installed_modeling_source_sha256'] or
        m['generation_config_sha256']!=refs['generation_config_sha256'] or
        {p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')}!=refs['generation_config_sha256'] or
        not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(registered['registered_utc'])<
            datetime.fromisoformat(m['started_utc'])<datetime.fromisoformat(m['completed_utc'])):
        raise ValueError('Registered scope, native tensors, actual runtime, source closure or chronology differs')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Preserve all sealed live/archived runtime sources')
    if m['smoke']:
        if m['smoke_equivalence_gate'] is not None:raise ValueError('Operational smoke cannot use a primary gate')
    else:
        gate=m['smoke_equivalence_gate'];eq=json.loads(Path(gate['path']).read_text())
        if (checked_gate(Path(gate['path']),m['dataset_manifest_sha256'])!=gate or eq['runtime_source_sha256']!=m['source_sha256'] or
            any(datetime.fromisoformat(t)>=datetime.fromisoformat(m['started_utc']) for t in eq['smoke_completed_utc'])):
            raise ValueError('Primary run preceded exact order smoke or changed its sources')
    if [r['id'] for r in m['native_anchors']]!=samples:raise ValueError('Actual pre-controller native anchors incomplete')
    anchors={r['id']:r['readout'] for r in m['native_anchors']}
    for key,value in anchors.items():audit_readout(value,queries[key])
    all_rows={};counts={'physical_readouts':0,'related_bindings':len(bindings)*len(ORDER),
        'native_before_attach_replays':0,'detached_native_replays':0,'attached_none_policy_identities':0,
        'question_native_prefix_checks':0,'shared_question_all_distinct_native_prefixes':0,'actual_forward_receipts':0}
    for name in order:
        state=m['models'][name];arm=plan['evaluation_arms'][name];spec=plan['models'][arm['model']];policy=arm['token_policy']
        if (state['model_key']!=arm['model'] or state['token_policy']!=policy or state['completed_queries']!=len(physical) or not state['hooks_restored']):
            raise ValueError('Actual complete arm, checkpoint or timing coverage differs')
        if spec:
            training=json.loads((Path(spec['training_run'])/'manifest.json').read_text());description=training['arms'][spec['training_arm']]['description']
            targets=projection_targets(description,config)
            if (state['description']!=description or state['adapter_parameters']!=spec['parameters'] or state['checkpoint_sha256']!=spec['checkpoint_sha256'] or
                state['adapter_state_sha256_before']!=state['adapter_state_sha256_after'] or
                state['adapter_state_sha256_after']!=refs['training'][arm['model']]['checkpoint_state_sha256'] or
                not state['all_adapter_parameters_frozen'] or not state['arithmetic_hooks_removed'] or
                state['enabled_outputs']!=sorted(description['output_shapes']) or state['projection_targets']!=targets):
                raise ValueError('Actual frozen controller, complete enabled projection scope or cleanup differs')
            audit_affines(state,profile['profile'])
        else:
            targets={}
            if (state['enabled_outputs'] or state['projection_targets'] or state['none_identity_checks'] or
                any(k.startswith(('adapter_','checkpoint_','arithmetic','affine_','actual_affine')) for k in state)):
                raise ValueError('Native arm unexpectedly has adapter state')
        result_path=path/f'{name}.jsonl'
        if streaming_hash(result_path)!=state['results_sha256']:raise ValueError('Actual raw readouts changed')
        rows=[json.loads(line) for line in result_path.read_text().splitlines()]
        if [r['id'] for r in rows]!=list(queries):raise ValueError('Actual physical readout order incomplete')
        for row,q in zip(rows,physical):
            audit_readout(readout(row),q);audit_policy(row['policy_calls'],q,policy,targets)
            if (row['model']!=name or row['token_policy']!=policy or row['prefill']!='full' or row['regime']!='full' or
                row['registration_sha256']!=data_fingerprint(q) or row['arithmetic_profile']!=profile['profile']):
                raise ValueError('Actual full-history readout or canonical reference mislabeled')
            counts['actual_forward_receipts']+=len(row['policy_calls'])
        all_rows[name]=rows;counts['physical_readouts']+=len(rows)
        if name=='native' and any(readout(row)!=anchors[row['id']] for row in rows if row['id'] in anchors):raise ValueError('Native arm differs from anchors')
        for key in ('native_before_attach_checks','native_restore_checks'):audit_anchors(state[key],samples,anchors)
        counts['native_before_attach_replays']+=len(state['native_before_attach_checks']);counts['detached_native_replays']+=len(state['native_restore_checks'])
        if spec:
            if [p['id'] for p in state['none_identity_checks']]!=samples:raise ValueError('Actual attached none-policy controls incomplete')
            for p in state['none_identity_checks']:
                if p['readout']!=anchors[p['id']]:raise ValueError('Attached none policy is not exactly native')
                audit_policy(p['policy_calls'],queries[p['id']],'none',targets)
            counts['attached_none_policy_identities']+=len(state['none_identity_checks'])
        prefix_q=(unique_prefix_queries(physical) if name=='shared_question_timing' else
            [q for q in physical if q['id'] in safety] if policy=='question' else [])
        if m['prefix_query_ids'][name]!=[q['id'] for q in prefix_q]:raise ValueError('Registered actual prefix coverage differs')
        audit_prefix(state['prefix_checks'],prefix_q);counts['question_native_prefix_checks']+=len(prefix_q)
        if name=='shared_question_timing':counts['shared_question_all_distinct_native_prefixes']=len(prefix_q)
    for field,expected in {'physical_readouts':180 if m['smoke'] else 3360,'related_bindings':180 if m['smoke'] else 4320,
        'native_before_attach_replays':100,'detached_native_replays':100,'attached_none_policy_identities':90,
        'shared_question_all_distinct_native_prefixes':len(unique_prefix_queries(physical))}.items():
        if counts[field]!=expected:raise ValueError(f'Actual prescribed control count differs: {field}')
    ordered={name:all_rows[name] for name in ORDER}
    return {'status':'audited_unseen_shared_timing_smoke' if m['smoke'] else 'completed_and_audited_unseen_shared_timing',
        'source_run':str(path),'manifest_sha256':streaming_hash(path/'manifest.json'),
        'dataset_manifest_sha256':m['dataset_manifest_sha256'],'operational_controls_passed':True,
        'counts':counts,'model_order':order,'arithmetic_profile':profile['profile'],'source_files_verified':len(sources()),
        'metrics':{'primary_evaluated':False,'accuracy_gate':False} if m['smoke'] else summarize(ordered,cases,registry,registered),
        'prior_fresh_gates_unchanged':refs['prior_fresh_gates'],'interpretation':plan['interpretation_limits'],
        'analysis_source_sha256':streaming_hash(Path(__file__))}


def compare_observations(first,second,a,b):
    if (first['model_order']!=list(ORDER) or second['model_order']!=list(ORDER[::-1]) or
        first['selected_physical_ids']!=second['selected_physical_ids'] or len(first['selected_physical_ids'])!=18 or
        first['native_anchors']!=second['native_anchors']):raise ValueError('Both complete opposite smoke orders and native anchors required')
    count=0
    for name in ORDER:
        if ([r['id'] for r in a[name]]!=first['selected_physical_ids'] or [r['id'] for r in b[name]]!=first['selected_physical_ids'] or
            a[name]!=b[name]):raise ValueError(f'Exact complete readout or actual scope receipts differ across orders: {name}')
        count+=len(a[name]);x,y=first['models'][name],second['models'][name]
        for field in ('model_key','token_policy','prefix_checks','none_identity_checks','native_before_attach_checks','native_restore_checks',
            'projection_targets','enabled_outputs','adapter_state_sha256_before','adapter_state_sha256_after','description','arithmetic','actual_affine_calls'):
            if x.get(field)!=y.get(field):raise ValueError(f'Frozen state, prefix, scope or exact controls differ across orders: {name},{field}')
    if count!=180:raise ValueError('All180 complete readouts required per order')
    return {'exact_readouts':count,'exact_native_anchors':len(first['native_anchors']),
        'exact_question_prefix_checks':sum(len(s['prefix_checks']) for s in first['models'].values())}


def audit_equivalence(forward_audit,reverse_audit):
    manifests=[];rows=[]
    for path in (Path(forward_audit),Path(reverse_audit)):
        report=json.loads(path.read_text())
        if (report['status']!='audited_unseen_shared_timing_smoke' or report['counts']['physical_readouts']!=180 or
            streaming_hash(path.with_suffix('.source.py'))!=report['analysis_source_sha256'] or audit_run(Path(report['source_run']),True)!=report):
            raise ValueError('Both independently reconstructed180-readout operational audits required')
        run=Path(report['source_run']);manifests.append(json.loads((run/'manifest.json').read_text()))
        rows.append({n:[json.loads(line) for line in (run/f'{n}.jsonl').read_text().splitlines()] for n in ORDER})
    first,second=manifests
    if (first['reverse_order'] or not second['reverse_order'] or any(first[k]!=second[k] for k in
        ('source_sha256','dataset_manifest_sha256','references','arithmetic_profile','generation','generation_config_sha256','weights_before','independent_novelty_recheck'))):
        raise ValueError('Opposite operational orders changed frozen execution scope')
    exact=compare_observations(first,second,rows[0],rows[1])
    return {'status':'audited_unseen_shared_timing_smoke_order_equivalence','forward_audit':str(forward_audit),
        'forward_audit_sha256':streaming_hash(Path(forward_audit)),'reverse_audit':str(reverse_audit),
        'reverse_audit_sha256':streaming_hash(Path(reverse_audit)),'dataset_manifest_sha256':first['dataset_manifest_sha256'],
        'runtime_source_sha256':first['source_sha256'],'smoke_completed_utc':[m['completed_utc'] for m in manifests],
        **exact,'primary_evaluated':False,'accuracy_gate':False,'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path,nargs='?')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--allow-smoke',action='store_true')
    parser.add_argument('--equivalence',type=Path,nargs=2)
    args=parser.parse_args()
    if args.output.exists() or bool(args.run)==bool(args.equivalence):raise ValueError('Preserve audits; select exactly one run or two opposite smoke audits')
    report=audit_equivalence(*args.equivalence) if args.equivalence else audit_run(args.run,args.allow_smoke)
    write_json(args.output,report);args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:report[k] for k in ('status','counts','exact_readouts') if k in report}),flush=True)


if __name__=='__main__':main()
