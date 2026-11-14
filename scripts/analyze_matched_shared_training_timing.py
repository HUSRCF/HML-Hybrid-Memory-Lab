"""Independent paired-budget, checkpoint and complete-prefix training audit."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.analyze_fresh_copy_confirmation import audit_readout
from scripts.analyze_order_augmented_semantics import audit_order_evaluation,order_inputs
from scripts.analyze_paired_semantic_controllers import audit_safety,audit_checkpoint,audit_evaluation
from scripts.analyze_unseen_shared_timing import projection_targets,audit_prefix
from scripts.matched_shared_timing_common import ARMS,read_plan,references,sources,initial_fingerprint,scope_samples,query
from scripts.prepare_order_augmented_semantics import registered_augmented_dataset
from scripts.run_controller_trial import write_json
from scripts.unseen_shared_timing_runtime import audit_policy


def audit_guard(guard,arm,plan,data,registry,refs,runtime=None):
    runtime=runtime or tokenizer_runtime(plan['model_path'])
    policy=plan['arms'][arm]['token_policy'];description=refs['shared_description']
    config=json.loads((Path(plan['model_path'])/'config.json').read_text());config=config.get('text_config',config)
    targets=projection_targets(description,config);weights=guard['native_weights']
    if (guard['arm']!=arm or guard['policy']!=policy or guard['description']!=description or
        guard['enabled_outputs']!=sorted(description['output_shapes']) or
        weights['sha256']!=plan['native_loaded_weights_sha256'] or not weights['all_parameters_frozen'] or not weights['no_parameter_gradients'] or
        guard['adapter_state_before']!=guard['adapter_state_after'] or not guard['hooks_restored'] or not guard['rng_unchanged'] or
        guard['adapter_linear_profile']!='original_training_fp32_adapter_linears' or
        not guard['all_adapter_linear_weights_fp32'] or not guard['no_adapter_arithmetic_hooks'] or
        [p['index'] for p in guard['scope_checks']]!=scope_samples(data)):
        raise ValueError('Actual native weights, original training arithmetic or final frozen scope control differs')
    for item in guard['scope_checks']:
        i=item['index'];_,q=query(runtime,data['validation'][i],registry['validation'][i])
        if item['query']!=q or item['projection_targets']!=targets:raise ValueError('Actual final control input or projection coverage changed')
        audit_readout(item['readout'],q);audit_policy(item['policy_calls'],q,policy,targets)
    expected=[]
    if policy=='question':
        examples,permuted_registry=order_inputs(plan)
        for label,examples,reg in [('canonical',data['validation'],registry['validation']),('permuted',examples,permuted_registry)]:
            for i in [min(range(len(reg)),key=lambda i:(reg[i]['ledger_tokens'],i)),max(range(len(reg)),key=lambda i:(reg[i]['ledger_tokens'],i))]:
                _,q=query(runtime,examples[i],reg[i]);expected.append((label,i,q))
    if [(p['condition'],p['index']) for p in guard['complete_prefix_checks']]!=[(a,b) for a,b,_ in expected]:
        raise ValueError('Final four prescribed complete native prefixes missing')
    audit_prefix(guard['complete_prefix_checks'],[q for _,_,q in expected])


def audit_run(path,allow_smoke=False):
    path=Path(path);plan=read_plan();refs=references(plan);m=json.loads((path/'manifest.json').read_text())
    registered,data,registry=registered_augmented_dataset(Path(plan['data_source']))
    if (m['status']!='completed' or m['completed_arms']!=list(ARMS) or (m['smoke'] and not allow_smoke) or
        path!=Path(plan['training_smoke_output'] if m['smoke'] else plan['training_output']) or
        json.loads((path/'plan.json').read_text())!=plan or m['plan_sha256']!=data_fingerprint(plan) or
        json.loads((path/'reference_audits.json').read_text())!=refs or m['native_reference']!=refs['native'] or
        m['packages']!=plan['packages'] or m['model_path']!=plan['model_path'] or m['reader_policy']!='full' or
        m['data_manifest_sha256']!=plan['data_manifest_sha256'] or set(m['source_sha256'])!=sources() or
        not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(m['started_utc'])<datetime.fromisoformat(m['completed_utc'])):
        raise ValueError('Complete paired training schedule, runtime, inventory or chronology differs')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Preserve all140successful runtime sources')
    encoded=json.loads((Path(plan['data_source'])/'tokenized_train.json').read_text())
    if encoded!=json.loads((path/'tokenized_train.json').read_text()):raise ValueError('Actual optimized864examples changed')
    initial=initial_fingerprint(refs['shared_description'],plan['initialization_seed'])
    if any(initial!=plan['initial_state_sha256'][n] for n in ARMS):raise ValueError('Reconstructed paired initialization differs')
    models={}
    for arm in ARMS:
        spec=plan['arms'][arm];safety=json.loads((path/arm/'safety_check.json').read_text());audit_safety(safety,spec)
        for key in refs['shared_description']:
            if safety.get(key)!=refs['shared_description'][key]:raise ValueError('Actual safety scope differs from original shared model')
        if (set(safety['output_gradient_norms'])!=set(refs['shared_description']['output_shapes']) or
            set(safety['encoder_gradient_norms_after_one_step'])!={'shared'}):raise ValueError('Actual shared gradients omit a projection')
        if m['smoke']:
            models[arm]={'status':'one_update_safety_passed','parameters':694272};continue
        state=m['arms'][arm]
        if (state['description']!=refs['shared_description'] or state['initial_state_sha256']!=initial or
            state['completed_checkpoints']!=[144,288,432] or not state['native_restore_exact'] or
            len((path/arm/'training.jsonl').read_text().splitlines())!=432):raise ValueError('Paired trained scope or scheduled update count differs')
        records=[audit_checkpoint(path,arm,step,plan,encoded) for step in [144,288,432]]
        canonical=audit_evaluation(path,arm,plan,data,encoded,registry);permuted=audit_order_evaluation(path,arm,plan)
        if state['extra_validation']!=permuted:raise ValueError('Complete permuted evaluation changed')
        checkpoint=path/arm/'step-000432';guard=json.loads((checkpoint/'matched_timing_guard.json').read_text())
        audit_guard(guard,arm,plan,data,registry,refs)
        if guard['adapter_state_before']!=records[-1]['state_sha256']:raise ValueError('Final guard used another checkpoint')
        models[arm]={'canonical':canonical,'permuted':permuted,'teacher_fit':canonical['teacher_train_fit'],
            'checkpoint_sha256':records[-1]['controller_sha256'],'checkpoint_state_sha256':records[-1]['state_sha256'],
            'guard_sha256':streaming_hash(checkpoint/'matched_timing_guard.json'),'checkpoints_audited':3}
    return {'status':'audited_matched_shared_timing_smoke' if m['smoke'] else 'completed_and_audited_matched_shared_training_timing',
        'source_run':str(path),'manifest_sha256':streaming_hash(path/'manifest.json'),'protocol_sha256':streaming_hash(ROOT/'configs/matched_shared_training_timing_plan.json'),
        'operational_controls_passed':True,'paired_initialization_sha256':initial,'source_files_verified':len(sources()),'models':models,
        'optimizer_updates_audited':0 if m['smoke'] else 864,'checkpoints_audited':0 if m['smoke'] else 6,
        'teacher_train_records_audited':0 if m['smoke'] else 1728,'development_readouts_audited':0 if m['smoke'] else 1728,
        'prior_failed_gates_unchanged':refs['prior_failed_gates'],'previous_timing_criteria_unchanged':refs['previous_timing_criteria'],
        'unseen_confirmation_evaluated':False,'interpretation':plan['interpretation'],'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--allow-smoke',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous training audits')
    report=audit_run(args.run,args.allow_smoke);write_json(args.output,report);args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:report[k] for k in ['status','optimizer_updates_audited','checkpoints_audited','teacher_train_records_audited','development_readouts_audited']}),flush=True)


if __name__=='__main__':main()
