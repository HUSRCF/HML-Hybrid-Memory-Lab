"""Preserve the successful safety run and audit the complete trainer inventory."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_matched_shared_training_timing import audit_guard
from scripts.analyze_order_augmented_semantics import audit_order_evaluation
from scripts.analyze_paired_semantic_controllers import audit_safety,audit_checkpoint,audit_evaluation
from scripts.matched_shared_timing_common import ARMS,read_plan,references,initial_fingerprint
from scripts.prepare_order_augmented_semantics import registered_augmented_dataset
from scripts.run_controller_trial import write_json

ADDENDUM=ROOT/'configs/matched_shared_training_timing_source_addendum.json'
ADDENDUM_SHA256='5d97214bcda7f93ef087d77af0186095b7abfdfb4ebee552161772b107d2bc18'


def registered_addendum():
    if streaming_hash(ADDENDUM)!=ADDENDUM_SHA256:raise ValueError('Pre-training inventory correction changed')
    a=json.loads(ADDENDUM.read_text());plan=read_plan();run=Path(a['original_smoke_run'])
    m=json.loads((run/'manifest.json').read_text())
    if (not a['preserve_scientific_plan'] or a['changes_model_data_loss_budget_or_hypotheses'] or
        a['scientific_protocol_sha256']!=streaming_hash(ROOT/a['scientific_protocol']) or
        run!=Path(plan['training_smoke_output']) or streaming_hash(run/'manifest.json')!=a['original_smoke_manifest_sha256'] or
        m['status']!='completed' or not m['smoke'] or m['completed_arms']!=list(ARMS) or
        m['source_sha256']!=a['original_actual_source_sha256'] or len(m['source_sha256'])!=146 or
        len(a['six_original_trainer_dependencies'])!=6 or len(a['extra_verified_runtime_files'])!=3):
        raise ValueError('Only the evidenced missing trainer inventory may be corrected')
    old=Path(plan['augmentation_reference_run']);prior=json.loads((old/'manifest.json').read_text())
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(run/'sources'/name)!=digest:
            raise ValueError('Preserve every146original live/archived safety source')
    for name,digest in a['six_original_trainer_dependencies'].items():
        if prior['source_sha256'][name]!=digest or streaming_hash(old/'sources'/name)!=digest:
            raise ValueError('Missing trainer dependency differs from original completed training')
    return a


def runtime_sources():
    a=registered_addendum();result=set(a['original_actual_source_sha256'])|set(a['extra_verified_runtime_files'])
    if len(result)!=149:raise ValueError('Complete verified training runtime must contain149files')
    return result


def audit_run(path,allow_smoke=False):
    path=Path(path);plan=read_plan();refs=references(plan);a=registered_addendum();m=json.loads((path/'manifest.json').read_text())
    registered,data,registry=registered_augmented_dataset(Path(plan['data_source']))
    expected_sources=set(a['original_actual_source_sha256']) if m['smoke'] else runtime_sources()
    if (m['status']!='completed' or m['completed_arms']!=list(ARMS) or (m['smoke'] and not allow_smoke) or
        path!=Path(plan['training_smoke_output'] if m['smoke'] else plan['training_output']) or
        json.loads((path/'plan.json').read_text())!=plan or m['plan_sha256']!=data_fingerprint(plan) or
        json.loads((path/'reference_audits.json').read_text())!=refs or m['native_reference']!=refs['native'] or
        m['packages']!=plan['packages'] or m['model_path']!=plan['model_path'] or m['reader_policy']!='full' or
        m['data_manifest_sha256']!=plan['data_manifest_sha256'] or set(m['source_sha256'])!=expected_sources or
        not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(m['started_utc'])<datetime.fromisoformat(m['completed_utc'])):
        raise ValueError('Complete paired training schedule, runtime, actual inventory or chronology differs')
    if not m['smoke'] and not datetime.fromisoformat(a['registered_utc'])<datetime.fromisoformat(m['started_utc']):
        raise ValueError('Scheduled training preceded the verified inventory addendum')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Actual/archived complete runtime source changed')
    encoded=json.loads((Path(plan['data_source'])/'tokenized_train.json').read_text())
    if encoded!=json.loads((path/'tokenized_train.json').read_text()):raise ValueError('Actual optimized864examples changed')
    initial=initial_fingerprint(refs['shared_description'],plan['initialization_seed'])
    if any(initial!=plan['initial_state_sha256'][n] for n in ARMS):raise ValueError('Reconstructed paired initialization differs')
    models={}
    for arm in ARMS:
        spec=plan['arms'][arm];safety=json.loads((path/arm/'safety_check.json').read_text());audit_safety(safety,spec)
        for key,value in refs['shared_description'].items():
            if safety.get(key)!=value:raise ValueError('Actual safety scope differs from original shared model')
        if (set(safety['output_gradient_norms'])!=set(refs['shared_description']['output_shapes']) or
            set(safety['encoder_gradient_norms_after_one_step'])!={'shared'}):raise ValueError('Shared gradients omit a projection')
        if m['smoke']:
            models[arm]={'status':'one_update_safety_passed','parameters':694272};continue
        state=m['arms'][arm]
        if (state['description']!=refs['shared_description'] or state['initial_state_sha256']!=initial or
            state['completed_checkpoints']!=[144,288,432] or not state['native_restore_exact'] or
            len((path/arm/'training.jsonl').read_text().splitlines())!=432):raise ValueError('Paired scope or scheduled update count differs')
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
        'source_run':str(path),'manifest_sha256':streaming_hash(path/'manifest.json'),'protocol_sha256':streaming_hash(ROOT/a['scientific_protocol']),
        'source_inventory_addendum_sha256':ADDENDUM_SHA256,'original_safety_sources_preserved':146,
        'operational_controls_passed':True,'paired_initialization_sha256':initial,'source_files_verified':len(expected_sources),'models':models,
        'optimizer_updates_audited':0 if m['smoke'] else 864,'checkpoints_audited':0 if m['smoke'] else 6,
        'teacher_train_records_audited':0 if m['smoke'] else 1728,'development_readouts_audited':0 if m['smoke'] else 1728,
        'prior_failed_gates_unchanged':refs['prior_failed_gates'],'previous_timing_criteria_unchanged':refs['previous_timing_criteria'],
        'unseen_confirmation_evaluated':False,'interpretation':plan['interpretation'],'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--allow-smoke',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous audits')
    report=audit_run(args.run,args.allow_smoke);write_json(args.output,report);args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:report[k] for k in ['status','source_files_verified','optimizer_updates_audited','checkpoints_audited','teacher_train_records_audited','development_readouts_audited']}),flush=True)


if __name__=='__main__':main()
