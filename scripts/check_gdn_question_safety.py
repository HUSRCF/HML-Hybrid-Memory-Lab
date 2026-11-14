"""Independently check actual one-update safety; reuse the run's deep references."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.calibration import file_hash
from sas_memory.training_data import data_fingerprint
from scripts.analyze_gdn_question_timing_control import ARM,SPEC,check_plan_pair
from scripts.analyze_paired_semantic_controllers import audit_safety
from scripts.analyze_semantic_single_path_controls import check_scope,fresh_initial_fingerprint,model_config
from scripts.run_controller_trial import write_json


def check(path):
    import torch
    torch.set_num_threads(4)
    manifest=json.loads((path/'manifest.json').read_text())
    plan=json.loads((path/'plan.json').read_text())
    references=json.loads((path/'reference_audits.json').read_text())
    component=Path(plan['component_reference_run'])
    component_manifest=json.loads((component/'manifest.json').read_text())
    check_plan_pair(plan,json.loads((component/'plan.json').read_text()),component_manifest)
    if (manifest['status']!='completed' or manifest['smoke'] is not True or manifest['completed_arms']!=[ARM] or
        manifest['reader_policy']!='full' or manifest['plan_sha256']!=data_fingerprint(plan) or
        manifest['native_reference']!=references['native'] or manifest['model_path']!=component_manifest['model_path'] or
        manifest['packages']!=component_manifest['packages'] or
        datetime.fromisoformat(plan['registered_utc'])>=datetime.fromisoformat(manifest['started_utc'])):
        raise ValueError('Incomplete or changed actual-model safety smoke')
    for directory,prefix,report_name,manifest_key,plan_key in (
        (component,'component','component','component_reference_manifest_sha256','component_reference_plan_sha256'),
        (Path(plan['native_path_reference_run']),'native_path','native_paths','native_path_reference_manifest_sha256','native_path_plan_sha256')):
        report=references[report_name]
        if (report['status']!='completed_and_audited' or report['manifest_sha256']!=plan[manifest_key] or
            file_hash(directory/'manifest.json')!=plan[manifest_key] or file_hash(directory/'plan.json')!=plan[plan_key]):
            raise ValueError('Previously completed deep reference audit/manifest changed')
    if references['component']['validation_records_audited']!=1728 or references['native_paths']['records_audited']!=1728:
        raise ValueError('Safety smoke did not run complete deep reference gates')
    for relative,digest in manifest['source_sha256'].items():
        if file_hash(path/'sources'/relative)!=digest or file_hash(ROOT/relative)!=digest:
            raise ValueError('Actual or archived safety source changed')
    dataset=Path(plan['data_source'])
    if file_hash(dataset/'manifest.json')!=plan['data_manifest_sha256'] or manifest['data_manifest_sha256']!=plan['data_manifest_sha256']:
        raise ValueError('Pinned safety dataset manifest changed')
    encoded=json.loads((path/'tokenized_train.json').read_text())
    if (encoded!=json.loads((dataset/'tokenized_train.json').read_text()) or len(encoded)!=plan['unique_training_examples'] or
        data_fingerprint([{key:item[key] for key in ('input_ids','answer_ids')} for item in encoded])!=plan['tokenized_input_answer_sha256'] or
        sum(len(item['input_ids']) for item in encoded)!=plan['optimized_input_token_budgets']['432'] or
        sum(len(item['answer_ids']) for item in encoded)!=plan['supervised_answer_token_budgets']['432']):
        raise ValueError('Actual intended optimization inputs/budgets changed')
    safety=json.loads((path/ARM/'safety_check.json').read_text())
    audit_safety(safety,SPEC)
    config=model_config(manifest['model_path'])
    check_scope(safety,SPEC,config,plan['group_index'])
    initial=fresh_initial_fingerprint(config,SPEC,plan['initialization_seed'])
    if initial!=plan['initial_state_sha256'][ARM] or initial!=references['component']['arms']['gdn_only']['initial_state_sha256']:
        raise ValueError('Reconstructed paired initial weights differ')
    return {'status':'independent_scope_prefix_safety_audited','source_run':str(path),'manifest_sha256':file_hash(path/'manifest.json'),
        'parameters':SPEC['parameters'],'paired_initial_state_sha256':initial,'native_prefix_exact_after_scratch_update':True,
        'backbone_gradients_absent':True,'intended_training_inputs_verified':len(encoded),'scheduled_training_completed':False,
        'reference_audits_reexecuted':False,'recorded_deep_reference_audits_sha256':file_hash(path/'reference_audits.json'),
        'interpretation':'Checks actual projection shapes/counts, finite gradients, zero-initial native identity and native prefix identity after one scratch update, with pinned source and intended token budgets. Reuses the complete deep reference audits performed by the smoke before model loading; does not independently re-execute those references or claim completed scheduled training.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=check(args.run)
    report['analysis_source_sha256']=file_hash(Path(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status':report['status'],'parameters':report['parameters'],'native_prefix_exact_after_scratch_update':report['native_prefix_exact_after_scratch_update']}))


if __name__=='__main__':
    main()
