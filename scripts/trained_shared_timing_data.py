"""Untouched 63000 cohort gated on the two completed, audited training arms."""

from datetime import datetime
import copy
import json
from pathlib import Path

from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.analyze_matched_shared_training_timing_verified import audit_run as audit_training,runtime_sources
from scripts.fresh_copy_data import make_cases,build_registry,check_novelty,prior_index,sample_ids,safety_samples,selected_inputs
from scripts.matched_shared_timing_common import ROOT,PROTOCOL,PROTOCOL_SHA256,ARMS,read_plan
from scripts.unseen_shared_timing_data import unique_prefix_queries,references as legacy_references,read_plan as legacy_plan

DESTINATION=Path('/media/990Pro/Code/SAS/datasets/memory_update/fresh-trained-shared-timing-v1')
ORDER=('native','shared','shared_question_timing','shared_ledger_timing',
    *[f'{arm}_trained_{p}' for arm in ARMS for p in ('all','question','ledger')],
    'independent','local','local_question','attention_only','gdn_all_tokens','gdn_question_only')
SHARED_MODELS=('shared',*ARMS)
DATA_SOURCES=runtime_sources()|{'scripts/trained_shared_timing_data.py','scripts/prepare_trained_shared_timing.py'}


def sources():
    return DATA_SOURCES|{'scripts/trained_shared_timing_metrics.py','scripts/run_trained_shared_timing.py',
        'scripts/analyze_trained_shared_timing.py'}


def references(plan):
    report_path=Path(plan['training_audit']);report=json.loads(report_path.read_text())
    run=Path(plan['training_output']);manifest=json.loads((run/'manifest.json').read_text())
    if (report!=audit_training(run) or report['status']!='completed_and_audited_matched_shared_training_timing' or
        report['optimizer_updates_audited']!=864 or report['checkpoints_audited']!=6 or
        report['source_files_verified']!=155 or not report['operational_controls_passed'] or
        streaming_hash(report_path.with_suffix('.source.py'))!=report['analysis_source_sha256']):
        raise ValueError('Both final432 models and independent complete training audit required before new data')
    refs=copy.deepcopy(legacy_references(legacy_plan()));models=copy.deepcopy(plan['legacy_models'])
    for arm in ARMS:
        record=report['models'][arm];state=manifest['arms'][arm];checkpoint=run/arm/'step-000432'
        metadata=checkpoint/'checkpoint.json'
        if not metadata.exists():raise ValueError('Actual final checkpoint metadata missing')
        models[arm]={'training_run':str(run),'training_arm':arm,'checkpoint':str(checkpoint/'controller.safetensors'),
            'checkpoint_sha256':record['checkpoint_sha256'],'checkpoint_metadata_sha256':streaming_hash(metadata),
            'manifest_sha256':report['manifest_sha256'],'audit':str(report_path),'audit_sha256':streaming_hash(report_path),
            'final_step':432,'token_policy':plan['arms'][arm]['token_policy'],'mode':'shared','rank':64,
            'group_index':3,'adapt_gdn_keys':True,'parameters':694272,'initial_state_sha256':state['initial_state_sha256'],
            'teacher_train_fit':record['teacher_fit']}
        refs['training'][arm]={'checkpoint_state_sha256':record['checkpoint_state_sha256'],
            'checkpoint_sha256':record['checkpoint_sha256'],'training_audit_sha256':streaming_hash(report_path)}
    refs['matched_training']={'audit_path':str(report_path),'audit_sha256':streaming_hash(report_path),
        'manifest_sha256':report['manifest_sha256'],'completed_utc':manifest['completed_utc'],
        'source_files_verified':155,'optimizer_updates_audited':864,'checkpoints_audited':6,
        'paired_initialization_sha256':report['paired_initialization_sha256']}
    if (Path(plan['confirmation']['dataset_destination'])!=DESTINATION or
        plan['confirmation']['evaluation_order']!=list(ORDER)):
        raise ValueError('Single prospective cohort and all16 fixed arms required')
    return refs,models


def recheck_prior(path,registration,runtime=None):
    runtime=runtime or tokenizer_runtime(read_plan()['model_path'])
    index=prior_index(runtime,excluded=DESTINATION)
    if index!=json.loads((Path(path)/'prior_inputs.json').read_text()) or data_fingerprint(index)!=registration['prior_inputs_sha256']:
        raise ValueError('Actual complete prior input inventory changed')
    return {'prior_inputs_sha256':data_fingerprint(index),'registered_directories':len(index['registered_directories']),
        'input_files':len(index['source_files_sha256']),'complete_prompts':len(index['complete_prompt_ids_sha256']),
        'complete_prefixes':len(index['complete_prefix_ids_sha256'])}


def registered_inputs(path=DESTINATION):
    path=Path(path);plan=read_plan();refs,models=references(plan);conf=plan['confirmation']
    manifest=json.loads((path/'manifest.json').read_text())
    cases,registry,index=[json.loads((path/f'{n}.json').read_text()) for n in ('cases','registry','prior_inputs')]
    runtime=tokenizer_runtime(plan['model_path']);physical,bindings=selected_inputs(registry)
    if (path!=DESTINATION or manifest['status']!='registered_before_inference' or manifest['training'] or manifest['model_inference'] or
        manifest['protocol_sha256']!=PROTOCOL_SHA256 or manifest['counts']!=conf['counts'] or
        manifest['references']!=refs or manifest['models']!=models or json.loads((path/'plan.json').read_text())!=plan or
        cases!=make_cases(conf) or registry!=build_registry(cases,runtime) or
        any(manifest[f'{n}_sha256']!=data_fingerprint(value) for n,value in [('cases',cases),('registry',registry),('prior_inputs',index)]) or
        manifest['freshness']!=check_novelty(cases,registry,index) or
        manifest['sample_query_ids']!=sample_ids(cases,registry) or manifest['native_prefix_sample_ids']!=safety_samples(registry) or
        manifest['all_distinct_prefix_ids']!=[q['source_key'] for q in unique_prefix_queries(physical)] or
        (len(cases),len(physical),len(bindings))!=(48,336,432) or set(manifest['source_sha256'])!=DATA_SOURCES or
        not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(refs['matched_training']['completed_utc'])<
            datetime.fromisoformat(manifest['generation_started_utc'])<datetime.fromisoformat(manifest['registered_utc'])):
        raise ValueError('Untouched cohort, training gate, all16 models or chronology changed')
    for name,digest in manifest['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Registered data source changed')
    for name,digest in index['source_files_sha256'].items():
        if streaming_hash(Path(name))!=digest:raise ValueError('Prior novelty input changed')
    recheck_prior(path,manifest,runtime)
    return manifest,plan,cases,registry
