"""One fixed prospective cohort and unchanged checkpoint/source provenance."""

from datetime import datetime
import json
from pathlib import Path

from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.fresh_copy_data import (registered_inputs as fresh_inputs,make_cases,build_registry,
    check_novelty,prior_index,sample_ids,safety_samples,selected_inputs)
from scripts.state_continuation_controls import sources as state_sources

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'configs/unseen_shared_timing_plan.json'
PROTOCOL_SHA256='1fe0317cb6ab1563d16ed44a1fa130dcb42453b7d96606bb6ba075d80c5b316a'
DESTINATION=Path('/media/990Pro/Code/SAS/datasets/memory_update/fresh-shared-timing-v1')
NATURAL=('native','shared','independent','local','local_question','attention_only','gdn_all_tokens','gdn_question_only')
ORDER=('native','shared','shared_question_timing','shared_ledger_timing',*NATURAL[2:])
DATA_SOURCES=state_sources()|{'configs/unseen_shared_timing_plan.json',
    'scripts/unseen_shared_timing_data.py','scripts/prepare_unseen_shared_timing.py'}


def validate_plan(plan):
    expected={name:{'model':name,'token_policy':plan['models'][name]['token_policy'] if plan['models'][name] else 'none',
        'new_training':False} for name in NATURAL}
    expected.update(shared_question_timing={'model':'shared','token_policy':'question','new_training':False},
        shared_ledger_timing={'model':'shared','token_policy':'ledger','new_training':False})
    if (plan['status']!='registered_before_unseen_shared_timing_data_generation_or_inference' or
        not plan['prospective_unseen_streams'] or not plan['local_files_only'] or
        any(plan[k] for k in ('architecture_effectiveness_claim','explicit_semantic_controller_implemented',
            'can_change_prior_gates','new_training','new_models','new_environment')) or
        list(plan['models'])!=list(NATURAL) or plan['evaluation_arms']!=expected or plan['evaluation_order']!=list(ORDER) or
        plan['fresh_seeds']!=list(range(62000,62008)) or plan['dependency_depths']!=[1,2,3] or
        plan['question_styles']!=['original','entity_first'] or plan['distractor_updates_by_seed_index']!=[8,64]*4 or
        Path(plan['dataset_destination'])!=DESTINATION or plan['preserved_runtime_source_count']!=129 or
        plan['counts']!={'base_streams':8,'related_cases':48,'physical_queries':336,'related_bindings':432,
            'evaluation_arms':10,'primary_physical_full_answer_readouts':3360,'primary_related_bindings':4320,
            'new_checkpoints':0,'new_training_steps':0} or
        plan['operational_smokes']['physical_readouts_per_order']!=180):
        raise ValueError('Fixed unseen cohort, complete natural baselines or timing scope changed')


def read_plan():
    if streaming_hash(PROTOCOL)!=PROTOCOL_SHA256:raise ValueError('Pre-generation registration changed')
    plan=json.loads(PROTOCOL.read_text());validate_plan(plan);return plan


def references(plan):
    run=Path(plan['completed_state_run']);manifest=json.loads((run/'manifest.json').read_text())
    audit_path=Path(plan['completed_state_audit']);audit=json.loads(audit_path.read_text())
    old=json.loads(Path(plan['prior_fresh_plan']).read_text())
    fresh=json.loads(Path(plan['prior_fresh_audit']).read_text())
    if (streaming_hash(run/'manifest.json')!=plan['completed_state_manifest_sha256'] or
        streaming_hash(audit_path)!=plan['completed_state_audit_sha256'] or manifest['status']!='completed' or
        not audit['operational_controls_passed'] or audit['counts']['cached_continuation_readouts']!=20 or
        audit['manifest_sha256']!=plan['completed_state_manifest_sha256'] or
        streaming_hash(Path(plan['completed_state_plan']))!=plan['completed_state_plan_sha256'] or
        streaming_hash(Path(plan['prior_fresh_plan']))!=plan['prior_fresh_plan_sha256'] or
        streaming_hash(Path(plan['prior_fresh_audit']))!=plan['prior_fresh_audit_sha256'] or
        streaming_hash(ROOT/plan['precision_addendum'])!=plan['precision_addendum_sha256'] or
        any(plan[k]!=old[k] for k in ('models','model_path','packages','native_loaded_weights_sha256')) or
        set(manifest['source_sha256'])!=state_sources() or len(manifest['source_sha256'])!=129 or
        not datetime.fromisoformat(manifest['completed_utc'])<datetime.fromisoformat(plan['registered_utc'])):
        raise ValueError('Completed source evidence, arithmetic, checkpoints or chronology changed')
    for name,digest in manifest['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(run/'sources'/name)!=digest:
            raise ValueError('Preserve all129sealed source files')
    previous,source_plan,_,_,_=fresh_inputs()
    if source_plan['models']!=plan['models'] or fresh['dataset_manifest_sha256']!=streaming_hash(Path(old['dataset_destination'])/'manifest.json'):
        raise ValueError('Actual first-fresh registration or fixed training family changed')
    for name,spec in plan['models'].items():
        if spec and streaming_hash(Path(spec['checkpoint']))!=spec['checkpoint_sha256']:
            raise ValueError('Frozen controller checkpoint changed')
    prior_plan=json.loads(Path(plan['completed_state_plan']).read_text())
    installed=Path(prior_plan['installed_modeling_source']);source_hash=manifest['installed_modeling_source_sha256']
    if streaming_hash(installed)!=source_hash or streaming_hash(run/'installed_modeling_source.py')!=source_hash:
        raise ValueError('Installed native model implementation changed')
    return {'training':previous['references']['training'],
        'completed_state_audit_sha256':plan['completed_state_audit_sha256'],
        'prior_fresh_audit_sha256':plan['prior_fresh_audit_sha256'],
        'prior_fresh_gates':{k:fresh['metrics'][k] for k in ('generic_shared_controller','frozen_prefix_sufficiency','whole_record_necessity')},
        'installed_modeling_source':str(installed),'installed_modeling_source_sha256':source_hash,
        'actual_eos_token_id':manifest['actual_eos_token_id'],
        'generation_config_sha256':manifest['generation_config_sha256']}


def unique_prefix_queries(physical):
    seen=set();result=[]
    for query in physical:
        if query['source_key'] not in seen:seen.add(query['source_key']);result.append(query)
    return result


def recheck_prior(path,registration,runtime=None):
    runtime=runtime or tokenizer_runtime(read_plan()['model_path'])
    index=prior_index(runtime,excluded=DESTINATION)
    if index!=json.loads((Path(path)/'prior_inputs.json').read_text()) or data_fingerprint(index)!=registration['prior_inputs_sha256']:
        raise ValueError('Independent actual prior-input novelty reconstruction changed')
    return {'prior_inputs_sha256':data_fingerprint(index),'registered_directories':len(index['registered_directories']),
        'input_files':len(index['source_files_sha256']),'complete_prompts':len(index['complete_prompt_ids_sha256']),
        'complete_prefixes':len(index['complete_prefix_ids_sha256'])}


def registered_inputs(path=DESTINATION):
    path=Path(path);plan=read_plan();refs=references(plan)
    manifest=json.loads((path/'manifest.json').read_text())
    cases,registry,index=[json.loads((path/f'{n}.json').read_text()) for n in ('cases','registry','prior_inputs')]
    runtime=tokenizer_runtime(plan['model_path']);physical,bindings=selected_inputs(registry)
    if (path!=DESTINATION or manifest['status']!='registered_before_inference' or manifest['training'] or manifest['model_inference'] or
        manifest['protocol_sha256']!=PROTOCOL_SHA256 or manifest['counts']!=plan['counts'] or
        manifest['references']!=refs or json.loads((path/'plan.json').read_text())!=plan or
        cases!=make_cases(plan) or registry!=build_registry(cases,runtime) or
        any(manifest[f'{n}_sha256']!=data_fingerprint(value) for n,value in [('cases',cases),('registry',registry),('prior_inputs',index)]) or
        manifest['freshness']!=check_novelty(cases,registry,index) or
        manifest['sample_query_ids']!=sample_ids(cases,registry) or manifest['native_prefix_sample_ids']!=safety_samples(registry) or
        manifest['all_distinct_prefix_ids']!=[q['source_key'] for q in unique_prefix_queries(physical)] or
        (len(cases),len(physical),len(bindings))!=(48,336,432) or set(manifest['source_sha256'])!=DATA_SOURCES or
        not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(manifest['generation_started_utc'])<datetime.fromisoformat(manifest['registered_utc'])):
        raise ValueError('Frozen prospective data, truth, complete registry or chronology changed')
    for name,digest in manifest['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Preserve registered data sources')
    for name,digest in index['source_files_sha256'].items():
        if streaming_hash(Path(name))!=digest:raise ValueError('Prior novelty inputs changed')
    recheck_prior(path,manifest,runtime)
    return manifest,plan,cases,registry
