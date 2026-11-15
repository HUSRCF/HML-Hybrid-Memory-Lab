"""Frozen references and unchanged original queries for single-gate controls."""
from datetime import datetime
import json
from pathlib import Path
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.oracle_write_hold_data import (sources as previous_sources,registered_inputs as previous_data,
    control_ids,selected_inputs,actual_prompt,unique_prefixes)

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'configs/gate_component_factorial_plan.json'
PROTOCOL_SHA256='25971ba4eb6a75e929ba83cf06c01fba3cb26ee475e49b477ba13db0fc670b68'
DESTINATION=Path('/media/990Pro/Code/SAS/datasets/memory_update/gate-component-factorial-v1')
MODELS=('native','shared','shared_question_trained_all','shared_question_trained_question','gdn_all_tokens')
REGIMES=('decay_full','delta_full','decay_oracle','delta_oracle')
ARCHIVED=('full','hold','oracle','hold_oracle')
UNIFIED=('full','decay_full','delta_full','hold','oracle','decay_oracle','delta_oracle','hold_oracle')
DATA_SOURCES=previous_sources()|{'configs/gate_component_factorial_plan.json','scripts/gate_component_data.py','scripts/prepare_gate_components.py'}

def sources():
    return DATA_SOURCES|{'scripts/gate_component_runtime.py','scripts/run_gate_components.py',
        'scripts/analyze_gate_components.py','scripts/gate_component_metrics.py','tests/test_gate_components.py'}

def read_plan():
    if streaming_hash(PROTOCOL)!=PROTOCOL_SHA256:raise ValueError('Registered component protocol changed')
    p=json.loads(PROTOCOL.read_text())
    if (p['model_order']!=list(MODELS) or p['regimes']!=list(REGIMES) or p['archived_regimes']!=list(ARCHIVED) or
        p['unified_order']!=list(UNIFIED) or Path(p['dataset_destination'])!=DESTINATION or
        p['counts']['new_primary_physical_readouts']!=6720 or p['counts']['archived_physical_readouts']!=6720 or
        p['data_source_files']!=183 or p['runtime_source_files']!=188 or
        any(p[k] for k in ('training','new_adapters','new_models','new_task_cases','new_environment',
            'prospective_unseen_confirmation','architecture_effectiveness_claim','can_change_prior_gates')) or not p['local_files_only']):
        raise ValueError('All5models and complete reused-cohort component design required')
    return p

def previous_inputs(plan):
    registered,cases,registry,queries,policies,_=previous_data(Path(plan['source_dataset']))
    run=Path(plan['source_run']);m=json.loads((run/'manifest.json').read_text());ap=Path(plan['source_audit']);a=json.loads(ap.read_text())
    if (m['status']!='completed' or m['smoke'] or streaming_hash(run/'manifest.json')!=plan['source_manifest_sha256'] or
        streaming_hash(ap)!=plan['source_audit_sha256'] or a['status']!='completed_and_audited_oracle_write_hold' or
        a['new_physical_readouts_audited']!=6720 or a['exact_previous_readout_replays']!=3360 or
        streaming_hash(Path(plan['source_dataset'])/'manifest.json')!=plan['source_dataset_manifest_sha256'] or
        streaming_hash(ROOT/plan['source_protocol'])!=plan['source_protocol_sha256'] or
        m['model_path']!=plan['model_path'] or m['packages']!=plan['packages'] or
        m['weights_before']!=m['weights_after'] or m['weights_after']['sha256']!=plan['native_loaded_weights_sha256'] or
        set(m['source_sha256'])!=previous_sources() or len(m['source_sha256'])!=180 or
        any(a['prior_timing_criteria_unchanged'].values()) or
        not datetime.fromisoformat(m['completed_utc'])<datetime.fromisoformat(plan['registered_utc'])):
        raise ValueError('Complete preceding results, chronology or frozen scope changed')
    for n,d in m['source_sha256'].items():
        if streaming_hash(ROOT/n)!=d or streaming_hash(run/'sources'/n)!=d:raise ValueError('Preserve every180prior source')
    previous_plan=json.loads((run/'plan.json').read_text())
    if previous_plan['models']!=plan['models']:raise ValueError('Fixed checkpoint/deployment allocation changed')
    rows={};raw={}
    for name in MODELS:
        rows[name]={};raw[name]={}
        for regime in ARCHIVED:
            path=run/f'{name}.{regime}.jsonl';digest=streaming_hash(path)
            if digest!=m['models'][name]['results_sha256'][regime]:raise ValueError('All20prior raw files must remain complete')
            rows[name][regime]=[json.loads(line) for line in path.read_text().splitlines()];raw[name][regime]=digest
    refs={**registered['references'],'previous_hold_manifest_sha256':streaming_hash(run/'manifest.json'),
        'previous_hold_audit_sha256':streaming_hash(ap),'all20_archived_raw_sha256':raw,
        'archived_origin':'completed_oracle_write_hold_primary','prior_timing_criteria_unchanged':a['prior_timing_criteria_unchanged']}
    return cases,registry,queries,policies,rows,refs

def registered_inputs(path=DESTINATION):
    path=Path(path);plan=read_plan();cases,registry,queries,policies,rows,refs=previous_inputs(plan)
    m=json.loads((path/'manifest.json').read_text())
    if (path!=DESTINATION or m['status']!='registered_before_inference' or m['training'] or m['model_inference'] or not m['previous_full_audit_reexecuted'] or
        m['protocol_sha256']!=PROTOCOL_SHA256 or m['references']!=refs or m['counts']!=plan['counts'] or
        json.loads((path/'plan.json').read_text())!=plan or m['control_query_ids']!=control_ids(cases,registry) or
        set(m['source_sha256'])!=DATA_SOURCES or m['queries_sha256']!=data_fingerprint(queries) or
        m['policies_sha256']!=data_fingerprint(policies) or
        not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(m['copy_started_utc'])<datetime.fromisoformat(m['registered_utc'])):
        raise ValueError('Frozen original data registration or chronology changed')
    for name,expected in [('cases',cases),('registry',registry),('queries',queries),('policies',policies)]:
        p=path/(name+'.json')
        if (json.loads(p.read_text())!=expected or streaming_hash(p)!=m['original_file_sha256'][name+'.json'] or
            p.read_bytes()!=(Path(plan['source_dataset'])/(name+'.json')).read_bytes()):
            raise ValueError('Original data must be byte-identical, not regenerated')
    for n,d in m['source_sha256'].items():
        if streaming_hash(ROOT/n)!=d or streaming_hash(path/'sources'/n)!=d:raise ValueError('Frozen183data sources changed')
    return m,cases,registry,queries,policies,rows
