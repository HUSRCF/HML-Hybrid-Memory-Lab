"""Unfiltered normal histories and questions for local question-phase holds."""
from datetime import datetime
import json
import re
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.version_query_data import registered_inputs as previous_inputs,selected_contexts,context_queries
from scripts.finalize_version_query import recovery_descriptor
from sas_memory.write_locality_data import streaming_hash

PROTOCOL=ROOT/'configs/question_state_freeze_plan.json'
PROTOCOL_SHA256='4a7bdf4e2e19c1a8bb6cf5c64094fdf6fc8ee550115ea8160c749bb7df266630'
DESTINATION=Path('/media/990Pro/Code/SAS/datasets/memory_update/question-state-freeze-v1')
MODEL='gdn_all_tokens'
ARMS={'normal':[],**{f'freeze_group{i}':[4*i,4*i+1,4*i+2] for i in range(6)},
    'freeze_all18':[i for i in range(24) if i%4!=3]}
NEW_SOURCES={'configs/question_state_freeze_plan.json','scripts/question_state_freeze_data.py',
    'scripts/question_state_freeze_runtime.py','scripts/prepare_question_state_freeze.py',
    'scripts/run_question_state_freeze.py','scripts/analyze_question_state_freeze.py','tests/test_question_state_freeze.py'}
SEAL=Path('/media/990Pro/Code/SAS/artifacts/memory_update/question-state-freeze-seal-20261115-01.json')


def read_plan():
    if streaming_hash(PROTOCOL)!=PROTOCOL_SHA256:raise ValueError('Registered local-hold protocol changed')
    p=json.loads(PROTOCOL.read_text())
    if (p['model']!=MODEL or {a['name']:a['layers'] for a in p['arms']}!=ARMS or
        [a['name'] for a in p['arms']]!=list(ARMS) or
        any(a['gate_mode']!=('none' if a['name']=='normal' else 'both') for a in p['arms']) or
        p['primary_target_arm']!='freeze_group3' or p['counts']['new_primary_readouts']!=6912 or
        p['counts']['archived_reference_readouts']!=864 or p['smoke']['readouts_each_order']!=432 or
        p['smoke']['accuracy_gate'] or p['input_policy']['candidate_failure_filter'] or
        not p['controls']['all864_normal_readouts_must_exactly_replay_predecessor'] or
        p['predecessor_sources_immutable']!=254 or p['planned_total_sources']!=261 or
        set(p['new_sources'])!=NEW_SOURCES or Path(p['registration_destination'])!=DESTINATION or
        any(p[k] for k in ('code_started','data_packet_generated','model_inference_started','new_training',
            'new_adapters','new_models','new_environment','oracle_evidence','donor_forward',
            'can_change_prior_gates','architecture_effectiveness_claim','fresh_confirmation'))):
        raise ValueError('Complete normal-cache eight-arm question-only diagnostic required')
    return p


def predecessor_sources():
    p=read_plan();path=Path(p['source_run']);m=json.loads((path/'manifest.json').read_text())
    if streaming_hash(path/'manifest.json')!=p['source_manifest_sha256']:raise ValueError('Published predecessor changed')
    result={}
    for folder,digests in (('sources',m['source_sha256']),
        ('serialization-correction-sources',m['serialization_correction']['source_sha256']),
        ('terminal-validation-sources',m['terminal_validation_recovery']['source_sha256'])):
        for name,digest in digests.items():
            if name in result or streaming_hash(ROOT/name)!=digest or streaming_hash(path/folder/name)!=digest:
                raise ValueError('Preserve every separately sealed254source byte')
            result[name]=digest
    if len(result)!=254:raise ValueError('Complete254source predecessor closure required')
    return result


def sources():return set(predecessor_sources())|NEW_SOURCES


def reference_inputs():
    p=read_plan();path=Path(p['source_run']);m=json.loads((path/'manifest.json').read_text())
    audit=json.loads(Path(p['source_audit']).read_text());pub=json.loads(Path(p['source_publication']).read_text())
    if (streaming_hash(path/'manifest.json')!=p['source_manifest_sha256'] or
        streaming_hash(Path(p['source_audit']))!=p['source_audit_sha256'] or
        streaming_hash(Path(p['source_publication']))!=p['source_publication_sha256'] or
        streaming_hash(Path(p['source_observations']))!=p['source_observations_sha256'] or
        audit['status']!='completed_and_audited_version_query' or audit['total_source_files_verified']!=254 or
        audit['primary_readouts_audited']!=1728 or audit['native_inference_repeated'] or
        any(audit['prior_failed_criteria_unchanged'].values()) or
        pub['status']!='published_remote_assets_roundtrip_byte_verified' or pub['release_url']!=p['source_release'] or
        pub['audit_sha256']!=p['source_audit_sha256'] or len(pub['remote_assets'])!=3 or
        not all(r['downloaded_bytes_verified'] for r in pub['remote_assets']) or
        not datetime.fromisoformat(pub['completed_utc'])<datetime.fromisoformat(p['registered_utc'])):
        raise ValueError('Completed published wording diagnostic required')
    recovery_descriptor();predecessor_sources()
    old,cases,registry,queries,legacy=previous_inputs(Path(p['source_packet']))
    if streaming_hash(Path(p['source_packet'])/'manifest.json')!=p['source_packet_manifest_sha256']:
        raise ValueError('Exact paired-wording packet required')
    state=m['models'][MODEL];raw=path/f'{MODEL}.jsonl'
    if streaming_hash(raw)!=state['results_sha256']:raise ValueError('Preserve every predecessor answer')
    rows=[json.loads(line) for line in raw.read_text().splitlines()]
    if len(rows)!=864 or [r['id'] for r in rows]!=list(queries) or any(r['reader']!='full' for r in rows):
        raise ValueError('All864existing normal Full answers required without error selection')
    refs={k:legacy[k] for k in ('legacy_ids','legacy_queries','legacy_full','runtime','prior_failed_criteria')}
    refs.update(model_state={k:state[k] for k in ('description','adapter_state_sha256_before')},
        source_caches={k:state[k] for k in ('prefixes','sources')},normal_full=rows)
    if (len(registry['contexts'])!=144 or len(queries)!=864 or
        {len(q['input_ids'])-q['ledger_tokens'] for q in queries.values()}!={24}):
        raise ValueError('All unchanged histories, roles and24token paired questions required')
    return cases,registry,queries,refs


def registered_inputs(path=DESTINATION):
    path=Path(path);p=read_plan();m=json.loads((path/'manifest.json').read_text())
    if (m['status']!='registered' or m['model_inference'] or m['training'] or
        m['protocol_sha256']!=PROTOCOL_SHA256 or m['counts']!=p['counts'] or
        set(m['source_sha256'])!=sources() or len(sources())!=261 or
        not datetime.fromisoformat(p['registered_utc'])<datetime.fromisoformat(m['created_utc'])):
        raise ValueError('Complete261source packet required before inference')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:
            raise ValueError('Sealed local-hold source changed')
    for name,digest in m['payload_sha256'].items():
        if streaming_hash(path/name)!=digest:raise ValueError('Registered payload changed')
    cases,registry,queries,refs=reference_inputs()
    for name,value in (('plan',p),('cases',cases),('registry',registry),('queries',queries),('references',refs)):
        if json.loads((path/(name+'.json')).read_text())!=value:
            raise ValueError('Independent unfiltered input reconstruction differs')
    return m,cases,registry,queries,refs


def verified_seal(packet_manifest,packet_digest):
    seal=json.loads(SEAL.read_text());log=Path(seal['full_tests_log'])
    match=re.search(r'Ran (\d+) tests in [\d.]+s\s+OK\s*$',log.read_text())
    if (seal['status']!='sealed_before_any_question_state_freeze_native_inference' or
        seal['source_sha256']!=packet_manifest['source_sha256'] or
        seal['packet_manifest_sha256']!=packet_digest or streaming_hash(log)!=seal['full_tests_log_sha256'] or
        not match or int(match[1])!=seal['full_tests_passed'] or seal['full_tests_passed']<331 or
        not datetime.fromisoformat(packet_manifest['created_utc'])<datetime.fromisoformat(seal['created_utc'])):
        raise ValueError('Complete CPU regression and exact261source seal required before native inference')
    return {'path':str(SEAL),'sha256':streaming_hash(SEAL),'created_utc':seal['created_utc'],
        'full_tests_passed':seal['full_tests_passed']}
