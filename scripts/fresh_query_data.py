"""Fresh executable inputs and immutable222source registration provenance."""
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.post_tail_path_data import sources as prior_sources,registered_inputs as prior_inputs
from scripts.post_encoding_data import build_registry,select_seeds,selected_contexts,context_queries
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime

PROTOCOL=ROOT/'configs/fresh_question_query_mediation_plan.json'
PROTOCOL_SHA256='b99755922d42236d6077c3a0d16bcc3e8cadc61ad34d093b8f2946432bac791d'
DESTINATION=Path('/media/990Pro/Code/SAS/datasets/memory_update/fresh-question-query-mediation-v1')
MODELS=('native','gdn_all_tokens');READERS=('full','oracle');ORIGINS=('N','H');CONDITIONS=('NN','NH','HN','HH')
NEW_SOURCES={'configs/fresh_question_query_mediation_plan.json','scripts/fresh_query_data.py','scripts/prepare_fresh_query.py',
 'scripts/fresh_query_runtime.py','scripts/run_fresh_query.py','scripts/fresh_query_metrics.py','scripts/analyze_fresh_query.py','tests/test_fresh_query.py'}


def sources():return prior_sources()|NEW_SOURCES


def read_plan():
    if streaming_hash(PROTOCOL)!=PROTOCOL_SHA256:raise ValueError('Registered fresh Query protocol changed')
    p=json.loads(PROTOCOL.read_text())
    if (p['model_order']!=list(MODELS) or p['after_conditions']!=list(CONDITIONS) or p['reader_modes']!=list(READERS) or
        p['counts']['all_primary_readouts']!=3264 or p['counts']['same_source_query_replacement_replays']!=1536 or
        p['counts']['question_only_donor_capture_forwards']!=1536 or p['counts']['captured_query_tensors']!=9216 or
        p['smoke']['primary_readouts_each']!=204 or p['smoke']['accuracy_gate'] or
        p['operational_controls']['source_files_planned']!=222 or set(p['operational_controls']['new_sources'])!=NEW_SOURCES or
        Path(p['registration_destination'])!=DESTINATION or p['seed_rule']['start']!=75000 or p['seed_rule']['count']!=8 or
        any(p[k] for k in ('training','new_adapters','new_models','new_environment','can_change_prior_gates','architecture_effectiveness_claim','significance_claim','fresh_data_generated','model_inference_started'))):
        raise ValueError('Complete frozen fresh-stream Query intervention scope required')
    return p


def prior_inventory():
    p=read_plan();ip=Path(p['prior_input_inventory']);index=json.loads(ip.read_text())
    if streaming_hash(ip)!=p['prior_input_inventory_sha256'] or index['model_inference'] or len(index['source_files_sha256'])!=122:
        raise ValueError('Frozen prior-input inventory required')
    for name,digest in index['source_files_sha256'].items():
        if streaming_hash(Path(name))!=digest:raise ValueError('Prior registered input changed')
    return index


def reference_inputs():
    p=read_plan();run=Path(p['source_run']);m=json.loads((run/'manifest.json').read_text())
    audit=json.loads(Path(p['source_audit']).read_text());pub=json.loads(Path(p['source_publication']).read_text())
    if (streaming_hash(run/'manifest.json')!=p['source_manifest_sha256'] or streaming_hash(Path(p['source_audit']))!=p['source_audit_sha256'] or
        streaming_hash(Path(p['source_publication']))!=p['source_publication_sha256'] or
        audit['status']!='completed_and_audited_post_tail_path' or audit['new_primary_readouts_audited']!=4608 or
        audit['endpoint_operational_replays_audited']!=1536 or audit['source_files_verified']!=214 or
        any(audit['prior_failed_criteria_unchanged'].values()) or m['status']!='completed' or
        m['weights_before']!=m['weights_after'] or not m['hooks_restored'] or
        pub['status']!='published_remote_assets_roundtrip_byte_verified' or pub['release_url']!=p['source_release'] or
        pub['target_commit']!=p['source_release_target'] or len(pub['remote_assets'])!=3 or
        not all(v['downloaded_bytes_verified'] for v in pub['remote_assets']) or set(m['source_sha256'])!=prior_sources()):
        raise ValueError('Complete byte-verified predecessor required')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(run/'sources'/name)!=digest:raise ValueError('All214prior measurement sources must remain unchanged')
    packet,*_,refs=prior_inputs(Path(p['source_packet']))
    if streaming_hash(Path(p['source_packet'])/'manifest.json')!=p['source_packet_manifest_sha256']:raise ValueError('Frozen predecessor source packet changed')
    old=json.loads((run/'plan.json').read_text())
    if any(p[k]!=old[k] for k in ('models','model_order','model_path','packages','native_loaded_weights_sha256','precision_addendum','arithmetic_profile','generation')):
        raise ValueError('Same frozen models and runtime required')
    return {**{k:refs[k] for k in ('legacy_ids','legacy_queries','legacy_full','runtime','prior_failed_criteria')},
        'model_states':{n:{k:v for k,v in m['models'][n].items() if k not in ('sources','memories','prefixes')} for n in MODELS}}


def registered_inputs(path=DESTINATION):
    path=Path(path);p=read_plan();m=json.loads((path/'manifest.json').read_text());index=prior_inventory()
    if (m['status']!='registered' or m['model_inference'] or m['training'] or not m['fresh_streams'] or
        m['protocol_sha256']!=PROTOCOL_SHA256 or m['counts']!=p['counts'] or
        set(m['source_sha256'])!=sources() or len(sources())!=222 or
        not datetime.fromisoformat(p['registered_utc'])<datetime.fromisoformat(m['created_utc'])):
        raise ValueError('Complete fresh222source packet required before inference')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Sealed Query source changed')
    for name,digest in m['payload_sha256'].items():
        if streaming_hash(path/name)!=digest:raise ValueError('Registered input bytes changed')
    if json.loads((path/'plan.json').read_text())!=p or json.loads((path/'prior_inputs.json').read_text())!=index:
        raise ValueError('Frozen protocol/prior inventory changed')
    cases,registry,queries=build_registry(tokenizer_runtime(p['model_path']),p,index);refs=reference_inputs()
    for name,value in (('cases',cases),('registry',registry),('queries',queries),('references',refs)):
        if json.loads((path/(name+'.json')).read_text())!=value:raise ValueError('Independent fresh input reconstruction differs')
    if m['seeds']!=select_seeds(index,p):raise ValueError('No fresh-seed filtering permitted')
    return m,cases,registry,queries,refs
