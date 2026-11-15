"""Reuse every registered stream; pin old endpoints and complete source caches."""
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.post_encoding_data import sources as prior_sources,registered_inputs as prior_inputs
from scripts.post_encoding_data import context_queries
from scripts.post_tail_path_runtime import CONDITIONS
from scripts.copy_readout_runtime import readout

PROTOCOL=ROOT/'configs/post_tail_path_crossover_plan.json'
PROTOCOL_SHA256='ef761441d8bef022d1096d7bc851022c6199d219d544315f4c6793e72ad2a305'
DESTINATION=Path('/media/990Pro/Code/SAS/datasets/memory_update/post-tail-path-crossover-v1')
MODELS=('native','gdn_all_tokens');READERS=('full','oracle');ENDPOINTS={'NNN':'none','HHH':'both'}
PRIMARY=tuple(v for v in CONDITIONS if v not in ENDPOINTS)
NEW_SOURCES={'configs/post_tail_path_crossover_plan.json','scripts/post_tail_path_data.py','scripts/prepare_post_tail_path.py',
 'scripts/post_tail_path_runtime.py','scripts/run_post_tail_path.py','scripts/post_tail_path_metrics.py','scripts/analyze_post_tail_path.py','tests/test_post_tail_path.py'}


def sources():return prior_sources()|NEW_SOURCES


def read_plan():
    if streaming_hash(PROTOCOL)!=PROTOCOL_SHA256:raise ValueError('Registered crossover protocol changed')
    p=json.loads(PROTOCOL.read_text())
    if (p['model_order']!=list(MODELS) or p['conditions']!=list(CONDITIONS) or p['new_primary_conditions']!=list(PRIMARY) or
        p['archived_endpoint_conditions']!=ENDPOINTS or p['counts']['new_primary_readouts']!=4608 or
        p['counts']['archived_endpoint_references']!=1536 or p['counts']['unified_factorial_readouts']!=6144 or
        p['operational_controls']['source_files_planned']!=214 or set(p['operational_controls']['new_sources'])!=NEW_SOURCES or
        p['smoke']['new_primary_readouts_each']!=288 or p['smoke']['accuracy_gate'] or
        Path(p['registration_destination'])!=DESTINATION or
        any(p[k] for k in ('training','new_adapters','new_models','new_environment','can_change_prior_gates','architecture_effectiveness_claim','significance_claim'))):
        raise ValueError('Complete frozen reused-cohort crossover scope required')
    return p


def selected_contexts(registry,smoke=False):
    return [c for c in registry['contexts'] if c['episode']!='before' and (not smoke or c['seed']==74000)]


def reference_inputs():
    p=read_plan();run=Path(p['source_run']);m=json.loads((run/'manifest.json').read_text())
    audit=json.loads(Path(p['source_audit']).read_text());pub=json.loads(Path(p['source_publication']).read_text())
    if (streaming_hash(run/'manifest.json')!=p['source_manifest_sha256'] or streaming_hash(Path(p['source_audit']))!=p['source_audit_sha256'] or
        streaming_hash(Path(p['source_publication']))!=p['source_publication_sha256'] or
        audit['status']!='completed_and_audited_post_encoding' or audit['manifest_sha256']!=p['source_manifest_sha256'] or
        audit['primary_readouts_audited']!=8160 or audit['source_caches_audited']!=680 or audit['source_files_verified']!=206 or
        m['status']!='completed' or m['weights_before']!=m['weights_after'] or not m['hooks_restored'] or
        any(audit['prior_timing_criteria_unchanged'].values()) or
        pub['status']!='published_remote_assets_roundtrip_byte_verified' or pub['release_url']!=p['source_release'] or
        pub['target_commit']!=p['source_release_target'] or len(pub['remote_assets'])!=3 or
        not all(a['downloaded_bytes_verified'] for a in pub['remote_assets']) or set(m['source_sha256'])!=prior_sources()):
        raise ValueError('Complete byte-verified original phase required')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(run/'sources'/name)!=digest:raise ValueError('All206prior sources must remain unchanged')
    dm,cases,registry,queries,legacy=prior_inputs(Path(p['source_dataset']))
    if streaming_hash(Path(p['source_dataset'])/'manifest.json')!=p['source_dataset_manifest_sha256']:raise ValueError('Reuse original complete registered data')
    previous=json.loads((run/'plan.json').read_text());contexts=selected_contexts(registry)
    if (dm['seeds']!=p['seeds'] or len(contexts)!=32 or len({c['seed'] for c in contexts})!=8 or
        any(p[k]!=previous[k] for k in ('model_path','packages','native_loaded_weights_sha256','precision_addendum')) or
        any(p['models'][n]!=previous['models'][n] for n in MODELS) or p['arithmetic_profile']!=m['arithmetic_profile']):
        raise ValueError('Keep all original cases and fixed model/checkpoint settings')
    selected_queries={key:queries[key] for c in contexts for key in c['query_ids']};endpoints={};source_caches={};prefixes={}
    for name in MODELS:
        state=m['models'][name];rp=run/f'{name}.jsonl'
        if streaming_hash(rp)!=state['results_sha256']:raise ValueError('Preserve all old raw bytes')
        rows=[json.loads(line) for line in rp.read_text().splitlines()];index={(r['id'],r['mode'],r['reader']):r for r in rows}
        endpoints[name]={label:[{'id':q['id'],'reader':reader,**readout(index[(q['id'],mode,reader)])}
            for c in contexts for q in context_queries(c,queries) for reader in READERS] for label,mode in ENDPOINTS.items()}
        source_caches[name]={c['id']:{label:state['memories'][c['id']+'.'+mode] for label,mode in ENDPOINTS.items()} for c in contexts}
        prefixes[name]=state['prefixes']
    selection={'contexts':contexts,'prefixes':registry['prefixes'],'query_order':list(selected_queries)}
    refs={'source_caches':source_caches,'prefix_caches':prefixes,'model_states':{n:{k:v for k,v in m['models'][n].items()
            if k not in ('memories','prefixes','noop_controls')} for n in MODELS},
        'runtime':{'actual_eos_token_id':m['actual_eos_token_id'],'generation_config_sha256':m['generation_config_sha256'],
            'installed_modeling_source':previous['installed_modeling_source'],'installed_modeling_source_sha256':previous['installed_modeling_source_sha256']},
        'legacy_ids':legacy['legacy_ids'],'legacy_queries':legacy['legacy_queries'],
        'legacy_full':{n:legacy['legacy_full'][n] for n in MODELS},'prior_failed_criteria':audit['prior_timing_criteria_unchanged']}
    if len(selected_queries)!=192 or sum(len(v) for model in endpoints.values() for v in model.values())!=1536:
        raise ValueError('All192questions and1536endpoint references required without filtering')
    return cases,selection,selected_queries,endpoints,refs


def registered_inputs(path=DESTINATION):
    path=Path(path);p=read_plan();m=json.loads((path/'manifest.json').read_text())
    if (m['status']!='registered' or m['model_inference'] or m['training'] or m['fresh_confirmation'] or
        m['protocol_sha256']!=PROTOCOL_SHA256 or m['counts']!=p['counts'] or set(m['source_sha256'])!=sources() or len(sources())!=214 or
        not datetime.fromisoformat(p['registered_utc'])<datetime.fromisoformat(m['created_utc'])):
        raise ValueError('Complete no-new-inference214source packet required')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Sealed crossover source changed')
    for name,digest in m['payload_sha256'].items():
        if streaming_hash(path/name)!=digest:raise ValueError('Registered packet bytes changed')
    if json.loads((path/'plan.json').read_text())!=p:raise ValueError('Registered protocol changed')
    values=reference_inputs()
    for name,value in zip(('cases','registry','queries','endpoints','references'),values):
        if json.loads((path/(name+'.json')).read_text())!=value:raise ValueError('Independent reused-data reconstruction changed')
    return (m,*values)
