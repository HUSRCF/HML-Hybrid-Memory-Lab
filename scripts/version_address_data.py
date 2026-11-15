"""Unfiltered reused and fixed prospective inputs for passive record observations."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.question_state_freeze_data import registered_inputs as previous_inputs
from scripts.version_query_data import build_registry as wording_registry
from scripts.natural_load_data import build_registry as natural_registry
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash

PROTOCOL=ROOT/'configs/version_address_audit_plan.json'
PROTOCOL_SHA256='f3e1867a24b83bacbc5d5fbab0b2be682d818b27808a2f1d52082dc780da3bbe'
DESTINATION=Path('/media/990Pro/Code/SAS/datasets/memory_update/version-address-observation-v1')
MODELS=('native','gdn_all_tokens');LAYERS=(3,7,11,15,19,23);POSITIONS=(11,23)
NEW_SOURCES={'configs/version_address_audit_plan.json','scripts/version_address_data.py',
    'scripts/version_address_runtime.py','scripts/prepare_version_address.py','scripts/run_version_address.py',
    'scripts/analyze_version_address.py','tests/test_version_address.py'}
SEAL=Path('/media/990Pro/Code/SAS/artifacts/memory_update/version-address-seal-20261115-01.json')


def read_plan():
    if streaming_hash(PROTOCOL)!=PROTOCOL_SHA256:raise ValueError('Registered passive protocol changed')
    p=json.loads(PROTOCOL.read_text())
    if (p['model_order']!=list(MODELS) or p['fresh_seeds']!=list(range(82000,82008)) or
        p['observation']['attention_layers']!=list(LAYERS) or
        list(p['observation']['known_suffix_positions'].values())!=list(POSITIONS) or
        p['counts']['physical_readouts']!=3456 or p['predecessor_sources_immutable']!=261 or
        p['planned_total_sources']!=268 or set(p['new_sources'])!=NEW_SOURCES or
        p['smoke']['accuracy_gate'] or len(p['prior_packets'])!=28 or
        Path(p['registration_destination'])!=DESTINATION or any(p[k] for k in ('code_started',
        'data_packet_generated','model_inference_started','training','new_adapters','new_models','new_environment',
        'oracle_reading','donor_forward','changes_recurrent_updates','can_change_prior_gates',
        'architecture_effectiveness_claim','causal_attention_claim'))):
        raise ValueError('Complete unchanged observational design required')
    return p


def predecessor_sources():
    p=read_plan();run=Path(p['source_run']);m=json.loads((run/'manifest.json').read_text())
    if streaming_hash(run/'manifest.json')!=p['source_manifest_sha256'] or len(m['source_sha256'])!=261:
        raise ValueError('Published261source predecessor changed')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(run/'sources'/name)!=digest:
            raise ValueError('Preserve all261sealed predecessor sources')
    return m['source_sha256']


def sources():return set(predecessor_sources())|NEW_SOURCES


def prior_index():
    p=read_plan();seeds=set();ledgers=set();prompts=set()
    def visit(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('seed','base_seed') and type(v) is int:seeds.add(v)
                elif k in ('prefix_key','source_key') and isinstance(v,str):ledgers.add(v)
                elif k=='prompt_ids_sha256' and isinstance(v,str):prompts.add(v)
                if k=='prefixes' and isinstance(v,dict):
                    ledgers.update(x for x in v if re.fullmatch('[a-f0-9]{64}',x))
                if isinstance(v,(dict,list)):visit(v)
        elif isinstance(value,list):
            for v in value:
                if isinstance(v,(dict,list)):visit(v)
    for packet in p['prior_packets']:
        path=Path(packet['manifest'])
        if streaming_hash(path)!=packet['manifest_sha256']:raise ValueError('Pinned prior registration changed')
        for name,digest in packet['files'].items():
            f=path.parent/name
            if streaming_hash(f)!=digest:raise ValueError('Pinned prior input payload changed')
            visit(json.loads(f.read_text()))
    if set(p['fresh_seeds'])&seeds:raise ValueError('Fixed prospective seed was already used; never replace silently')
    return {'excluded_seeds':sorted(seeds),'excluded_prefix_hashes':sorted(ledgers),
        'excluded_prompt_hashes':sorted(prompts),'prior_packet_inventory_sha256':data_fingerprint(p['prior_packets'])}


def reference_inputs():
    p=read_plan();a=json.loads(Path(p['source_audit']).read_text());pub=json.loads(Path(p['source_publication']).read_text())
    if (streaming_hash(Path(p['source_audit']))!=p['source_audit_sha256'] or
        streaming_hash(Path(p['source_publication']))!=p['source_publication_sha256'] or
        a['status']!='completed_and_audited_question_state_freeze' or a['primary_readouts_audited']!=6912 or
        a['source_files_verified']!=261 or any(a['prior_failed_criteria_unchanged'].values()) or
        pub['status']!='published_remote_assets_roundtrip_byte_verified' or pub['release_url']!=p['source_release'] or
        pub['audit_sha256']!=p['source_audit_sha256'] or len(pub['remote_assets'])!=3 or
        not all(v['downloaded_bytes_verified'] for v in pub['remote_assets']) or
        not datetime.fromisoformat(pub['completed_utc'])<datetime.fromisoformat(p['registered_utc'])):
        raise ValueError('Complete published predecessor required')
    predecessor_sources();_,cases,registry,queries,old=previous_inputs()
    run=Path(p['wording_run']);m=json.loads((run/'manifest.json').read_text())
    if streaming_hash(run/'manifest.json')!=p['wording_manifest_sha256'] or m['status']!='completed':
        raise ValueError('Completed wording reference changed')
    refs={k:old[k] for k in ('legacy_ids','legacy_queries','legacy_full','runtime','prior_failed_criteria')}
    refs.update(normal_full={},model_states={},source_caches={})
    for name in MODELS:
        state=m['models'][name];raw=run/(name+'.jsonl')
        if streaming_hash(raw)!=state['results_sha256']:raise ValueError('Preserve all prior normal answers')
        rows=[json.loads(line) for line in raw.read_text().splitlines()]
        if len(rows)!=864 or [r['id'] for r in rows]!=list(queries):raise ValueError('All prior questions required')
        refs['normal_full'][name]=rows
        refs['model_states'][name]={k:state[k] for k in ('description','adapter_state_sha256_before') if k in state}
        refs['source_caches'][name]={k:state[k] for k in ('prefixes','sources')}
    refs.update(prior_index())
    if sum(not r['greedy_exact'] for r in refs['normal_full']['gdn_all_tokens'])!=50:
        raise ValueError('All50previous errors required')
    return cases,registry,queries,refs


def build_inputs(runtime):
    p=read_plan();cases,registry,queries,refs=reference_inputs()
    natural_plan={'seed_rule':{'start':82000,'count':8}}
    fresh_cases,fresh_registry,fresh_queries=natural_registry(runtime,natural_plan,refs)
    if sorted({c['seed'] for c in fresh_cases})!=p['fresh_seeds']:raise ValueError('Fixed seed rule changed')
    old_wording=json.loads((Path(p['wording_run'])/'plan.json').read_text())
    fresh_registry,fresh_queries=wording_registry(runtime,old_wording,fresh_registry,fresh_queries)
    combined=deepcopy(registry);combined['cohort_by_context']={c['id']:'reused' for c in registry['contexts']}
    combined['cohort_by_context'].update({c['id']:'fresh' for c in fresh_registry['contexts']})
    for field in ('prefixes',):
        if set(combined[field])&set(fresh_registry[field]):raise ValueError('Fresh prefix overlaps reused prefix')
        combined[field].update(fresh_registry[field])
    if set(queries)&set(fresh_queries):raise ValueError('Fresh question identity overlaps')
    combined['contexts']+=fresh_registry['contexts'];combined_queries={**queries,**fresh_queries}
    combined['query_order']=list(combined_queries)
    if (len(cases+fresh_cases),len(combined['contexts']),len(combined_queries))!=(32,288,1728):
        raise ValueError('Complete old and fresh input grid required')
    for c in combined['contexts']:
        if len(c['records']) not in (19,83,531) or [b['start'] for b in c['records']]!=list(range(c['prelude_tokens'],c['ledger_tokens'],14)):
            raise ValueError('Every contiguous14token record required')
        for key in c['query_ids']:
            q=combined_queries[key];known=q['input_ids'][c['ledger_tokens']:]
            if len(known)!=24 or runtime.ids('?')!=[known[11]] or runtime.ids('\n\n')!=[known[23]]:
                raise ValueError('Both registered causal observation positions required')
            if q['different_question_token_index']!=5:raise ValueError('Single semantic token control changed')
    return cases+fresh_cases,combined,combined_queries,refs


def selected_contexts(registry,smoke):
    first={cohort:min(c['seed'] for c in registry['contexts'] if registry['cohort_by_context'][c['id']]==cohort)
        for cohort in ('reused','fresh')}
    return [c for c in registry['contexts'] if not smoke or
        (c['seed']==first[registry['cohort_by_context'][c['id']]] and c['naming']=='original')]


def registered_inputs(path=DESTINATION):
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    p=read_plan();path=Path(path);m=json.loads((path/'manifest.json').read_text())
    if (m['status']!='registered' or m['model_inference'] or m['training'] or
        m['protocol_sha256']!=PROTOCOL_SHA256 or m['counts']!=p['counts'] or len(sources())!=268 or
        set(m['source_sha256'])!=sources() or not datetime.fromisoformat(p['registered_utc'])<datetime.fromisoformat(m['created_utc'])):
        raise ValueError('Complete268source input packet required')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:
            raise ValueError('Sealed observation source changed')
    for name,digest in m['payload_sha256'].items():
        if streaming_hash(path/name)!=digest:raise ValueError('Registered input changed')
    cases,registry,queries,refs=build_inputs(tokenizer_runtime(p['model_path']))
    for name,value in (('plan',p),('cases',cases),('registry',registry),('queries',queries),('references',refs)):
        if json.loads((path/(name+'.json')).read_text())!=value:raise ValueError('Independent full input reconstruction differs')
    return m,cases,registry,queries,refs


def verified_seal(packet,digest):
    seal=json.loads(SEAL.read_text());log=Path(seal['full_tests_log']);match=re.search(r'Ran (\d+) tests in [\d.]+s\s+OK\s*$',log.read_text())
    if (seal['status']!='sealed_before_version_address_native_inference' or
        seal['source_sha256']!=packet['source_sha256'] or seal['packet_manifest_sha256']!=digest or
        streaming_hash(log)!=seal['full_tests_log_sha256'] or not match or int(match[1])!=seal['full_tests_passed'] or
        seal['full_tests_passed']<343 or not datetime.fromisoformat(packet['created_utc'])<datetime.fromisoformat(seal['created_utc'])):
        raise ValueError('Complete CPU tests and source seal required before inference')
    return {'path':str(SEAL),'sha256':streaming_hash(SEAL),'created_utc':seal['created_utc'],'full_tests_passed':seal['full_tests_passed']}
