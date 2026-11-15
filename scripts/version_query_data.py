"""Registered one-token version questions on every unchanged normal history."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.natural_load_data import registered_inputs as prior_inputs, selected_contexts

PROTOCOL=ROOT/'configs/version_query_control_plan.json'
PROTOCOL_SHA256='34d0ae71adf9507bb345bcc1c051bf09e33278d26e4706f4a919a3c3b9a56807'
DESTINATION=Path('/media/990Pro/Code/SAS/datasets/memory_update/version-query-control-v1')
MODELS=('native','gdn_all_tokens');READERS=('full',)
STYLES=('state_term','event_term');ROLES=('target_current','target_initial','protected_current')
NEW_SOURCES={'configs/version_query_control_plan.json','scripts/version_query_data.py',
    'scripts/prepare_version_query.py','scripts/run_version_query.py',
    'scripts/analyze_version_query.py','tests/test_version_query.py'}


def read_plan():
    if streaming_hash(PROTOCOL)!=PROTOCOL_SHA256:
        raise ValueError('Registered one-token protocol changed')
    p=json.loads(PROTOCOL.read_text())
    if (p['model_order']!=list(MODELS) or p['question_styles']!=list(STYLES) or
        p['roles']!=list(ROLES) or p['reader_modes']!=list(READERS) or
        p['counts']['new_primary_readouts']!=1728 or p['counts']['archived_full_reference_readouts']!=1728 or
        p['counts']['paired_state_to_event_comparisons']!=864 or p['smoke']['new_readouts_each']!=108 or
        p['smoke']['accuracy_gate'] or not p['smoke']['primary_same_subset_exact_replay'] or
        p['operational_controls']['source_files_planned']!=246 or
        set(p['operational_controls']['new_sources'])!=NEW_SOURCES or
        Path(p['registration_destination'])!=DESTINATION or
        any(p[k] for k in ('training','new_adapters','new_models','new_environment','can_change_prior_gates',
            'architecture_effectiveness_claim','fresh_confirmation','new_code_started','data_packet_generated','model_inference_started'))):
        raise ValueError('Complete unfiltered one-token diagnostic required')
    return p


def predecessor_sources():
    p=read_plan();run=Path(p['source_run'])
    if streaming_hash(run/'manifest.json')!=p['source_manifest_sha256']:
        raise ValueError('Published predecessor manifest changed')
    return json.loads((run/'manifest.json').read_text())['source_sha256']


def sources():return set(predecessor_sources())|NEW_SOURCES


def reference_inputs():
    p=read_plan();run=Path(p['source_run']);m=json.loads((run/'manifest.json').read_text())
    a=json.loads(Path(p['source_audit']).read_text());pub=json.loads(Path(p['source_publication']).read_text())
    if (streaming_hash(run/'manifest.json')!=p['source_manifest_sha256'] or
        streaming_hash(Path(p['source_audit']))!=p['source_audit_sha256'] or
        streaming_hash(Path(p['source_publication']))!=p['source_publication_sha256'] or
        streaming_hash(Path(p['source_version_attribution']))!=p['source_version_attribution_sha256'] or
        a['status']!='completed_and_audited_natural_load' or a['primary_readouts_audited']!=3456 or
        a['source_files_verified']!=240 or any(a['prior_failed_criteria_unchanged'].values()) or
        m['status']!='completed' or m['weights_before']!=m['weights_after'] or not m['hooks_restored'] or
        pub['status']!='published_remote_assets_roundtrip_byte_verified' or pub['release_url']!=p['source_release'] or
        pub['target_commit']!=p['source_release_target'] or pub['audit_sha256']!=p['source_audit_sha256'] or
        len(pub['remote_assets'])!=3 or not all(v['downloaded_bytes_verified'] for v in pub['remote_assets'])):
        raise ValueError('Complete byte-verified normal-update predecessor required')
    old,cases,registry,queries,legacy=prior_inputs(Path(p['source_packet']))
    if streaming_hash(Path(p['source_packet'])/'manifest.json')!=p['source_packet_manifest_sha256'] or len(old['source_sha256'])!=240:
        raise ValueError('Complete immutable240source predecessor packet required')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(run/'sources'/name)!=digest:
            raise ValueError('All240previous measurement sources must remain unchanged')
    if m['source_sha256']!=old['source_sha256']:
        raise ValueError('Published run and registered source closure differ')
    oldplan=json.loads((run/'plan.json').read_text())
    if any(p[k]!=oldplan[k] for k in ('models','model_order','model_path','packages','generation',
        'native_loaded_weights_sha256','precision_addendum','arithmetic_profile')):
        raise ValueError('Frozen model/runtime/adapter profile changed')
    refs={**{k:legacy[k] for k in ('legacy_ids','legacy_queries','legacy_full','runtime','prior_failed_criteria')},
        'model_states':{n:{k:m['models'][n][k] for k in ('description','adapter_state_sha256_before')
            if k in m['models'][n]} for n in MODELS},
        'source_caches':{n:{k:m['models'][n][k] for k in ('prefixes','sources')} for n in MODELS},
        'archived_full':{},'archived_queries':{}}
    for model in MODELS:
        raw=run/f'{model}.jsonl'
        if streaming_hash(raw)!=m['models'][model]['results_sha256']:
            raise ValueError('Complete predecessor raw bytes changed')
        rows=[json.loads(line) for line in raw.read_text().splitlines()]
        refs['archived_full'][model]=[r for r in rows if r['reader']=='full']
        if len(rows)!=1728 or len(refs['archived_full'][model])!=864:
            raise ValueError('Retain ALL1728archived Full readouts')
        for row in refs['archived_full'][model]:
            q=queries[row['id']]
            if row['registration_sha256']!=data_fingerprint(q):raise ValueError('Predecessor query binding changed')
            refs['archived_queries'][q['id']]={k:q[k] for k in ('id','case_id','seed','naming','episode','role','question_style','answer')}
            refs['archived_queries'][q['id']]['question_tokens']=len(q['input_ids'])-q['ledger_tokens']
    return cases,registry,queries,refs


def context_queries(context,queries,smoke=False):
    # Both treatment words are required in smoke as well as the primary run.
    return [queries[k] for k in context['query_ids']]


def build_registry(runtime,p,prior_registry,prior_queries):
    registry=deepcopy(prior_registry);queries={};prior_by_role={}
    for q in prior_queries.values():
        if q['question_style']=='original':prior_by_role[(q['context_id'],q['role'])]=q
    lengths={};pair_count=0
    for context in registry['contexts']:
        context['query_ids']=[];role_queries={}
        for style in STYLES:
            for role in ROLES:
                old=prior_by_role[(context['id'],role)]
                # Entity identity is taken from the executable record supporting this role.
                entity=context['records'][old['support_event_index']]['target']
                term=p['terms'][style][role];text=p['templates'][style].format(term=term,entity=entity)
                prompt=context['ledger']+runtime.query_suffix(text);ids=runtime.ids(prompt)
                history=runtime.ids(context['ledger'])
                if ids[:len(history)]!=history or len(history)!=context['ledger_tokens']:
                    raise ValueError('Known question crossed unchanged history boundary')
                q=deepcopy(old);qid=data_fingerprint([p['profile'],context['id'],style,role])
                q.update(id=qid,question_style=style,question=text,input_ids=ids,prompt_ids_sha256=data_fingerprint(ids),
                    answer_ids=runtime.continuation_ids(prompt,old['answer']),control_term=term,source_truth_query_id=old['id'])
                if q['answer_ids']!=old['answer_ids'] or len(q['answer_ids'])!=3:
                    raise ValueError('Unchanged three-token gold target required')
                length=len(ids)-len(history);key=(role,style)
                if key in lengths and lengths[key]!=length:raise ValueError('Question lengths differ across names/contexts')
                lengths[key]=length;queries[qid]=q;context['query_ids'].append(qid);role_queries[(style,role)]=q
        for role in ROLES:
            left,right=(role_queries[(style,role)] for style in STYLES)
            if len(left['input_ids'])!=len(right['input_ids']):raise ValueError('Literal pair is not token-length matched')
            differences=[i for i,(x,y) in enumerate(zip(left['input_ids'],right['input_ids'])) if x!=y]
            if len(differences)!=1 or differences[0]<context['ledger_tokens']:
                raise ValueError('ExactlyONE known-question token must differ; no silent pair filtering')
            index=differences[0]
            for q,other in ((left,right),(right,left)):
                term_ids=runtime.ids(' '+q['control_term'])
                if len(term_ids)!=1 or q['input_ids'][index]!=term_ids[0]:
                    raise ValueError('Only registered semantic word may change its token')
                q.update(paired_query_id=other['id'],different_token_index=index,
                    different_question_token_index=index-context['ledger_tokens'])
            if any(left[k]!=right[k] for k in ('answer','answer_ids','support_event_index','history_keep_bits','source_key')):
                raise ValueError('Question pair changed truth/support/history')
            pair_count+=1
    registry['query_order']=list(queries)
    if len(registry['contexts'])!=144 or len(queries)!=864 or pair_count!=432 or len({q['prompt_ids_sha256'] for q in queries.values()})!=864:
        raise ValueError('Complete864question unfiltered paired grid required')
    return registry,queries


def registered_inputs(path=DESTINATION):
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    path=Path(path);p=read_plan();m=json.loads((path/'manifest.json').read_text())
    if (m['status']!='registered' or m['model_inference'] or m['training'] or m['fresh_confirmation'] or
        m['protocol_sha256']!=PROTOCOL_SHA256 or m['counts']!=p['counts'] or
        set(m['source_sha256'])!=sources() or len(sources())!=246 or
        not datetime.fromisoformat(p['registered_utc'])<datetime.fromisoformat(m['created_utc'])):
        raise ValueError('Complete246source packet required before inference')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:
            raise ValueError('Sealed version-query source changed')
    for name,digest in m['payload_sha256'].items():
        if streaming_hash(path/name)!=digest:raise ValueError('Registered payload bytes changed')
    cases,old_registry,old_queries,refs=reference_inputs()
    registry,queries=build_registry(tokenizer_runtime(p['model_path']),p,old_registry,old_queries)
    for name,value in (('plan',p),('cases',cases),('registry',registry),('queries',queries),('references',refs)):
        if json.loads((path/(name+'.json')).read_text())!=value:
            raise ValueError('Independent paired input reconstruction differs')
    return m,cases,registry,queries,refs
