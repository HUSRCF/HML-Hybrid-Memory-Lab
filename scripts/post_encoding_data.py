"""Registered fresh input construction; all write contexts precede their questions."""
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.tasks import INSTRUCTIONS,Operation
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.update_budget_data import sources as previous_sources, registered_inputs as previous_inputs
from scripts.post_encoding_tasks import make_case,question,truth,ROLES,STYLES

PROTOCOL=ROOT/'configs/post_encoding_update_plan.json'
PROTOCOL_SHA256='ccd180ff352e56064f1eb9bbd017e41c285f49cdf16ce39764992467d9a8bfd8'
DESTINATION=Path('/media/990Pro/Code/SAS/datasets/memory_update/post-encoding-updates-v1')
MODES=('none','decay','delta','both');READERS=('full','oracle')
NEW_SOURCES={'configs/post_encoding_update_plan.json','scripts/post_encoding_tasks.py','scripts/post_encoding_data.py',
 'scripts/prepare_post_encoding.py','scripts/post_encoding_runtime.py','scripts/run_post_encoding.py',
 'scripts/analyze_post_encoding.py','scripts/post_encoding_metrics.py','tests/test_post_encoding.py'}


def sources():return previous_sources()|NEW_SOURCES


def read_plan():
    if streaming_hash(PROTOCOL)!=PROTOCOL_SHA256:raise ValueError('Registered post-encoding protocol changed')
    p=json.loads(PROTOCOL.read_text())
    if (p['counts']['primary_readouts']!=8160 or p['smoke']['readouts_each']!=510 or
        p['operational_controls']['source_files']!=206 or set(p['operational_controls']['new_sources'])!=NEW_SOURCES or
        p['gate_modes']!=list(MODES) or p['reader_modes']!=list(READERS) or p['roles']!=list(ROLES) or
        p['question_styles']!=list(STYLES) or Path(p['dataset_destination'])!=DESTINATION or
        any(p[k] for k in ('training','new_adapters','new_models','new_environment','can_change_prior_gates','architecture_effectiveness_claim'))):
        raise ValueError('Complete fixed post-encoding scope required')
    return p


def reference_inputs():
    p=read_plan();run=Path(p['source_run']);m=json.loads((run/'manifest.json').read_text());ap=Path(p['source_audit']);a=json.loads(ap.read_text())
    if (streaming_hash(run/'manifest.json')!=p['source_manifest_sha256'] or streaming_hash(ap)!=p['source_audit_sha256'] or
        m['status']!='completed' or a['status']!='completed_and_audited_update_budget' or
        a['new_physical_readouts_audited']!=13440 or a['source_files_verified']!=197 or
        a['manifest_sha256']!=p['source_manifest_sha256'] or set(m['source_sha256'])!=previous_sources() or
        m['weights_before']!=m['weights_after'] or not m['hooks_restored'] or any(a['prior_timing_criteria_unchanged'].values())):
        raise ValueError('Completed published predecessor audit changed')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(run/'sources'/name)!=digest:
            raise ValueError('Preserve all197 prior measurement sources')
    old,cases,registry,queries,policies,archived=previous_inputs()
    oldplan=json.loads((ROOT/'configs/update_budget_controls_plan.json').read_text())
    if any(p[k]!=oldplan[k] for k in ('models','model_order','model_path','packages','native_loaded_weights_sha256','precision_addendum','generation')):
        raise ValueError('Fixed models/checkpoints/arithmetic changed')
    ids=old['control_query_ids']
    if len(ids)!=12:raise ValueError('All12fixed legacy anchors required')
    return {'predecessor_manifest':m,'predecessor_audit':a,'legacy_ids':ids,'legacy_queries':{k:queries[k] for k in ids},
        'legacy_full':{n:{r['id']:{k:r[k] for k in ('answer','answer_nll','greedy_exact','generated','query_tokens')}
            for r in archived[n]['full'] if r['id'] in ids} for n in p['model_order']}}


def select_seeds(index,plan):
    known=set(index['seeds']);seed=plan['seed_rule']['start'];result=[]
    while len(result)<plan['seed_rule']['count']:
        if seed not in known:result.append(seed)
        seed+=1
    return result


def spans(runtime,prelude,events):
    text=prelude;ids=runtime.ids(text);result=[]
    for i,event in enumerate(events):
        addition=Operation(**event).text();new=runtime.ids(text+addition)
        if new[:len(ids)]!=ids or len(new)<=len(ids):raise ValueError('Record token boundary crosses cached prefix')
        result.append({'index':i,'start':len(ids),'stop':len(new),'kind':event['kind'],'target':event['target']})
        text+=addition;ids=new
    return text,ids,result


def build_registry(runtime,plan,index):
    cases=[make_case(seed) for seed in select_seeds(index,plan)];contexts=[];queries={};prefixes={}
    old_prefixes=set(index['complete_prefix_ids_sha256']);old_prompts=set(index['complete_prompt_ids_sha256'])
    old_ids=set(index['identifiers']);prelude=runtime.prompt_start+INSTRUCTIONS;prelude_tokens=len(runtime.ids(prelude))
    for case in cases:
        if case['id'] in old_ids:raise ValueError('New case identity overlaps prior registered inputs')
        prefix,prefix_ids,_=spans(runtime,prelude,case['prefix_events']);prefix_key=data_fingerprint(prefix_ids)
        if prefix_key in old_prefixes or prefix_key in prefixes:raise ValueError('Fresh base prefix is not novel')
        prefixes[prefix_key]={'case_id':case['id'],'text':prefix,'input_ids':prefix_ids,'tokens':len(prefix_ids)}
        paired={}
        for episode in case['episodes']:
            events=case['prefix_events']+episode['tail'];ledger,ids,blocks=spans(runtime,prelude,events)
            if ids[:len(prefix_ids)]!=prefix_ids:raise ValueError('Shared normally encoded prefix changed')
            key=case['id']+'.'+episode['id'];source_key=data_fingerprint(ids)
            if source_key in old_prefixes:raise ValueError('New ledger overlaps prior exact input')
            context={'id':key,'case_id':case['id'],'seed':case['seed'],'episode':episode['id'],'kind':episode['kind'],
                'tail_records':episode['record_count'],'prefix_key':prefix_key,'prefix_text':prefix,'prefix_tokens':len(prefix_ids),
                'tail_text':ledger[len(prefix):],'tail_tokens':len(ids)-len(prefix_ids),'ledger':ledger,'ledger_tokens':len(ids),
                'source_key':source_key,'prelude_tokens':prelude_tokens,'records':blocks,'query_ids':[]}
            for style in STYLES:
                for role in ROLES:
                    qid=data_fingerprint([key,style,role]);q=question(case,role,style);answer,support=truth(case,episode,role)
                    prompt=ledger+runtime.query_suffix(q);prompt_ids=runtime.ids(prompt)
                    if prompt_ids[:len(ids)]!=ids:raise ValueError('Question token boundary changed ledger')
                    prompt_key=data_fingerprint(prompt_ids)
                    if prompt_key in old_prompts:raise ValueError('Fresh exact prompt overlaps registered prior input')
                    keep=['0']*len(ids);keep[:prelude_tokens]=['1']*prelude_tokens
                    b=blocks[support];keep[b['start']:b['stop']]=['1']*(b['stop']-b['start'])
                    answer_ids=runtime.continuation_ids(prompt,answer)
                    if not answer_ids:raise ValueError('Nonempty gold-only NLL target required')
                    queries[qid]={'id':qid,'context_id':key,'case_id':case['id'],'seed':case['seed'],'episode':episode['id'],
                        'kind':episode['kind'],'tail_records':episode['record_count'],'role':role,'question_style':style,
                        'question':q,'answer':answer,'support_event_index':support,'prefix_evidence':support<9,
                        'history_keep_bits':''.join(keep),'prelude_tokens':prelude_tokens,'ledger_tokens':len(ids),
                        'input_ids':prompt_ids,'prompt_ids_sha256':prompt_key,'source_key':source_key,'answer_ids':answer_ids}
                    context['query_ids'].append(qid)
            if episode['record_count']:
                count=episode['record_count'];signature=(len(ids),context['tail_tokens'],[(b['start'],b['stop']) for b in blocks])
                if count in paired and paired[count]!=signature:raise ValueError('Related/unrelated tokens or record positions differ')
                paired[count]=signature
            contexts.append(context)
    if (len(cases)!=8 or len(prefixes)!=8 or len(contexts)!=40 or len(queries)!=240 or
        len({q['prompt_ids_sha256'] for q in queries.values()})!=240):raise ValueError('Full unique fresh task grid required')
    return cases,{'prefixes':prefixes,'contexts':contexts,'query_order':list(queries)},queries


def selected_contexts(registry,smoke=False):
    first=registry['contexts'][0]['seed']
    return [c for c in registry['contexts'] if not smoke or c['seed']==first]


def context_queries(context,queries,smoke=False):
    return [queries[k] for k in context['query_ids'] if not smoke or queries[k]['question_style']=='original']


def registered_inputs(path=DESTINATION):
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    path=Path(path);plan=read_plan();m=json.loads((path/'manifest.json').read_text());index=json.loads((path/'prior_inputs.json').read_text())
    if m['status']!='registered' or m['model_inference'] or m['training'] or m['protocol_sha256']!=PROTOCOL_SHA256:
        raise ValueError('Immutable no-model-inference dataset registration required')
    if not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(m['created_utc']):raise ValueError('Protocol must precede fresh data')
    for name,digest in m['payload_sha256'].items():
        if streaming_hash(path/name)!=digest:raise ValueError('Registered data bytes changed')
    if json.loads((path/'plan.json').read_text())!=plan or streaming_hash(Path(plan['prior_input_inventory']))!=m['payload_sha256']['prior_inputs.json']:
        raise ValueError('Protocol or frozen prior-input inventory changed')
    for name,digest in index['source_files_sha256'].items():
        if streaming_hash(Path(name))!=digest:raise ValueError('Prior registered inputs changed')
    if set(m['source_sha256'])!=sources() or len(sources())!=206:raise ValueError('Complete206source closure required')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Sealed post-encoding source changed')
    cases,registry,queries=build_registry(tokenizer_runtime(plan['model_path']),plan,index)
    if any(json.loads((path/(name+'.json')).read_text())!=value for name,value in (('cases',cases),('registry',registry),('queries',queries))):
        raise ValueError('Independent fresh input reconstruction differs')
    if m['seeds']!=select_seeds(index,plan) or m['counts']!=plan['counts']:raise ValueError('Fixed fresh data grid changed')
    return m,cases,registry,queries,reference_inputs()
