"""Whole-record write budgets with independent, unchanged semantic KV reading."""
from collections import Counter,defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from sas_memory.tasks import Operation,INSTRUCTIONS
from scripts.gate_component_data import (sources as previous_sources,registered_inputs as previous_data,
    MODELS,control_ids,selected_inputs,unique_prefixes,actual_prompt)
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'configs/update_budget_controls_plan.json'
PROTOCOL_SHA256='46bb14453d6f67fa58bcafaba0b78232501ff3b86ce18bbc8c0b9b3a0f20da71'
DESTINATION=Path('/media/990Pro/Code/SAS/datasets/memory_update/update-budget-controls-v1')
FAMILIES=('random0','random1','early','recent')
REGIMES=tuple(f'{f}_{r}' for f in FAMILIES for r in ('full','oracle'))
ARCHIVED=('full','delta_full','oracle','delta_oracle')
UNIFIED=('full','delta_full','random0_full','random1_full','early_full','recent_full',
    'oracle','delta_oracle','random0_oracle','random1_oracle','early_oracle','recent_oracle')
DATA_SOURCES=previous_sources()|{'configs/gate_component_kernel_reference_addendum.json',
    'configs/update_budget_controls_plan.json','scripts/update_budget_data.py','scripts/prepare_update_budget.py'}

def sources():
    return DATA_SOURCES|{'scripts/update_budget_runtime.py','scripts/run_update_budget.py',
        'scripts/analyze_update_budget.py','scripts/update_budget_metrics.py','tests/test_update_budget.py'}

def read_plan():
    if streaming_hash(PROTOCOL)!=PROTOCOL_SHA256:raise ValueError('Registered budget protocol changed')
    p=json.loads(PROTOCOL.read_text())
    if (p['model_order']!=list(MODELS) or p['families']!=list(FAMILIES) or p['regimes']!=list(REGIMES) or
        p['archived_regimes']!=list(ARCHIVED) or p['unified_order']!=list(UNIFIED) or
        Path(p['dataset_destination'])!=DESTINATION or p['data_source_files']!=192 or p['runtime_source_files']!=197 or
        p['counts']['new_primary_physical_readouts']!=13440 or p['counts']['archived_physical_readouts']!=6720 or
        p['random_hash_seeds']!={'random0':41001,'random1':41002} or
        any(p[k] for k in ('training','new_adapters','new_models','new_task_cases','new_environment',
            'prospective_unseen_confirmation','architecture_effectiveness_claim','can_change_prior_gates')) or not p['local_files_only']):
        raise ValueError('Complete all5configuration matched-record design required')
    return p

def previous_inputs(plan):
    registered,cases,registry,queries,oracle,older=previous_data(Path(plan['source_dataset']))
    run=Path(plan['source_run']);m=json.loads((run/'manifest.json').read_text());ap=Path(plan['source_audit']);a=json.loads(ap.read_text())
    if (m['status']!='completed' or m['smoke'] or streaming_hash(run/'manifest.json')!=plan['source_manifest_sha256'] or
        streaming_hash(ap)!=plan['source_audit_sha256'] or a['status']!='completed_and_audited_gate_components' or
        a['new_physical_readouts_audited']!=6720 or a['archived_reference_readouts_verified']!=6720 or
        streaming_hash(Path(plan['source_dataset'])/'manifest.json')!=plan['source_dataset_manifest_sha256'] or
        streaming_hash(ROOT/plan['source_protocol'])!=plan['source_protocol_sha256'] or
        streaming_hash(ROOT/plan['kernel_validation_addendum'])!=plan['kernel_validation_addendum_sha256'] or
        m['model_path']!=plan['model_path'] or m['packages']!=plan['packages'] or
        m['weights_before']!=m['weights_after'] or m['weights_after']['sha256']!=plan['native_loaded_weights_sha256'] or
        set(m['source_sha256'])!=previous_sources() or len(m['source_sha256'])!=188 or
        a['analysis_source_sha256']!=streaming_hash(ap.with_suffix('.source.py')) or
        any(a['prior_timing_criteria_unchanged'].values()) or
        not datetime.fromisoformat(m['completed_utc'])<datetime.fromisoformat(plan['registered_utc'])):
        raise ValueError('Preserve complete previous component evidence and chronology')
    for n,h in m['source_sha256'].items():
        if streaming_hash(ROOT/n)!=h or streaming_hash(run/'sources'/n)!=h:raise ValueError('Preserve every188completed measurement source')
    if json.loads((run/'plan.json').read_text())['models']!=plan['models']:raise ValueError('Fixed checkpoints/deployments changed')
    all_rows={};raw={};references={}
    for name in MODELS:
        all_rows[name]=dict(older[name]);raw[name]={}
        for regime,h in m['models'][name]['results_sha256'].items():
            path=run/f'{name}.{regime}.jsonl'
            if streaming_hash(path)!=h:raise ValueError('All20component raw files required')
            all_rows[name][regime]=[json.loads(s) for s in path.read_text().splitlines()];raw[name][regime]=h
        references[name]={r:all_rows[name][r] for r in ARCHIVED}
    refs={**registered['references'],'previous_component_manifest_sha256':streaming_hash(run/'manifest.json'),
        'previous_component_audit_sha256':streaming_hash(ap),'all20_component_raw_sha256':raw,
        'kernel_validation_addendum_sha256':plan['kernel_validation_addendum_sha256'],
        'prior_timing_criteria_unchanged':a['prior_timing_criteria_unchanged']}
    return cases,registry,queries,oracle,references,refs

def record_layout(q,runtime):
    texts=[Operation(**e).text() for e in q['events']];prelude=runtime.prompt_start+INSTRUCTIONS
    if q['ledger_text']!=prelude+''.join(texts):raise ValueError('Original record text changed')
    enc=runtime.tokenizer(q['ledger_text'],add_special_tokens=False,return_offsets_mapping=True)
    if enc['input_ids']!=q['prompt_ids'][:q['ledger_tokens']]:raise ValueError('Original prefix token IDs changed')
    records=[];start=len(prelude)
    for i,text in enumerate(texts):
        end=start+len(text);indices=[j for j,(a,b) in enumerate(enc['offset_mapping']) if a<b and a<end and b>start]
        if not indices or indices!=list(range(indices[0],indices[-1]+1)):raise ValueError('Noncontiguous record tokens')
        records.append({'index':i,'kind':q['events'][i]['kind'],'start':indices[0],'stop':indices[-1]+1,'length':len(indices)})
        start=end
    if [j for b in records for j in range(b['start'],b['stop'])]!=list(range(q['prelude_tokens'],q['ledger_tokens'])):
        raise ValueError('Record partition crosses original token boundaries')
    if any((b['kind'],b['length']) not in (('SET',17),('COPY',21)) for b in records):
        raise ValueError('Registered SET17/COPY21 inventory mismatch; do not change matching')
    return records

def choose_records(records,budget,source_key,family):
    """Only structural descriptors, prefix identity and count signature enter ranking."""
    if family not in FAMILIES:raise ValueError('Fixed random0/random1/early/recent families required')
    buckets=defaultdict(list)
    for b in records:buckets[(b['kind'],b['length'])].append(b['index'])
    selected=[]
    for (kind,length),count in sorted(budget.items()):
        if type(count) is not int or count<0 or count>len(buckets[kind,length]):raise ValueError('Infeasible whole-record budget')
        indices=buckets[kind,length]
        if family.startswith('random'):
            seed={'random0':41001,'random1':41002}[family]
            def rank(i):
                value=json.dumps([seed,source_key,kind,length,i],separators=(',',':'))
                return hashlib.sha256(value.encode()).hexdigest(),i
            ordered=sorted(indices,key=rank)
        else:ordered=sorted(indices,reverse=family=='recent')
        selected.extend(ordered[:count])
    return sorted(selected)

def build_policies(queries,oracle,model_path):
    runtime=tokenizer_runtime(model_path);cache={};groups={};result={};inventory=Counter();collisions=Counter()
    for key,q in queries.items():
        if q['source_key'] not in cache:cache[q['source_key']]=record_layout(q,runtime)
        records=cache[q['source_key']];original=oracle[key];indices=set(original['oracle_event_indices'])
        budget=Counter((b['kind'],b['length']) for b in records if b['index'] in indices)
        if not budget or sum(budget.values())!=len(indices):raise ValueError('Original whole-record support invalid')
        bits=original['history_keep_bits']
        if any(set(bits[b['start']:b['stop']])!={('1' if b['index'] in indices else '0')} for b in records):
            raise ValueError('Oracle must retain or hold each complete record')
        signature=[{'kind':k,'length':length,'count':count} for (k,length),count in sorted(budget.items())]
        selected={};write={};overlap={}
        for family in FAMILIES:
            chosen=choose_records(records,budget,q['source_key'],family);chosen_set=set(chosen)
            keep=['1']*q['ledger_tokens']
            for b in records:
                if b['index'] not in chosen_set:keep[b['start']:b['stop']]=['0']*b['length']
            keep=''.join(keep)
            if keep.count('0')!=original['held_token_count']:raise ValueError('Exact permitted-write token budget mismatch')
            group=(q['source_key'],data_fingerprint(signature),family)
            if group in groups and groups[group]!=keep:raise ValueError('Question/style-dependent control mask forbidden within prefix/budget')
            groups[group]=keep;write[family]=keep;selected[family]=chosen
            exact=keep==bits;collisions[family]+=exact
            overlap[family]={'support_records_retained':len(chosen_set&indices),'support_record_count':len(indices),
                'support_tokens_retained':sum(b['length'] for b in records if b['index'] in chosen_set&indices),
                'support_token_count':sum(length*count for (_,length),count in budget.items()),'exact_oracle_write_mask':exact}
        for b in records:inventory[b['kind'],b['length']]+=1
        result[key]={'id':key,'source_key':q['source_key'],'ledger_tokens':q['ledger_tokens'],'prelude_tokens':q['prelude_tokens'],
            'history_keep_bits':bits,'held_token_count':original['held_token_count'],'oracle_event_indices':sorted(indices),
            'budget_signature':signature,'record_blocks':records,'write_keep_bits':write,'selected_event_indices':selected,'support_overlap':overlap}
    summary={'physical_queries':len(result),'unique_prefixes':len(cache),'prefix_budget_groups':len(groups)//4,
        'prefix_budget_family_groups':len(groups),'record_kind_token_lengths':{f'{k}:{n}':v for (k,n),v in sorted(inventory.items())},
        'accidental_oracle_mask_collisions':{f:collisions[f] for f in FAMILIES},'all_controls_prefix_budget_consistent':True,
        'one_global_query_independent_policy_tested':False,'model_inference':False}
    return result,summary

def registered_inputs(path=DESTINATION):
    path=Path(path);plan=read_plan();cases,registry,queries,oracle,references,refs=previous_inputs(plan)
    policies,summary=build_policies(queries,oracle,plan['model_path']);m=json.loads((path/'manifest.json').read_text())
    if (path!=DESTINATION or m['status']!='registered_before_inference' or m['training'] or m['model_inference'] or
        not m['previous_full_audit_reexecuted'] or m['protocol_sha256']!=PROTOCOL_SHA256 or m['references']!=refs or
        m['counts']!=plan['counts'] or json.loads((path/'plan.json').read_text())!=plan or
        m['control_query_ids']!=control_ids(cases,registry) or set(m['source_sha256'])!=DATA_SOURCES or
        m['queries_sha256']!=data_fingerprint(queries) or m['policies_sha256']!=data_fingerprint(policies) or
        json.loads((path/'policies.json').read_text())!=policies or json.loads((path/'mask_summary.json').read_text())!=summary or
        summary['unique_prefixes']!=56 or summary['prefix_budget_groups']!=104 or
        not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(m['generation_started_utc'])<datetime.fromisoformat(m['registered_utc'])):
        raise ValueError('Registered complete matching, source closure or chronology changed')
    for name,expected in (('cases',cases),('registry',registry),('queries',queries),('oracle_policies',oracle)):
        p=path/(name+'.json');src=Path(plan['source_dataset'])/('policies.json' if name=='oracle_policies' else name+'.json')
        if json.loads(p.read_text())!=expected or p.read_bytes()!=src.read_bytes() or streaming_hash(p)!=m['original_file_sha256'][name+'.json']:
            raise ValueError('Original task/oracle source bytes must stay unchanged')
    for n,h in m['source_sha256'].items():
        if streaming_hash(ROOT/n)!=h or streaming_hash(path/'sources'/n)!=h:raise ValueError('Frozen192data sources changed')
    return m,cases,registry,queries,policies,references
