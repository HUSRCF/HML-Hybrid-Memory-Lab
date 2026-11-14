"""Pinned references and truth-preserving text controls; no model-generated labels."""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from sas_memory.tasks import INSTRUCTIONS,Operation,execute
from sas_memory.oracle import evidence_slice
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.copy_readout_runtime import readout
from scripts.copy_readout_data import sample_ids,selected_inputs
from scripts.trained_timing_evidence_access_data import (MODELS,sources as previous_sources,
    registered_inputs as previous_inputs,require_exact_full)
from scripts.trained_timing_evidence_access_metrics import supplemental_depth2_queries

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'configs/text_competence_factorial_plan.json'
PROTOCOL_SHA256='fce95c001170418fdd485495d5cfde0120c89b751e136816fc8fc1d86621f00d'
DESTINATION=Path('/media/990Pro/Code/SAS/datasets/memory_update/text-competence-factorial-v1')
REGIMES=('full','support_text','compiled_full','compiled_support')
DATA_SOURCES=previous_sources()|{'configs/text_competence_factorial_plan.json',
    'scripts/text_competence_factorial_data.py','scripts/prepare_text_competence_factorial.py'}


def sources():
    return DATA_SOURCES|{'scripts/run_text_competence_factorial.py','scripts/analyze_text_competence_factorial.py',
                        'scripts/text_competence_factorial_metrics.py'}


def read_plan():
    if streaming_hash(PROTOCOL)!=PROTOCOL_SHA256:raise ValueError('Pre-transformation protocol changed')
    p=json.loads(PROTOCOL.read_text())
    if (p['regimes']!=list(REGIMES) or p['model_order']!=list(MODELS) or list(p['models'])!=list(MODELS) or
        p['counts']!={'base_streams':8,'cases':48,'physical_queries':336,'related_bindings':432,'models':5,'regimes':4,
                      'primary_physical_readouts':6720,'primary_related_bindings':8640,'fresh_full_replays':1680,'transformed_text_readouts':5040} or
        any(p[k] for k in ('training','new_adapters','new_models','new_task_cases','new_environment',
                          'prospective_unseen_confirmation','architecture_effectiveness_claim','can_change_prior_gates')) or
        not p['local_files_only'] or p['smoke_readouts_per_order']!=360 or p['data_source_files']!=169 or
        p['runtime_source_files']!=172 or Path(p['dataset_destination'])!=DESTINATION):
        raise ValueError('Fixed complete-cohort factorial scope changed')
    return p


def source_inputs(plan):
    registered,_,cases,registry,_,canonical=previous_inputs()
    run=Path(plan['source_run']);m=json.loads((run/'manifest.json').read_text())
    ap=Path(plan['source_audit']);a=json.loads(ap.read_text());old=json.loads((run/'plan.json').read_text())
    if (streaming_hash(ap)!=plan['source_audit_sha256'] or streaming_hash(run/'manifest.json')!=plan['source_manifest_sha256'] or
        streaming_hash(ROOT/plan['source_protocol'])!=plan['source_protocol_sha256'] or
        m['status']!='completed' or m['smoke'] or a['status']!='completed_and_audited_trained_timing_evidence_access' or
        a['new_physical_readouts_audited']!=5040 or a['manifest_sha256']!=plan['source_manifest_sha256'] or
        set(m['source_sha256'])!=previous_sources() or len(m['source_sha256'])!=166 or
        old['models']!=plan['models'] or m['model_order']!=list(MODELS) or m['weights_before']!=m['weights_after'] or
        m['weights_after']['sha256']!=plan['native_loaded_weights_sha256'] or m['arithmetic_profile']!=plan['arithmetic_profile'] or
        a['analysis_source_sha256']!=streaming_hash(ap.with_suffix('.source.py')) or
        any(a['prior_timing_criteria_unchanged'].values()) or
        data_fingerprint(cases)!=plan['source_cases_sha256'] or data_fingerprint(registry)!=plan['source_registry_sha256'] or
        streaming_hash(Path(plan['source_dataset'])/'manifest.json')!=plan['source_dataset_manifest_sha256'] or
        not datetime.fromisoformat(m['completed_utc'])<datetime.fromisoformat(plan['registered_utc'])):
        raise ValueError('Complete previous diagnostic, fixed checkpoints or original truth changed')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(run/'sources'/name)!=digest:raise ValueError('Preserve all166 prior sources')
    rows={};hashes={}
    for name in MODELS:
        for regime,digest in m['models'][name]['results_sha256'].items():
            p=run/f'{name}.{regime}.jsonl'
            if streaming_hash(p)!=digest:raise ValueError('Retain all15 previous raw result files')
            hashes[p.name]=digest
            if regime=='full':rows[name]=[json.loads(s) for s in p.read_text().splitlines()]
        if [readout(r) for r in rows[name]]!=[readout(r) for r in canonical[name]]:raise ValueError('Both original canonical full profiles must agree')
    refs={**registered['references'],'previous_diagnostic_audit_sha256':streaming_hash(ap),
          'previous_diagnostic_manifest_sha256':streaming_hash(run/'manifest.json'),'all15_previous_raw_sha256':hashes,
          'prior_timing_criteria_unchanged':a['prior_timing_criteria_unchanged']}
    return cases,registry,rows,refs


def compile_events(events):
    """Execute online: preserve snapshot COPY semantics, not future reference semantics."""
    current={};compiled=[]
    for e in events:
        if e.kind=='SET':value=e.value;replacement=e
        elif e.kind=='COPY':value=current[e.source];replacement=Operation('SET',e.target,value)
        else:raise ValueError('Registered task contains only SET and COPY')
        current[e.target]=value;compiled.append(replacement)
    # Independently use the existing interpreter on every prefix, not only final gold.
    for end in range(1,len(events)+1):
        if execute(events[:end])!=execute(compiled[:end]):raise ValueError('Compilation changed an entity current/initial state at an intermediate step')
    return compiled


def control_ids(cases,registry):
    ids=set(sample_ids(cases,registry))|{q['id'] for q in supplemental_depth2_queries(cases,registry)}
    result=[q['id'] for q in registry['physical'] if q['id'] in ids]
    if len(result)!=12:raise ValueError('All12 fixed all-role and depth2 controls required')
    return result


def build_transformations(cases,registry,runtime):
    result={};case_map={c['id']:c for c in cases}
    for original in registry['physical']:
        ref=original['representative'];case=case_map[ref['case_id']]
        query=next(q for q in case['variants'][ref['variant']]['queries'] if q['role']==ref['role'])
        events=[Operation(**e) for e in case['prefix']+case['variants'][ref['variant']]['tail']]
        support=evidence_slice(events,query['entity'],query['version']);identities={id(e) for e in support}
        support_indices=[i for i,e in enumerate(events) if id(e) in identities]
        sequences={'full':events,'support_text':support,'compiled_full':compile_events(events),'compiled_support':compile_events(support)}
        result[original['id']]={}
        for regime,sequence in sequences.items():
            if str(execute(sequence)[query['version']][query['entity']])!=original['answer']:
                raise ValueError('Transformed sequence changed its own-world query truth')
            ledger=runtime.prompt_start+INSTRUCTIONS+''.join(e.text() for e in sequence)
            prompt=ledger+runtime.query_suffix(original['question']);ids=runtime.ids(prompt);ledger_ids=runtime.ids(ledger)
            answer_ids=runtime.continuation_ids(prompt,original['answer'])
            q={**original,'regime':regime,'source_key':data_fingerprint(ledger_ids),'prompt_ids_sha256':data_fingerprint(ids),
                'prompt_tokens':len(ids),'ledger_tokens':len(ledger_ids),'question_ids':ids[len(ledger_ids):],
                'question_tokens':len(ids)-len(ledger_ids),'answer_ids':answer_ids,'prompt_ids':ids,
                'ledger_text':ledger,'events':[asdict(e) for e in sequence],
                'original_event_indices':list(range(len(events))) if regime in ('full','compiled_full') else support_indices,
                'original_event_count':len(events),'compiled':regime in ('compiled_full','compiled_support')}
            if (ids[:len(ledger_ids)]!=ledger_ids or len(ids)>4096 or answer_ids!=original['answer_ids'] or
                q['question_ids']!=original['question_ids'] or not 0<q['prelude_tokens']<q['ledger_tokens']):
                raise ValueError('Actual prompt/answer tokens, original question or context budget changed')
            if regime=='full' and any(q[k]!=v for k,v in original.items()):raise ValueError('Original full prompt registration must be unchanged')
            result[original['id']][regime]=q
    return result


def collision_registry(transforms,registry):
    by_regime={};within=[]
    for regime in REGIMES:
        groups={}
        for original in registry['physical']:
            q=transforms[original['id']][regime];key=q['prompt_ids_sha256']
            if key in groups:
                other=transforms[groups[key][0]][regime]
                if any(q[k]!=other[k] for k in ('prompt_ids','ledger_tokens','answer','answer_ids','question_tokens')):
                    raise ValueError('Identical transformed prompt has contradictory boundary or truth')
            groups.setdefault(key,[]).append(original['id'])
        by_regime[regime]={'unique_prompts':len(groups),'collisions':{k:v for k,v in groups.items() if len(v)>1}}
    for original in registry['physical']:
        q=transforms[original['id']]
        pairs=[[a,b] for i,a in enumerate(REGIMES) for b in REGIMES[i+1:] if q[a]['prompt_ids']==q[b]['prompt_ids']]
        within.append({'id':original['id'],'identical_regime_pairs':pairs})
    return {'by_regime':by_regime,'within_query_identical_inputs':within,
            'note':'Original task IDs preserve paired coverage; collisions are not independent repetitions.'}


def queries_for(transforms,cases,registry,regime,smoke=False,with_controls=False):
    physical,_=selected_inputs(registry,smoke);ids={q['id'] for q in physical}
    if with_controls:ids.update(control_ids(cases,registry))
    return [transforms[q['id']][regime] for q in registry['physical'] if q['id'] in ids]


def unique_prefixes(queries):
    seen=set();result=[]
    for q in queries:
        if q['source_key'] not in seen:seen.add(q['source_key']);result.append(q)
    return result


def actual_prompt(runtime,q):
    ledger=runtime.prompt_start+INSTRUCTIONS+''.join(Operation(**e).text() for e in q['events'])
    ids=runtime.ids(ledger+runtime.query_suffix(q['question']));ledger_ids=runtime.ids(ledger)
    if (ledger!=q['ledger_text'] or ids!=q['prompt_ids'] or data_fingerprint(ids)!=q['prompt_ids_sha256'] or
        data_fingerprint(ledger_ids)!=q['source_key'] or len(ledger_ids)!=q['ledger_tokens'] or
        ids[len(ledger_ids):]!=q['question_ids'] or runtime.continuation_ids(ledger+runtime.query_suffix(q['question']),q['answer'])!=q['answer_ids']):
        raise ValueError('Actual registered transformed text or tokens changed')
    return ledger


def registered_inputs(path=DESTINATION):
    path=Path(path);plan=read_plan();cases,registry,references,refs=source_inputs(plan)
    m=json.loads((path/'manifest.json').read_text());transforms=json.loads((path/'transformations.json').read_text())
    collisions=json.loads((path/'collisions.json').read_text())
    if (path!=DESTINATION or m['status']!='registered_before_inference' or m['training'] or m['model_inference'] or
        not m['complete_previous_audit_reexecuted'] or json.loads((path/'cases.json').read_text())!=cases or
        json.loads((path/'registry.json').read_text())!=registry or
        m['protocol_sha256']!=PROTOCOL_SHA256 or m['counts']!=plan['counts'] or m['references']!=refs or
        json.loads((path/'plan.json').read_text())!=plan or m['cases_sha256']!=data_fingerprint(cases) or
        m['registry_sha256']!=data_fingerprint(registry) or m['transformations_sha256']!=data_fingerprint(transforms) or
        m['collisions_sha256']!=data_fingerprint(collisions) or m['control_query_ids']!=control_ids(cases,registry) or
        set(m['source_sha256'])!=DATA_SOURCES or
        not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(m['generation_started_utc'])<datetime.fromisoformat(m['registered_utc'])):
        raise ValueError('Frozen text registration, truth or chronology changed')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Text generation source closure changed')
    if (transforms!=build_transformations(cases,registry,tokenizer_runtime(plan['model_path'])) or
        collisions!=collision_registry(transforms,registry) or m['distinct_prefixes_per_regime']!={
            r:len(unique_prefixes(queries_for(transforms,cases,registry,r))) for r in REGIMES}):
        raise ValueError('All actual transformed text, execution truth or collisions changed')
    return m,cases,registry,transforms,references
