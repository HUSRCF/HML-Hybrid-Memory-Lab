"""Immutable original inputs for the preregistered recurrent hold diagnostic."""

from datetime import datetime
import json
from pathlib import Path

from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.text_competence_factorial_data import (
    sources as prior_sources, registered_inputs as text_inputs, control_ids,
    selected_inputs, actual_prompt, unique_prefixes,
)
from scripts.trained_timing_evidence_access_data import registered_inputs as access_inputs

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/oracle_write_hold_factorial_plan.json'
PROTOCOL_SHA256 = 'b05fc0ee3b1a1134e54e2a0c43ee815a650b08599c9654122e87682d1f9710e1'
DESTINATION = Path('/media/990Pro/Code/SAS/datasets/memory_update/oracle-write-hold-factorial-v1')
MODELS = ('native', 'shared', 'shared_question_trained_all', 'shared_question_trained_question', 'gdn_all_tokens')
REGIMES = ('full', 'hold', 'oracle', 'hold_oracle')
DATA_SOURCES = prior_sources() | {'configs/oracle_write_hold_factorial_plan.json',
    'scripts/oracle_write_hold_data.py', 'scripts/prepare_oracle_write_hold.py'}


def sources():
    return DATA_SOURCES | {'scripts/oracle_write_hold_runtime.py', 'scripts/run_oracle_write_hold.py',
        'scripts/oracle_write_hold_metrics.py', 'scripts/analyze_oracle_write_hold.py',
        'tests/test_oracle_write_hold.py'}


def read_plan():
    if streaming_hash(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Registered protocol changed')
    p = json.loads(PROTOCOL.read_text())
    if (p['model_order'] != list(MODELS) or p['regimes'] != list(REGIMES) or
        Path(p['dataset_destination']) != DESTINATION or
        any(p[k] for k in ('training', 'new_adapters', 'new_models', 'new_task_cases', 'new_environment',
            'prospective_unseen_confirmation', 'architecture_effectiveness_claim', 'can_change_prior_gates')) or
        not p['local_files_only'] or p['counts']['primary_physical_readouts'] != 6720):
        raise ValueError('Fixed full-cohort factorial allocation changed')
    return p


def previous_inputs(plan):
    tm, cases, registry, transforms, _ = text_inputs(Path(plan['source_dataset']))
    am, _, ac, ar, masks, _ = access_inputs(Path(plan['source_mask_dataset']))
    run = Path(plan['source_run']); m = json.loads((run/'manifest.json').read_text())
    ap = Path(plan['source_audit']); a = json.loads(ap.read_text())
    if (streaming_hash(run/'manifest.json') != plan['source_manifest_sha256'] or
        streaming_hash(ap) != plan['source_audit_sha256'] or
        streaming_hash(ROOT/plan['source_protocol']) != plan['source_protocol_sha256'] or
        streaming_hash(Path(plan['source_dataset'])/'manifest.json') != plan['source_dataset_manifest_sha256'] or
        streaming_hash(Path(plan['source_mask_dataset'])/'manifest.json') != plan['source_mask_manifest_sha256'] or
        streaming_hash(Path(plan['source_mask_dataset'])/'masks.json') != plan['source_masks_sha256'] or
        m['status'] != 'completed' or m['smoke'] or cases != ac or registry != ar or
        a['status'] != 'completed_and_audited_text_competence_factorial' or
        a['new_physical_readouts_audited'] != 6720 or len(m['source_sha256']) != 172 or
        set(m['source_sha256']) != prior_sources() or
        not datetime.fromisoformat(m['completed_utc']) < datetime.fromisoformat(plan['registered_utc'])):
        raise ValueError('Previous complete evidence, original inputs or chronology changed')
    for name, digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name) != digest or streaming_hash(run/'sources'/name) != digest:
            raise ValueError(f'Preserve all172prior sources: {name}')
    # Keep every raw cell, including the other three text conditions.
    for name, state in m['models'].items():
        for r, digest in state['results_sha256'].items():
            if streaming_hash(run/f'{name}.{r}.jsonl') != digest:
                raise ValueError('Prior complete text readouts changed')
    refs = {**tm['references'], 'previous_text_audit_sha256': streaming_hash(ap),
        'previous_text_manifest_sha256': streaming_hash(run/'manifest.json'),
        'previous_text_all20_raw_sha256': {n: s['results_sha256'] for n,s in m['models'].items()},
        'original_access_manifest_sha256': plan['source_mask_manifest_sha256']}
    original = {q['id']: transforms[q['id']]['full'] for q in registry['physical']}
    access_run = Path(json.loads((Path(plan['source_dataset'])/'plan.json').read_text())['source_run'])
    access_manifest = json.loads((access_run/'manifest.json').read_text())
    references = {}
    for n in MODELS:
        references[n] = {}
        for r in ('full', 'oracle'):
            path = access_run/f'{n}.{r}.jsonl'
            if streaming_hash(path) != access_manifest['models'][n]['results_sha256'][r]:
                raise ValueError('Pinned full/oracle baseline bytes changed')
            references[n][r] = [json.loads(line) for line in path.read_text().splitlines()]
    return cases, registry, original, masks, references, refs


def build_policies(queries, masks):
    result = {}
    for key,q in queries.items():
        mask = masks[key]; bits = mask['keep_bits']['oracle']
        if (len(bits) != q['ledger_tokens'] or any(b != '1' for b in bits[:q['prelude_tokens']]) or
            not {'0', '1'} <= set(bits[q['prelude_tokens']:])):
            raise ValueError('Original prelude/support/non-support token span changed')
        hold = [i for i,b in enumerate(bits) if b == '0']
        result[key] = {'id': key, 'history_keep_bits': bits, 'held_positions': hold,
            'first_held_position': hold[0], 'held_token_count': len(hold),
            'oracle_event_indices': mask['oracle_event_indices'], 'event_count': mask['event_count'],
            'prompt_ids_sha256': q['prompt_ids_sha256'], 'ledger_tokens': q['ledger_tokens'],
            'prelude_tokens': q['prelude_tokens']}
    return result


def registered_inputs(path=DESTINATION):
    path = Path(path); plan = read_plan()
    cases,registry,queries,masks,references,refs = previous_inputs(plan)
    m = json.loads((path/'manifest.json').read_text()); policies = json.loads((path/'policies.json').read_text())
    if (path != DESTINATION or m['status'] != 'registered_before_inference' or
        m['protocol_sha256'] != PROTOCOL_SHA256 or m['references'] != refs or
        json.loads((path/'plan.json').read_text()) != plan or
        json.loads((path/'cases.json').read_text()) != cases or json.loads((path/'registry.json').read_text()) != registry or
        json.loads((path/'queries.json').read_text()) != queries or policies != build_policies(queries,masks) or
        m['policies_sha256'] != data_fingerprint(policies) or m['queries_sha256'] != data_fingerprint(queries) or
        m['control_query_ids'] != control_ids(cases,registry) or not m['previous_full_audit_reexecuted'] or
        set(m['source_sha256']) != DATA_SOURCES or
        not datetime.fromisoformat(plan['registered_utc']) < datetime.fromisoformat(m['generation_started_utc']) <
            datetime.fromisoformat(m['registered_utc'])):
        raise ValueError('Original fixed-input data, policies or source registration changed')
    for n,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/n) != digest or streaming_hash(path/'sources'/n) != digest:
            raise ValueError('Registered data source changed')
    return m,cases,registry,queries,policies,references
