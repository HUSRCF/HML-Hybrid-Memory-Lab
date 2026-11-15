"""Literal observed fit inputs; no prospective confirmation data generation."""
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash

PROTOCOL = ROOT / 'configs/readout_adapter_training_plan.json'
PROTOCOL_SHA256 = 'd8c7564a1dcaf73629a9d4a36fbd8a36f1bf566d4144b8a0adc1cfa664ae049b'
ARMS = ('query', 'gate', 'query_gate')
NEW_SOURCES = {'configs/readout_adapter_training_plan.json', 'sas_memory/readout_adapters.py',
    'scripts/readout_adapter_data.py', 'scripts/readout_adapter_runtime.py',
    'scripts/prepare_readout_adapters.py', 'scripts/run_readout_adapter_training.py',
    'scripts/analyze_readout_adapter_training.py', 'tests/test_readout_adapters.py'}


def read_plan():
    if streaming_hash(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Registered training design changed')
    p = json.loads(PROTOCOL.read_text())
    if (p['training']['optimizer_steps_per_arm'] != 432 or p['training']['examples'] != 1728
        or p['training']['gradient_accumulation'] != 4 or p['training']['order'] != list(ARMS)
        or p['attention_layer'] != 15 or p['planned_total_sources'] != 276
        or set(p['new_sources']) != NEW_SOURCES
        or any(p['arms'][a]['new_trainable_parameters'] != 131072 for a in ARMS)
        or any(p['prior_failed_criteria_unchanged'].values())
        or any(p[k] for k in ('implementation_started', 'packet_generated', 'native_smoke_started',
            'optimizer_updates_started', 'new_backbone', 'new_environment', 'oracle_reading',
            'donor_forward', 'changes_fact_prefixes', 'can_change_prior_gates', 'architecture_effectiveness_claim'))):
        raise ValueError('Complete fixed learned-readout controls required')
    return p


def load_sources():
    p = read_plan()
    run, packet = Path(p['source_run']), Path(p['source_packet'])
    for path, digest in ((run / 'manifest.json', p['source_manifest_sha256']),
                         (packet / 'manifest.json', p['source_packet_manifest_sha256']),
                         (Path(p['source_audit']), p['source_audit_sha256']),
                         (Path(p['source_publication']), p['source_publication_sha256'])):
        if streaming_hash(path) != digest:
            raise ValueError('Published normal-history source changed')
    m = json.loads((run / 'manifest.json').read_text())
    audit = json.loads(Path(p['source_audit']).read_text())
    pub = json.loads(Path(p['source_publication']).read_text())
    if (m['status'] != 'completed' or len(m['source_sha256']) != 268
        or audit['status'] != 'completed_and_audited_version_address'
        or audit['physical_readouts_audited'] != 3456 or audit['source_files_verified'] != 268
        or audit['manifest_sha256'] != p['source_manifest_sha256']
        or pub['status'] != 'published_remote_assets_roundtrip_byte_verified'
        or pub['audit_sha256'] != p['source_audit_sha256'] or pub['release_url'] != p['source_release']
        or len(pub['remote_assets']) != 3 or not all(v['downloaded_bytes_verified'] for v in pub['remote_assets'])
        or audit['prior_failed_criteria_unchanged'] != p['prior_failed_criteria_unchanged']
        or not datetime.fromisoformat(pub['completed_utc']) < datetime.fromisoformat(p['registered_utc'])):
        raise ValueError('Complete published predecessor required')
    for name, digest in m['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / 'sources' / name) != digest:
            raise ValueError('All268predecessor sources stay immutable')
    for name, digest in p['source_payload_sha256'].items():
        if streaming_hash(packet / name) != digest:
            raise ValueError('Literal observed input changed')
    cases, registry, queries = [json.loads((packet / (name + '.json')).read_text())
                                for name in ('cases', 'registry', 'queries')]
    raw = run / 'gdn_all_tokens.jsonl'
    if streaming_hash(raw) != m['models']['gdn_all_tokens']['results_sha256']:
        raise ValueError('Complete strong-baseline outputs changed')
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    if (len(cases) != 32 or len(registry['contexts']) != 288 or len(queries) != 1728
        or [r['id'] for r in rows] != list(queries)
        or m['models']['gdn_all_tokens']['adapter_state_sha256_before'] != p['fixed_gdn_state_sha256']):
        raise ValueError('All observed fit examples required, including all95errors')
    return p, m, cases, registry, queries, rows


def sources():
    _, m, *_ = load_sources()
    return set(m['source_sha256']) | NEW_SOURCES


def build_examples(runtime, cases, registry, queries):
    cases = {c['id']: c for c in cases}
    contexts = {c['id']: c for c in registry['contexts']}
    examples = []
    for qid, q in queries.items():
        c, case = contexts[q['context_id']], cases[q['case_id']]
        events = case['prefix_events'] + next(e['tail'] for e in case['episodes'] if e['id'] == q['episode'])
        if any(e['kind'] != 'SET' for e in events):
            raise ValueError('Observed adaptation data must be the fixed SET tasks')
        entity = case['protected'] if q['role'] == 'protected_current' else case['target']
        own = [(i, e) for i, e in enumerate(events) if e['target'] == entity]
        support, fact = own[0 if q['role'] == 'target_initial' else -1]
        prompt = c['ledger'] + runtime.query_suffix(q['question'])
        ids = runtime.ids(prompt)
        answer = runtime.continuation_ids(prompt, q['answer'] + '\n')
        if (str(fact['value']) != q['answer'] or support != q['support_event_index']
            or ids != q['input_ids'] or len(answer) != 4 or answer[:3] != q['answer_ids']
            or len(ids) - q['ledger_tokens'] != 24):
            raise ValueError('Independent truth, canonical teacher tokens or known boundary changed')
        examples.append({'id': qid, 'context_id': c['id'], 'case_id': case['id'],
                         'ledger_tokens': q['ledger_tokens'], 'known_ids': ids[q['ledger_tokens']:],
                         'answer_ids': answer, 'teacher_ids': ids[q['ledger_tokens']:] + answer[:-1],
                         'literal_prompt_ids_sha256': data_fingerprint(ids),
                         'observed_source_cohort': registry['cohort_by_context'][c['id']]})
    if len(examples) != 1728 or any(len(e['teacher_ids']) != 27 for e in examples):
        raise ValueError('One full fixed1728example pass required')
    return examples


def registered_inputs(path=None):
    p, old, cases, registry, queries, rows = load_sources()
    path = Path(path or p['registration_destination'])
    if path.resolve() != Path(p['registration_destination']).resolve():
        raise ValueError('Use the canonical sealed fit packet')
    m = json.loads((path / 'manifest.json').read_text())
    if (m['status'] != 'registered_observed_fit_inputs_before_training'
        or m['protocol_sha256'] != PROTOCOL_SHA256 or m['source_manifest_sha256'] != p['source_manifest_sha256']
        or set(m['source_sha256']) != set(old['source_sha256']) | NEW_SOURCES
        or len(m['source_sha256']) != 276 or m['examples'] != 1728
        or m['training'] or m['model_inference'] or m['new_confirmation_inputs_generated']
        or json.loads((path / 'plan.json').read_text()) != p
        or not datetime.fromisoformat(p['registered_utc']) < datetime.fromisoformat(m['created_utc'])):
        raise ValueError('Sealed observed-fit packet required')
    for name, digest in m['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / 'sources' / name) != digest:
            raise ValueError('New training source changed after registration')
    for name, digest in m['payload_sha256'].items():
        if streaming_hash(path / name) != digest:
            raise ValueError('Registered fit payload changed')
    examples = json.loads((path / 'examples.json').read_text())
    if [e['id'] for e in examples] != list(queries) or any(
        e['context_id'] != queries[e['id']]['context_id'] or e['case_id'] != queries[e['id']]['case_id']
        or e['ledger_tokens'] != queries[e['id']]['ledger_tokens']
        or e['literal_prompt_ids_sha256'] != queries[e['id']]['prompt_ids_sha256']
        or e['observed_source_cohort'] != registry['cohort_by_context'][e['context_id']]
        or e['known_ids'] != queries[e['id']]['input_ids'][e['ledger_tokens']:]
        or e['teacher_ids'] != e['known_ids'] + e['answer_ids'][:-1]
        or e['answer_ids'][:3] != queries[e['id']]['answer_ids']
        or e['answer_ids'][-1] != m['newline_token_id'] for e in examples):
        raise ValueError('Literal order, source questions or answer tokens changed')
    return p, m, old, cases, registry, queries, examples, rows


def verify_seal(packet):
    p = read_plan()
    path = Path(p['registration_seal'])
    seal = json.loads(path.read_text())
    log = Path(seal['full_tests_log'])
    match = re.search(r'Ran (\d+) tests in [\d.]+s\s+OK\s*$', log.read_text())
    if (seal['status'] != 'sealed_before_learned_readout_native_smoke'
        or seal['source_sha256'] != packet['source_sha256']
        or seal['packet_manifest_sha256'] != streaming_hash(Path(p['registration_destination']) / 'manifest.json')
        or not match or int(match[1]) != seal['full_tests_passed'] or seal['full_tests_passed'] <= 352
        or streaming_hash(log) != seal['full_tests_log_sha256']
        or not datetime.fromisoformat(packet['created_utc']) < datetime.fromisoformat(seal['created_utc'])):
        raise ValueError('Full prior and new CPU regression seal required before native smoke')
    return {'path': str(path), 'sha256': streaming_hash(path), 'created_utc': seal['created_utc'],
            'full_tests_passed': seal['full_tests_passed']}
