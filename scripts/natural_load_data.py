"""Prospective normal-update inputs; executable truth and tokenizer-only checks."""
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import random
import string
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.tasks import INSTRUCTIONS, Operation, execute
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.post_encoding_tasks import question

PROTOCOL = ROOT / 'configs/natural_update_load_plan.json'
PROTOCOL_SHA256 = 'c0f32de5f7fc43b587300e82554e8609cfcf9a74da963659d9e1ea049b463903'
DESTINATION = Path('/media/990Pro/Code/SAS/datasets/memory_update/natural-update-load-v1')
MODELS = ('native', 'gdn_all_tokens')
READERS = ('full', 'oracle')
ROLES = ('target_current', 'target_initial', 'protected_current')
STYLES = ('original', 'entity_first')
NEW_SOURCES = {'configs/natural_update_load_plan.json', 'scripts/natural_load_data.py',
    'scripts/prepare_natural_load.py', 'scripts/run_natural_load.py',
    'scripts/analyze_natural_load.py', 'tests/test_natural_load.py'}


def read_plan():
    if streaming_hash(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Prospectively registered natural-load protocol changed')
    p = json.loads(PROTOCOL.read_text())
    if (p['model_order'] != list(MODELS) or p['reader_modes'] != list(READERS) or
            p['roles'] != list(ROLES) or p['question_styles'] != list(STYLES) or
            p['tail_records'] != [64, 512] or p['active_tail_entities'] != [2, 16] or
            p['counts']['primary_readouts'] != 3456 or p['counts']['primary_cells'] != 36 or
            p['smoke']['readouts_each'] != 108 or p['smoke']['accuracy_gate'] or
            p['operational_controls']['source_files_planned'] != 240 or
            set(p['operational_controls']['new_sources']) != NEW_SOURCES or
            Path(p['registration_destination']) != DESTINATION or
            any(p[k] for k in ('training', 'new_adapters', 'new_models', 'new_environment',
                'can_change_prior_gates', 'architecture_effectiveness_claim', 'fresh_confirmation',
                'new_code_started', 'data_packet_generated', 'model_inference_started'))):
        raise ValueError('Entire fixed pilot grid required')
    return p


def predecessor_sources():
    p = read_plan()
    m = json.loads((Path(p['source_run']) / 'manifest.json').read_text())
    return {**m['source_sha256'], **m['audit_correction']['source_sha256']}


def sources():
    return set(predecessor_sources()) | NEW_SOURCES


def reference_inputs():
    p = read_plan(); run = Path(p['source_run'])
    m = json.loads((run / 'manifest.json').read_text())
    a = json.loads(Path(p['source_audit']).read_text())
    pub = json.loads(Path(p['source_publication']).read_text())
    if (streaming_hash(run / 'manifest.json') != p['source_manifest_sha256'] or
            streaming_hash(Path(p['source_audit'])) != p['source_audit_sha256'] or
            streaming_hash(Path(p['source_publication'])) != p['source_publication_sha256'] or
            a['status'] != 'completed_and_audited_query_layer' or a['new_primary_readouts_audited'] != 4608 or
            a['total_source_files_verified'] != 234 or any(a['prior_failed_criteria_unchanged'].values()) or
            m['status'] != 'completed' or m['weights_before'] != m['weights_after'] or not m['hooks_restored'] or
            pub['status'] != 'published_remote_assets_roundtrip_byte_verified' or
            pub['release_url'] != p['source_release'] or pub['audit_sha256'] != p['source_audit_sha256'] or
            len(pub['remote_assets']) != 3 or not all(v['downloaded_bytes_verified'] for v in pub['remote_assets'])):
        raise ValueError('Complete published predecessor required')
    digests = predecessor_sources()
    if len(digests) != 234:
        raise ValueError('Preserve all234 prior sources')
    for name, digest in digests.items():
        sub = 'sources' if name in m['source_sha256'] else 'audit-correction-sources'
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / sub / name) != digest:
            raise ValueError('Immutable predecessor source changed')
    packet = Path(p['prior_fresh_packet']); pm = json.loads((packet / 'manifest.json').read_text())
    if streaming_hash(packet / 'manifest.json') != p['prior_fresh_manifest_sha256']:
        raise ValueError('Fresh predecessor packet changed')
    for name, digest in pm['payload_sha256'].items():
        if streaming_hash(packet / name) != digest:
            raise ValueError('Prior fresh input bytes changed')
    prior = json.loads((packet / 'prior_inputs.json').read_text())
    cases = json.loads((packet / 'cases.json').read_text())
    registry = json.loads((packet / 'registry.json').read_text())
    queries = json.loads((packet / 'queries.json').read_text())
    legacy = json.loads((packet / 'references.json').read_text())
    old = json.loads((run / 'plan.json').read_text())
    if any(p[k] != old[k] for k in ('models', 'model_order', 'model_path', 'packages', 'generation',
            'native_loaded_weights_sha256', 'precision_addendum', 'arithmetic_profile')):
        raise ValueError('Preserve existing frozen model and arithmetic')
    return {**{k: legacy[k] for k in ('legacy_ids', 'legacy_queries', 'legacy_full', 'runtime', 'prior_failed_criteria')},
        'model_states': {n: {k: m['models'][n][k] for k in ('description', 'adapter_state_sha256_before')
            if k in m['models'][n]} for n in MODELS},
        'excluded_seeds': sorted(set(prior['seeds']) | {c['seed'] for c in cases}),
        'excluded_prefix_hashes': sorted(set(prior['complete_prefix_ids_sha256']) | set(registry['prefixes'])),
        'excluded_prompt_hashes': sorted(set(prior['complete_prompt_ids_sha256']) | {q['prompt_ids_sha256'] for q in queries.values()})}


def select_seeds(p, refs):
    result = []; seed = p['seed_rule']['start']
    while len(result) < p['seed_rule']['count']:
        if seed not in refs['excluded_seeds']:
            result.append(seed)
        seed += 1
    return result


def make_cases(seed):
    rng = random.Random(seed); names = rng.sample(list(string.ascii_uppercase), 18)
    target, protected = names[:2]; pool = names[2:]
    values = rng.sample(range(100, 1000), 531)
    initial = [Operation('SET', name, value=values[i]) for i, name in enumerate(names)]
    rng.shuffle(initial); prefix = initial + [Operation('SET', target, value=values[18])]
    masters = {}
    for load in (2, 16):
        chosen = rng.sample(pool, load); sequence = []
        for _ in range(512 // load):
            cycle = chosen.copy(); rng.shuffle(cycle); sequence.extend(cycle)
        masters[load] = [Operation('SET', name, value=v) for name, v in zip(sequence, values[19:])]
    letters = list(string.ascii_uppercase); rng.shuffle(letters)
    renaming = dict(zip(string.ascii_uppercase, letters)); result = []
    for naming in ('original', 'permuted'):
        mapping = renaming if naming == 'permuted' else dict(zip(string.ascii_uppercase, string.ascii_uppercase))
        def mapped(events):
            return [asdict(Operation(e.kind, mapping[e.target], value=e.value)) for e in events]
        episodes = [{'id': 'before', 'kind': 'before', 'record_count': 0, 'active_entities': 0, 'tail': []}]
        for count in (64, 512):
            for load in (2, 16):
                for kind in ('unrelated', 'related'):
                    tail = masters[load][:count].copy()
                    if kind == 'related':
                        tail[-1] = Operation('SET', target, value=tail[-1].value)
                    episodes.append({'id': f'{kind}-{count}-load{load}', 'kind': kind, 'record_count': count,
                        'active_entities': load, 'tail': mapped(tail)})
        result.append({'id': f'natural-load-s{seed}-{naming}', 'seed': seed, 'naming': naming,
            'target': mapping[target], 'protected': mapping[protected], 'prefix_events': mapped(prefix),
            'account_mapping': mapping, 'episodes': episodes})
    return result


def truth(case, episode, role):
    events = [Operation(**e) for e in case['prefix_events'] + episode['tail']]
    state = execute(events); entity = case['protected'] if role == 'protected_current' else case['target']
    field = 'initial' if role == 'target_initial' else 'current'
    support = [i for i, e in enumerate(events) if e.target == entity]
    support = support[0] if field == 'initial' else support[-1]
    answer = str(state[field][entity])
    if str(events[support].value) != answer:
        raise ValueError('Interpreter and exact SET support disagree')
    return answer, support


def record_spans(runtime, prelude, events):
    text = prelude; ids = runtime.ids(prelude); spans = []
    for event in events:
        piece = Operation(**event).text(); tokens = runtime.ids(piece)
        if len(tokens) != 14:
            raise ValueError('Every complete SET record must have14tokens; never substitute names/seeds')
        spans.append({'index': len(spans), 'start': len(ids), 'stop': len(ids) + len(tokens),
            'kind': event['kind'], 'target': event['target']})
        text += piece; ids.extend(tokens)
    if runtime.ids(text) != ids:
        raise ValueError('Full tokenization must verify every concatenated record boundary')
    return text, ids, spans


def build_registry(runtime, p, refs):
    cases = [c for seed in select_seeds(p, refs) for c in make_cases(seed)]
    prefixes = {}; contexts = []; queries = {}; signatures = {}; question_lengths = {}
    prior_prefixes = set(refs['excluded_prefix_hashes']); prior_prompts = set(refs['excluded_prompt_hashes'])
    prelude = runtime.prompt_start + INSTRUCTIONS; prelude_tokens = len(runtime.ids(prelude))
    for case in cases:
        prefix, prefix_ids, _ = record_spans(runtime, prelude, case['prefix_events'])
        prefix_key = data_fingerprint(prefix_ids)
        if prefix_key in prior_prefixes or prefix_key in prefixes:
            raise ValueError('Every complete common prefix must be fresh and unique')
        prefixes[prefix_key] = {'case_id': case['id'], 'input_ids': prefix_ids, 'text': prefix}
        baseline = {r: truth(case, case['episodes'][0], r)[0] for r in ROLES}
        for episode in case['episodes']:
            ledger, ids, spans = record_spans(runtime, prelude, case['prefix_events'] + episode['tail'])
            if ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError('Fixed prefix token boundary changed')
            signature = (len(ids), [(b['start'], b['stop']) for b in spans])
            key = (case['seed'], episode['record_count'])
            if key in signatures and signatures[key] != signature:
                raise ValueError('Length, load, kind and renaming record positions must match')
            signatures[key] = signature
            if len(ids) != 351 + 14 * episode['record_count']:
                raise ValueError('Actual fixed BF16 prefill length differs from registered expectation')
            cid = case['id'] + '.' + episode['id']; source_key = data_fingerprint(ids)
            if source_key in prior_prefixes:
                raise ValueError('New ledger overlaps prior exact input')
            c = {'id': cid, 'case_id': case['id'], 'seed': case['seed'], 'naming': case['naming'],
                'episode': episode['id'], 'kind': episode['kind'], 'tail_records': episode['record_count'],
                'active_entities': episode['active_entities'], 'prefix_key': prefix_key, 'prefix_text': prefix,
                'prefix_tokens': len(prefix_ids), 'tail_text': ledger[len(prefix):], 'tail_tokens': len(ids)-len(prefix_ids),
                'ledger': ledger, 'ledger_tokens': len(ids), 'prelude_tokens': prelude_tokens,
                'source_key': source_key, 'records': spans, 'query_ids': []}
            for style in STYLES:
                for role in ROLES:
                    qid = data_fingerprint([cid, style, role]); text = question(case, role, style)
                    answer, support = truth(case, episode, role)
                    if (episode['kind'] != 'related' or role != 'target_current') and answer != baseline[role]:
                        raise ValueError('Unrelated queried truth changed')
                    if episode['kind'] == 'related' and role == 'target_current' and answer == baseline[role]:
                        raise ValueError('Related update must require new answer')
                    prompt = ledger + runtime.query_suffix(text); prompt_ids = runtime.ids(prompt)
                    prompt_key = data_fingerprint(prompt_ids)
                    if prompt_ids[:len(ids)] != ids or prompt_key in prior_prompts:
                        raise ValueError('Question boundary or exact prompt novelty failed')
                    qkey = (role, style); length = len(prompt_ids)-len(ids)
                    if qkey in question_lengths and question_lengths[qkey] != length:
                        raise ValueError('Question lengths must match across entity names')
                    question_lengths[qkey] = length
                    keep = ['0'] * len(ids); keep[:prelude_tokens] = ['1'] * prelude_tokens
                    b = spans[support]; keep[b['start']:b['stop']] = ['1'] * (b['stop']-b['start'])
                    queries[qid] = {'id': qid, 'context_id': cid, 'case_id': case['id'], 'seed': case['seed'],
                        'naming': case['naming'], 'episode': episode['id'], 'kind': episode['kind'],
                        'tail_records': episode['record_count'], 'active_entities': episode['active_entities'],
                        'role': role, 'question_style': style, 'question': text, 'answer': answer,
                        'support_event_index': support, 'prefix_evidence': support < 19,
                        'history_keep_bits': ''.join(keep), 'prelude_tokens': prelude_tokens,
                        'ledger_tokens': len(ids), 'input_ids': prompt_ids, 'prompt_ids_sha256': prompt_key,
                        'source_key': source_key, 'answer_ids': runtime.continuation_ids(prompt, answer)}
                    c['query_ids'].append(qid)
            contexts.append(c)
    if (len(cases), len(prefixes), len(contexts), len(queries)) != (16, 16, 144, 864) or len({q['prompt_ids_sha256'] for q in queries.values()}) != 864:
        raise ValueError('Complete unfiltered natural-update grid required')
    return cases, {'prefixes': prefixes, 'contexts': contexts, 'query_order': list(queries)}, queries


def selected_contexts(registry, smoke):
    first = registry['contexts'][0]['seed']
    return [c for c in registry['contexts'] if not smoke or (c['seed'] == first and c['naming'] == 'original')]


def context_queries(context, queries, smoke):
    return [queries[k] for k in context['query_ids'] if not smoke or queries[k]['question_style'] == 'original']


def registered_inputs(path=DESTINATION):
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    path = Path(path); p = read_plan(); m = json.loads((path / 'manifest.json').read_text())
    if (m['status'] != 'registered' or m['model_inference'] or m['training'] or
            m['protocol_sha256'] != PROTOCOL_SHA256 or m['counts'] != p['counts'] or
            set(m['source_sha256']) != sources() or len(sources()) != 240 or
            not datetime.fromisoformat(p['registered_utc']) < datetime.fromisoformat(m['created_utc'])):
        raise ValueError('Complete240source packet must precede inference')
    for name, digest in m['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / 'sources' / name) != digest:
            raise ValueError('Sealed natural-load source changed')
    for name, digest in m['payload_sha256'].items():
        if streaming_hash(path / name) != digest:
            raise ValueError('Registered payload bytes changed')
    refs = reference_inputs()
    if json.loads((path / 'plan.json').read_text()) != p or json.loads((path / 'references.json').read_text()) != refs:
        raise ValueError('Sealed protocol/reference reconstruction differs')
    values = build_registry(tokenizer_runtime(p['model_path']), p, refs)
    for name, value in zip(('cases', 'registry', 'queries'), values):
        if json.loads((path / (name + '.json')).read_text()) != value:
            raise ValueError('Independent tokenizer/interpreter input reconstruction differs')
    return m, *values, refs
