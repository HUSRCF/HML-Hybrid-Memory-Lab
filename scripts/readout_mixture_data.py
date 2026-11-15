"""Deterministic observed-input registry for paired Gate task-coverage controls."""
from collections import Counter, OrderedDict
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.tasks import Operation, execute
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.readout_adapter_data import registered_inputs as old_inputs
from scripts.run_controller_trial import prompt_for

PROTOCOL = ROOT / 'configs/readout_mixture_training_plan.json'
PROTOCOL_SHA256 = 'ad9ee63e5e367e7e9ee4850585625102becc43cd624d03ad21950ca43b730f6e'
REGIMES = ('set_replay', 'set_copy_mix')
NEW_SOURCES = {'configs/readout_mixture_training_plan.json', 'scripts/readout_mixture_data.py',
               'scripts/prepare_readout_mixture.py', 'scripts/readout_mixture_runtime.py',
               'scripts/run_readout_mixture_training.py', 'scripts/analyze_readout_mixture_training.py',
               'tests/test_readout_mixture.py'}


def read_plan():
    if streaming_hash(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Registered paired mixture design changed')
    p = json.loads(PROTOCOL.read_text())
    if (p['regimes'] != list(REGIMES) or p['mode'] != 'gate' or p['rank'] != 32
        or p['attention_layer'] != 15 or p['new_trainable_parameters'] != 131072
        or p['planned_total_sources'] != 296 or p['required_full_cpu_tests'] != 380
        or set(p['new_sources']) != NEW_SOURCES
        or p['training']['optimizer_steps_per_regime'] != 432
        or p['training']['exposures_per_regime'] != 1728
        or p['confirmation']['set_seeds'] != list(range(160000, 160008))
        or p['confirmation']['copy_seeds'] != list(range(161000, 161008))
        or any(p[k] for k in ('implementation_started', 'packet_generated', 'optimizer_updates_started',
                              'confirmation_generated', 'new_environment', 'new_backbone', 'oracle',
                              'donor', 'changes_fact_prefixes', 'architecture_superiority_claim'))):
        raise ValueError('Fixed Gate regimes, budgets and future cohorts required')
    return p


def sources():
    p = read_plan()
    seal_path = Path(p['parent_source_seal'])
    if streaming_hash(seal_path) != p['parent_source_seal_sha256']:
        raise ValueError('The predecessor source seal changed')
    prior = json.loads(seal_path.read_text())['source_sha256']
    if len(prior) != 289:
        raise ValueError('All289predecessor sources required')
    for name, digest in prior.items():
        if streaming_hash(ROOT / name) != digest:
            raise ValueError('Sealed predecessor changed: ' + name)
    return set(prior) | NEW_SOURCES


def observed_gate(reexecute=False):
    p = read_plan()
    for name, digest in p['observed_files_sha256'].items():
        if streaming_hash(Path(name)) != digest:
            raise ValueError('Observed source changed: ' + name)
    audit = json.loads(Path(p['parent_confirmation_audit']).read_text())
    publication = json.loads(Path(p['parent_publication']).read_text())
    if (audit['status'] != 'completed_and_audited_frozen_readout_confirmation'
        or audit['physical_readouts_audited'] != 6000 or audit['source_files_verified'] != 289
        or publication['status'] != 'published_confirmation_backup_remote_assets_roundtrip_byte_verified'
        or len(publication['remote_assets']) != 3
        or not all(r['downloaded_bytes_verified'] for r in publication['remote_assets'])
        or not datetime.fromisoformat(publication['completed_utc']) < datetime.fromisoformat(p['registered_utc'])):
        raise ValueError('Complete published confirmation must precede mixture registration')
    if reexecute:
        from scripts.analyze_readout_confirmation import audit_run
        if audit != audit_run(Path(audit['source_run'])):
            raise ValueError('Complete parent confirmation audit no longer reproduces')
    return p


def example(runtime, qid, context, question, answer, task):
    ids = runtime.ids(context['ledger'] + runtime.query_suffix(question))
    boundary = context['ledger_tokens']
    if ids[:boundary] != runtime.ids(context['ledger']):
        raise ValueError('Literal ledger/question boundary changed')
    answer_ids = runtime.continuation_ids(context['ledger'] + runtime.query_suffix(question), answer + '\n')
    known = ids[boundary:]
    if (len(answer_ids) != 4 or len(known) not in ((24,) if task == 'set' else (26, 27))
        or runtime.ids(answer) != answer_ids[:3]):
        raise ValueError('Numeric targets or native teacher lengths changed')
    return {'id': qid, 'context_id': context['id'], 'task': task, 'ledger_tokens': boundary,
            'known_ids': known, 'teacher_ids': known + answer_ids[:-1], 'answer_ids': answer_ids,
            'question': question, 'answer': answer, 'literal_prompt_ids_sha256': data_fingerprint(ids)}


def exposure_schedule(set_chunks, copy_chunks, regime):
    if regime not in REGIMES or len(set_chunks) != 144 or len(copy_chunks) != 144:
        raise ValueError('Both fixed144chunk streams required')
    result = []
    for index, (left, right) in enumerate(zip(set_chunks, copy_chunks)):
        if len(left['query_ids']) != 6 or len(right['query_ids']) != 6:
            raise ValueError('Complete six-question chunks required')
        for part, chunk in enumerate((left, left if regime == 'set_replay' else right)):
            for qid in chunk['query_ids']:
                result.append({'exposure_id': f'{regime}:{len(result):04d}', 'query_id': qid,
                               'context_id': chunk['context_id'], 'block': index, 'part': part})
    if len(result) != 1728 or len({r['exposure_id'] for r in result}) != 1728:
        raise ValueError('Exactly1728distinct exposures required')
    return result


def build_inputs(runtime):
    """Rebind only already observed packets; never call a case generator."""
    p = observed_gate()
    bp, _, _, _, old_registry, old_queries, old_examples, _ = old_inputs()
    seeds = set(p['set_core']['seeds'])
    contexts = [c for c in old_registry['contexts'] if old_queries[c['query_ids'][0]]['seed'] in seeds]
    set_chunks = [{'context_id': c['id'], 'query_ids': c['query_ids']} for c in contexts]
    all_contexts = {c['id']: {**c, 'task': 'set'} for c in contexts}
    examples, queries = {}, {}
    old_by_id = {e['id']: e for e in old_examples}
    for c in contexts:
        for qid in c['query_ids']:
            q = old_queries[qid]
            e = example(runtime, qid, c, q['question'], q['answer'], 'set')
            if any(e[k] != old_by_id[qid][k] for k in ('known_ids', 'teacher_ids', 'answer_ids',
                                                      'literal_prompt_ids_sha256')):
                raise ValueError('An observed SET example was rewritten')
            examples[qid], queries[qid] = e, {**q, 'task': 'set'}
    if len(contexts) != 144 or len(examples) != 864:
        raise ValueError('Complete observed SET core required')
    data_root = Path(p['dataset']).parent
    first = data_root / 'fresh-copy-memory-confirmation-v1'
    second = json.loads((data_root / 'readout-adapter-confirmation-v1/inputs.json').read_text())
    cohorts = [(json.loads((first / 'cases.json').read_text()), json.loads((first / 'registry.json').read_text())),
               (second['copy_cases'], second['copy_registry'])]
    copy_chunks, copy_bindings = [], []
    for cohort, (cases, registry) in enumerate(cohorts):
        if sorted({c['seed'] for c in cases}) != p['copy_core']['cohort_seeds'][cohort]:
            raise ValueError('Fixed observed COPY seeds changed')
        case_by_id = {c['id']: c for c in cases}
        physical = {q['id']: q for q in registry['physical']}
        groups = OrderedDict()
        for binding in registry['bindings']:
            key = (binding['base_case_id'], binding['dependency_depth'], binding['variant'])
            groups.setdefault(key, []).append(binding)
            copy_bindings.append(binding)
        for key, bindings in groups.items():
            qids, ledger_key = [], None
            styles, roles = Counter(), Counter()
            for binding in bindings:
                qid = binding['physical_id']
                q, case = physical[qid], case_by_id[binding['case_id']]
                variant = case['variants'][binding['variant']]
                truth_query = next(x for x in variant['queries'] if x['role'] == binding['role'])
                truth = execute([Operation(**e) for e in case['prefix'] + variant['tail']])
                version = 'initial' if binding['role'] == 'target_initial' else 'current'
                if str(truth[version][truth_query['entity']]) != q['answer']:
                    raise ValueError('Executable COPY snapshot truth changed')
                ledger, prompt = prompt_for(runtime, case, binding['variant'], truth_query)
                cid = 'mixture-copy:' + q['source_key']
                context = {'id': cid, 'task': 'copy', 'ledger': ledger, 'prefix_text': ledger,
                           'ledger_tokens': q['ledger_tokens'], 'source_key': q['source_key']}
                if data_fingerprint(runtime.ids(ledger)) != q['source_key'] or data_fingerprint(runtime.ids(prompt)) != qid:
                    raise ValueError('Literal observed COPY input changed')
                e = example(runtime, qid, context, q['question'], q['answer'], 'copy')
                if e['known_ids'] != q['question_ids'] or e['answer_ids'][:3] != q['answer_ids']:
                    raise ValueError('Observed COPY suffix or answer tokens changed')
                if qid in examples and examples[qid] != e:
                    raise ValueError('Conflicting duplicated physical prompt')
                if cid in all_contexts and all_contexts[cid] != context:
                    raise ValueError('Shared COPY ledger changed across bindings')
                examples[qid], queries[qid], all_contexts[cid] = e, {**q, 'context_id': cid, 'task': 'copy'}, context
                if ledger_key is not None and ledger_key != cid:
                    raise ValueError('A training chunk mixes distinct fact histories')
                ledger_key = cid
                qids.append(qid)
                styles[binding['question_style']] += 1
                roles[binding['role']] += 1
            if len(qids) != 6 or len(set(qids)) != 6 or sorted(styles.values()) != [3, 3] or sorted(roles.values()) != [2, 2, 2]:
                raise ValueError('Both COPY styles and every role required per chunk')
            copy_chunks.append({'context_id': ledger_key, 'query_ids': qids, 'base_case_id': key[0],
                                'dependency_depth': key[1], 'variant': key[2]})
    result = {'set_registry': {'contexts': contexts}, 'contexts': all_contexts, 'queries': queries,
              'examples': examples, 'set_chunks': set_chunks, 'copy_chunks': copy_chunks,
              'copy_bindings': copy_bindings,
              'schedules': {r: exposure_schedule(set_chunks, copy_chunks, r) for r in REGIMES}}
    validate_inputs(result)
    return result


def validate_inputs(data):
    if len(data['examples']) != 1536 or len(data['contexts']) != 256 or len(data['copy_bindings']) != 864:
        raise ValueError('Complete physical inputs and logical COPY bindings required')
    for regime in REGIMES:
        expected = exposure_schedule(data['set_chunks'], data['copy_chunks'], regime)
        if data['schedules'][regime] != expected:
            raise ValueError('Missing, duplicated or reordered exposure')
        es = [data['examples'][r['query_id']] for r in expected]
        if (len({e['id'] for e in es}) != (864 if regime == 'set_replay' else 1536)
            or sum(len(e['teacher_ids']) for e in es) != (46656 if regime == 'set_replay' else 48816)
            or len({e['context_id'] for e in es}) != (144 if regime == 'set_replay' else 256)):
            raise ValueError('Registered exposure, physical, history or token budget differs')
    for qid, e in data['examples'].items():
        q, c = data['queries'][qid], data['contexts'][e['context_id']]
        if (e['id'] != qid or len(e['answer_ids']) != 4
            or e['teacher_ids'] != e['known_ids'] + e['answer_ids'][:-1]
            or e['answer'] != q['answer'] or e['question'] != q['question']
            or e['ledger_tokens'] != c['ledger_tokens']
            or len(e['known_ids']) not in ((24,) if e['task'] == 'set' else (26, 27))):
            raise ValueError('Teacher targets, literal input or cache boundary differs')


def verify_seal():
    p = read_plan()
    s = json.loads(Path(p['source_seal']).read_text())
    log = Path(s['full_tests_log'])
    match = re.search(r'Ran (\d+) tests in [\d.]+s\s+OK\s*$', log.read_text())
    if (s['status'] != 'sealed296mixture_sources_before_native_smoke'
        or s['source_sha256'].keys() != dict.fromkeys(sources()).keys()
        or not match or int(match[1]) != 380 or s['full_tests_passed'] != 380
        or streaming_hash(log) != s['full_tests_log_sha256']
        or s['protocol_sha256'] != PROTOCOL_SHA256):
        raise ValueError('Complete380test296source seal required')
    for name, digest in s['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest:
            raise ValueError('Measured mixture source changed: ' + name)
    return s


def registered_inputs():
    p = observed_gate()
    seal = verify_seal()
    path = Path(p['dataset'])
    m = json.loads((path / 'manifest.json').read_text())
    if (m['status'] != 'registered_observed_mixture_inputs_before_native'
        or m['protocol_sha256'] != PROTOCOL_SHA256 or m['source_sha256'] != seal['source_sha256']
        or json.loads((path / 'plan.json').read_text()) != p
        or not datetime.fromisoformat(p['registered_utc']) < datetime.fromisoformat(m['created_utc'])
        or any(m[k] for k in ('training', 'future_confirmation_generated'))):
        raise ValueError('Registered observed mixture packet required')
    for name, digest in m['payload_sha256'].items():
        if streaming_hash(path / name) != digest:
            raise ValueError('Registered mixture input changed')
    for name, digest in m['source_sha256'].items():
        if streaming_hash(path / 'sources' / name) != digest:
            raise ValueError('Registered source snapshot changed')
    data = json.loads((path / 'inputs.json').read_text())
    validate_inputs(data)
    return p, m, data
