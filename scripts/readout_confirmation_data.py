"""Post-training gates and fixed prospective input construction."""
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts import natural_load_data as natural
from scripts import version_query_data as wording
from scripts import fresh_copy_data as copy
from scripts.copy_transfer_data import build_registry as copy_registry
from scripts.run_controller_trial import prompt_for

PROTOCOL = ROOT / 'configs/readout_adapter_confirmation_plan.json'
PROTOCOL_SHA256 = '30fe0e1107b57ff48dec43a2015ddd29c4e8f9c168b2b8e4dd1895c326a8fb33'
NEW_SOURCES = {'configs/readout_adapter_confirmation_plan.json', 'scripts/readout_confirmation_data.py',
    'scripts/prepare_readout_confirmation.py', 'scripts/readout_confirmation_runtime.py',
    'scripts/run_readout_confirmation.py', 'scripts/analyze_readout_confirmation.py',
    'tests/test_readout_confirmation.py'}
MODELS = ('native', 'fixed_gdn', 'query', 'gate', 'query_gate')


def read_plan():
    if streaming_hash(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Registered confirmation execution changed')
    p = json.loads(PROTOCOL.read_text())
    if (p['set_seeds'] != list(range(88000, 88008)) or p['copy_seeds'] != list(range(89000, 89008))
        or p['models'] != list(MODELS) or p['planned_sources'] != 289
        or set(p['new_sources']) != NEW_SOURCES or p['full_cpu_tests_required'] != 372
        or any(p[k] for k in ('training', 'new_models', 'new_environment',
            'confirmation_code_started', 'confirmation_generated', 'new_inference_started'))
        or streaming_hash(ROOT / p['parent_training_protocol']) != p['parent_training_protocol_sha256']
        or streaming_hash(Path(p['parent_source_seal'])) != p['parent_source_seal_sha256']):
        raise ValueError('Fixed models, seeds, scope and predecessor must remain')
    return p


def sources():
    p = read_plan()
    prior = json.loads(Path(p['parent_source_seal']).read_text())['source_sha256']
    if len(prior) != 282:
        raise ValueError('All282predecessor files required')
    for name, digest in prior.items():
        if streaming_hash(ROOT / name) != digest:
            raise ValueError('Sealed predecessor changed: ' + name)
    return set(prior) | NEW_SOURCES


def training_gate():
    """Never construct a prospective case before independent full fit audit."""
    from scripts.analyze_readout_adapter_cache_training import audit_run, correction_audit
    p = read_plan()
    path, run = Path(p['training_audit']), Path(p['training_run'])
    if not path.is_file() or not path.with_suffix('.correction.json').is_file():
        raise ValueError('All three final training arms and independent audit must finish first')
    audit = json.loads(path.read_text())
    if (audit['status'] != 'completed_and_audited_readout_adapter_training'
        or audit != audit_run(run) or audit['optimizer_updates_audited'] != 1296
        or json.loads(path.with_suffix('.correction.json').read_text()) != correction_audit(run, audit)
        or any(v['fit_questions'] != 1728 for v in audit['arms'].values())):
        raise ValueError('Complete three-arm training audit does not reproduce')
    files = [run / 'manifest.json', path, path.with_suffix('.correction.json')]
    for arm in ('query', 'gate', 'query_gate'):
        files.extend(run / arm / n for n in ('training.jsonl', 'final-fit.jsonl'))
        for step in (144, 288, 432):
            files.extend(run / arm / f'step-{step:06d}' / n
                         for n in ('controller.safetensors', 'checkpoint.json', 'optimizer.pt'))
    return {'status': 'all_final_weights_and_complete_fit_independently_audited',
            'training_run': str(run), 'completed_utc': audit['completed_utc'],
            'files_sha256': {str(f): streaming_hash(f) for f in files},
            'final_states': {k: v['checkpoints'][-1]['state_sha256'] for k, v in audit['arms'].items()},
            'prior_failed_criteria_unchanged': audit['prior_failed_criteria_unchanged']}


def source_seal():
    p = read_plan()
    path = Path(p['source_seal'])
    s = json.loads(path.read_text())
    match = re.search(r'Ran (\d+) tests in [\d.]+s\s+OK\s*$', Path(s['full_tests_log']).read_text())
    if (s['status'] != 'sealed289confirmation_sources_before_prospective_generation'
        or set(s['source_sha256']) != sources() or len(s['source_sha256']) != 289
        or not match or int(match[1]) != 372 or s['full_tests_passed'] != 372
        or streaming_hash(Path(s['full_tests_log'])) != s['full_tests_log_sha256']
        or s['protocol_sha256'] != PROTOCOL_SHA256):
        raise ValueError('Complete372test confirmation source seal required')
    for name, digest in s['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest:
            raise ValueError('Measured confirmation source changed')
    return s


def prior_index(runtime, data_root=None, run_root=None, excluded=None):
    p = read_plan()
    data_root = Path(data_root or '/media/990Pro/Code/SAS/datasets/memory_update')
    run_root = Path(run_root or '/media/990Pro/Code/SAS/runs/memory_update')
    excluded = Path(excluded or p['dataset'])
    index = copy.prior_index(runtime, data_root, excluded)
    files = dict(index['source_files_sha256'])
    names = {'data.json', 'cases.json', 'examples.json', 'registry.json', 'queries.json', 'training_examples.json'}
    for f in sorted(run_root.rglob('*.json')):
        if f.name not in names or any('sources' in part for part in f.relative_to(run_root).parts):
            continue
        files[str(f)] = streaming_hash(f)
    seeds, ids = set(index['seeds']), set(index['identifiers'])
    prompts, prefixes = set(index['complete_prompt_ids_sha256']), set(index['complete_prefix_ids_sha256'])
    standard = {}
    for name, digest in files.items():
        f = Path(name)
        if streaming_hash(f) != digest:
            raise ValueError('Actual prior input changed during inventory')
        values = [json.loads(line) for line in f.read_text().splitlines()] if f.suffix == '.jsonl' else json.loads(f.read_text())
        for item in copy.walk(values):
            for key in ('seed', 'base_seed'):
                if type(item.get(key)) is int:
                    seeds.add(item[key])
            for key in ('id', 'case_id', 'base_case_id', 'base_stream_id'):
                if isinstance(item.get(key), str):
                    ids.add(item[key])
            for key in ('literal_prompt_ids_sha256', 'prompt_ids_sha256'):
                if isinstance(item.get(key), str):
                    prompts.add(item[key])
            for key in ('ledger_ids_sha256', 'source_key', 'prefix_key'):
                if isinstance(item.get(key), str):
                    prefixes.add(item[key])
            if isinstance(item.get('prefixes'), dict):
                prefixes.update(k for k in item['prefixes'] if re.fullmatch('[a-f0-9]{64}', k))
            tokens, boundary = item.get('input_ids'), item.get('ledger_tokens')
            if isinstance(tokens, list) and tokens and all(type(v) is int for v in tokens):
                prompts.add(data_fingerprint(tokens))
                if type(boundary) is int and 0 < boundary <= len(tokens):
                    prefixes.add(data_fingerprint(tokens[:boundary]))
            if isinstance(item.get('prefix'), list) and isinstance(item.get('variants'), dict):
                standard[data_fingerprint(item)] = item
    for case in standard.values():
        for variant, value in case['variants'].items():
            if not isinstance(value, dict) or not isinstance(value.get('tail'), list):
                continue
            for query in value.get('queries', []):
                ledger, prompt = prompt_for(runtime, case, variant, query)
                prefixes.add(data_fingerprint(runtime.ids(ledger)))
                prompts.add(data_fingerprint(runtime.ids(prompt)))
    return {**index, 'source_files_sha256': files, 'seeds': sorted(seeds), 'identifiers': sorted(ids),
            'complete_prompt_ids_sha256': sorted(prompts), 'complete_prefix_ids_sha256': sorted(prefixes),
            'all_dataset_and_run_standard_cases_reconstructed': len(standard)}


def build_inputs(runtime, set_seeds, copy_seeds, index):
    if set(set_seeds + copy_seeds) & set(index['seeds']):
        raise ValueError('Fixed prospective seed already observed; never resample')
    refs = {'excluded_seeds': index['seeds'], 'excluded_prefix_hashes': index['complete_prefix_ids_sha256'],
            'excluded_prompt_hashes': index['complete_prompt_ids_sha256']}
    cfg = {'seed_rule': {'start': set_seeds[0], 'count': len(set_seeds)}}
    cases, reg, queries = natural.build_registry(runtime, cfg, refs)
    if sorted({c['seed'] for c in cases}) != set_seeds:
        raise ValueError('Frozen seed selection silently changed')
    reg, queries = wording.build_registry(runtime, json.loads(wording.PROTOCOL.read_text()), reg, queries)
    cp = json.loads(copy.PROTOCOL.read_text())
    cp['fresh_seeds'] = copy_seeds
    copy_cases = copy.make_cases(cp)
    cr = copy_registry(copy_cases, runtime)
    case_map = {c['id']: c for c in copy_cases}
    copy_contexts, copy_queries = {}, {}
    for q in cr['physical']:
        representative = q['representative']
        case = case_map[representative['case_id']]
        query = next(x for x in case['variants'][representative['variant']]['queries'] if x['role'] == representative['role'])
        ledger, prompt = prompt_for(runtime, case, representative['variant'], query)
        tokens = runtime.ids(prompt)
        if data_fingerprint(tokens) != q['id']:
            raise ValueError('Frozen COPY physical prompt differs')
        cid = 'copy:' + q['source_key']
        c = copy_contexts.setdefault(cid, {'id': cid, 'ledger': ledger, 'ledger_tokens': q['ledger_tokens'],
                                          'source_key': q['source_key'], 'query_ids': []})
        if c['ledger'] != ledger:
            raise ValueError('COPY prefix token identity has different text')
        c['query_ids'].append(q['id'])
        copy_queries[q['id']] = {**q, 'context_id': cid, 'input_ids': tokens}
    return {'set_cases': cases, 'set_registry': reg, 'set_queries': queries, 'copy_cases': copy_cases,
            'copy_registry': cr, 'copy_contexts': list(copy_contexts.values()), 'copy_queries': copy_queries,
            'query_order': [k for c in reg['contexts'] for k in c['query_ids']] +
                           [k for c in copy_contexts.values() for k in c['query_ids']]}


def check_novelty(inputs, index):
    cases = inputs['set_cases'] + inputs['copy_cases']
    queries = list(inputs['set_queries'].values()) + list(inputs['copy_queries'].values())
    seeds = {c['seed'] for c in cases}
    ids = {c['id'] for c in cases} | {c['base_case_id'] for c in inputs['copy_cases']}
    prompts = {data_fingerprint(q['input_ids']) for q in queries}
    prefixes = {q['source_key'] for q in queries}
    overlap = {'seeds': sorted(seeds & set(index['seeds'])),
               'ids': sorted(ids & set(index['identifiers'])),
               'prompts': sorted(prompts & set(index['complete_prompt_ids_sha256'])),
               'prefixes': sorted(prefixes & set(index['complete_prefix_ids_sha256']))}
    if any(overlap.values()):
        raise ValueError('Actual prior inputs overlap fixed confirmation; never resample: ' + str(overlap))
    if (len(inputs['set_cases']), len(inputs['set_registry']['contexts']), len(inputs['set_queries']),
        len(inputs['copy_cases']), len(inputs['copy_contexts']), len(inputs['copy_queries']),
        len(inputs['copy_registry']['bindings']), len(inputs['query_order'])) != (16, 144, 864, 48, 56, 336, 432, 1200):
        raise ValueError('Every fixed condition and physical question required')
    if len(prompts) != 1200 or len(set(inputs['query_order'])) != 1200:
        raise ValueError('Unexpected physical prompt overlap')
    return {'overlap': overlap, 'base_seeds': sorted(seeds), 'physical_prompts': 1200,
            'contexts': 200, 'prior_input_files': len(index['source_files_sha256'])}


def registered_inputs():
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    p = read_plan()
    packet = Path(p['dataset'])
    m = json.loads((packet / 'manifest.json').read_text())
    seal = source_seal()
    lock = json.loads((packet / 'training-lock.json').read_text())
    index = json.loads((packet / 'prior-inputs.json').read_text())
    inputs = json.loads((packet / 'inputs.json').read_text())
    if (m['status'] != 'registered_fixed_confirmation_before_inference' or m['protocol_sha256'] != PROTOCOL_SHA256
        or m['training'] or m['model_inference'] or m['counts'] != p['counts']
        or m['source_seal_sha256'] != streaming_hash(Path(p['source_seal']))
        or m['source_sha256'] != seal['source_sha256'] or lock['training'] != training_gate()
        or json.loads((packet / 'plan.json').read_text()) != p
        or not datetime.fromisoformat(lock['training']['completed_utc']) < datetime.fromisoformat(lock['locked_utc'])
        or not datetime.fromisoformat(seal['created_utc']) < datetime.fromisoformat(lock['locked_utc'])
        or not datetime.fromisoformat(lock['locked_utc']) < datetime.fromisoformat(m['generation_started_utc'])
        or not datetime.fromisoformat(m['generation_started_utc']) < datetime.fromisoformat(m['registered_utc'])):
        raise ValueError('Complete post-training fixed confirmation registration required')
    for name, digest in m['payload_sha256'].items():
        if streaming_hash(packet / name) != digest:
            raise ValueError('Registered confirmation payload changed')
    for name, digest in m['source_sha256'].items():
        if streaming_hash(packet / 'sources' / name) != digest:
            raise ValueError('Confirmation source snapshot changed')
    for name, digest in index['source_files_sha256'].items():
        if streaming_hash(Path(name)) != digest:
            raise ValueError('Actual prior input inventory changed')
    rebuilt = build_inputs(tokenizer_runtime(json.loads((ROOT / p['parent_training_protocol']).read_text())['model_path']),
                           p['set_seeds'], p['copy_seeds'], index)
    if inputs != rebuilt or m['novelty'] != check_novelty(inputs, index):
        raise ValueError('Complete literal tokenizer/interpreter reconstruction differs')
    return p, m, inputs, lock
