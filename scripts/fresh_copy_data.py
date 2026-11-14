"""A fixed new COPY cohort, with actual prior-input novelty and frozen masks."""

from datetime import datetime
import json
from pathlib import Path

from sas_memory.generalization_tasks import build_generalization_cases, validate_generalization_cases
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.canonical_oracle_common import sources as previous_sources, registered_inputs as previous_inputs
from scripts.copy_readout_data import build_masks as oracle_masks, sample_ids, safety_samples, selected_inputs
from scripts.copy_necessity_data import build_masks as removal_masks
from scripts.copy_transfer_data import MODELS, ROLES, VARIANTS, build_registry
from scripts.run_controller_trial import prompt_for

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/fresh_copy_memory_confirmation_plan.json'
PROTOCOL_SHA256 = 'e310e4690db41d0c673c93f0785301bd8503c33cfd91c176416010ad672c54da'
DESTINATION = Path('/media/990Pro/Code/SAS/datasets/memory_update/fresh-copy-memory-confirmation-v1')
DATA_ROOT = DESTINATION.parent
PRIMARY = ('local_question', 'gdn_question_only')
MASKED_REGIMES = ('oracle', 'remove_support', 'remove_matched_other')
DATA_SOURCES = tuple(sorted(previous_sources() | {
    'configs/fresh_copy_memory_confirmation_plan.json', 'scripts/fresh_copy_data.py',
    'scripts/prepare_fresh_copy_confirmation.py',
}))
RUNTIME_SOURCES = tuple(sorted(set(DATA_SOURCES) | {
    'scripts/fresh_copy_metrics.py', 'scripts/fresh_copy_runtime.py',
    'scripts/run_fresh_copy_confirmation.py', 'scripts/analyze_fresh_copy_confirmation.py',
    'scripts/compare_fresh_copy_smokes.py',
}))


def sources():
    return set(RUNTIME_SOURCES)


def read_plan():
    plan = json.loads(PROTOCOL.read_text())
    if (streaming_hash(PROTOCOL) != PROTOCOL_SHA256 or plan['model_order'] != list(MODELS) or
        plan['primary_readout_models'] != list(PRIMARY) or plan['fresh_seeds'] != list(range(61000, 61008)) or
        any(plan[k] for k in ('training', 'new_adapters', 'new_models', 'new_environment', 'case_generation_started', 'model_inference_started')) or
        not plan['local_files_only'] or
        plan['regimes_by_model'] != {m: ['full', *MASKED_REGIMES] if m in PRIMARY else ['full'] for m in MODELS}):
        raise ValueError('Pre-generation fixed fresh-cohort scope changed')
    return plan


def model_order(reverse=False):
    return list(reversed(MODELS)) if reverse else list(MODELS)


def prior_references(plan):
    old_registration, old_plan, _, _, _, _ = previous_inputs()
    if (plan['models'] != old_plan['models'] or streaming_hash(Path(plan['fixed_checkpoint_protocol'])) != plan['fixed_checkpoint_protocol_sha256'] or
        old_plan['model_path'] != plan['model_path'] or old_plan['packages'] != plan['packages'] or
        streaming_hash(ROOT / plan['profile_source']) != plan['profile_source_sha256'] or
        streaming_hash(ROOT / 'sas_memory/generalization_tasks.py') != plan['case_generator_source_sha256'] or
        streaming_hash(Path(plan['precision_addendum'])) != plan['precision_addendum_sha256']):
        raise ValueError('Fixed checkpoint family, generator or canonical arithmetic changed')
    refs = {**old_registration['references']}
    for kind, required, count in (('oracle', 'completed_and_audited_canonical_oracle', 1344),
                                   ('necessity', 'completed_and_audited_canonical', 8064)):
        file = Path(plan[f'prior_canonical_{kind}_audit']); report = json.loads(file.read_text())
        run = Path(report['source_run']); manifest = json.loads((run / 'manifest.json').read_text())
        gate = 'frozen_prefix_sufficiency' if kind == 'oracle' else 'whole_record_necessity'
        if (streaming_hash(file) != plan[f'prior_canonical_{kind}_audit_sha256'] or report['status'] != required or
            report['new_physical_readouts_audited'] != count or not report['metrics'][gate]['passed'] or
            not report['metrics']['prior_shared_history_repeat_primaries_remain_failed'] or
            report['weights_sha256'] != plan['native_loaded_weights_sha256'] or report['arithmetic_profile'] != plan['arithmetic_profile'] or
            streaming_hash(file.with_suffix('.source.py')) != report['analysis_source_sha256'] or
            streaming_hash(run / 'manifest.json') != report['manifest_sha256'] or
            not datetime.fromisoformat(manifest['completed_utc']) < datetime.fromisoformat(plan['registered_utc'])):
            raise ValueError('Completed prior numerical-profile evidence changed')
        for name, digest in manifest['source_sha256'].items():
            if streaming_hash(ROOT / name) != digest or streaming_hash(run / 'sources' / name) != digest:
                raise ValueError('Completed prior live/archived source changed')
        for model, state in manifest['models'].items():
            for regime, digest in state['results_sha256'].items():
                if streaming_hash(run / f'{model}.{regime}.jsonl') != digest:
                    raise ValueError('Completed prior actual observations changed')
        refs[f'prior_canonical_{kind}_audit'] = str(file)
        refs[f'prior_canonical_{kind}_audit_sha256'] = streaming_hash(file)
        refs[f'prior_canonical_{kind}_manifest_sha256'] = report['manifest_sha256']
    return old_plan, refs


def walk(value):
    if isinstance(value, dict):
        yield value
        for item in value.values(): yield from walk(item)
    elif isinstance(value, list):
        for item in value: yield from walk(item)


def prior_index(runtime, data_root=DATA_ROOT, excluded=DESTINATION):
    """Use actual registered input files, not names or historical score summaries."""
    seeds, identifiers, prompts, prefixes, cases = set(), set(), set(), set(), {}
    files, directories = {}, []
    for folder in sorted(Path(data_root).iterdir()):
        if not folder.is_dir() or folder.resolve() == Path(excluded).resolve() or not (folder / 'manifest.json').is_file(): continue
        directories.append(str(folder))
        for file in sorted(folder.rglob('*')):
            if not file.is_file() or file.suffix not in ('.json', '.jsonl') or 'sources' in file.relative_to(folder).parts: continue
            files[str(file)] = streaming_hash(file)
            values = [json.loads(line) for line in file.read_text().splitlines()] if file.suffix == '.jsonl' else json.loads(file.read_text())
            for item in walk(values):
                if type(item.get('seed')) is int: seeds.add(item['seed'])
                for field in ('id', 'case_id', 'base_case_id', 'base_stream_id'):
                    if isinstance(item.get(field), str): identifiers.add(item[field])
                for field in ('prompt_ids_sha256',):
                    if isinstance(item.get(field), str): prompts.add(item[field])
                for field in ('ledger_ids_sha256', 'source_key'):
                    if isinstance(item.get(field), str): prefixes.add(item[field])
                ids = item.get('input_ids'); boundary = item.get('ledger_tokens')
                if isinstance(ids, list) and ids and all(type(i) is int for i in ids):
                    prompts.add(data_fingerprint(ids))
                    if type(boundary) is int and 0 < boundary <= len(ids): prefixes.add(data_fingerprint(ids[:boundary]))
                if isinstance(item.get('prefix'), list) and isinstance(item.get('variants'), dict):
                    cases.setdefault(data_fingerprint(item), item)
    reconstructed = 0
    for case in cases.values():
        for variant, detail in case['variants'].items():
            if not isinstance(detail, dict) or not isinstance(detail.get('tail'), list): continue
            for query in detail.get('queries', []):
                ledger, prompt = prompt_for(runtime, case, variant, query)
                prefixes.add(data_fingerprint(runtime.ids(ledger))); prompts.add(data_fingerprint(runtime.ids(prompt)))
                reconstructed += 1
    return {'registered_directories': directories, 'source_files_sha256': files,
        'seeds': sorted(seeds), 'identifiers': sorted(identifiers),
        'complete_prompt_ids_sha256': sorted(prompts), 'complete_prefix_ids_sha256': sorted(prefixes),
        'reconstructed_case_queries': reconstructed, 'standard_cases_reconstructed': len(cases)}


def check_novelty(cases, registry, index):
    seeds = {c['seed'] for c in cases}
    ids = {c[k] for c in cases for k in ('id', 'base_case_id')}
    prompts = {q['id'] for q in registry['physical']}; prefixes = {q['source_key'] for q in registry['physical']}
    collisions = {'seeds': sorted(seeds & set(index['seeds'])), 'identifiers': sorted(ids & set(index['identifiers'])),
                  'complete_prompts': sorted(prompts & set(index['complete_prompt_ids_sha256'])),
                  'complete_prefixes': sorted(prefixes & set(index['complete_prefix_ids_sha256']))}
    if any(collisions.values()): raise ValueError(f'Fixed fresh cohort overlaps actual prior inputs; do not resample: {collisions}')
    return {'overlap': collisions, 'new_seeds': sorted(seeds), 'new_case_identifiers': len(ids),
            'new_physical_prompts': len(prompts), 'new_distinct_complete_prefixes': len(prefixes),
            'prior_registered_directories': len(index['registered_directories']),
            'prior_input_files': len(index['source_files_sha256']),
            'prior_complete_prompts': len(index['complete_prompt_ids_sha256']),
            'prior_complete_prefixes': len(index['complete_prefix_ids_sha256'])}


def make_cases(plan):
    cases = build_generalization_cases(seeds=tuple(plan['fresh_seeds']), depths=tuple(plan['dependency_depths']),
                                       question_styles=tuple(plan['question_styles']))
    counts = validate_generalization_cases(cases)
    if (counts['cases'] != 48 or counts['base_cases'] != 8 or
        [c['distractor_updates'] for c in cases[::6]] != plan['distractor_updates_by_seed_index']):
        raise ValueError('Complete fixed fresh depth/style/load grid changed')
    for case in cases:
        queries = {v: {q['role']: q for q in case['variants'][v]['queries']} for v in VARIANTS}
        if queries['control'][ROLES[0]]['answer'] == queries['treatment'][ROLES[0]]['answer']:
            raise ValueError('Order must change COPY target truth')
        if any(len({queries[v][r]['answer'] for v in VARIANTS}) != 1 for r in ROLES[1:]):
            raise ValueError('COPY order must preserve unrelated and historical truth')
    return cases


def make_masks(cases, registry, runtime):
    oracle, removal = oracle_masks(cases, registry, runtime), removal_masks(cases, registry, runtime)
    masks = {}
    for key in oracle:
        if oracle[key]['oracle_event_indices'] != removal[key]['oracle_event_indices']:
            raise ValueError('Oracle and necessary leaf differ in symbolic dependency path')
        masks[key] = {'id': key, 'oracle_registration': oracle[key], 'removal_registration': removal[key],
            'keep_bits': {'oracle': oracle[key]['keep_bits']['oracle'], **removal[key]['keep_bits']}}
    return masks


def registered_inputs(path=DESTINATION):
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    plan = read_plan(); path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    cases, registry, masks, index = [json.loads((path / f'{name}.json').read_text()) for name in ('cases','registry','masks','prior_inputs')]
    old_plan, refs = prior_references(plan)
    runtime = tokenizer_runtime(plan['model_path'])
    if (path != Path(plan['dataset_destination']) or manifest['status'] != 'registered_before_inference' or
        manifest['training'] or manifest['model_inference'] or manifest['counts'] != plan['counts'] or
        manifest['protocol_sha256'] != streaming_hash(PROTOCOL) or manifest['references'] != refs or
        json.loads((path / 'plan.json').read_text()) != plan or cases != make_cases(plan) or
        registry != build_registry(cases, runtime) or masks != make_masks(cases, registry, runtime) or
        manifest['cases_sha256'] != data_fingerprint(cases) or manifest['registry_sha256'] != data_fingerprint(registry) or
        manifest['mask_registry_sha256'] != data_fingerprint(masks) or manifest['prior_inputs_sha256'] != data_fingerprint(index) or
        manifest['freshness'] != check_novelty(cases, registry, index) or
        manifest['sample_query_ids'] != sample_ids(cases, registry) or manifest['native_prefix_sample_ids'] != safety_samples(registry) or
        set(manifest['source_sha256']) != set(DATA_SOURCES) or
        not datetime.fromisoformat(plan['registered_utc']) < datetime.fromisoformat(manifest['generation_started_utc']) < datetime.fromisoformat(manifest['registered_utc'])):
        raise ValueError('Actual fresh cases, complete masks, novelty, references or registration changed')
    for file, digest in index['source_files_sha256'].items():
        if streaming_hash(Path(file)) != digest: raise ValueError('Prior novelty input changed')
    for file, digest in manifest['source_sha256'].items():
        if streaming_hash(ROOT / file) != digest or streaming_hash(path / 'sources' / file) != digest:
            raise ValueError('Actual/archived fresh data source changed')
    return manifest, old_plan, cases, registry, masks
