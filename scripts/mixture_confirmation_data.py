"""Fixed seven-model, observed/fresh inputs gated by both complete training audits."""
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.readout_confirmation_data import (build_inputs, check_novelty, prior_index,
                                               training_gate as original_training_gate)

PROTOCOL = ROOT / 'configs/readout_mixture_confirmation_plan.json'
PROTOCOL_SHA256 = '8b5c777474e14bcc2debcef76aa6c9cf16e46d0843dd999709548369ac0dda69'
OLD_MODELS = ('native', 'fixed_gdn', 'query', 'gate', 'query_gate')
NEW_MODELS = ('gate_set_replay', 'gate_set_copy_mix')
MODELS = OLD_MODELS + NEW_MODELS
REGIMES = dict(zip(NEW_MODELS, ('set_replay', 'set_copy_mix')))
NEW_SOURCES = {'configs/readout_mixture_confirmation_plan.json', 'scripts/mixture_confirmation_data.py',
               'scripts/prepare_mixture_confirmation.py', 'scripts/mixture_confirmation_runtime.py',
               'scripts/run_mixture_confirmation.py', 'scripts/analyze_mixture_confirmation.py',
               'tests/test_mixture_confirmation.py'}


def read_plan():
    if streaming_hash(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Registered mixture confirmation design changed')
    p = json.loads(PROTOCOL.read_text())
    if (p['models'] != list(MODELS) or p['observed_new_models'] != list(NEW_MODELS)
        or p['set_seeds'] != list(range(160000, 160008)) or p['copy_seeds'] != list(range(161000, 161008))
        or p['planned_sources'] != 303 or p['full_cpu_tests_required'] != 388
        or set(p['new_sources']) != NEW_SOURCES
        or streaming_hash(ROOT / p['parent_training_protocol']) != p['parent_training_protocol_sha256']
        or any(p[k] for k in ('new_environment', 'new_backbone', 'training', 'oracle', 'donor',
                              'implementation_started', 'confirmation_generated', 'inference_started',
                              'architecture_superiority_claim'))):
        raise ValueError('Fixed seven models, cohort scope and no-training confirmation required')
    return p


def sources():
    p = read_plan()
    path = Path(p['parent_source_seal'])
    if streaming_hash(path) != p['parent_source_seal_sha256']:
        raise ValueError('Parent296source seal changed')
    prior = json.loads(path.read_text())['source_sha256']
    if len(prior) != 296:
        raise ValueError('All296predecessor sources required')
    for name, digest in prior.items():
        if streaming_hash(ROOT / name) != digest:
            raise ValueError('Sealed predecessor changed: ' + name)
    return set(prior) | NEW_SOURCES


def training_gate():
    p = read_plan()
    path, run = Path(p['mixture_training_audit']), Path(p['mixture_training_run'])
    if not path.is_file():
        raise ValueError('Both final mixture regimes and independent full audit must finish first')
    from scripts.analyze_readout_mixture_training import audit_run
    audit = json.loads(path.read_text())
    if (audit != audit_run(run) or audit['status'] != 'completed_and_audited_paired_mixture_training'
        or audit['optimizer_updates_audited'] != 864 or audit['physical_observed_readouts_audited'] != 2400
        or {k: v['fit_questions'] for k, v in audit['arms'].items()} != {'set_replay': 864, 'set_copy_mix': 1536}):
        raise ValueError('Complete paired training and every final observed readout required')
    files = [run / 'manifest.json', path, run / 'normal-history-references.json']
    for regime in ('set_replay', 'set_copy_mix'):
        files.extend(run / regime / n for n in ('training.jsonl', 'final-fit.jsonl'))
        for step in (144, 288, 432):
            files.extend(run / regime / f'step-{step:06d}' / n
                         for n in ('controller.safetensors', 'checkpoint.json', 'optimizer.pt'))
    mixture = {'training_run': str(run), 'completed_utc': audit['completed_utc'],
               'files_sha256': {str(f): streaming_hash(f) for f in files},
               'final_states': {name: v['checkpoints'][-1]['state_sha256'] for name, v in audit['arms'].items()}}
    return {'mixture_training': mixture, 'original_training': original_training_gate()}


def observed_inputs():
    p = read_plan()
    for name, digest in p['observed_files_sha256'].items():
        if streaming_hash(Path(name)) != digest:
            raise ValueError('Locked observed input or archived baseline changed')
    run, packet = Path(p['observed_run']), Path(p['observed_packet'])
    manifest = json.loads((run / 'manifest.json').read_text())
    audit = json.loads(Path(p['observed_audit']).read_text())
    registration = json.loads((packet / 'manifest.json').read_text())
    if (manifest['status'] != 'completed' or manifest['model_order'] != list(OLD_MODELS)
        or audit['status'] != 'completed_and_audited_frozen_readout_confirmation'
        or audit['physical_readouts_audited'] != 6000 or audit['manifest_sha256'] != streaming_hash(run / 'manifest.json')
        or manifest['packet_manifest_sha256'] != streaming_hash(packet / 'manifest.json')):
        raise ValueError('Complete locked observed five-model reference required')
    inputs = json.loads((packet / 'inputs.json').read_text())
    if len(inputs['query_order']) != 1200:
        raise ValueError('Every old question must remain in development')
    for name in OLD_MODELS:
        if streaming_hash(run / (name + '.jsonl')) != manifest['models'][name]['results_sha256']:
            raise ValueError('An archived baseline answer file changed')
    return inputs, manifest


def model_spec(name, lock):
    if name not in MODELS:
        raise ValueError('Unknown frozen confirmation model')
    if name in ('native', 'fixed_gdn'):
        return {'name': name, 'parameters': 0, 'checkpoint': None}
    p = read_plan()
    if name in NEW_MODELS:
        regime = REGIMES[name]
        training, folder, mode, rank, state_key = lock['mixture_training'], regime, 'gate', 32, regime
    else:
        training, folder, mode = lock['original_training'], name, name
        rank, state_key = (16 if name == 'query_gate' else 32), name
    path = Path(training['training_run']) / folder / 'step-000432/controller.safetensors'
    return {'name': name, 'parameters': 131072, 'mode': mode, 'rank': rank,
            'checkpoint': str(path), 'checkpoint_sha256': training['files_sha256'][str(path)],
            'state_sha256': training['final_states'][state_key]}


def source_seal():
    p = read_plan()
    s = json.loads(Path(p['source_seal']).read_text())
    log = Path(s['full_tests_log'])
    match = re.search(r'Ran (\d+) tests in [\d.]+s\s+OK\s*$', log.read_text())
    if (s['status'] != 'sealed303mixture_confirmation_sources_before_generation'
        or set(s['source_sha256']) != sources() or len(s['source_sha256']) != 303
        or not match or int(match[1]) != 388 or s['full_tests_passed'] != 388
        or streaming_hash(log) != s['full_tests_log_sha256'] or s['protocol_sha256'] != PROTOCOL_SHA256):
        raise ValueError('Complete388test303source confirmation seal required')
    for name, digest in s['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest:
            raise ValueError('Measured confirmation source changed')
    return s


def registered_inputs():
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    p = read_plan()
    seal = source_seal()
    path = Path(p['dataset'])
    m = json.loads((path / 'manifest.json').read_text())
    lock = json.loads((path / 'training-lock.json').read_text())
    if (m['status'] != 'registered_mixture_confirmation_after_complete_training'
        or m['protocol_sha256'] != PROTOCOL_SHA256 or m['source_sha256'] != seal['source_sha256']
        or m['source_seal_sha256'] != streaming_hash(Path(p['source_seal']))
        or set(m['payload_sha256']) != {'plan.json', 'training-lock.json', 'inputs.json', 'prior-inputs.json'}
        or lock['training'] != training_gate() or m['counts'] != p['counts']
        or any(m[k] for k in ('training', 'model_inference'))
        or json.loads((path / 'plan.json').read_text()) != p
        or not datetime.fromisoformat(lock['training']['mixture_training']['completed_utc']) < datetime.fromisoformat(lock['locked_utc'])
        or not datetime.fromisoformat(seal['created_utc']) < datetime.fromisoformat(lock['locked_utc'])
        or not datetime.fromisoformat(lock['locked_utc']) < datetime.fromisoformat(m['generation_started_utc']) < datetime.fromisoformat(m['registered_utc'])):
        raise ValueError('Complete post-training observed/fresh confirmation registration required')
    for name, digest in m['payload_sha256'].items():
        if streaming_hash(path / name) != digest:
            raise ValueError('Registered confirmation payload changed')
    for name, digest in m['source_sha256'].items():
        if streaming_hash(path / 'sources' / name) != digest:
            raise ValueError('Registered confirmation source snapshot changed')
    index = json.loads((path / 'prior-inputs.json').read_text())
    for name, digest in index['source_files_sha256'].items():
        if streaming_hash(Path(name)) != digest:
            raise ValueError('Actual prior input changed')
    data = json.loads((path / 'inputs.json').read_text())
    observed, _ = observed_inputs()
    if data['observed'] != observed:
        raise ValueError('Observed reference queries were altered')
    base = json.loads((ROOT / 'configs/readout_adapter_training_plan.json').read_text())
    rebuilt = build_inputs(tokenizer_runtime(base['model_path']), p['set_seeds'], p['copy_seeds'], index)
    if data['fresh'] != rebuilt or m['novelty'] != check_novelty(rebuilt, index):
        raise ValueError('Complete fixed fresh cohort no longer reconstructs')
    return p, m, data, lock['training']
