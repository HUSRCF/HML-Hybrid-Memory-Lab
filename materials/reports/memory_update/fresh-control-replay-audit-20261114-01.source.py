"""Independent audit of the fixed posthoc control-change operational replay."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_fresh_copy_confirmation import audit_readout, audit_proofs
from scripts.canonical_necessity_common import audit_affines
from scripts.copy_readout_runtime import readout
from scripts.run_controller_trial import write_json
from scripts.run_fresh_control_replay import PROTOCOL, PROTOCOL_SHA256, input_scope, sources, trials


def audit_model(scope, name, rows, state):
    plan = scope['plan']
    selected = [q for q in plan['queries'] if q['model'] == name]
    originals = {q['id']: q for q in selected}
    expected_trials = trials(plan, name)
    if (len(rows) != 12 or state['completed_trials'] != 6 or state['recorded_rows'] != 12 or
        len(state['proofs']) != 6 or 'active_trial' in state or state['token_policy'] != 'question'):
        raise ValueError('Both queries, all three repetitions and both readouts required')
    if [p['id'] for p in state['prefix_checks']] != [q['id'] for q in selected]:
        raise ValueError('All four actual question-only native prefixes required')
    native_hashes = {}
    for p, entry in zip(state['prefix_checks'], selected):
        q = scope['queries'][entry['id']]
        if (p['source_key'] != q['source_key'] or p['ledger_tokens'] != q['ledger_tokens'] or
            not p['complete_KV_recurrent_conv_bookkeeping_exact'] or not p['last_prefix_logits_exact'] or
            not re.fullmatch(r'[0-9a-f]{64}', p['prefix_sha256'])):
            raise ValueError('Actual native recurrent/conv/KV prefix or logits changed')
        native_hashes[p['id']] = p['prefix_sha256']
    for index, trial in enumerate(expected_trials):
        proof = state['proofs'][index]
        if {k: proof[k] for k in trial} != trial:
            raise ValueError('Prescribed trial order or complete proof coverage changed')
        entry, q = originals[trial['id']], scope['queries'][trial['id']]
        full, masked = rows[2 * index:2 * index + 2]
        regime = trial['masked_regime']
        for actual_regime, row, expected in (('full', full, entry['expected_full']),
                                              (regime, masked, entry['expected_intervened'])):
            audit_readout(readout(row), q)
            if (any(row[k] != v for k, v in trial.items()) or row['model'] != name or
                row['regime'] != actual_regime or row['token_policy'] != 'question' or row['prefill'] != 'full' or
                row['arithmetic_profile'] != plan['arithmetic_profile'] or
                row['registration_sha256'] != data_fingerprint(q) or
                row['mask_registration_sha256'] != (None if actual_regime == 'full' else entry['mask_registration_sha256']) or
                row['keep_bits_sha256'] != (None if actual_regime == 'full' else entry['keep_bits_sha256']) or
                not row['exact_original_replay'] or readout(row) != expected):
                raise ValueError('Exact original five-field replay, query, regime or mask changed')
            if actual_regime == 'full' and row['hook_calls'] is not None:
                raise ValueError('Unmasked full reference unexpectedly masked')
            if actual_regime != 'full' and (row['hook_calls']['prefill'] != 1 or row['hook_calls']['cached'] < 1):
                raise ValueError('Actual masked prefill/cached generation incomplete')
        audit_proofs([proof['proof']], [q['id']], {q['id']: full}, {regime: {q['id']: masked}}, (regime,))
        if proof['proof']['regimes'][regime]['prefix_sha256'] != native_hashes[q['id']]:
            raise ValueError('Masked replay prefix differs from the actual original native prefix')
        if (full['greedy_exact'] == masked['greedy_exact'] or
            ('harm' if full['greedy_exact'] else 'repair') != trial['direction']):
            raise ValueError('Selected original correctness flip no longer reproduces')
    return {'recorded_readouts': len(rows), 'trials': len(expected_trials),
        'unmasked_exact_repeats': len(expected_trials), 'all_history_exact_controls': len(expected_trials),
        'actual_masked_attention_and_complete_prefix_proofs': len(expected_trials),
        'question_only_native_prefix_checks': len(native_hashes)}


def audit_run(path):
    path = Path(path)
    scope = input_scope(); plan = scope['plan']
    manifest = json.loads((path / 'manifest.json').read_text())
    before, after = manifest['weights_before'], manifest['weights_after']
    if (manifest['status'] != 'completed' or any(manifest[k] for k in
        ('training', 'new_adapters', 'new_data', 'independent_scientific_confirmation', 'can_change_primary_gates')) or
        not manifest['posthoc'] or manifest['protocol_sha256'] != PROTOCOL_SHA256 or
        json.loads((path / 'plan.json').read_text()) != plan or
        json.loads((path / 'precision_addendum.json').read_text()) != scope['profile'] or
        manifest['original_primary_gates_unchanged'] != scope['primary_gates'] or
        any(manifest[k] != plan[k] for k in ('model_path', 'dataset_manifest_sha256', 'mask_registry_sha256',
            'arithmetic_profile', 'precision_addendum_sha256', 'packages', 'completed_fresh_audit_sha256',
            'completed_fresh_manifest_sha256', 'control_change_report_sha256')) or
        manifest['actual_eos_token_id'] != scope['primary_manifest']['actual_eos_token_id'] or
        manifest['generation_config_sha256'] != scope['primary_manifest']['generation_config_sha256'] or
        manifest['generation_config_sha256'] != {p.name: streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')} or
        manifest['model_order'] != plan['models'] or manifest['completed_models'] != plan['models'] or
        list(manifest['models']) != plan['models'] or set(manifest['source_sha256']) != sources() or
        before != after or after['sha256'] != plan['native_loaded_weights_sha256'] or
        not after['all_parameters_frozen'] or not after['no_parameter_gradients'] or not manifest['hooks_restored'] or
        not datetime.fromisoformat(plan['registered_utc']) < datetime.fromisoformat(manifest['started_utc']) <
            datetime.fromisoformat(manifest['completed_utc'])):
        raise ValueError('Frozen posthoc scope, source closure, original gates, chronology or native weights changed')
    for name, digest in manifest['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / 'sources' / name) != digest:
            raise ValueError('Actual or archived replay source changed')
    keys = [q['id'] for q in plan['queries']]
    if [r['id'] for r in manifest['native_anchors']] != keys:
        raise ValueError('Complete four-query pre-controller native anchors required')
    anchors = {}
    for row in manifest['native_anchors']:
        audit_readout(row['readout'], scope['queries'][row['id']])
        if not row['exact_completed_fresh_native_replay'] or row['readout'] != scope['native_references'][row['id']]:
            raise ValueError('Native anchor failed exact completed-primary replay')
        anchors[row['id']] = row['readout']
    counts = {k: 0 for k in ('recorded_readouts', 'trials', 'unmasked_exact_repeats', 'all_history_exact_controls',
        'actual_masked_attention_and_complete_prefix_proofs', 'question_only_native_prefix_checks')}
    detach_count = 0
    for name in plan['models']:
        state = manifest['models'][name]
        spec = scope['source_plan']['models'][name]
        if ({k: state[k] for k in plan['fixed_checkpoints'][name]} != plan['fixed_checkpoints'][name] or
            state['adapter_parameters'] != spec['parameters'] or
            state['adapter_state_sha256_after'] != state['adapter_state_sha256_before'] or
            not state['all_adapter_parameters_frozen'] or not state['arithmetic_hooks_removed'] or not state['hooks_restored']):
            raise ValueError('Actual frozen adapter bytes, gradients, scope or cleanup changed')
        audit_affines(state, plan['arithmetic_profile'])
        result_path = path / f'{name}.jsonl'
        if streaming_hash(result_path) != state['results_sha256']:
            raise ValueError('Actual replay readout file changed')
        rows = [json.loads(line) for line in result_path.read_text().splitlines()]
        model_counts = audit_model(scope, name, rows, state)
        for key, value in model_counts.items():
            counts[key] += value
        if ([r['id'] for r in state['native_restore_checks']] != keys or
            any(not r['exact'] or r['readout'] != anchors[r['id']] for r in state['native_restore_checks'])):
            raise ValueError('All four native anchors must replay after each controller arm')
        detach_count += len(state['native_restore_checks'])
    required = {'recorded_readouts': 24, 'trials': 12, 'unmasked_exact_repeats': 12,
        'all_history_exact_controls': 12, 'actual_masked_attention_and_complete_prefix_proofs': 12,
        'question_only_native_prefix_checks': 4}
    if counts != required or detach_count != 8:
        raise ValueError('Complete prescribed replay and control coverage required')
    counts.update(precontroller_native_anchors=len(anchors), detached_native_replays=detach_count,
        measure_calls_including_controls=counts['recorded_readouts'] + counts['unmasked_exact_repeats'] +
            counts['all_history_exact_controls'] + len(anchors) + detach_count)
    if counts['measure_calls_including_controls'] != plan['planned_total_measure_calls_including_controls']:
        raise ValueError('Measure-call coverage differs from the registered complete controls')
    return {'status': 'completed_and_audited_posthoc_control_replay', 'operational_replay_passed': True,
        'posthoc': True, 'independent_scientific_confirmation': False, 'can_change_primary_gates': False,
        'source_run': str(path), 'manifest_sha256': streaming_hash(path / 'manifest.json'),
        'protocol_sha256': streaming_hash(PROTOCOL), 'dataset_manifest_sha256': plan['dataset_manifest_sha256'],
        'mask_registry_sha256': plan['mask_registry_sha256'], 'completed_fresh_audit_sha256': plan['completed_fresh_audit_sha256'],
        'control_change_report_sha256': plan['control_change_report_sha256'],
        'arithmetic_profile': plan['arithmetic_profile'], 'weights_sha256': after['sha256'], 'counts': counts,
        'original_primary_gates_unchanged': scope['primary_gates'],
        'selected_changes': [{'model': q['model'], 'id': q['id'], 'regime': q['regime'], 'direction': q['direction'],
            'bindings': q['bindings'], 'full': q['expected_full'], 'intervened': q['expected_intervened'],
            'exact_repetitions': plan['repeats']} for q in plan['queries']],
        'interpretation': 'All3harms and1repair operationally replayed with original prefix memories. This reused-sample diagnostic neither repairs the failed original gates nor establishes recurrent storage loss, semantic dependency coordination or a new architecture.',
        'analysis_source_sha256': streaming_hash(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior independent audits')
    report = audit_run(args.run)
    write_json(args.output, report)
    source = args.output.with_suffix('.source.py'); source.write_bytes(Path(__file__).read_bytes())
    rows = [json.loads(line) for name in ('local_question', 'gdn_question_only')
            for line in (args.run / f'{name}.jsonl').read_text().splitlines()]
    table = args.output.with_suffix('.csv')
    with table.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(args.output.with_suffix('.provenance.json'), {'status': 'complete_posthoc_replay_artifacts',
        'manifest_sha256': report['manifest_sha256'], 'counts': report['counts'],
        'artifacts_sha256': {str(p): streaming_hash(p) for p in (args.output, source, table)},
        'runtime_sources_sha256': json.loads((args.run / 'manifest.json').read_text())['source_sha256']})
    print(json.dumps({'status': report['status'], 'counts': report['counts'],
        'original_scientific_gates_unchanged': {k: v['passed'] for k, v in report['original_primary_gates_unchanged'].items()}}), flush=True)
