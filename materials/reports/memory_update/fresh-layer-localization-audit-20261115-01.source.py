"""Independently audit all registered layer interventions without an accuracy gate."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_fresh_copy_confirmation import audit_readout
from scripts.canonical_necessity_common import audit_affines
from scripts.copy_readout_runtime import readout
from scripts.fresh_layer_controls import (PROTOCOL_SHA256, LAYERS, input_scope, sources,
    audit_receipts, audit_probabilities)
from scripts.run_controller_trial import write_json


def sha(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('Actual tensor/cache SHA256 required')


def audit_first(value, query):
    for key in ('argmax_id', 'gold_id', 'non_gold_id'):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError('Actual first-token integer IDs required')
    if (value['gold_id'] != query['answer_ids'][0] or value['non_gold_id'] == value['gold_id'] or
        not isinstance(value['argmax_text'], str) or
        any(not math.isfinite(value[k]) for k in ('gold_logit', 'non_gold_logit', 'gold_margin')) or
        value['gold_margin'] != value['gold_logit'] - value['non_gold_logit']):
        raise ValueError('Fixed gold first-token identity or actual logit margin changed')
    expected = value['gold_id'] if value['gold_margin'] > 0 else value['non_gold_id']
    if value['gold_margin'] == 0:
        expected = min(value['gold_id'], value['non_gold_id'])
    if value['argmax_id'] != expected:
        raise ValueError('First-token argmax inconsistent with strongest non-gold logit')
    sha(value['logits_sha256'])


def audit_query(scope, entry, rows, proofs, ref, original, all_history, equivalence, native_prefix):
    plan, q = scope['plan'], scope['queries'][entry['id']]
    order = plan['condition_order']
    if (len(rows) != 14 or len(proofs) != 14 or [r['condition'] for r in rows] != order or
        [p['condition'] for p in proofs] != order or any(s['id'] != q['id'] for s in
            (ref, original, all_history, equivalence)) or not equivalence['cache_input_logits_readout_exact']):
        raise ValueError('All14ordered conditions and both operator controls required for every query')
    for key in ('prefix_sha256', 'full_question_cache_sha256', 'attention23_input_sha256'):
        sha(ref[key])
    audit_first(ref['first_token'], q)
    if ref['prefix_sha256'] != native_prefix:
        raise ValueError('Reference is not the actual native question-only prefix')
    audit_readout(original['readout'], q); audit_readout(all_history['readout'], q)
    if (original['readout'] != entry['expected_intervened'] or original['historical_kv_fields_exact'] != 12 or
        original['calls']['prefill'] != 3 or original['calls']['cached'] < 1 or
        all_history['readout'] != entry['expected_full'] or not all_history['complete_cache_input_logits_exact'] or
        all_history['calls']['cached'] < 1):
        raise ValueError('Original global-mask or all-history exact readout control failed')
    for key in ('full_question_cache_sha256', 'attention23_input_sha256', 'logits_sha256'):
        sha(original[key])
    audit_receipts(all_history['calls'], all_history['receipts'], LAYERS, 3)
    for row, proof in zip(rows, proofs):
        label, layers = row['condition'], plan['conditions'][row['condition']]
        audit_readout(readout(row), q); audit_first(row['first_token'], q)
        if (row['id'] != q['id'] or row['model'] != entry['model'] or row['source_regime'] != entry['regime'] or
            row['direction'] != entry['direction'] or row['selected_layers'] != layers or
            row['token_policy'] != 'question' or row['prefill'] != 'full' or
            row['arithmetic_profile'] != plan['arithmetic_profile'] or
            row['registration_sha256'] != data_fingerprint(q) or
            any(row[k] != entry[k] for k in ('mask_registration_sha256', 'keep_bits_sha256')) or
            proof['id'] != q['id'] or proof['selected_layers'] != layers or
            not proof['complete_prefix_memory_exact'] or not proof['prefix_logits_exact'] or
            proof['prefix_sha256'] != native_prefix or proof['masked_prefix_sha256'] != native_prefix or
            proof['historical_kv_fields_exact'] != 12 or not proof['hooks_restored'] or
            proof['proof_calls'] != {'prefill': 2, 'cached': 0}):
            raise ValueError('Actual query, intervention, native prefix, historicalKV or cleanup proof changed')
        audit_receipts(proof['measure_calls'], proof['measure_receipts'], layers, 1)
        audit_receipts(proof['proof_calls'], proof['proof_receipts'], layers, 2)
        audit_probabilities(proof['attention_stats'], layers, proof['measure_calls'])
        for key in ('full_question_cache_sha256', 'attention23_input_sha256'):
            sha(proof[key])
        if (proof['full_question_cache_equals_unmasked'] !=
                (proof['full_question_cache_sha256'] == ref['full_question_cache_sha256']) or
            proof['attention23_input_exact_to_unmasked'] !=
                (proof['attention23_input_sha256'] == ref['attention23_input_sha256'])):
            raise ValueError('Actual full-question cache or final-reader input hash contradicts equality receipt')
        if label in ('full', 'only_layer_23') and (not proof['full_question_cache_equals_unmasked'] or
                not proof['attention23_input_exact_to_unmasked']):
            raise ValueError('Full/only-final-layer condition changed full-question cache or final-reader input')
        if label == 'full' and (readout(row) != entry['expected_full'] or row['first_token'] != ref['first_token']):
            raise ValueError('Full original five-field readout or first logits drifted')
        if label == 'all_layers' and (readout(row) != entry['expected_intervened'] or
                readout(row) != original['readout'] or
                proof['full_question_cache_sha256'] != original['full_question_cache_sha256'] or
                proof['attention23_input_sha256'] != original['attention23_input_sha256'] or
                row['first_token']['logits_sha256'] != original['logits_sha256']):
            raise ValueError('All-layer operator differs from exact original global operator')
    by = {r['condition']: r for r in rows}
    baseline, all_layers = by['full'], by['all_layers']
    last = by['only_layer_23']
    return {'model': entry['model'], 'id': q['id'], 'source_regime': entry['regime'],
        'direction': entry['direction'], 'bindings': entry['bindings'],
        'full': readout(baseline), 'all_layers': readout(all_layers),
        'only_final_layer': {'readout': readout(last), 'complete_question_cache_and_input_exact': True,
            'first_token_argmax_changed': last['first_token']['argmax_id'] != baseline['first_token']['argmax_id'],
            'full_generation_changed': last['generated'] != baseline['generated'],
            'first_token': last['first_token']},
        'single_layer': [{'layer': i, 'readout': readout(by[f'only_layer_{i}']),
            'reproduces_correctness_flip': by[f'only_layer_{i}']['greedy_exact'] == all_layers['greedy_exact'],
            'exact_generated_match_to_all_layers': by[f'only_layer_{i}']['generated'] == all_layers['generated'],
            'first_token': by[f'only_layer_{i}']['first_token']} for i in LAYERS],
        'conditional_leave_one_out': [{'omitted_layer': i, 'readout': readout(by[f'all_except_layer_{i}']),
            'removes_correctness_flip': by[f'all_except_layer_{i}']['greedy_exact'] != all_layers['greedy_exact'],
            'restores_full_generation': by[f'all_except_layer_{i}']['generated'] == baseline['generated'],
            'first_token': by[f'all_except_layer_{i}']['first_token']} for i in LAYERS]}


def audit_run(path):
    path = Path(path); scope = input_scope(); plan = scope['plan']
    manifest = json.loads((path / 'manifest.json').read_text())
    after = manifest['weights_after']
    if (manifest['status'] != 'completed' or not manifest['posthoc'] or not manifest['no_accuracy_gate'] or
        any(manifest[k] for k in ('training', 'new_adapters', 'new_data', 'independent_scientific_confirmation',
            'can_change_primary_gates')) or manifest['protocol_sha256'] != PROTOCOL_SHA256 or
        json.loads((path / 'plan.json').read_text()) != plan or
        json.loads((path / 'precision_addendum.json').read_text()) != scope['profile'] or
        manifest['original_primary_gates_unchanged'] != scope['primary_gates'] or
        any(manifest[k] != plan[k] for k in ('model_path', 'dataset_manifest_sha256', 'mask_registry_sha256',
            'arithmetic_profile', 'precision_addendum_sha256', 'packages', 'prior_replay_audit_sha256',
            'prior_replay_manifest_sha256', 'completed_fresh_audit_sha256')) or
        manifest['actual_eos_token_id'] != scope['primary_manifest']['actual_eos_token_id'] or
        manifest['generation_config_sha256'] != scope['primary_manifest']['generation_config_sha256'] or
        manifest['generation_config_sha256'] != {p.name: streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')} or
        manifest['model_order'] != plan['models'] or manifest['completed_models'] != plan['models'] or
        list(manifest['models']) != plan['models'] or set(manifest['source_sha256']) != sources() or
        manifest['weights_before'] != after or after['sha256'] != plan['native_loaded_weights_sha256'] or
        not after['all_parameters_frozen'] or not after['no_parameter_gradients'] or not manifest['hooks_restored'] or
        not datetime.fromisoformat(plan['registered_utc']) < datetime.fromisoformat(manifest['started_utc']) <
            datetime.fromisoformat(manifest['completed_utc'])):
        raise ValueError('Frozen posthoc scope, prior gates, source closure, native weights or chronology changed')
    for name, digest in manifest['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / 'sources' / name) != digest:
            raise ValueError('Actual or archived runtime source changed')
    keys = [q['id'] for q in plan['queries']]
    if [r['id'] for r in manifest['native_anchors']] != keys:
        raise ValueError('Complete four-query native anchors required')
    anchors = {}
    for row in manifest['native_anchors']:
        audit_readout(row['readout'], scope['queries'][row['id']])
        if not row['exact_completed_fresh_native_replay'] or row['readout'] != scope['native_references'][row['id']]:
            raise ValueError('Native original readout drifted')
        anchors[row['id']] = row['readout']
    summaries = []; row_count = proof_count = prefix_count = detach_count = 0
    for name in plan['models']:
        state = manifest['models'][name]; spec = scope['source_plan']['models'][name]
        selected = [e for e in plan['queries'] if e['model'] == name]; ids = [e['id'] for e in selected]
        if ({k: state[k] for k in plan['fixed_checkpoints'][name]} != plan['fixed_checkpoints'][name] or
            state['adapter_parameters'] != spec['parameters'] or
            state['adapter_state_sha256_after'] != state['adapter_state_sha256_before'] or
            not state['all_adapter_parameters_frozen'] or not state['arithmetic_hooks_removed'] or not state['hooks_restored'] or
            state['completed_queries'] != 2 or state['recorded_rows'] != 28 or 'active_query' in state or
            state['token_policy'] != 'question' or len(state['proofs']) != 28):
            raise ValueError('Frozen adapter bytes, complete execution or cleanup changed')
        audit_affines(state, plan['arithmetic_profile'])
        for field in ('prefix_checks', 'references', 'original_operator_controls', 'all_history_controls', 'operator_equivalence'):
            if [r['id'] for r in state[field]] != ids:
                raise ValueError('Complete fixed per-query control coverage required')
        result_path = path / f'{name}.jsonl'
        if streaming_hash(result_path) != state['results_sha256']:
            raise ValueError('Actual layer-readout file changed')
        rows = [json.loads(line) for line in result_path.read_text().splitlines()]
        if len(rows) != 28:
            raise ValueError('All28physical readouts per model required')
        for index, entry in enumerate(selected):
            p = state['prefix_checks'][index]; q = scope['queries'][entry['id']]
            if (p['source_key'] != q['source_key'] or p['ledger_tokens'] != q['ledger_tokens'] or
                not p['complete_KV_recurrent_conv_bookkeeping_exact'] or not p['last_prefix_logits_exact']):
                raise ValueError('Actual native recurrent/conv/KV prefix or logits changed')
            sha(p['prefix_sha256'])
            sl = slice(index * 14, (index + 1) * 14)
            summaries.append(audit_query(scope, entry, rows[sl], state['proofs'][sl],
                state['references'][index], state['original_operator_controls'][index],
                state['all_history_controls'][index], state['operator_equivalence'][index], p['prefix_sha256']))
        if ([r['id'] for r in state['native_restore_checks']] != keys or
            any(not r['exact'] or r['readout'] != anchors[r['id']] for r in state['native_restore_checks'])):
            raise ValueError('All four native anchors required after each controller detachment')
        row_count += len(rows); proof_count += len(state['proofs']); prefix_count += len(state['prefix_checks'])
        detach_count += len(state['native_restore_checks'])
    counts = {'recorded_readouts': row_count, 'actual_mask_and_complete_prefix_proofs': proof_count,
        'question_only_native_prefix_checks': prefix_count, 'only_final_layer_complete_question_identity_checks': 4,
        'original_global_operator_exact_controls': 4, 'all_history_exact_controls': 4,
        'precontroller_native_anchors': len(anchors), 'detached_native_replays': detach_count,
        'measure_calls_including_controls': row_count + 4 + 4 + len(anchors) + detach_count}
    if (row_count, proof_count, prefix_count, detach_count, len(summaries)) != (56, 56, 4, 8, 4):
        raise ValueError('Complete prescribed56readout localization required')
    return {'status': 'completed_and_audited_posthoc_layer_localization', 'operational_controls_passed': True,
        'posthoc': True, 'independent_scientific_confirmation': False, 'can_change_primary_gates': False,
        'no_accuracy_gate': True, 'source_run': str(path), 'manifest_sha256': streaming_hash(path / 'manifest.json'),
        'protocol_sha256': PROTOCOL_SHA256, 'prior_replay_audit_sha256': plan['prior_replay_audit_sha256'],
        'arithmetic_profile': plan['arithmetic_profile'], 'weights_sha256': after['sha256'], 'counts': counts,
        'original_primary_gates_unchanged': scope['primary_gates'], 'cases': summaries,
        'interpretation': 'All four reused control flips and all layer conditions are retained. Single-layer sufficiency and leave-one-out necessity are case-specific, with the latter conditional on other masked layers. Final-layer first-token changes under identical complete question cache/input demonstrate a direct readout route only for those cases. No prefix-write destruction, independent confirmation or architecture benefit is established.',
        'analysis_source_sha256': streaming_hash(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous independent audits')
    report = audit_run(args.run); write_json(args.output, report)
    source = args.output.with_suffix('.source.py'); source.write_bytes(Path(__file__).read_bytes())
    rows = [json.loads(line) for name in ('local_question', 'gdn_question_only')
        for line in (args.run / f'{name}.jsonl').read_text().splitlines()]
    table = args.output.with_suffix('.csv')
    with table.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows([{**r, 'first_token': json.dumps(r['first_token'], sort_keys=True)} for r in rows])
    write_json(args.output.with_suffix('.provenance.json'), {'status': 'complete_posthoc_layer_artifacts',
        'manifest_sha256': report['manifest_sha256'], 'counts': report['counts'],
        'artifacts_sha256': {str(p): streaming_hash(p) for p in (args.output, source, table)},
        'runtime_sources_sha256': json.loads((args.run / 'manifest.json').read_text())['source_sha256']})
    print(json.dumps({'status': report['status'], 'counts': report['counts'],
        'only_final_layer_first_argmax_changes': [c['only_final_layer']['first_token_argmax_changed'] for c in report['cases']]}), flush=True)
