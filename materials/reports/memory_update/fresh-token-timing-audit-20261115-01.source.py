"""Audit all timing conditions and compare causal first-token effects case by case."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_fresh_copy_confirmation import audit_readout
from scripts.analyze_fresh_layer_localization import audit_first, sha
from scripts.canonical_necessity_common import audit_affines
from scripts.fresh_timing_controls import (PROTOCOL_SHA256, LAYERS, input_scope, sources,
    audit_operator, audit_weights)
from scripts.run_controller_trial import write_json


def audit_query(scope, entry, rows, proofs, ref, controls, native_prefix):
    plan, q = scope['plan'], scope['queries'][entry['id']]
    order = [(t, c) for t in plan['timing_order'] for c in plan['condition_order']]
    if (len(rows) != 24 or len(proofs) != 24 or
        [(r['timing'], r['condition']) for r in rows] != order or
        [(p['timing'], p['condition']) for p in proofs] != order or ref['id'] != q['id'] or
        [(c['id'], c['timing']) for c in controls] != [(q['id'], t) for t in plan['timing_order']]):
        raise ValueError('Every case, timing and all eight layer conditions must be retained in fixed order')
    for key in ('prefix_sha256', 'full_question_cache_sha256', 'attention23_input_sha256',
                'preceding_question_cache_sha256', 'preceding_question_logits_sha256'):
        sha(ref[key])
    audit_first(ref['first_token'], q)
    if ref['prefix_sha256'] != native_prefix:
        raise ValueError('Question-only reference must preserve actual native ledger-prefix memory')
    for control in controls:
        if not control['complete_cache_input_logits_exact']:
            raise ValueError('Timing all-history cache/input/logit identity failed')
        audit_operator(control['calls'], control['receipts'], control['batches'], LAYERS, q,
            control['timing'], [q['prompt_tokens']])
    prior = {(r['model'], r['id'], r['condition']): r for r in plan['prior_references']}
    for row, proof in zip(rows, proofs):
        timing, label = row['timing'], row['condition']; layers = plan['conditions'][label]
        audit_first(row['first_token'], q)
        if (row['model'] != entry['model'] or row['id'] != q['id'] or
            row['source_regime'] != entry['regime'] or row['direction'] != entry['direction'] or
            row['selected_layers'] != layers or row['token_policy'] != 'question' or row['prefill'] != 'full' or
            row['arithmetic_profile'] != plan['arithmetic_profile'] or
            row['registration_sha256'] != data_fingerprint(q) or
            row['prompt_ids_sha256'] != q['prompt_ids_sha256'] or
            any(row[k] != entry[k] for k in ('mask_registration_sha256', 'keep_bits_sha256')) or
            proof['id'] != q['id'] or proof['selected_layers'] != layers or
            not proof['complete_prefix_memory_exact'] or not proof['prefix_logits_exact'] or
            proof['prefix_sha256'] != native_prefix or proof['masked_prefix_sha256'] != native_prefix or
            proof['historical_kv_fields_exact'] != 12 or not proof['hooks_restored']):
            raise ValueError('Original token input, intervention, native prefix, historicalKV or cleanup proof changed')
        lengths = [q['prompt_tokens'], q['ledger_tokens']] + (
            [q['prompt_tokens'] - 1] if timing == 'last_question_token' else [])
        audit_operator(proof['calls'], proof['receipts'], proof['batches'], layers, q, timing, lengths)
        audit_weights(proof['attention_stats'], layers, q, timing)
        for key in ('full_question_cache_sha256', 'attention23_input_sha256'):
            sha(proof[key])
        if (proof['full_question_cache_equals_unmasked'] !=
                (proof['full_question_cache_sha256'] == ref['full_question_cache_sha256']) or
            proof['attention23_input_exact_to_unmasked'] !=
                (proof['attention23_input_sha256'] == ref['attention23_input_sha256'])):
            raise ValueError('Actual complete-question cache/input hashes contradict equality receipts')
        if label in ('full', 'only_layer_23') and (not proof['full_question_cache_equals_unmasked'] or
                not proof['attention23_input_exact_to_unmasked']):
            raise ValueError('Full/only-last-layer condition must preserve complete-question cache and input')
        if label == 'full' and row['first_token'] != ref['first_token']:
            raise ValueError('Unmasked full first logits changed with timing policy')
        preceding = proof['preceding_question_proof']
        if timing == 'last_question_token':
            if (preceding is None or not preceding['complete_memory_exact'] or not preceding['last_logits_exact'] or
                preceding['cache_sha256'] != ref['preceding_question_cache_sha256'] or
                preceding['logits_sha256'] != ref['preceding_question_logits_sha256'] or
                preceding['earlier_question_kv_fields_exact'] != 12):
                raise ValueError('Final-token-only intervention changed preceding question memory/logits/KV')
        elif preceding is not None:
            raise ValueError('Preceding-question proof mislabeled as final-token-only')
        if timing == 'complete_question':
            actual = {'model': entry['model'], 'id': q['id'], 'condition': label, 'first_token': row['first_token'],
                **{k: proof[k] for k in ('prefix_sha256', 'full_question_cache_sha256', 'attention23_input_sha256')}}
            if actual != prior[(entry['model'], q['id'], label)]:
                raise ValueError('Original completed-layer first-token/cache/input reference must replay exactly')
    by = {(r['timing'], r['condition']): r['first_token'] for r in rows}
    base = by[('complete_question', 'full')]
    summaries = []
    for label in plan['condition_order']:
        complete, last, earlier = [by[(timing, label)] for timing in plan['timing_order']]
        summaries.append({'condition': label, 'complete_question': complete, 'last_question_token': last,
            'earlier_question_tokens': earlier, 'complete_changes_argmax_from_full': complete['argmax_id'] != base['argmax_id'],
            'last_changes_argmax_from_full': last['argmax_id'] != base['argmax_id'],
            'earlier_changes_argmax_from_full': earlier['argmax_id'] != base['argmax_id'],
            'last_reproduces_complete_argmax': last['argmax_id'] == complete['argmax_id'],
            'earlier_reproduces_complete_argmax': earlier['argmax_id'] == complete['argmax_id'],
            'last_logits_exact_to_complete': last['logits_sha256'] == complete['logits_sha256'],
            'earlier_logits_exact_to_complete': earlier['logits_sha256'] == complete['logits_sha256'],
            'last_gold_first_token_correct': last['argmax_id'] == last['gold_id'],
            'earlier_gold_first_token_correct': earlier['argmax_id'] == earlier['gold_id']})
    return {'model': entry['model'], 'id': q['id'], 'source_regime': entry['regime'],
        'original_full_answer_direction': entry['direction'], 'bindings': entry['bindings'],
        'full_first_token': base, 'conditions': summaries,
        'earlier_only_final_layer_logits_equal_full':
            by[('earlier_question_tokens', 'only_layer_23')]['logits_sha256'] == base['logits_sha256']}


def audit_run(path):
    path = Path(path); scope = input_scope(); plan = scope['plan']
    manifest = json.loads((path / 'manifest.json').read_text()); after = manifest['weights_after']
    if (manifest['status'] != 'completed' or not manifest['posthoc'] or not manifest['no_accuracy_gate'] or
        any(manifest[k] for k in ('training', 'new_adapters', 'new_data', 'full_generation_scored',
            'independent_scientific_confirmation', 'can_change_primary_gates')) or
        manifest['protocol_sha256'] != PROTOCOL_SHA256 or json.loads((path / 'plan.json').read_text()) != plan or
        json.loads((path / 'precision_addendum.json').read_text()) != scope['profile'] or
        manifest['original_primary_gates_unchanged'] != scope['primary_gates'] or
        any(manifest[k] != plan[k] for k in ('model_path', 'dataset_manifest_sha256', 'mask_registry_sha256',
            'arithmetic_profile', 'precision_addendum_sha256', 'packages',
            'prior_localization_audit_sha256', 'prior_localization_manifest_sha256')) or
        manifest['actual_eos_token_id'] != scope['primary_manifest']['actual_eos_token_id'] or
        manifest['generation_config_sha256'] != scope['primary_manifest']['generation_config_sha256'] or
        manifest['generation_config_sha256'] != {p.name: streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')} or
        manifest['model_order'] != plan['models'] or manifest['completed_models'] != plan['models'] or
        list(manifest['models']) != plan['models'] or set(manifest['source_sha256']) != sources() or
        manifest['weights_before'] != after or after['sha256'] != plan['native_loaded_weights_sha256'] or
        not after['all_parameters_frozen'] or not after['no_parameter_gradients'] or not manifest['hooks_restored'] or
        not datetime.fromisoformat(plan['registered_utc']) < datetime.fromisoformat(manifest['started_utc']) <
            datetime.fromisoformat(manifest['completed_utc'])):
        raise ValueError('Frozen timing scope, prior gates, source closure, chronology or native weights changed')
    for name, digest in manifest['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / 'sources' / name) != digest:
            raise ValueError('Actual or archived timing runtime source changed')
    keys = [q['id'] for q in plan['queries']]
    if [r['id'] for r in manifest['native_anchors']] != keys:
        raise ValueError('Complete four-query original native anchors required')
    anchors = {}
    for row in manifest['native_anchors']:
        audit_readout(row['readout'], scope['queries'][row['id']])
        if not row['exact_completed_fresh_native_replay'] or row['readout'] != scope['native_references'][row['id']]:
            raise ValueError('Native original five-field score drifted')
        anchors[row['id']] = row['readout']
    cases = []; rows_count = proofs_count = controls_count = prefix_count = detach_count = 0
    for name in plan['models']:
        state = manifest['models'][name]; spec = scope['source_plan']['models'][name]
        selected = [e for e in plan['queries'] if e['model'] == name]; ids = [e['id'] for e in selected]
        if ({k: state[k] for k in plan['fixed_checkpoints'][name]} != plan['fixed_checkpoints'][name] or
            state['adapter_parameters'] != spec['parameters'] or
            state['adapter_state_sha256_after'] != state['adapter_state_sha256_before'] or
            not state['all_adapter_parameters_frozen'] or not state['arithmetic_hooks_removed'] or not state['hooks_restored'] or
            state['completed_queries'] != 2 or state['recorded_rows'] != 48 or 'active_query' in state or
            state['token_policy'] != 'question' or len(state['proofs']) != 48 or
            [r['id'] for r in state['prefix_checks']] != ids or [r['id'] for r in state['references']] != ids or
            [(r['id'], r['timing']) for r in state['all_history_controls']] !=
                [(key, t) for key in ids for t in plan['timing_order']]):
            raise ValueError('Frozen adapter, complete question/timing control coverage or cleanup changed')
        audit_affines(state, plan['arithmetic_profile'])
        result_path = path / f'{name}.jsonl'
        if streaming_hash(result_path) != state['results_sha256']:
            raise ValueError('Actual first-token result file changed')
        rows = [json.loads(line) for line in result_path.read_text().splitlines()]
        if len(rows) != 48:
            raise ValueError('All48physical first-token readouts required per model')
        for index, entry in enumerate(selected):
            p = state['prefix_checks'][index]; q = scope['queries'][entry['id']]
            if (p['source_key'] != q['source_key'] or p['ledger_tokens'] != q['ledger_tokens'] or
                not p['complete_KV_recurrent_conv_bookkeeping_exact'] or not p['last_prefix_logits_exact']):
                raise ValueError('Actual native recurrent/conv/KV ledger prefix or logits changed')
            sha(p['prefix_sha256']); sl = slice(24 * index, 24 * (index + 1))
            cases.append(audit_query(scope, entry, rows[sl], state['proofs'][sl], state['references'][index],
                state['all_history_controls'][3 * index:3 * (index + 1)], p['prefix_sha256']))
        if ([r['id'] for r in state['native_restore_checks']] != keys or
            any(not r['exact'] or r['readout'] != anchors[r['id']] for r in state['native_restore_checks'])):
            raise ValueError('All original native anchors required after each controller detachment')
        rows_count += len(rows); proofs_count += len(state['proofs'])
        controls_count += len(state['all_history_controls']); prefix_count += len(state['prefix_checks'])
        detach_count += len(state['native_restore_checks'])
    if (rows_count, proofs_count, controls_count, prefix_count, detach_count) != (96, 96, 12, 4, 8):
        raise ValueError('Complete prescribed96readout timing coverage required')
    counts = {'physical_first_token_readouts': rows_count, 'exact_prior_first_token_references': 32,
        'new_timing_counterfactual_first_token_readouts': 64, 'actual_row_mask_and_complete_prefix_proofs': proofs_count,
        'preceding_question_complete_memory_logits_and_KV_checks': 32, 'only_final_layer_complete_question_identity_checks': 12,
        'all_history_exact_controls': controls_count, 'question_only_native_prefix_checks': prefix_count,
        'precontroller_native_anchors': len(anchors), 'detached_native_replays': detach_count,
        'full_generation_scientific_readouts': 0}
    return {'status': 'completed_and_audited_posthoc_token_timing', 'operational_controls_passed': True,
        'posthoc': True, 'independent_scientific_confirmation': False, 'can_change_primary_gates': False,
        'no_accuracy_gate': True, 'full_generation_scored': False, 'source_run': str(path),
        'manifest_sha256': streaming_hash(path / 'manifest.json'), 'protocol_sha256': PROTOCOL_SHA256,
        'prior_localization_audit_sha256': plan['prior_localization_audit_sha256'],
        'arithmetic_profile': plan['arithmetic_profile'], 'weights_sha256': after['sha256'], 'counts': counts,
        'original_primary_gates_unchanged': scope['primary_gates'], 'cases': cases,
        'interpretation': 'All reused cases, all single layers and all three timings are retained. A final-token-only first-argmax flip under identical preceding-question memory establishes an effect initiated on that final token for that case. It can still propagate through downstream GDN, residuals, MLPs and Attention within that token. Earlier-token-only effects also do not identify GDN uniquely. First-token correctness is not full-answer correctness; no prefix-write destruction, new-data confirmation or shared semantic-block benefit is established.',
        'analysis_source_sha256': streaming_hash(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve all prior independent audits')
    report = audit_run(args.run); write_json(args.output, report)
    source = args.output.with_suffix('.source.py'); source.write_bytes(Path(__file__).read_bytes())
    rows = [json.loads(line) for name in ('local_question', 'gdn_question_only')
        for line in (args.run / f'{name}.jsonl').read_text().splitlines()]
    table = args.output.with_suffix('.csv')
    with table.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows([{**r, 'first_token': json.dumps(r['first_token'], sort_keys=True)} for r in rows])
    write_json(args.output.with_suffix('.provenance.json'), {'status': 'complete_posthoc_timing_artifacts',
        'manifest_sha256': report['manifest_sha256'], 'counts': report['counts'],
        'artifacts_sha256': {str(p): streaming_hash(p) for p in (args.output, source, table)},
        'runtime_sources_sha256': json.loads((args.run / 'manifest.json').read_text())['source_sha256']})
    print(json.dumps({'status': report['status'], 'counts': report['counts']}), flush=True)
