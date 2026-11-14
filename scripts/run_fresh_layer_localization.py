"""Run every registered layer-localization condition with original-prefix proofs."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.kv_readout import QuestionKVReadout
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.canonical_necessity_common import check_affine_scope
from scripts.copy_readout_runtime import SelectionWeights, readout
from scripts.fresh_layer_controls import (PROTOCOL_SHA256, LAYERS, input_scope, sources, LayerReadout,
    Attention23Input, tensor_hash, first_token, audit_receipts, audit_probabilities)
from scripts.run_controller_trial import state_fingerprint, write_json
from scripts.run_copy_dependency_transfer import actual_prompt
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_adapter_transfer import verify_question_prefix
from scripts.run_write_locality_load_screen import hook_state, weights_fingerprint
from scripts.semantic_order_path_common import assert_complete_cache_equal, cache_fingerprint
from scripts.stable_adapter_linears import StableAdapterLinears


def history_fields(reference, actual, boundary):
    import torch
    fields = 0
    for left, right in zip(reference.layers, actual.layers):
        for name in ('keys', 'values'):
            expected, observed = getattr(left, name, None), getattr(right, name, None)
            if isinstance(expected, torch.Tensor):
                if not isinstance(observed, torch.Tensor) or not torch.equal(expected[..., :boundary, :], observed[..., :boundary, :]):
                    raise ValueError('Historical original K/V slices changed')
                fields += 1
    if fields != 12:
        raise ValueError('All twelve historical key/value fields required')
    return fields


def reference(runtime, ledger, query):
    prompt = ledger + runtime.query_suffix(query['question'])
    tokens = query['ledger_tokens'] + query['question_tokens']
    with Attention23Input(runtime.model, tokens) as capture:
        prefix, prefix_logits, h = runtime.advance(ledger)
        question, logits, n = runtime.advance(prompt)
        hidden = capture.hidden
    if h != query['ledger_tokens'] or n != tokens or hidden is None or capture.calls != 1:
        raise ValueError('Actual complete prompt or Attention23 input capture changed')
    return {'prefix': prefix, 'prefix_logits': prefix_logits, 'question': question, 'logits': logits,
        'hidden': hidden, 'prefix_sha256': cache_fingerprint(prefix),
        'question_sha256': cache_fingerprint(question), 'hidden_sha256': tensor_hash(hidden)}


def original_operator(runtime, ledger, query, keep, ref):
    import torch
    prompt = ledger + runtime.query_suffix(query['question'])
    with QuestionKVReadout(runtime.model, keep) as operator:
        with Attention23Input(runtime.model, query['ledger_tokens'] + query['question_tokens']) as capture:
            prefix, prefix_logits, _ = runtime.advance(ledger)
            actual, logits, _ = runtime.advance(prompt)
            hidden = capture.hidden
        observed = measure(runtime, ledger, query)
    assert_complete_cache_equal(prefix, ref['prefix'])
    if hidden is None or capture.calls != 1 or not torch.equal(prefix_logits, ref['prefix_logits']):
        raise ValueError('Original global operator changed prefix or missed Attention23 input')
    return {'cache': actual, 'logits': logits, 'hidden': hidden, 'readout': observed,
        'calls': operator.calls, 'historical_kv_fields_exact': history_fields(ref['question'], actual, query['ledger_tokens'])}


def all_history_control(runtime, ledger, query, ref):
    import torch
    prompt = ledger + runtime.query_suffix(query['question'])
    with LayerReadout(runtime.model, [True] * query['ledger_tokens'], LAYERS) as operator:
        with Attention23Input(runtime.model, query['ledger_tokens'] + query['question_tokens']) as capture:
            prefix, prefix_logits, _ = runtime.advance(ledger)
            question, logits, _ = runtime.advance(prompt)
            hidden = capture.hidden
        observed = measure(runtime, ledger, query)
    assert_complete_cache_equal(prefix, ref['prefix']); assert_complete_cache_equal(question, ref['question'])
    if hidden is None or capture.calls != 1 or not torch.equal(hidden, ref['hidden']) or not torch.equal(prefix_logits, ref['prefix_logits']) or not torch.equal(logits, ref['logits']):
        raise ValueError('All-history layer operator changed actual cache, input or logits')
    audit_receipts(operator.calls, operator.receipts, LAYERS, 3)
    return {'id': query['id'], 'readout': observed, 'complete_cache_input_logits_exact': True,
        'calls': operator.calls, 'receipts': operator.receipts}


def condition(runtime, ledger, query, keep, selected, label, ref):
    import torch
    hooks = hook_state(runtime.model)
    with LayerReadout(runtime.model, keep, selected) as operator:
        with SelectionWeights(runtime.model, {'original_keep': keep}, query['prelude_tokens']) as observer:
            observed = measure(runtime, ledger, query)
    audit_receipts(operator.calls, operator.receipts, selected, 1)
    stats = observer.stats['original_keep']; audit_probabilities(stats, selected, operator.calls)
    if hook_state(runtime.model) != hooks:
        raise ValueError('Scored layer mask or probability observer failed cleanup')
    prompt = ledger + runtime.query_suffix(query['question'])
    with LayerReadout(runtime.model, keep, selected) as proof_operator:
        with Attention23Input(runtime.model, query['ledger_tokens'] + query['question_tokens']) as capture:
            prefix, prefix_logits, _ = runtime.advance(ledger)
            actual, logits, _ = runtime.advance(prompt)
            hidden = capture.hidden
    assert_complete_cache_equal(prefix, ref['prefix'])
    if hidden is None or capture.calls != 1 or not torch.equal(prefix_logits, ref['prefix_logits']) or proof_operator.calls != {'prefill': 2, 'cached': 0}:
        raise ValueError('Complete original prefix/logits or full-question capture changed')
    audit_receipts(proof_operator.calls, proof_operator.receipts, selected, 2)
    prefix_hash, question_hash, hidden_hash = cache_fingerprint(prefix), cache_fingerprint(actual), tensor_hash(hidden)
    cache_same = question_hash == ref['question_sha256']
    hidden_same = hidden.dtype == ref['hidden'].dtype and hidden.shape == ref['hidden'].shape and torch.equal(hidden, ref['hidden'])
    if label in ('full', 'only_layer_23'):
        assert_complete_cache_equal(actual, ref['question'])
        if not cache_same or not hidden_same:
            raise ValueError('Only-last-layer intervention changed full-question memory or Attention23 input')
    if label == 'full' and not torch.equal(logits, ref['logits']):
        raise ValueError('Unmasked per-layer operator changed logits')
    if hook_state(runtime.model) != hooks:
        raise ValueError('Layer proof failed to restore active hooks')
    proof = {'id': query['id'], 'condition': label, 'selected_layers': list(selected),
        'measure_calls': operator.calls, 'measure_receipts': operator.receipts, 'attention_stats': stats,
        'proof_calls': proof_operator.calls, 'proof_receipts': proof_operator.receipts,
        'complete_prefix_memory_exact': True, 'prefix_logits_exact': True,
        'prefix_sha256': ref['prefix_sha256'], 'masked_prefix_sha256': prefix_hash,
        'historical_kv_fields_exact': history_fields(ref['question'], actual, query['ledger_tokens']),
        'full_question_cache_sha256': question_hash, 'full_question_cache_equals_unmasked': cache_same,
        'attention23_input_sha256': hidden_hash, 'attention23_input_exact_to_unmasked': bool(hidden_same),
        'hooks_restored': True}
    return observed, first_token(runtime, logits, query), proof, actual, logits, hidden


def execute(scope, output):
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    from sas_memory.runtime import TextRuntime

    torch.set_num_threads(4)
    plan, profile = scope['plan'], scope['profile']
    manifest = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'training': False, 'new_adapters': False, 'new_data': False, 'posthoc': True,
        'independent_scientific_confirmation': False, 'can_change_primary_gates': False,
        'no_accuracy_gate': True, 'protocol_sha256': PROTOCOL_SHA256,
        **{k: plan[k] for k in ('model_path', 'dataset_manifest_sha256', 'mask_registry_sha256',
            'arithmetic_profile', 'precision_addendum_sha256', 'prior_replay_audit_sha256',
            'prior_replay_manifest_sha256', 'completed_fresh_audit_sha256')},
        'original_primary_gates_unchanged': scope['primary_gates'],
        'source_sha256': {n: streaming_hash(ROOT / n) for n in sorted(sources())},
        'model_order': plan['models'], 'completed_models': [], 'models': {}, 'native_anchors': []}
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'manifest.json'
    write_json(output / 'plan.json', plan); write_json(output / 'precision_addendum.json', profile)
    for name in manifest['source_sha256']:
        target = output / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((ROOT / name).read_bytes())
    write_json(path, manifest)
    runtime, native_hooks = None, None
    try:
        runtime = TextRuntime(plan['model_path'], 'it')
        manifest.update(python=sys.executable, gpu=torch.cuda.get_device_name(0),
            packages={n: importlib.metadata.version(n) for n in plan['packages']},
            actual_eos_token_id=runtime.model.generation_config.eos_token_id,
            generation_config_sha256={p.name: streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')})
        if (manifest['packages'] != plan['packages'] or
            manifest['actual_eos_token_id'] != scope['primary_manifest']['actual_eos_token_id'] or
            manifest['generation_config_sha256'] != scope['primary_manifest']['generation_config_sha256'] or
            runtime.model.config.layer_types != (['linear_attention'] * 3 + ['full_attention']) * 6):
            raise ValueError('Frozen original runtime, generation configuration or layout changed')
        native_hooks = hook_state(runtime.model)
        before = manifest['weights_before'] = weights_fingerprint(runtime.model)
        if (before['sha256'] != plan['native_loaded_weights_sha256'] or
            not before['all_parameters_frozen'] or not before['no_parameter_gradients']):
            raise ValueError('Actual native checkpoint/freeze changed')
        cases = {c['id']: c for c in scope['cases']}
        anchors = {}
        with torch.inference_mode():
            for entry in plan['queries']:
                q = scope['queries'][entry['id']]
                actual = measure(runtime, actual_prompt(runtime, cases, q), q)
                manifest['native_anchors'].append({'id': q['id'], 'readout': actual,
                    'exact_completed_fresh_native_replay': actual == scope['native_references'][q['id']]})
                write_json(path, manifest)
                if actual != scope['native_references'][q['id']]:
                    raise ValueError('Native anchor differs from completed fresh primary')
                anchors[q['id']] = actual
        for name in plan['models']:
            spec = scope['source_plan']['models'][name]
            selected = [e for e in plan['queries'] if e['model'] == name]
            state = manifest['models'][name] = {'token_policy': 'question', 'proofs': [], 'prefix_checks': [],
                'references': [], 'original_operator_controls': [], 'all_history_controls': [],
                'native_restore_checks': [], 'operator_equivalence': [],
                'completed_queries': 0, 'recorded_rows': 0}
            controller, arithmetic = None, None
            result_path = output / f'{name}.jsonl'
            try:
                controller = GroupController(runtime.model, spec['group_index'], spec['rank'], spec['mode'],
                    adapt_gdn_keys=spec['adapt_gdn_keys'])
                controller.load_state_dict(load_file(spec['checkpoint'], device=str(runtime.model.device)), strict=True)
                controller.eval().requires_grad_(False)
                state.update(description=controller.description(), checkpoint_sha256=streaming_hash(Path(spec['checkpoint'])),
                    adapter_state_sha256_before=state_fingerprint(controller), adapter_parameters=sum(p.numel() for p in controller.parameters()))
                if ({k: state[k] for k in plan['fixed_checkpoints'][name]} != plan['fixed_checkpoints'][name] or
                    state['adapter_parameters'] != spec['parameters']):
                    raise ValueError('Actual frozen controller checkpoint or allocation changed')
                state['affine_scope'] = check_affine_scope(controller, runtime.model)
                arithmetic = StableAdapterLinears(controller); state['arithmetic'] = arithmetic.description()
                with torch.inference_mode(), result_path.open('w', buffering=1) as stream:
                    for entry in selected:
                        q = scope['queries'][entry['id']]
                        p = verify_question_prefix(runtime, controller, actual_prompt(runtime, cases, q), q)
                        state['prefix_checks'].append({'id': q['id'], **p})
                    for entry in selected:
                        q = scope['queries'][entry['id']]; ledger = actual_prompt(runtime, cases, q)
                        controller.set_token_policy('question', q['ledger_tokens'])
                        state['active_query'] = q['id']; write_json(path, manifest)
                        ref = reference(runtime, ledger, q)
                        native_prefix = next(p['prefix_sha256'] for p in state['prefix_checks'] if p['id'] == q['id'])
                        if native_prefix != ref['prefix_sha256']:
                            raise ValueError('Localization reference differs from actual native prefix')
                        state['references'].append({'id': q['id'], 'prefix_sha256': ref['prefix_sha256'],
                            'full_question_cache_sha256': ref['question_sha256'],
                            'attention23_input_sha256': ref['hidden_sha256'], 'first_token': first_token(runtime, ref['logits'], q)})
                        keep = [v == '1' for v in scope['masks'][q['id']]['keep_bits'][entry['regime']]]
                        original = original_operator(runtime, ledger, q, keep, ref)
                        state['original_operator_controls'].append({'id': q['id'], 'readout': original['readout'],
                            'calls': original['calls'], 'historical_kv_fields_exact': original['historical_kv_fields_exact'],
                            'full_question_cache_sha256': cache_fingerprint(original['cache']),
                            'attention23_input_sha256': tensor_hash(original['hidden']),
                            'logits_sha256': tensor_hash(original['logits'])})
                        write_json(path, manifest)
                        if original['readout'] != entry['expected_intervened']:
                            raise ValueError('Old global mask no longer exactly replays original readout')
                        control = all_history_control(runtime, ledger, q, ref)
                        state['all_history_controls'].append(control); write_json(path, manifest)
                        if control['readout'] != entry['expected_full']:
                            raise ValueError('All-history layer operator changed full readout')
                        for label in plan['condition_order']:
                            layers = plan['conditions'][label]
                            observed, first, proof, cache, logits, hidden = condition(runtime, ledger, q, keep, layers, label, ref)
                            row = {'model': name, 'id': q['id'], 'condition': label, 'source_regime': entry['regime'],
                                'direction': entry['direction'], 'selected_layers': layers,
                                'token_policy': 'question', 'prefill': 'full', 'arithmetic_profile': plan['arithmetic_profile'],
                                'registration_sha256': data_fingerprint(q),
                                'mask_registration_sha256': entry['mask_registration_sha256'],
                                'keep_bits_sha256': entry['keep_bits_sha256'], **observed, 'first_token': first}
                            stream.write(json.dumps(row) + '\n'); state['proofs'].append(proof)
                            state['recorded_rows'] += 1; write_json(path, manifest)
                            if label == 'full' and observed != entry['expected_full']:
                                raise ValueError('Full per-layer operator changed original five-field score')
                            if label == 'all_layers':
                                assert_complete_cache_equal(cache, original['cache'])
                                if (observed != entry['expected_intervened'] or observed != original['readout'] or
                                    not torch.equal(logits, original['logits']) or not torch.equal(hidden, original['hidden'])):
                                    raise ValueError('New all-layer and original global operators differ')
                                state['operator_equivalence'].append({'id': q['id'], 'cache_input_logits_readout_exact': True})
                            print(json.dumps({'event': 'layer_localization', 'model': name,
                                'id': q['id'], 'condition': label, 'rows': state['recorded_rows']}), flush=True)
                            del cache, logits, hidden
                        del ref, original
                        state['completed_queries'] += 1; state.pop('active_query'); write_json(path, manifest)
            finally:
                if controller is not None:
                    state.update(adapter_state_sha256_after=state_fingerprint(controller),
                        all_adapter_parameters_frozen=all(not p.requires_grad and p.grad is None for p in controller.parameters()))
                if arithmetic is not None:
                    state['actual_affine_calls'] = dict(arithmetic.calls)
                    arithmetic.close(); state['arithmetic_hooks_removed'] = all(not m._forward_hooks for m in controller.modules())
                if controller is not None:
                    controller.close()
                if result_path.exists():
                    state['results_sha256'] = streaming_hash(result_path)
                state['hooks_restored'] = hook_state(runtime.model) == native_hooks
            if (state['completed_queries'] != 2 or state['recorded_rows'] != 28 or not state['hooks_restored'] or
                not state['arithmetic_hooks_removed'] or not state['all_adapter_parameters_frozen'] or
                state['adapter_state_sha256_after'] != state['adapter_state_sha256_before'] or
                any(v < 1 for v in state['actual_affine_calls'].values())):
                raise ValueError('Frozen controller execution or cleanup changed')
            with torch.inference_mode():
                for key, expected_anchor in anchors.items():
                    q = scope['queries'][key]; actual = measure(runtime, actual_prompt(runtime, cases, q), q)
                    state['native_restore_checks'].append({'id': key, 'exact': actual == expected_anchor, 'readout': actual})
                    write_json(path, manifest)
                    if actual != expected_anchor:
                        raise ValueError('Detached native original anchor replay changed')
            manifest['completed_models'].append(name); write_json(path, manifest)
        manifest['weights_after'] = weights_fingerprint(runtime.model)
        if manifest['weights_after'] != before or hook_state(runtime.model) != native_hooks:
            raise ValueError('Native model weights, gradients or hooks changed')
        manifest.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if runtime is not None and native_hooks is not None:
            manifest['hooks_restored'] = hook_state(runtime.model) == native_hooks
        write_json(path, manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous runs; select a new output directory')
    execute(input_scope(), args.output)
