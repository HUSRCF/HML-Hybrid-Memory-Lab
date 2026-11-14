"""Execute all96registered first-token timing observations on frozen models."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.canonical_necessity_common import check_affine_scope
from scripts.fresh_layer_controls import Attention23Input, tensor_hash, first_token, LAYERS
from scripts.fresh_timing_controls import (PROTOCOL_SHA256, TimingReadout, TimingWeights,
    input_scope, sources, audit_operator, audit_weights)
from scripts.run_controller_trial import state_fingerprint, write_json
from scripts.run_copy_dependency_transfer import actual_prompt
from scripts.run_fresh_layer_localization import reference, history_fields
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_adapter_transfer import verify_question_prefix
from scripts.run_write_locality_load_screen import hook_state, weights_fingerprint
from scripts.semantic_order_path_common import assert_complete_cache_equal, cache_fingerprint
from scripts.stable_adapter_linears import StableAdapterLinears


def advance_ids(runtime, ids):
    if not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('Exact nonempty original token-ID prefix required')
    result = runtime.model(input_ids=runtime.tensor([ids]), past_key_values=None,
        use_cache=True, logits_to_keep=1)
    return result.past_key_values, result.logits[:, -1].float()


def query_reference(runtime, ledger, query):
    ref = reference(runtime, ledger, query)
    ids = runtime.ids(ledger + runtime.query_suffix(query['question']))
    if len(ids) != query['prompt_tokens'] or data_fingerprint(ids) != query['prompt_ids_sha256']:
        raise ValueError('Original full input IDs changed')
    preceding, logits = advance_ids(runtime, ids[:-1])
    ref.update(ids=ids, preceding=preceding, preceding_logits=logits,
        preceding_sha256=cache_fingerprint(preceding), preceding_logits_sha256=tensor_hash(logits))
    return ref


def all_history(runtime, query, timing, ref):
    import torch
    with TimingReadout(runtime.model, [True] * query['ledger_tokens'], LAYERS, query['question_tokens'], timing) as op:
        with Attention23Input(runtime.model, query['prompt_tokens']) as cap:
            cache, logits = advance_ids(runtime, ref['ids'])
    assert_complete_cache_equal(cache, ref['question'])
    if cap.calls != 1 or not torch.equal(cap.hidden, ref['hidden']) or not torch.equal(logits, ref['logits']):
        raise ValueError('Timing all-history control changed complete question cache, input or logits')
    audit_operator(op.calls, op.receipts, op.batches, LAYERS, query, timing, [query['prompt_tokens']])
    return {'id': query['id'], 'timing': timing, 'complete_cache_input_logits_exact': True,
        'calls': op.calls, 'receipts': op.receipts, 'batches': op.batches}


def condition(runtime, query, keep, layers, timing, label, ref):
    import torch
    h, n = query['ledger_tokens'], query['prompt_tokens']
    hooks = hook_state(runtime.model)
    preceding = preceding_logits = None
    with TimingReadout(runtime.model, keep, layers, query['question_tokens'], timing) as op:
        with TimingWeights(runtime.model, keep, query['prelude_tokens'], query['question_tokens'], timing) as observer:
            with Attention23Input(runtime.model, n) as cap:
                cache, logits = advance_ids(runtime, ref['ids'])
        prefix, prefix_logits = advance_ids(runtime, ref['ids'][:h])
        if timing == 'last_question_token':
            preceding, preceding_logits = advance_ids(runtime, ref['ids'][:-1])
    assert_complete_cache_equal(prefix, ref['prefix'])
    if not torch.equal(prefix_logits, ref['prefix_logits']) or cap.calls != 1 or cap.hidden is None:
        raise ValueError('Complete original ledger prefix/logits or full-question input capture changed')
    lengths = [n, h] + ([n - 1] if timing == 'last_question_token' else [])
    audit_operator(op.calls, op.receipts, op.batches, layers, query, timing, lengths)
    audit_weights(observer.stats, layers, query, timing)
    cache_sha, hidden_sha = cache_fingerprint(cache), tensor_hash(cap.hidden)
    cache_equal = cache_sha == ref['question_sha256']
    input_equal = hidden_sha == ref['hidden_sha256']
    if label in ('full', 'only_layer_23'):
        assert_complete_cache_equal(cache, ref['question'])
        if not cache_equal or not input_equal or not torch.equal(cap.hidden, ref['hidden']):
            raise ValueError('Full/only-final-layer timing condition changed full-question cache or input')
    if label == 'full' and not torch.equal(logits, ref['logits']):
        raise ValueError('Full first-token reference drifted')
    prior = None
    if timing == 'last_question_token':
        assert_complete_cache_equal(preceding, ref['preceding'])
        if not torch.equal(preceding_logits, ref['preceding_logits']):
            raise ValueError('Final-token-only mask changed preceding question memory/logits')
        prior = {'complete_memory_exact': True, 'last_logits_exact': True,
            'cache_sha256': cache_fingerprint(preceding), 'logits_sha256': tensor_hash(preceding_logits),
            'earlier_question_kv_fields_exact': history_fields(ref['question'], cache, n - 1)}
    if hook_state(runtime.model) != hooks:
        raise ValueError('Timing operator, observer or final-input hook failed cleanup')
    proof = {'id': query['id'], 'condition': label, 'timing': timing, 'selected_layers': list(layers),
        'calls': op.calls, 'receipts': op.receipts, 'batches': op.batches, 'attention_stats': observer.stats,
        'complete_prefix_memory_exact': True, 'prefix_logits_exact': True, 'prefix_sha256': ref['prefix_sha256'],
        'masked_prefix_sha256': cache_fingerprint(prefix),
        'historical_kv_fields_exact': history_fields(ref['question'], cache, h),
        'full_question_cache_sha256': cache_sha, 'full_question_cache_equals_unmasked': cache_equal,
        'attention23_input_sha256': hidden_sha, 'attention23_input_exact_to_unmasked': input_equal,
        'preceding_question_proof': prior, 'hooks_restored': True}
    return first_token(runtime, logits, query), proof


def execute(scope, output):
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    from sas_memory.runtime import TextRuntime

    torch.set_num_threads(4); plan = scope['plan']
    manifest = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'training': False, 'new_adapters': False, 'new_data': False, 'posthoc': True, 'no_accuracy_gate': True,
        'independent_scientific_confirmation': False, 'can_change_primary_gates': False, 'full_generation_scored': False,
        'protocol_sha256': PROTOCOL_SHA256, **{k: plan[k] for k in ('model_path', 'dataset_manifest_sha256',
            'mask_registry_sha256', 'arithmetic_profile', 'precision_addendum_sha256',
            'prior_localization_audit_sha256', 'prior_localization_manifest_sha256')},
        'original_primary_gates_unchanged': scope['primary_gates'],
        'source_sha256': {n: streaming_hash(ROOT / n) for n in sorted(sources())},
        'model_order': plan['models'], 'completed_models': [], 'models': {}, 'native_anchors': []}
    output.mkdir(parents=True, exist_ok=False); path = output / 'manifest.json'
    write_json(output / 'plan.json', plan); write_json(output / 'precision_addendum.json', scope['profile'])
    for name in manifest['source_sha256']:
        target = output / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((ROOT / name).read_bytes())
    write_json(path, manifest); runtime = native_hooks = None
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
            raise ValueError('Frozen runtime, generation configuration or hybrid layout changed')
        native_hooks = hook_state(runtime.model); before = manifest['weights_before'] = weights_fingerprint(runtime.model)
        if (before['sha256'] != plan['native_loaded_weights_sha256'] or
            not before['all_parameters_frozen'] or not before['no_parameter_gradients']):
            raise ValueError('Native checkpoint/freeze changed')
        cases = {c['id']: c for c in scope['cases']}; anchors = {}
        with torch.inference_mode():
            for entry in plan['queries']:
                q = scope['queries'][entry['id']]; actual = measure(runtime, actual_prompt(runtime, cases, q), q)
                manifest['native_anchors'].append({'id': q['id'], 'readout': actual,
                    'exact_completed_fresh_native_replay': actual == scope['native_references'][q['id']]})
                write_json(path, manifest)
                if actual != scope['native_references'][q['id']]:
                    raise ValueError('Native original anchor drifted')
                anchors[q['id']] = actual
        oldrefs = {(r['model'], r['id'], r['condition']): r for r in plan['prior_references']}
        for name in plan['models']:
            spec = scope['source_plan']['models'][name]; entries = [e for e in plan['queries'] if e['model'] == name]
            state = manifest['models'][name] = {'token_policy': 'question', 'proofs': [], 'prefix_checks': [],
                'references': [], 'all_history_controls': [], 'native_restore_checks': [],
                'completed_queries': 0, 'recorded_rows': 0}
            controller = arithmetic = None; result_path = output / f'{name}.jsonl'
            try:
                controller = GroupController(runtime.model, spec['group_index'], spec['rank'], spec['mode'],
                    adapt_gdn_keys=spec['adapt_gdn_keys'])
                controller.load_state_dict(load_file(spec['checkpoint'], device=str(runtime.model.device)), strict=True)
                controller.eval().requires_grad_(False)
                state.update(description=controller.description(), checkpoint_sha256=streaming_hash(Path(spec['checkpoint'])),
                    adapter_state_sha256_before=state_fingerprint(controller), adapter_parameters=sum(p.numel() for p in controller.parameters()))
                if ({k: state[k] for k in plan['fixed_checkpoints'][name]} != plan['fixed_checkpoints'][name] or state['adapter_parameters'] != spec['parameters']):
                    raise ValueError('Frozen controller checkpoint or allocation changed')
                state['affine_scope'] = check_affine_scope(controller, runtime.model)
                arithmetic = StableAdapterLinears(controller); state['arithmetic'] = arithmetic.description()
                with torch.inference_mode(), result_path.open('w', buffering=1) as stream:
                    for entry in entries:
                        q = scope['queries'][entry['id']]
                        state['prefix_checks'].append({'id': q['id'],
                            **verify_question_prefix(runtime, controller, actual_prompt(runtime, cases, q), q)})
                    for entry in entries:
                        q = scope['queries'][entry['id']]; controller.set_token_policy('question', q['ledger_tokens'])
                        state['active_query'] = q['id']; write_json(path, manifest)
                        ref = query_reference(runtime, actual_prompt(runtime, cases, q), q)
                        if ref['prefix_sha256'] != next(p['prefix_sha256'] for p in state['prefix_checks'] if p['id'] == q['id']):
                            raise ValueError('Timing reference differs from actual native ledger-prefix state')
                        state['references'].append({'id': q['id'], 'prefix_sha256': ref['prefix_sha256'],
                            'full_question_cache_sha256': ref['question_sha256'],
                            'attention23_input_sha256': ref['hidden_sha256'], 'first_token': first_token(runtime, ref['logits'], q),
                            'preceding_question_cache_sha256': ref['preceding_sha256'],
                            'preceding_question_logits_sha256': ref['preceding_logits_sha256']})
                        keep = [bit == '1' for bit in scope['masks'][q['id']]['keep_bits'][entry['regime']]]
                        for timing in plan['timing_order']:
                            state['all_history_controls'].append(all_history(runtime, q, timing, ref)); write_json(path, manifest)
                            for label in plan['condition_order']:
                                layers = plan['conditions'][label]
                                first, proof = condition(runtime, q, keep, layers, timing, label, ref)
                                row = {'model': name, 'id': q['id'], 'condition': label, 'timing': timing,
                                    'source_regime': entry['regime'], 'direction': entry['direction'], 'selected_layers': layers,
                                    'token_policy': 'question', 'prefill': 'full', 'arithmetic_profile': plan['arithmetic_profile'],
                                    'registration_sha256': data_fingerprint(q), 'prompt_ids_sha256': q['prompt_ids_sha256'],
                                    'mask_registration_sha256': entry['mask_registration_sha256'],
                                    'keep_bits_sha256': entry['keep_bits_sha256'], 'first_token': first}
                                stream.write(json.dumps(row) + '\n'); state['proofs'].append(proof)
                                state['recorded_rows'] += 1; write_json(path, manifest)
                                if timing == 'complete_question':
                                    actual = {'model': name, 'id': q['id'], 'condition': label, 'first_token': first,
                                        **{k: proof[k] for k in ('prefix_sha256', 'full_question_cache_sha256', 'attention23_input_sha256')}}
                                    if actual != oldrefs[(name, q['id'], label)]:
                                        raise ValueError('Complete-question first logits, cache or input differ from completed layer run')
                                print(json.dumps({'event': 'token_timing', 'model': name, 'id': q['id'],
                                    'timing': timing, 'condition': label, 'rows': state['recorded_rows']}), flush=True)
                        del ref; state['completed_queries'] += 1; state.pop('active_query'); write_json(path, manifest)
            finally:
                if controller is not None:
                    state.update(adapter_state_sha256_after=state_fingerprint(controller),
                        all_adapter_parameters_frozen=all(not p.requires_grad and p.grad is None for p in controller.parameters()))
                if arithmetic is not None:
                    state['actual_affine_calls'] = dict(arithmetic.calls); arithmetic.close()
                    state['arithmetic_hooks_removed'] = all(not m._forward_hooks for m in controller.modules())
                if controller is not None: controller.close()
                if result_path.exists(): state['results_sha256'] = streaming_hash(result_path)
                state['hooks_restored'] = hook_state(runtime.model) == native_hooks
            if (state['completed_queries'] != 2 or state['recorded_rows'] != 48 or not state['hooks_restored'] or
                not state['arithmetic_hooks_removed'] or not state['all_adapter_parameters_frozen'] or
                state['adapter_state_sha256_after'] != state['adapter_state_sha256_before'] or
                any(v < 1 for v in state['actual_affine_calls'].values())):
                raise ValueError('Frozen timing execution or cleanup failed')
            with torch.inference_mode():
                for key, expected in anchors.items():
                    q = scope['queries'][key]; actual = measure(runtime, actual_prompt(runtime, cases, q), q)
                    state['native_restore_checks'].append({'id': key, 'exact': actual == expected, 'readout': actual})
                    write_json(path, manifest)
                    if actual != expected: raise ValueError('Detached native anchor drifted')
            manifest['completed_models'].append(name); write_json(path, manifest)
        manifest['weights_after'] = weights_fingerprint(runtime.model)
        if manifest['weights_after'] != before or hook_state(runtime.model) != native_hooks:
            raise ValueError('Native weights, gradients or hooks changed')
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
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve all prior runs; choose a new directory')
    execute(input_scope(), args.output)
