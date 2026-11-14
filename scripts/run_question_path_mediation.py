"""Execute all registered GDN-output versus question-KV oracle path conditions."""

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
from scripts.fresh_layer_controls import first_token, tensor_hash
from scripts.fresh_timing_controls import TimingReadout, TimingWeights, audit_operator, audit_weights
from scripts.question_path_controls import (PROTOCOL_SHA256, CONTEXTS, PATCHES, PathTrace,
    input_scope, sources, recurrent_hashes, audit_receipts)
from scripts.run_controller_trial import state_fingerprint, write_json
from scripts.run_copy_dependency_transfer import actual_prompt
from scripts.run_fresh_layer_localization import history_fields
from scripts.run_fresh_token_timing import advance_ids
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_adapter_transfer import verify_question_prefix
from scripts.run_write_locality_load_screen import hook_state, weights_fingerprint
from scripts.semantic_order_path_common import assert_complete_cache_equal, cache_fingerprint
from scripts.stable_adapter_linears import StableAdapterLinears


def kv_hashes(cache, stop):
    import torch
    result = {f'{i}:{name}': tensor_hash(getattr(layer, name)[..., :stop, :])
        for i, layer in enumerate(cache.layers) for name in ('keys', 'values')
        if isinstance(getattr(layer, name, None), torch.Tensor)}
    if len(result) != 12: raise ValueError('All twelve original K/V fields required')
    return result


def condition(runtime, query, ids, keep, context, patch, prefix_ref, baseline=None, context_base=None):
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    n, h = query['prompt_tokens'], query['ledger_tokens']; hooks = hook_state(runtime.model)
    layers = CONTEXTS[context]['selected_layers']; timing = 'earlier_question_tokens'
    with TimingReadout(runtime.model, keep, layers, query['question_tokens'], timing) as op:
        with PathTrace(runtime.model, query, patch, baseline['trace'] if baseline else None) as trace:
            with TimingWeights(runtime.model, keep, query['prelude_tokens'], query['question_tokens'], timing) as weights:
                cache, logits = advance_ids(runtime, ids)
            prefix, prefix_logits = advance_ids(runtime, ids[:h])
    assert_complete_cache_equal(prefix, prefix_ref['cache'])
    if not torch.equal(prefix_logits, prefix_ref['logits']):
        raise ValueError('Oracle patch or readout intervention changed native ledger-prefix logits')
    audit_operator(op.calls, op.receipts, op.batches, layers, query, timing, [n, h])
    audit_weights(weights.stats, layers, query, timing)
    audit_receipts(trace.receipts, trace.input_calls, query, patch,
        baseline['trace'].receipts if baseline else trace.receipts)
    result = {'cache': cache, 'logits': logits, 'trace': trace, 'states': recurrent_hashes(cache)}
    unmasked, ctx = baseline or result, context_base or result
    if len(result['states']) != 36 or not torch.equal(trace.input20, unmasked['trace'].input20):
        raise ValueError('Earlier-only19 changed final-row input to GDN20 or omitted linear cache states')
    if not torch.equal(trace.input23[:, :-1], ctx['trace'].input23[:, :-1]):
        raise ValueError('Current-output or internal KV patch changed earlier Attention23 input rows')
    if PATCHES[patch]['gdn_current_outputs'] and not torch.equal(trace.input23[:, -1:], unmasked['trace'].input23[:, -1:]):
        raise ValueError('GDN current-output patches failed to restore final Attention23 input')
    if patch == 'question_kv' and not torch.equal(trace.input23, ctx['trace'].input23):
        raise ValueError('Internal question-KV patch changed Attention23 input')
    if context == 'unmasked':
        assert_complete_cache_equal(cache, unmasked['cache'])
        if not torch.equal(logits, unmasked['logits']) or not torch.equal(trace.input23, unmasked['trace'].input23):
            raise ValueError('Unmasked oracle self-to-self sham changed cache/input/first logits')
    kv = cache.layers[23]; native_kv = unmasked['cache'].layers[23]
    qkv = {name: tensor_hash(getattr(kv, name)[..., h:n-1, :]) for name in ('keys', 'values')}
    if PATCHES[patch]['question_kv'] and any(not torch.equal(getattr(kv, name)[..., h:n-1, :],
            getattr(native_kv, name)[..., h:n-1, :]) for name in ('keys', 'values')):
        raise ValueError('Restored Knorm/V projection did not restore actual question K/V after RoPE')
    original_key = trace.key_before_last.transpose(1, 2)
    _, rotated_key = apply_rotary_pos_emb(original_key, original_key, trace.cos_last, trace.sin_last)
    original_value = trace.value_before_last.view(1, 1, kv.values.shape[1], kv.values.shape[-1]).transpose(1, 2)
    if not torch.equal(kv.keys[..., -1:, :], rotated_key) or not torch.equal(kv.values[..., -1:, :], original_value):
        raise ValueError('Question-KV patch changed actual current-row K/V outside its target')
    state20 = lambda data: {k: v for k, v in data.items() if k.startswith('(20,')}
    if len(state20(result['states'])) != 2 or state20(result['states']) != state20(ctx['states']):
        raise ValueError('Output-only patches changed GDN20 already-written conv/recurrent states')
    earlier = kv_hashes(cache, n - 1)
    if patch == 'gdn_current_outputs':
        history_fields(ctx['cache'], cache, n - 1)
        if earlier != kv_hashes(ctx['cache'], n - 1):
            raise ValueError('Current-output patch changed previous-position K/V')
    if hook_state(runtime.model) != hooks:
        raise ValueError('Mediation hook cleanup failed')
    proof = {'id': query['id'], 'context': context, 'patch': patch, 'selected_layers': layers,
        'calls': op.calls, 'receipts': op.receipts, 'batches': op.batches, 'attention_stats': weights.stats,
        'patch_receipts': trace.receipts, 'input_calls': trace.input_calls,
        'complete_prefix_memory_exact': True, 'prefix_logits_exact': True,
        'prefix_sha256': prefix_ref['sha256'], 'masked_prefix_sha256': cache_fingerprint(prefix),
        'historical_kv_fields_exact': history_fields(unmasked['cache'], cache, h),
        'full_question_cache_sha256': cache_fingerprint(cache),
        'attention23_input_sha256': tensor_hash(trace.input23),
        'attention23_last_input_sha256': tensor_hash(trace.input23[:, -1:]),
        'attention23_earlier_input_sha256': tensor_hash(trace.input23[:, :-1]),
        'layer20_last_input_sha256': tensor_hash(trace.input20),
        'question_kv23_sha256': qkv, 'earlier_position_kv_sha256': earlier,
        'final_row_kv23_original_projection_exact_fields': 2,
        'recurrent_conv_sha256': result['states'], 'gdn20_written_states_unchanged': True,
        'hooks_restored': True}
    result.update(proof=proof, first_token=first_token(runtime, logits, query))
    return result


def execute(scope, output):
    import torch
    from safetensors.torch import load_file, save_file
    from sas_memory.controllers import GroupController
    from sas_memory.runtime import TextRuntime

    torch.set_num_threads(4); plan = scope['plan']
    manifest = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'training': False, 'new_adapters': False, 'new_data': False, 'posthoc': True, 'selection_is_posthoc': True,
        'independent_scientific_confirmation': False, 'can_change_primary_gates': False,
        'no_accuracy_gate': True, 'full_generation_scored': False, 'save_full_first_logits': True,
        'protocol_sha256': PROTOCOL_SHA256,
        **{k: plan[k] for k in ('model_path', 'dataset_manifest_sha256', 'mask_registry_sha256', 'arithmetic_profile',
            'precision_addendum_sha256', 'prior_timing_audit_sha256', 'prior_timing_manifest_sha256',
            'installed_modeling_source_sha256')},
        'original_primary_gates_unchanged': scope['primary_gates'],
        'source_sha256': {n: streaming_hash(ROOT / n) for n in sorted(sources())},
        'model_order': plan['models'], 'completed_models': [], 'models': {}, 'native_anchors': []}
    output.mkdir(parents=True, exist_ok=False); path = output / 'manifest.json'
    write_json(output / 'plan.json', plan); write_json(output / 'precision_addendum.json', scope['profile'])
    (output / 'installed_modeling_source.py').write_bytes(Path(plan['installed_modeling_source']).read_bytes())
    for name in manifest['source_sha256']:
        target = output / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((ROOT / name).read_bytes())
    write_json(path, manifest); runtime = native_hooks = None
    try:
        runtime = TextRuntime(plan['model_path'], 'it')
        manifest.update(python=sys.executable, gpu=torch.cuda.get_device_name(0),
            packages={n: importlib.metadata.version(n) for n in plan['packages']},
            actual_vocab_size=runtime.model.get_output_embeddings().weight.shape[0],
            actual_eos_token_id=runtime.model.generation_config.eos_token_id,
            generation_config_sha256={p.name: streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')})
        if (manifest['packages'] != plan['packages'] or
            manifest['actual_eos_token_id'] != scope['primary_manifest']['actual_eos_token_id'] or
            manifest['generation_config_sha256'] != scope['primary_manifest']['generation_config_sha256'] or
            runtime.model.config.layer_types != (['linear_attention'] * 3 + ['full_attention']) * 6):
            raise ValueError('Frozen runtime, hybrid layout or generation configuration changed')
        native_hooks = hook_state(runtime.model); before = manifest['weights_before'] = weights_fingerprint(runtime.model)
        if (before['sha256'] != plan['native_loaded_weights_sha256'] or
            not before['all_parameters_frozen'] or not before['no_parameter_gradients']):
            raise ValueError('Actual native checkpoint/freeze changed')
        cases = {c['id']: c for c in scope['cases']}; anchors = {}
        with torch.inference_mode():
            for entry in plan['queries']:
                q = scope['queries'][entry['id']]; actual = measure(runtime, actual_prompt(runtime, cases, q), q)
                manifest['native_anchors'].append({'id': q['id'], 'readout': actual,
                    'exact_completed_fresh_native_replay': actual == scope['native_references'][q['id']]})
                write_json(path, manifest)
                if actual != scope['native_references'][q['id']]: raise ValueError('Native original anchor drifted')
                anchors[q['id']] = actual
        prior = {(r['model'], r['id'], r['context']): r for r in plan['prior_references']}
        for name in plan['models']:
            spec = scope['source_plan']['models'][name]; entries = [e for e in plan['queries'] if e['model'] == name]
            state = manifest['models'][name] = {'token_policy': 'question', 'proofs': [], 'prefix_checks': [],
                'native_restore_checks': [], 'completed_queries': 0, 'recorded_rows': 0}
            controller = arithmetic = None; vectors = {}
            result_path, logits_path = output / f'{name}.jsonl', output / f'{name}.logits.safetensors'
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
                        q = scope['queries'][entry['id']]; ledger = actual_prompt(runtime, cases, q)
                        ids = runtime.ids(ledger + runtime.query_suffix(q['question']))
                        if data_fingerprint(ids) != q['prompt_ids_sha256'] or len(ids) != q['prompt_tokens']:
                            raise ValueError('Original full input IDs changed')
                        controller.set_token_policy('none'); native_prefix, native_logits = advance_ids(runtime, ids[:q['ledger_tokens']])
                        controller.set_token_policy('question', q['ledger_tokens'])
                        prefix_ref = {'cache': native_prefix, 'logits': native_logits, 'sha256': cache_fingerprint(native_prefix)}
                        if prefix_ref['sha256'] != next(p['prefix_sha256'] for p in state['prefix_checks'] if p['id'] == q['id']):
                            raise ValueError('Actual native ledger-prefix reference changed')
                        keep = [v == '1' for v in scope['masks'][q['id']]['keep_bits'][entry['regime']]]
                        state['active_query'] = q['id']; write_json(path, manifest)
                        unmasked = None
                        fixed_competitor = prior[(name, q['id'], 'unmasked')]['first_token']['non_gold_id']
                        for context in plan['context_order']:
                            ctx = gdn_states = None
                            for patch in plan['patch_order']:
                                result = condition(runtime, q, ids, keep, context, patch, prefix_ref, unmasked, ctx)
                                if unmasked is None: unmasked = result
                                if ctx is None: ctx = result
                                if patch == 'gdn_current_outputs': gdn_states = result['states']
                                expected_states = ctx['states'] if patch == 'question_kv' else gdn_states if patch == 'both' else None
                                if expected_states is not None and result['states'] != expected_states:
                                    raise ValueError('Internal KV patch changed previously computed recurrent/conv states')
                                result['proof']['kv_patch_recurrent_states_unchanged'] = True if expected_states is not None else None
                                closure = context == 'earlier_only_layer19' and patch == 'both'
                                result['proof']['combined_paths_logits_exact'] = bool(torch.equal(result['logits'], unmasked['logits'])) if closure else None
                                key = f"{q['id']}--{context}--{patch}"
                                vectors[key] = result['logits'].detach().cpu().contiguous()
                                row = {'model': name, 'id': q['id'], 'context': context, 'patch': patch,
                                    'source_regime': entry['regime'], 'original_direction': entry['direction'],
                                    'token_policy': 'question', 'prefill': 'full', 'arithmetic_profile': plan['arithmetic_profile'],
                                    'registration_sha256': data_fingerprint(q), 'prompt_ids_sha256': q['prompt_ids_sha256'],
                                    'mask_registration_sha256': entry['mask_registration_sha256'], 'keep_bits_sha256': entry['keep_bits_sha256'],
                                    'first_token': result['first_token'], 'logits_key': key, 'fixed_competitor_id': fixed_competitor,
                                    'fixed_pair_margin': result['logits'][0, q['answer_ids'][0]].item() - result['logits'][0, fixed_competitor].item()}
                                stream.write(json.dumps(row) + '\n'); state['proofs'].append(result['proof'])
                                state['recorded_rows'] += 1; write_json(path, manifest)
                                if patch == 'none':
                                    expected = prior[(name, q['id'], context)]
                                    if (row['first_token'] != expected['first_token'] or any(result['proof'][k] != expected[k]
                                        for k in ('prefix_sha256', 'full_question_cache_sha256', 'attention23_input_sha256'))):
                                        raise ValueError('Unpatched first logits, complete cache or input drifted from completed timing run')
                                if closure and not result['proof']['combined_paths_logits_exact']:
                                    raise ValueError('Combined oracle paths failed to close to unmasked first logits')
                                print(json.dumps({'event': 'question_path_mediation', 'model': name, 'id': q['id'],
                                    'context': context, 'patch': patch, 'rows': state['recorded_rows']}), flush=True)
                                if result is not ctx and result is not unmasked: del result
                            del ctx
                        del unmasked, prefix_ref, native_prefix
                        state['completed_queries'] += 1; state.pop('active_query'); write_json(path, manifest)
            finally:
                if vectors:
                    save_file(vectors, str(logits_path)); state['logits_file_sha256'] = streaming_hash(logits_path)
                    state['logits_tensors'] = len(vectors)
                if controller is not None:
                    state.update(adapter_state_sha256_after=state_fingerprint(controller),
                        all_adapter_parameters_frozen=all(not p.requires_grad and p.grad is None for p in controller.parameters()))
                if arithmetic is not None:
                    state['actual_affine_calls'] = dict(arithmetic.calls); arithmetic.close()
                    state['arithmetic_hooks_removed'] = all(not mod._forward_hooks for mod in controller.modules())
                if controller is not None: controller.close()
                if result_path.exists(): state['results_sha256'] = streaming_hash(result_path)
                state['hooks_restored'] = hook_state(runtime.model) == native_hooks
            if (state['completed_queries'] != 2 or state['recorded_rows'] != 16 or state['logits_tensors'] != 16 or
                not state['hooks_restored'] or not state['arithmetic_hooks_removed'] or not state['all_adapter_parameters_frozen'] or
                state['adapter_state_sha256_after'] != state['adapter_state_sha256_before'] or
                any(v < 1 for v in state['actual_affine_calls'].values())):
                raise ValueError('Frozen path execution, full-vector archival or cleanup failed')
            with torch.inference_mode():
                for key, expected in anchors.items():
                    q = scope['queries'][key]; actual = measure(runtime, actual_prompt(runtime, cases, q), q)
                    state['native_restore_checks'].append({'id': key, 'readout': actual, 'exact': actual == expected})
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
        if runtime is not None and native_hooks is not None: manifest['hooks_restored'] = hook_state(runtime.model) == native_hooks
        write_json(path, manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve all previous runs; choose a new directory')
    execute(input_scope(), args.output)
