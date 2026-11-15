"""Frozen normal-update diagnostic, gated by exact opposite-order smokes."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.natural_load_data import (read_plan, registered_inputs, sources, selected_contexts,
    context_queries, DESTINATION, PROTOCOL, MODELS, READERS)
from sas_memory.runtime import TextRuntime
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.post_encoding_runtime import CacheGuard, advance_tail, assert_isolated, measure_cached
from scripts.run_controller_trial import write_json, state_fingerprint
from scripts.run_oracle_kv_readout import measure
from scripts.update_budget_data import actual_prompt
from scripts.run_write_locality_load_screen import hook_state, weights_fingerprint
from scripts.canonical_necessity_common import check_affine_scope
from scripts.stable_adapter_linears import StableAdapterLinears
from scripts.gate_component_runtime import component_kernel_contract, validation_addendum, VALIDATION_ADDENDUM_SHA256


def checked_gate(path, packet_digest):
    from scripts.analyze_natural_load import audit_equivalence
    path = Path(path); g = json.loads(path.read_text())
    if (g != audit_equivalence(Path(g['forward_run']), Path(g['reverse_run'])) or
            g['packet_manifest_sha256'] != packet_digest or
            streaming_hash(path.with_suffix('.source.py')) != g['analysis_source_sha256']):
        raise ValueError('Both exact independently audited108readout smokes required')
    return {'path': str(path), 'sha256': streaming_hash(path)}


def run_model(runtime, name, plan, registry, queries, refs, smoke, output, state, anchors, progress):
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    spec = plan['models'][name]; controller = None; arithmetic = None
    native_hooks = hook_state(runtime.model); controls = refs['legacy_ids']; legacy = refs['legacy_queries']
    def activate(policy, boundary=0):
        if controller:
            controller.set_token_policy(policy, boundary)
    try:
        for key in controls:
            v = measure(runtime, actual_prompt(runtime, legacy[key]), legacy[key])
            if v != anchors[key]:
                raise ValueError('Native preattach endpoint changed')
            state['native_before_attach_checks'].append({'id': key, 'readout': v})
        if spec:
            controller = GroupController(runtime.model, spec['group_index'], spec['rank'], spec['mode'], adapt_gdn_keys=spec['adapt_gdn_keys'])
            controller.load_state_dict(load_file(spec['checkpoint'], device=str(runtime.model.device)), strict=True)
            controller.eval().requires_grad_(False)
            state.update(description=controller.description(), adapter_parameters=sum(p.numel() for p in controller.parameters()),
                checkpoint_sha256=streaming_hash(Path(spec['checkpoint'])), adapter_state_sha256_before=state_fingerprint(controller))
            if (state['description'] != refs['model_states'][name]['description'] or
                    state['adapter_parameters'] != spec['parameters'] or state['checkpoint_sha256'] != spec['checkpoint_sha256'] or
                    state['adapter_state_sha256_before'] != refs['model_states'][name]['adapter_state_sha256_before']):
                raise ValueError('Existing frozen adapter changed')
            state['affine_scope'] = check_affine_scope(controller, runtime.model)
            arithmetic = StableAdapterLinears(controller); state['arithmetic'] = arithmetic.description()
            for key in controls:
                activate('none', legacy[key]['ledger_tokens'])
                v = measure(runtime, actual_prompt(runtime, legacy[key]), legacy[key])
                if v != anchors[key]:
                    raise ValueError('Attached-none native endpoint changed')
                state['attached_none_checks'].append({'id': key, 'readout': v})
        activate('all' if spec else 'none')
        for key in controls:
            v = measure(runtime, actual_prompt(runtime, legacy[key]), legacy[key])
            if v != refs['legacy_full'][name][key]:
                raise ValueError('Frozen legacy endpoint changed')
            state['legacy_full_checks'].append({'id': key, 'readout': v})
        contexts = selected_contexts(registry, smoke)
        with (output/f'{name}.jsonl').open('w', buffering=1) as raw:
            for case_id in dict.fromkeys(c['case_id'] for c in contexts):
                group = [c for c in contexts if c['case_id'] == case_id]; before = group[0]
                if before['episode'] != 'before' or len(group) != 9:
                    raise ValueError('Common prefix and all8normal tails required')
                prefix, logits, count = runtime.advance(before['prefix_text'])
                if count != before['prefix_tokens']:
                    raise ValueError('Actual prefix boundary changed')
                parent = CacheGuard(prefix, logits); parent.record()
                children = {}; guards = {}; proofs = {}
                for context in group[1:]:
                    cache, last, proof = advance_tail(runtime, prefix, context['prefix_text'], context['tail_text'], context['prefix_tokens'], 'none')
                    parent.check(); guard = CacheGuard(cache, last); guard.record()
                    children[context['id']] = cache; guards[context['id']] = guard; proofs[context['id']] = proof
                actual = [prefix] + list(children.values())
                for i, left in enumerate(actual):
                    for right in actual[i+1:]:
                        assert_isolated(left, right)
                # All sources exist before even the first baseline question or answer.
                for context in group:
                    cid = context['id']; cache = prefix if context['episode'] == 'before' else children[cid]
                    for q in context_queries(context, queries, smoke):
                        for reader in READERS:
                            value, proof = measure_cached(runtime, context, cache, q, reader)
                            parent.check()
                            if cid in guards:
                                guards[cid].check()
                            raw.write(json.dumps({'id': q['id'], 'model': name, 'reader': reader,
                                'registration_sha256': data_fingerprint(q), 'source_key': q['source_key'],
                                'prompt_ids_sha256': q['prompt_ids_sha256'], 'prefill': 'cached',
                                'measurement_origin': 'normally_encoded_without_intervention', 'proof': proof, **value})+'\n')
                            state['completed_primary'] += 1
                    if cid in guards:
                        state['sources'][cid] = {'cache': guards[cid].record(), 'tail_proof': proofs[cid],
                            'all_sources_built_before_any_question': True, 'pairwise_storage_isolated': True}
                state['prefixes'][before['prefix_key']] = parent.record()
                del actual, children, guards, proofs, prefix, logits, parent, cache, guard, last
                progress()
                print(json.dumps({'event': 'normal_update_load', 'model': name, 'case_id': case_id,
                    'readouts': state['completed_primary']}), flush=True)
        if controller:
            state.update(adapter_state_sha256_after=state_fingerprint(controller), actual_affine_calls=dict(arithmetic.calls),
                all_adapter_parameters_frozen=all(not p.requires_grad and p.grad is None for p in controller.parameters()))
            if state['adapter_state_sha256_after'] != state['adapter_state_sha256_before'] or not state['all_adapter_parameters_frozen']:
                raise ValueError('Frozen adapter bytes or gradients changed')
    finally:
        if arithmetic:
            arithmetic.close(); state['arithmetic_hooks_removed'] = all(not p._forward_hooks for p in controller.modules())
        if controller:
            controller.close()
    if hook_state(runtime.model) != native_hooks:
        raise ValueError('Model hooks not restored')
    for key in controls:
        v = measure(runtime, actual_prompt(runtime, legacy[key]), legacy[key])
        if v != anchors[key]:
            raise ValueError('Detached native endpoint changed')
        state['native_restore_checks'].append({'id': key, 'readout': v})
    state.update(hooks_restored=True, results_sha256=streaming_hash(output/f'{name}.jsonl'))


def execute_run(args):
    import torch
    torch.set_num_threads(4)
    if args.output.exists() or args.smoke == bool(args.smoke_equivalence) or (args.reverse_model_order and not args.smoke):
        raise ValueError('New operational smoke or gated complete primary required')
    plan = read_plan(); registered, cases, registry, queries, refs = registered_inputs(args.packet)
    gate = None if args.smoke else checked_gate(args.smoke_equivalence, streaming_hash(args.packet/'manifest.json'))
    profile = json.loads(Path(plan['precision_addendum']).read_text()); correction = validation_addendum()
    order = list(MODELS[::-1] if args.reverse_model_order else MODELS)
    m = {'status':'running', 'started_utc':datetime.now(timezone.utc).isoformat(), 'training':False,
        'new_adapters':False, 'fresh_streams':True, 'smoke':args.smoke, 'reverse_model_order':args.reverse_model_order,
        'model_order':order, 'packet':str(args.packet), 'packet_manifest_sha256':streaming_hash(args.packet/'manifest.json'),
        'protocol_sha256':streaming_hash(PROTOCOL), 'source_sha256':{n:streaming_hash(ROOT/n) for n in sorted(sources())},
        'models':{}, 'completed_models':[], 'native_anchors':[], 'model_path':plan['model_path'],
        'generation':plan['generation'], 'arithmetic_profile':profile['profile'],
        'precision_addendum_sha256':streaming_hash(Path(plan['precision_addendum'])),
        'validation_addendum_sha256':VALIDATION_ADDENDUM_SHA256, 'smoke_equivalence_gate':gate,
        'generation_config_sha256':{p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')}}
    args.output.mkdir(parents=True, exist_ok=False); mp = args.output/'manifest.json'
    for name, value in (('plan', plan), ('precision_addendum', profile), ('kernel_validation_addendum', correction)):
        write_json(args.output/(name+'.json'), value)
    for name in m['source_sha256']:
        p = args.output/'sources'/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((ROOT/name).read_bytes())
    write_json(mp, m)
    try:
        runtime = TextRuntime(plan['model_path'], 'it')
        m.update(python=sys.executable, gpu=torch.cuda.get_device_name(0),
            packages={n:importlib.metadata.version(n) for n in ('torch','transformers','fla-core')},
            actual_eos_token_id=runtime.model.generation_config.eos_token_id)
        if (m['packages'] != plan['packages'] or m['actual_eos_token_id'] != refs['runtime']['actual_eos_token_id'] or
                m['generation_config_sha256'] != refs['runtime']['generation_config_sha256'] or
                runtime.model.config.layer_types != (['linear_attention']*3+['full_attention'])*6):
            raise ValueError('Actual native runtime changed')
        installed = Path(plan['installed_modeling_source'])
        if streaming_hash(installed) != plan['installed_modeling_source_sha256']:
            raise ValueError('Native installed source changed')
        (args.output/'installed_modeling_source.py').write_bytes(installed.read_bytes())
        native_hooks = hook_state(runtime.model); m['weights_before'] = weights_fingerprint(runtime.model)
        if (m['weights_before']['sha256'] != plan['native_loaded_weights_sha256'] or
                not m['weights_before']['all_parameters_frozen'] or not m['weights_before']['no_parameter_gradients']):
            raise ValueError('Native loaded weights or freeze changed')
        m['native_component_kernel_checks'] = component_kernel_contract(runtime)
        anchors = {}
        with torch.inference_mode():
            for key in refs['legacy_ids']:
                q = refs['legacy_queries'][key]; v = measure(runtime, actual_prompt(runtime, q), q)
                if v != refs['legacy_full']['native'][key]:
                    raise ValueError('Original native anchor changed')
                anchors[key] = v; m['native_anchors'].append({'id':key, 'readout':v})
            for name in order:
                state = {'token_policy':'all' if plan['models'][name] else 'none', 'completed_primary':0,
                    'native_before_attach_checks':[], 'attached_none_checks':[], 'legacy_full_checks':[],
                    'native_restore_checks':[], 'prefixes':{}, 'sources':{}}
                m['models'][name] = state; write_json(mp, m)
                run_model(runtime, name, plan, registry, queries, refs, args.smoke, args.output, state, anchors, lambda:write_json(mp,m))
                m['completed_models'].append(name); write_json(mp, m)
        m['weights_after'] = weights_fingerprint(runtime.model); m['hooks_restored'] = hook_state(runtime.model) == native_hooks
        if m['weights_after'] != m['weights_before'] or not m['hooks_restored']:
            raise ValueError('Final native weights or hooks changed')
        m.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as e:
        m.update(status='failed', error=f'{type(e).__name__}: {e}'); raise
    finally:
        write_json(mp, m)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, default=DESTINATION); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--reverse-model-order', action='store_true')
    p.add_argument('--smoke-equivalence', type=Path)
    execute_run(p.parse_args())


if __name__ == '__main__':
    main()
