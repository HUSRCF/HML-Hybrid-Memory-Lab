"""Exact posthoc replay of all four registered preservation-control flips."""

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
from scripts.copy_readout_runtime import prove_and_measure, readout
from scripts.fresh_copy_data import registered_inputs, selected_inputs, sources as fresh_sources, read_plan as fresh_plan
from scripts.run_controller_trial import state_fingerprint, write_json
from scripts.run_copy_dependency_transfer import actual_prompt
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_adapter_transfer import verify_question_prefix
from scripts.run_write_locality_load_screen import hook_state, weights_fingerprint
from scripts.stable_adapter_linears import StableAdapterLinears

PROTOCOL = ROOT / 'configs/fresh_control_change_replay_plan.json'
PROTOCOL_SHA256 = '1a3ba5fa85c2547ab5e915f874fc27d9908ec8b05a844cbe6f4f13259b55b1c1'
ADDITIONS = {'configs/fresh_control_change_replay_plan.json', 'scripts/run_fresh_control_replay.py',
    'scripts/analyze_fresh_control_replay.py', 'scripts/analyze_fresh_control_changes.py'}


def sources():
    return fresh_sources() | ADDITIONS


def trials(plan, model):
    selected = [q for q in plan['queries'] if q['model'] == model]
    result = []
    for repeat, order in enumerate(plan['query_order_within_model'], 1):
        if order not in ('registered', 'reverse_registered'):
            raise ValueError('Unregistered query ordering')
        for entry in (selected if order == 'registered' else selected[::-1]):
            result.append({'trial_index': len(result) + 1, 'repeat': repeat, 'id': entry['id'],
                'masked_regime': entry['regime'], 'direction': entry['direction']})
    return result


def input_scope():
    plan = json.loads(PROTOCOL.read_text())
    if (streaming_hash(PROTOCOL) != PROTOCOL_SHA256 or plan['status'] != 'registered_before_targeted_replay' or
        plan['models'] != ['local_question', 'gdn_question_only'] or plan['repeats'] != 3 or
        plan['query_order_within_model'] != ['registered', 'reverse_registered', 'registered'] or
        len(plan['queries']) != 4 or len({q['id'] for q in plan['queries']}) != 4 or
        sum(q['direction'] == 'harm' for q in plan['queries']) != 3 or
        sum(q['direction'] == 'repair' for q in plan['queries']) != 1):
        raise ValueError('Fixed four-change replay scope changed')
    audit_path, change_path = Path(plan['completed_fresh_audit']), Path(plan['control_change_report'])
    if (streaming_hash(audit_path) != plan['completed_fresh_audit_sha256'] or
        streaming_hash(change_path) != plan['control_change_report_sha256']):
        raise ValueError('Completed primary or posthoc selection changed')
    audit, change = json.loads(audit_path.read_text()), json.loads(change_path.read_text())
    run = Path(plan['completed_fresh_run'])
    manifest = json.loads((run / 'manifest.json').read_text())
    if (audit['status'] != 'completed_and_audited_fresh_copy' or manifest['status'] != 'completed' or
        audit['source_run'] != str(run) or streaming_hash(run / 'manifest.json') != plan['completed_fresh_manifest_sha256'] or
        audit['manifest_sha256'] != plan['completed_fresh_manifest_sha256'] or
        streaming_hash(audit_path.with_suffix('.source.py')) != audit['analysis_source_sha256'] or
        change['status'] != 'posthoc_descriptive_control_changes_no_new_inference' or
        change['audit_sha256'] != plan['completed_fresh_audit_sha256'] or change['counts'] != {'harm': 3, 'repair': 1} or
        change['selection_rule'] != plan['selection_rule'] or
        streaming_hash(change_path.with_suffix('.source.py')) != plan['control_change_analysis_source_sha256'] or
        streaming_hash(ROOT / 'scripts/analyze_fresh_control_changes.py') != plan['control_change_analysis_source_sha256'] or
        change['analysis_source_sha256'] != plan['control_change_analysis_source_sha256'] or
        not datetime.fromisoformat(manifest['completed_utc']) < datetime.fromisoformat(plan['registered_utc']) or
        set(manifest['source_sha256']) != fresh_sources()):
        raise ValueError('Actual prior source closure, selection provenance or chronology changed')
    for name, digest in manifest['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / 'sources' / name) != digest:
            raise ValueError('Preserve all109completed fresh runtime sources')
    dataset = Path(plan['dataset'])
    registered, source_plan, cases, registry, masks = registered_inputs(dataset)
    original = fresh_plan()
    profile = json.loads(Path(original['precision_addendum']).read_text())
    if (streaming_hash(dataset / 'manifest.json') != plan['dataset_manifest_sha256'] or
        registered['mask_registry_sha256'] != plan['mask_registry_sha256'] or
        streaming_hash(Path(original['precision_addendum'])) != plan['precision_addendum_sha256'] or
        profile['profile'] != plan['arithmetic_profile'] or
        any(plan[k] != original[k] for k in ('model_path', 'packages', 'arithmetic_profile')) or
        audit['weights_sha256'] != plan['native_loaded_weights_sha256']):
        raise ValueError('Actual frozen data, model or arithmetic profile changed')
    physical, bindings = selected_inputs(registry)
    queries = {q['id']: q for q in physical}
    all_rows = {}
    for model in plan['models']:
        spec, state = source_plan['models'][model], manifest['models'][model]
        fixed = {k: state[k] for k in ('checkpoint_sha256', 'adapter_state_sha256_before', 'description')}
        if fixed != plan['fixed_checkpoints'][model] or spec['token_policy'] != 'question':
            raise ValueError('Fixed original question-only checkpoint changed')
        all_rows[model] = {}
        for regime in ('full', 'oracle', 'remove_matched_other'):
            path = run / f'{model}.{regime}.jsonl'
            if streaming_hash(path) != state['results_sha256'][regime]:
                raise ValueError('Completed original observations changed')
            all_rows[model][regime] = {v['id']: v for v in map(json.loads, path.read_text().splitlines())}
    derived = []
    for model in plan['models']:
        for regime in ('oracle', 'remove_matched_other'):
            for key, row in all_rows[model][regime].items():
                full = all_rows[model]['full'][key]
                linked = [b for b in bindings if b['physical_id'] == key]
                if (row['greedy_exact'] == full['greedy_exact'] or
                    (regime == 'oracle' and all(b['role'] == 'target_current' for b in linked))):
                    continue
                q = queries[key]
                if (row['registration_sha256'] != data_fingerprint(q) or
                    row['mask_registration_sha256'] != data_fingerprint(masks[key]) or
                    row['keep_bits_sha256'] != data_fingerprint(masks[key]['keep_bits'][regime])):
                    raise ValueError('Original query or complete-record mask changed')
                derived.append({'model': model, 'regime': regime, 'id': key,
                    'direction': 'harm' if full['greedy_exact'] else 'repair', 'bindings': linked,
                    'expected_full': readout(full), 'expected_intervened': readout(row),
                    **{k: row[k] for k in ('registration_sha256', 'mask_registration_sha256', 'keep_bits_sha256')}})
    if derived != plan['queries'] or len(change['changes']) != len(derived):
        raise ValueError('Every actual correctness flip, including the repair, must be retained')
    for entry, descriptive in zip(derived, change['changes']):
        if (any(descriptive[k] != entry[k] for k in ('model', 'regime', 'id', 'direction', 'bindings')) or
            descriptive['full'] != entry['expected_full'] or descriptive['intervened'] != entry['expected_intervened'] or
            descriptive['mask_registration'] != masks[entry['id']]):
            raise ValueError('Posthoc report no longer matches actual original observations')
    native_path = run / 'native.full.jsonl'
    if streaming_hash(native_path) != manifest['models']['native']['results_sha256']['full']:
        raise ValueError('Completed original native observations changed')
    native = {r['id']: readout(r) for r in map(json.loads, native_path.read_text().splitlines())}
    return {'plan': plan, 'registration': registered, 'source_plan': source_plan, 'cases': cases,
        'queries': queries, 'masks': masks, 'profile': profile, 'audit': audit, 'primary_manifest': manifest,
        'native_references': {q['id']: native[q['id']] for q in plan['queries']},
        'primary_gates': {k: audit['metrics'][k] for k in
            ('generic_shared_controller', 'frozen_prefix_sufficiency', 'whole_record_necessity')}}


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
        'protocol_sha256': PROTOCOL_SHA256, 'model_path': plan['model_path'],
        'dataset_manifest_sha256': plan['dataset_manifest_sha256'], 'mask_registry_sha256': plan['mask_registry_sha256'],
        'arithmetic_profile': plan['arithmetic_profile'], 'precision_addendum_sha256': plan['precision_addendum_sha256'],
        'completed_fresh_audit_sha256': plan['completed_fresh_audit_sha256'],
        'completed_fresh_manifest_sha256': plan['completed_fresh_manifest_sha256'],
        'control_change_report_sha256': plan['control_change_report_sha256'],
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
                    raise ValueError(f"Native anchor differs from completed primary: {q['id']}")
                anchors[q['id']] = actual
        for name in plan['models']:
            spec = scope['source_plan']['models'][name]
            selected = [q for q in plan['queries'] if q['model'] == name]
            expected = {q['id']: q for q in selected}
            state = manifest['models'][name] = {'token_policy': 'question', 'proofs': [], 'prefix_checks': [],
                'native_restore_checks': [], 'completed_trials': 0, 'recorded_rows': 0}
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
                    for trial in trials(plan, name):
                        state['active_trial'] = trial; write_json(path, manifest)
                        q, entry = scope['queries'][trial['id']], expected[trial['id']]
                        ledger = actual_prompt(runtime, cases, q)
                        controller.set_token_policy('question', q['ledger_tokens'])
                        active_hooks = hook_state(runtime.model)
                        full = measure(runtime, ledger, q)

                        def record(regime, actual, calls=None):
                            original = entry['expected_full' if regime == 'full' else 'expected_intervened']
                            row = {'model': name, **trial, 'regime': regime, 'token_policy': 'question', 'prefill': 'full',
                                'arithmetic_profile': plan['arithmetic_profile'], 'registration_sha256': data_fingerprint(q),
                                'mask_registration_sha256': None if regime == 'full' else entry['mask_registration_sha256'],
                                'keep_bits_sha256': None if regime == 'full' else entry['keep_bits_sha256'],
                                'hook_calls': calls, 'exact_original_replay': actual == original, **actual}
                            stream.write(json.dumps(row) + '\n'); state['recorded_rows'] += 1
                            write_json(path, manifest)
                            if actual != original:
                                raise ValueError(f"Original readout drift: model={name},trial={trial},regime={regime},expected={original},actual={actual}")

                        record('full', full)
                        regime = trial['masked_regime']
                        keep = [v == '1' for v in scope['masks'][q['id']]['keep_bits'][regime]]
                        proof, actual = prove_and_measure(runtime, ledger, q, {regime: keep}, full)
                        state['proofs'].append({**trial, 'proof': proof})
                        record(regime, actual[regime], proof['regimes'][regime]['measure_hook_calls'])
                        if hook_state(runtime.model) != active_hooks:
                            raise ValueError('Replay or proof failed to restore controller hooks')
                        state['completed_trials'] += 1
                        state.pop('active_trial')
                        write_json(path, manifest)
                        print(json.dumps({'event': 'control_change_replay', 'model': name,
                            **trial, 'full_and_intervened_exact': True}), flush=True)
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
            if (state['completed_trials'] != 6 or state['recorded_rows'] != 12 or not state['hooks_restored'] or
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
        raise ValueError('Preserve all previous runs; select a new output directory')
    execute(input_scope(), args.output)
