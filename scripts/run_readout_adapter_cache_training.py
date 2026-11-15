"""Frozen matched readout protocol with registered teacher-cache commit correction.

Copied from the sealed runner; native arithmetic, examples and optimizer budgets
are unchanged. Explicit imports and safety receipts identify the correction.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.readout_adapter_modules import ReadoutAdapter
from sas_memory.calibration import save_checkpoint
from sas_memory.write_locality_data import streaming_hash
from sas_memory.training_data import data_fingerprint
from scripts.readout_adapter_data import ARMS, registered_inputs, verify_seal
from scripts.readout_adapter_runtime import (base, FixedGDN, normal_groups, normal_readout,
                                             legacy_controls, check_normal_source)
from scripts.run_controller_trial import write_json, state_fingerprint
from scripts.fresh_layer_controls import tensor_hash
from scripts.readout_adapter_cache_commit import (corrected_teacher as teacher,
    cache_commit_equivalence, verify_correction_seal)


def gradient_norms(adapter):
    return {name: None if value.grad is None else float(value.grad.detach().float().norm().item())
            for name, value in adapter.named_parameters()}


def freeze_check(runtime, fixed):
    if any(p.requires_grad or p.grad is not None for p in runtime.model.parameters()) or any(
        p.requires_grad or p.grad is not None for p in fixed.controller.parameters()):
        raise ValueError('Only newly declared readout parameters may have gradients')


def initialization(adapter):
    return {'description': adapter.description(), 'initial_state_sha256': state_fingerprint(adapter),
            'initial_branch_affines': {name: {part: tensor_hash(layer.weight) for part, layer in branch.items()}
                                        for name, branch in adapter.branches.items()}}


def optimizer_for(adapter, plan):
    c = plan['training']
    return torch.optim.AdamW(adapter.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])


def scratch(runtime, fixed, adapter, plan, registry, old_state, examples, queries, expected, destination):
    iterator = normal_groups(runtime, adapter, registry, old_state, smoke=True)
    context, source, guard = next(iterator)
    selected = [e for e in examples if e['context_id'] == context['id']]
    if len(selected) != 6 or selected[:4] != examples[:4]:
        raise ValueError('The six first-context questions and first four teacher examples are fixed')
    zero = []
    for e in selected:
        value = normal_readout(runtime, adapter, context, source, queries[e['id']])
        if value != {k: expected[e['id']][k] for k in value}:
            raise ValueError('Zero readout adapter changed an original normal answer')
        zero.append({'id': e['id'], 'readout': value})
        guard.check()
    before_equivalence = cache_commit_equivalence(runtime, adapter, source, selected[0])
    guard.check()
    optimizer = optimizer_for(adapter, plan)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for e in selected[:4]:
        loss, details = teacher(runtime, adapter, source, e, True)
        (loss / 4).backward()
        losses.append(details)
        del loss
        guard.check()
        freeze_check(runtime, fixed)
    first = gradient_norms(adapter)
    if any(first[f'branches.{name}.b.weight'] is None or
           not math.isfinite(first[f'branches.{name}.b.weight']) or first[f'branches.{name}.b.weight'] <= 0
           for name in adapter.branches):
        raise ValueError('Active output branch has no finite gradient')
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    log = destination / 'training.jsonl'
    log.write_text(json.dumps({'step': 1, 'ids': [e['id'] for e in selected[:4]],
                               'losses': losses, 'gradient_norm': float(norm.item())}) + '\n')
    checkpoint = save_checkpoint(destination, 1, adapter, optimizer, log,
                                 {'protocol': plan['profile'], 'scratch_only': True})
    loss, _ = teacher(runtime, adapter, source, selected[0], True)
    loss.backward()
    second = gradient_norms(adapter)
    del loss
    after_equivalence = cache_commit_equivalence(runtime, adapter, source, selected[0])
    guard.check()
    if any(second[f'branches.{name}.a.weight'] is None or
           not math.isfinite(second[f'branches.{name}.a.weight']) or second[f'branches.{name}.a.weight'] <= 0
           for name in adapter.branches):
        raise ValueError('Encoder gradient did not arrive after the throwaway update')
    optimizer.zero_grad(set_to_none=True)
    guard.check()
    adapter.set_policy(True, context['ledger_tokens'])
    with torch.inference_mode():
        updated, logits, _ = runtime.advance(context['prefix_text'])
        updated_guard = base.CacheGuard(updated, logits)
        rebuilt = check_normal_source(updated_guard, old_state['prefixes'][context['prefix_key']])
    result = {'status': 'passed_real_one_update_readout_safety', 'zero_normal_readouts': zero,
              'cache_commit_equivalence': {'before_update': before_equivalence, 'after_update': after_equivalence},
              'scratch_optimizer_updates': 1, 'second_backward_without_update': True,
              'first_gradient_norms': first, 'second_gradient_norms': second,
              'checkpoint_sha256': streaming_hash(checkpoint / 'controller.safetensors'),
              'checkpoint_metadata_sha256': streaming_hash(checkpoint / 'checkpoint.json'),
              'scratch_state_sha256': state_fingerprint(adapter), 'scope': dict(adapter.scope),
              'prefix_after_nonzero_update': rebuilt, 'fixed_parameters': fixed.verify()}
    iterator.close()
    return result


def scheduled(runtime, fixed, adapter, plan, registry, old_state, examples, queries, destination, baseline):
    by_id = {e['id']: e for e in examples}
    optimizer = optimizer_for(adapter, plan)
    optimizer.zero_grad(set_to_none=True)
    step, batch, cache_records, seen = 0, [], {}, []
    started = time.monotonic()
    baseline_path = destination / 'disabled_teacher_baseline.jsonl'
    baseline_handle = baseline_path.open('x', buffering=1) if baseline else None
    log = destination / 'training.jsonl'
    try:
        with log.open('x', buffering=1) as handle:
            for context, source, guard in normal_groups(runtime, adapter, registry, old_state):
                cache_records[context['id']] = dict(guard.normal_source_record)
                for qid in context['query_ids']:
                    e = by_id[qid]
                    if baseline_handle:
                        adapter.set_policy(False, e['ledger_tokens'])
                        _, detail = teacher(runtime, None, source, e, False)
                        baseline_handle.write(json.dumps(detail) + '\n')
                        guard.check()
                    loss, detail = teacher(runtime, adapter, source, e, True)
                    (loss / 4).backward()
                    del loss
                    guard.check()
                    freeze_check(runtime, fixed)
                    seen.append(qid)
                    batch.append(detail)
                    if len(batch) == 4:
                        gradients = gradient_norms(adapter)
                        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        step += 1
                        handle.write(json.dumps({'step': step, 'examples': batch,
                            'gradient_norm': float(norm.item()), 'parameter_gradient_norms': gradients,
                            'native_and_fixed_gdn_gradient_free': True,
                            'elapsed_seconds': time.monotonic() - started}) + '\n')
                        batch = []
                        if step in plan['training']['checkpoint_steps']:
                            save_checkpoint(destination, step, adapter, optimizer, log,
                                {'description': adapter.description(), 'fixed_gdn_state_sha256': fixed.before,
                                 'profile': plan['profile'], 'final_only_selection': True})
                        if step == 1 or step % 16 == 0:
                            print(json.dumps({'event': 'readout_training', 'arm': adapter.mode,
                                               'step': step, 'elapsed_seconds': time.monotonic() - started}), flush=True)
        if step != 432 or batch or seen != [e['id'] for e in examples] or len(cache_records) != 288:
            raise ValueError('One complete literal1728example pass required')
    finally:
        if baseline_handle:
            baseline_handle.close()
    return {'optimizer_steps': step, 'ordered_ids_sha256': data_fingerprint(seen),
            'normal_cache_records': cache_records, 'training_log_sha256': streaming_hash(log),
            'disabled_baseline_sha256': streaming_hash(baseline_path) if baseline else None}


def final_fit(runtime, adapter, registry, old_state, examples, queries, destination):
    by_id = {e['id']: e for e in examples}
    rows, cache_records = [], {}
    adapter.reset_scope()
    path = destination / 'final-fit.jsonl'
    with path.open('x', buffering=1) as handle:
        for context, source, guard in normal_groups(runtime, adapter, registry, old_state):
            cache_records[context['id']] = dict(guard.normal_source_record)
            for qid in context['query_ids']:
                _, detail = teacher(runtime, adapter, source, by_id[qid], False)
                result = normal_readout(runtime, adapter, context, source, queries[qid])
                row = {**detail, **result}
                rows.append(row)
                handle.write(json.dumps(row) + '\n')
                guard.check()
    if [r['id'] for r in rows] != [e['id'] for e in examples] or len(cache_records) != 288:
        raise ValueError('Every observed fit input and normal cache required')
    return {'examples': len(rows), 'teacher_nll_mean': sum(r['teacher_nll_per_token'] for r in rows) / len(rows),
            'teacher_all_tokens_correct': sum(r['teacher_all_tokens_correct'] for r in rows),
            'unconstrained_numeric_correct': sum(r['greedy_exact'] for r in rows),
            'file_sha256': streaming_hash(path), 'normal_cache_records': cache_records,
            'scope': dict(adapter.scope)}


def execute(args):
    p, packet, old, cases, registry, queries, examples, references = registered_inputs(args.packet)
    seal = verify_seal(packet)
    correction = verify_correction_seal()
    if args.output.exists() or args.smoke == bool(args.smoke_audit):
        raise ValueError('Use a new smoke, or a new complete run with its audited smoke')
    gate = None
    if args.smoke_audit:
        from scripts.analyze_readout_adapter_cache_training import audit_run
        gate = json.loads(args.smoke_audit.read_text())
        if gate != audit_run(Path(gate['source_run']), True) or gate['status'] != 'audited_readout_adapter_smoke':
            raise ValueError('Real safety must be independently audited before full training')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    m = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(), 'smoke': args.smoke,
         'packet': str(Path(p['registration_destination'])),
         'packet_manifest_sha256': streaming_hash(Path(p['registration_destination']) / 'manifest.json'),
         'protocol_sha256': packet['protocol_sha256'], 'source_sha256': packet['source_sha256'],
         'registration_seal': seal, 'cache_commit_correction': correction,
         'smoke_gate': None if gate is None else
         {'path': str(args.smoke_audit), 'sha256': streaming_hash(args.smoke_audit)}, 'arms': {}}
    for name in correction['source_sha256']:
        target = args.output / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write_json(args.output / 'plan.json', p)
    write_json(args.output / 'cache-commit-correction.json', correction)
    path = args.output / 'manifest.json'
    write_json(path, m)
    runtime = fixed = None
    try:
        runtime = base.TextRuntime(p['model_path'], 'it')
        hooks = base.hook_state(runtime.model)
        m['weights_before'] = base.weights_fingerprint(runtime.model)
        m['packages'] = {n: importlib.metadata.version(n) for n in p['packages']}
        m['gpu'] = torch.cuda.get_device_name(0)
        m['python'] = sys.executable
        m['python_version'] = sys.version
        if (m['weights_before']['sha256'] != p['native_loaded_weights_sha256'] or
            m['packages'] != p['packages'] or m['python'] != old['python'] or streaming_hash(Path(p['installed_modeling_source'])) != p['installed_modeling_source_sha256']):
            raise ValueError('Native loaded model or packages changed')
        m['native_legacy_before'] = legacy_controls(runtime, p, 'native')
        fixed = FixedGDN(runtime, p)
        m['fixed_legacy_before'] = legacy_controls(runtime, p, 'gdn_all_tokens')
        old_state = old['models']['gdn_all_tokens']
        expected = {r['id']: r for r in references}
        fixed_hooks = base.hook_state(runtime.model)
        for arm in ARMS:
            torch.manual_seed(1701)
            adapter = ReadoutAdapter(runtime.model, arm, p['arms'][arm]['rank'], p['attention_layer'])
            destination = args.output / arm
            destination.mkdir()
            record = initialization(adapter)
            m['arms'][arm] = record
            try:
                if record['description']['parameters'] != 131072:
                    raise ValueError('Exactly matched new parameter budget required')
                if args.smoke:
                    record['safety'] = scratch(runtime, fixed, adapter, p, registry, old_state, examples,
                                                queries, expected, destination)
                else:
                    if record['initial_state_sha256'] != gate['arms'][arm]['initial_state_sha256']:
                        raise ValueError('Discard scratch and reset the registered initialization')
                    record['training'] = scheduled(runtime, fixed, adapter, p, registry, old_state,
                                                    examples, queries, destination, arm == 'query')
                    final_sha = state_fingerprint(adapter)
                    record['final_fit'] = final_fit(runtime, adapter, registry, old_state, examples, queries, destination)
                    record['final_state_sha256'] = state_fingerprint(adapter)
                    if record['final_state_sha256'] != final_sha:
                        raise ValueError('Final fit changed trained weights')
                record['fixed_parameters'] = fixed.verify()
            finally:
                adapter.close()
            if base.hook_state(runtime.model) != fixed_hooks:
                raise ValueError('A readout adapter leaked hooks')
            record['hooks_restored'] = True
            write_json(path, m)
        m['fixed_legacy_after'] = legacy_controls(runtime, p, 'gdn_all_tokens')
        m['fixed_parameters_after'] = fixed.verify()
        fixed.close()
        fixed = None
        m['native_legacy_after'] = legacy_controls(runtime, p, 'native')
        m['weights_after'] = base.weights_fingerprint(runtime.model)
        m['hooks_restored'] = base.hook_state(runtime.model) == hooks
        if m['weights_before'] != m['weights_after'] or not m['hooks_restored']:
            raise ValueError('Native weights or hooks changed')
        m.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        m.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if fixed is not None:
            fixed.close()
        write_json(path, m)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-audit', type=Path)
    execute(parser.parse_args())


if __name__ == '__main__':
    main()
