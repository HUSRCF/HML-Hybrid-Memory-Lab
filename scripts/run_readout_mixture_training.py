"""Paired frozen-backbone Gate controls with complete observed-input exposure logs."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.readout_mixture_data import registered_inputs, verify_seal, REGIMES
from scripts.readout_mixture_runtime import (teacher, copy_history, context_stream,
                                           normal_reference, check_reference)
from scripts.readout_adapter_data import load_sources
from scripts.readout_adapter_modules import ReadoutAdapter
from scripts.readout_adapter_runtime import (base, FixedGDN, normal_groups, normal_readout, legacy_controls)
from scripts.readout_adapter_cache_commit import cache_commit_equivalence
from scripts.run_readout_adapter_cache_training import (gradient_norms, freeze_check, initialization,
                                                        optimizer_for)
from scripts.run_controller_trial import write_json, state_fingerprint
from sas_memory.calibration import save_checkpoint
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash


def scratch(runtime, fixed, adapter, plan, data, old, references, destination):
    stream = normal_groups(runtime, adapter, data['set_registry'], old)
    sc, source, sg = next(stream)
    cc = data['contexts'][data['copy_chunks'][0]['context_id']]
    copy_source, cg = copy_history(runtime, adapter, cc)
    check_reference(sc, sg, references)
    check_reference(cc, cg, references)
    set_es = [data['examples'][qid] for qid in data['set_chunks'][0]['query_ids']]
    copy_es = [data['examples'][qid] for qid in data['copy_chunks'][0]['query_ids']]
    zero = []
    for c, cache, guard, values in ((sc, source, sg, set_es), (cc, copy_source, cg, copy_es)):
        for e in values:
            adapter.set_policy(False, e['ledger_tokens'])
            expected = normal_readout(runtime, None, c, cache, data['queries'][e['id']])
            actual = normal_readout(runtime, adapter, c, cache, data['queries'][e['id']])
            if actual != expected:
                raise ValueError('Zero Gate changed an actual normal SET/COPY answer')
            guard.check()
            zero.append({'id': e['id'], 'readout': actual, 'disabled_readout': expected})
    varied = [(source, set_es[0]), (copy_source, copy_es[0]),
              (copy_source, next(e for e in copy_es if len(e['teacher_ids']) == 30))]
    if [len(e['teacher_ids']) for _, e in varied] != [27, 29, 30]:
        raise ValueError('All three native teacher lengths must be tested')
    before = [cache_commit_equivalence(runtime, adapter, s, e) for s, e in varied]
    optimizer = optimizer_for(adapter, plan)
    optimizer.zero_grad(set_to_none=True)
    train_es = [(source, sg, e) for e in set_es[:2]] + [(copy_source, cg, e) for e in copy_es[:2]]
    details = []
    for cache, guard, e in train_es:
        loss, detail = teacher(runtime, adapter, cache, e, True)
        (loss / 4).backward()
        del loss
        guard.check()
        freeze_check(runtime, fixed)
        details.append(detail)
    first = gradient_norms(adapter)
    if not first['branches.gate.b.weight'] or not all(torch.isfinite(p.grad).all() for p in adapter.parameters()):
        raise ValueError('No finite Gate gradient through actual native kernels')
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    log = destination / 'training.jsonl'
    log.write_text(json.dumps({'step': 1, 'examples': details, 'gradient_norm': float(norm.item())}) + '\n')
    checkpoint = save_checkpoint(destination, 1, adapter, optimizer, log, {'scratch_only': True})
    loss, detail = teacher(runtime, adapter, copy_source, varied[-1][1], True)
    loss.backward()
    del loss
    second = gradient_norms(adapter)
    if not second['branches.gate.a.weight'] or not all(torch.isfinite(p.grad).all() for p in adapter.parameters()):
        raise ValueError('Encoder gradient did not arrive on the30token COPY backward')
    optimizer.zero_grad(set_to_none=True)
    after = [cache_commit_equivalence(runtime, adapter, s, e) for s, e in varied]
    rebuilt = []
    for c in (sc, cc):
        adapter.set_policy(True, c['ledger_tokens'])
        with torch.inference_mode():
            cache, logits, count = runtime.advance(c['ledger'])
        if count != c['ledger_tokens']:
            raise ValueError('Rebuilt normal history has a different boundary')
        rebuilt.append(check_reference(c, base.CacheGuard(cache, logits), references))
    sg.check()
    cg.check()
    stream.close()
    return {'status': 'passed_real_paired_mixture_safety', 'zero_readouts': zero,
            'cache_commit_equivalence': {'before_update': before, 'after_update': after},
            'teacher_lengths_checked': [27, 29, 30], 'scratch_optimizer_updates': 1,
            'second_backward_teacher_tokens': detail['cached_teacher_tokens'],
            'first_gradient_norms': first, 'second_gradient_norms': second,
            'checkpoint_sha256': streaming_hash(checkpoint / 'controller.safetensors'),
            'checkpoint_metadata_sha256': streaming_hash(checkpoint / 'checkpoint.json'),
            'scratch_state_sha256': state_fingerprint(adapter),
            'prefix_after_nonzero_update': rebuilt, 'source_caches_unchanged': True,
            'scope': dict(adapter.scope)}


def scheduled(runtime, fixed, adapter, plan, data, old, references, regime, destination):
    optimizer = optimizer_for(adapter, plan)
    optimizer.zero_grad(set_to_none=True)
    seen, batch, matches, guards = [], [], {}, {}
    step = 0
    started = time.monotonic()
    schedule = data['schedules'][regime]
    log = destination / 'training.jsonl'
    with log.open('x', buffering=1) as handle:
        for index, (c, source, guard) in enumerate(normal_groups(runtime, adapter, data['set_registry'], old)):
            left = data['set_chunks'][index]
            if left['context_id'] != c['id']:
                raise ValueError('SET history and paired exposure block differ')
            if regime == 'set_replay':
                right, rc, rs, rg = left, c, source, guard
            else:
                right = data['copy_chunks'][index]
                rc = data['contexts'][right['context_id']]
                rs, rg = copy_history(runtime, adapter, rc)
            for part, (chunk, context, cache, check) in enumerate(((left, c, source, guard), (right, rc, rs, rg))):
                matches[context['id']] = check_reference(context, check, references)
                for qid in chunk['query_ids']:
                    exposure = schedule[len(seen)]
                    if (exposure['query_id'] != qid or exposure['context_id'] != context['id']
                        or exposure['block'] != index or exposure['part'] != part):
                        raise ValueError('A training exposure differs from the frozen schedule')
                    loss, detail = teacher(runtime, adapter, cache, data['examples'][qid], True)
                    (loss / 4).backward()
                    del loss
                    check.check()
                    guards[context['id']] = guards.get(context['id'], 0) + 1
                    freeze_check(runtime, fixed)
                    seen.append(exposure['exposure_id'])
                    batch.append({**detail, 'exposure_id': exposure['exposure_id'], 'context_id': context['id']})
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
                        if step in (144, 288, 432):
                            save_checkpoint(destination, step, adapter, optimizer, log,
                                {'regime': regime, 'description': adapter.description(), 'final_only_selection': True})
                        if step == 1 or step % 16 == 0:
                            print(json.dumps({'event': 'paired_mixture_training', 'regime': regime, 'step': step}), flush=True)
            guard.check()
            rg.check()
            del rs, rg
    if (step != 432 or batch or seen != [r['exposure_id'] for r in schedule]
        or len(matches) != plan['training']['unique_histories'][regime]):
        raise ValueError('Complete1728exposure432step paired schedule required')
    return {'optimizer_steps': step, 'exposures': len(seen), 'ordered_exposures_sha256': data_fingerprint(seen),
            'history_checks': matches, 'source_guard_checks': guards,
            'training_log_sha256': streaming_hash(log), 'elapsed_seconds': time.monotonic() - started}


def final_fit(runtime, fixed, adapter, plan, data, old, references, regime, destination):
    wanted = {r['query_id'] for r in data['schedules'][regime]}
    by_context = {}
    for qid in data['examples']:
        if qid in wanted:
            by_context.setdefault(data['examples'][qid]['context_id'], []).append(qid)
    rows, matches = [], {}
    adapter.reset_scope()
    with (destination / 'final-fit.jsonl').open('x', buffering=1) as handle:
        for c, source, guard in context_stream(runtime, adapter, data, old):
            if c['id'] not in by_context:
                continue
            matches[c['id']] = check_reference(c, guard, references)
            for qid in by_context[c['id']]:
                _, detail = teacher(runtime, adapter, source, data['examples'][qid], False)
                result = normal_readout(runtime, adapter, c, source, data['queries'][qid])
                row = {**detail, **result, 'context_id': c['id']}
                rows.append(row)
                handle.write(json.dumps(row) + '\n')
                guard.check()
                freeze_check(runtime, fixed)
    if (len(rows) != len(wanted) or {r['id'] for r in rows} != wanted
        or len(matches) != plan['training']['unique_histories'][regime]):
        raise ValueError('Every unique physical training input and normal history required')
    return {'physical_questions': len(rows), 'correct': sum(r['greedy_exact'] for r in rows),
            'results_sha256': streaming_hash(destination / 'final-fit.jsonl'),
            'history_checks': matches, 'scope': dict(adapter.scope)}


def execute(args):
    p, packet, data = registered_inputs()
    seal = verify_seal()
    bp, old, *_ = load_sources()
    if args.output.exists() or args.smoke == bool(args.smoke_audit):
        raise ValueError('Use a new smoke or an audited smoke for a new full run')
    gate = None
    if args.smoke_audit:
        from scripts.analyze_readout_mixture_training import audit_run
        gate = json.loads(args.smoke_audit.read_text())
        if gate != audit_run(Path(gate['source_run']), True) or gate['status'] != 'audited_paired_mixture_smoke':
            raise ValueError('Independent complete native smoke audit required')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    m = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(), 'smoke': args.smoke,
         'protocol_sha256': packet['protocol_sha256'], 'source_sha256': seal['source_sha256'],
         'packet_manifest_sha256': streaming_hash(Path(p['dataset']) / 'manifest.json'),
         'smoke_gate': None if gate is None else {'path': str(args.smoke_audit), 'sha256': streaming_hash(args.smoke_audit)},
         'training': not args.smoke, 'oracle': False, 'donor': False, 'future_confirmation_generated': False,
         'regime_order': list(REGIMES), 'arms': {}}
    path = args.output / 'manifest.json'
    for name in seal['source_sha256']:
        target = args.output / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write_json(args.output / 'plan.json', p)
    write_json(path, m)
    fixed = None
    try:
        runtime = base.TextRuntime(p['model_path'], 'it')
        hooks = base.hook_state(runtime.model)
        m['weights_before'] = base.weights_fingerprint(runtime.model)
        m['packages'] = {n: importlib.metadata.version(n) for n in p['packages']}
        m['gpu'] = torch.cuda.get_device_name(0)
        m['python'] = sys.executable
        if (m['weights_before']['sha256'] != p['native_loaded_weights_sha256']
            or m['packages'] != p['packages'] or streaming_hash(Path(p['installed_modeling_source'])) != p['installed_modeling_source_sha256']):
            raise ValueError('Existing native model, kernel source or environment changed')
        m['native_legacy_before'] = legacy_controls(runtime, bp, 'native')
        fixed = FixedGDN(runtime, bp)
        m['fixed_legacy_before'] = legacy_controls(runtime, bp, 'gdn_all_tokens')
        references = normal_reference(runtime, data, old['models']['gdn_all_tokens'])
        write_json(args.output / 'normal-history-references.json', references)
        m['normal_reference_sha256'] = streaming_hash(args.output / 'normal-history-references.json')
        fixed_hooks = base.hook_state(runtime.model)
        for regime in REGIMES:
            torch.manual_seed(1701)
            adapter = ReadoutAdapter(runtime.model, 'gate', 32, 15)
            destination = args.output / regime
            destination.mkdir()
            record = initialization(adapter)
            m['arms'][regime] = record
            if record['description']['parameters'] != 131072 or record['initial_state_sha256'] != p['initial_state_sha256']:
                adapter.close()
                raise ValueError('Matched original Gate initialization and parameter count required')
            try:
                if args.smoke:
                    record['safety'] = scratch(runtime, fixed, adapter, p, data,
                                              old['models']['gdn_all_tokens'], references, destination)
                else:
                    if record['initial_state_sha256'] != gate['arms'][regime]['initial_state_sha256']:
                        raise ValueError('Discard scratch and restore initial Gate weights')
                    record['training'] = scheduled(runtime, fixed, adapter, p, data,
                        old['models']['gdn_all_tokens'], references, regime, destination)
                    final_state = state_fingerprint(adapter)
                    record['final_fit'] = final_fit(runtime, fixed, adapter, p, data,
                        old['models']['gdn_all_tokens'], references, regime, destination)
                    record['final_state_sha256'] = state_fingerprint(adapter)
                    if record['final_state_sha256'] != final_state:
                        raise ValueError('Observed readout changed final weights')
                record['fixed_parameters'] = fixed.verify()
            finally:
                adapter.close()
            if base.hook_state(runtime.model) != fixed_hooks:
                raise ValueError('Gate training leaked a runtime hook')
            record['hooks_restored'] = True
            write_json(path, m)
        m['fixed_legacy_after'] = legacy_controls(runtime, bp, 'gdn_all_tokens')
        m['fixed_parameters_after'] = fixed.verify()
        fixed.close()
        fixed = None
        m['native_legacy_after'] = legacy_controls(runtime, bp, 'native')
        m['weights_after'] = base.weights_fingerprint(runtime.model)
        m['hooks_restored'] = base.hook_state(runtime.model) == hooks
        if m['weights_after'] != m['weights_before'] or not m['hooks_restored']:
            raise ValueError('Frozen model weights or hooks changed')
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-audit', type=Path)
    execute(parser.parse_args())


if __name__ == '__main__':
    main()
