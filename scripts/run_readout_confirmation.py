"""Frozen five-model confirmation, gated by observed-fit replay and full training audit."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.readout_confirmation_data import registered_inputs, source_seal, MODELS
from scripts.readout_confirmation_runtime import (base, frozen_model, normal_contexts,
    history_record, compare_history, observed_fit_smoke, measure_context, legacy_controls)
from scripts.readout_adapter_data import load_sources
from scripts.run_controller_trial import write_json
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash


def execute(args):
    p, packet, inputs, lock = registered_inputs()
    seal = source_seal()
    train_plan = json.loads((ROOT / p['parent_training_protocol']).read_text())
    if args.output.exists() or args.smoke == bool(args.smoke_audit):
        raise ValueError('New observed smoke or independently gated primary required')
    gate = None
    if args.smoke_audit:
        from scripts.analyze_readout_confirmation import audit_run
        gate = json.loads(args.smoke_audit.read_text())
        if (gate != audit_run(Path(gate['source_run']), True)
            or gate['status'] != 'audited_observed_fit_replay_before_confirmation'):
            raise ValueError('All final models must reproduce observed fit before confirmation')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    m = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(), 'smoke': args.smoke,
         'protocol_sha256': packet['protocol_sha256'], 'dataset': p['dataset'],
         'packet_manifest_sha256': streaming_hash(Path(p['dataset']) / 'manifest.json'),
         'source_sha256': seal['source_sha256'], 'training_lock_sha256': streaming_hash(Path(p['dataset']) / 'training-lock.json'),
         'model_order': list(MODELS), 'models': {}, 'smoke_gate': None if gate is None else
             {'path': str(args.smoke_audit), 'sha256': streaming_hash(args.smoke_audit)},
         'training': False, 'oracle': False, 'donor': False, 'normal_query_independent_histories': True}
    for name, digest in m['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest:
            raise ValueError('Sealed source changed before inference')
        target = args.output / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write_json(args.output / 'plan.json', p)
    write_json(args.output / 'training-lock.json', lock)
    path = args.output / 'manifest.json'
    write_json(path, m)
    try:
        runtime = base.TextRuntime(train_plan['model_path'], 'it')
        hooks = base.hook_state(runtime.model)
        m.update(weights_before=base.weights_fingerprint(runtime.model), python=sys.executable,
                 packages={n: importlib.metadata.version(n) for n in train_plan['packages']},
                 gpu=torch.cuda.get_device_name(0))
        if (m['weights_before']['sha256'] != train_plan['native_loaded_weights_sha256']
            or not m['weights_before']['all_parameters_frozen'] or not m['weights_before']['no_parameter_gradients']
            or m['packages'] != train_plan['packages']
            or streaming_hash(Path(train_plan['installed_modeling_source'])) != train_plan['installed_modeling_source_sha256']):
            raise ValueError('Frozen native checkpoint or installed runtime changed')
        old = old_registry = old_queries = None
        if args.smoke:
            _, old, _, old_registry, old_queries, _ = load_sources()
        with torch.inference_mode():
            m['native_legacy_before'] = legacy_controls(runtime, train_plan, 'native')
            for name in MODELS:
                state = {'histories': {}, 'source_guard_checks': {}, 'readouts': 0}
                m['models'][name] = state
                output = args.output / f'{name}.jsonl'
                with frozen_model(runtime, name, train_plan, lock) as (_, adapter, description):
                    state['model'] = description
                    with output.open('x', buffering=1) as handle:
                        if args.smoke:
                            rows, records, checks = observed_fit_smoke(runtime, name, adapter, train_plan,
                                p['training_run'], old, old_registry, old_queries)
                            for row in rows:
                                handle.write(json.dumps(row) + '\n')
                            state.update(histories=records, source_guard_checks=checks, readouts=6)
                        else:
                            seen = []
                            for task, context, source, guard in normal_contexts(runtime, adapter, inputs):
                                record = history_record(guard)
                                if name in ('query', 'gate', 'query_gate'):
                                    compare_history(record, m['models']['fixed_gdn']['histories'][context['id']])
                                state['histories'][context['id']] = record
                                for key in context['query_ids']:
                                    query = inputs[task + '_queries'][key]
                                    value = measure_context(runtime, adapter, context, source, query)
                                    guard.check()
                                    handle.write(json.dumps({'id': key, 'task': task, 'model': name,
                                        'context_id': context['id'], 'registration_sha256': data_fingerprint(query),
                                        'normal_history_sha256': record['sha256'], **value}) + '\n')
                                    seen.append(key)
                                    state['readouts'] += 1
                                state['source_guard_checks'][context['id']] = guard.checks
                                if len(state['histories']) % 16 == 0:
                                    write_json(path, m)
                                    print(json.dumps({'event': 'frozen_confirmation_progress',
                                        'model': name, 'readouts': state['readouts'],
                                        'normal_histories': len(state['histories'])}), flush=True)
                            if seen != inputs['query_order'] or len(state['histories']) != 200:
                                raise ValueError('All1200physical questions and200histories required')
                state['results_sha256'] = streaming_hash(output)
                state['hooks_restored'] = base.hook_state(runtime.model) == hooks
                if not state['hooks_restored']:
                    raise ValueError('Final model leaked inference hooks')
                state['native_legacy_after'] = legacy_controls(runtime, train_plan, 'native')
                write_json(path, m)
                print(json.dumps({'event': 'completed_frozen_readout_model', 'model': name,
                                   'smoke': args.smoke, 'readouts': state['readouts']}), flush=True)
        m['weights_after'] = base.weights_fingerprint(runtime.model)
        m['hooks_restored'] = base.hook_state(runtime.model) == hooks
        if m['weights_after'] != m['weights_before'] or not m['hooks_restored']:
            raise ValueError('Native model changed during frozen confirmation')
        m.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        m.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(path, m)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-audit', type=Path)
    execute(parser.parse_args())


if __name__ == '__main__':
    main()
