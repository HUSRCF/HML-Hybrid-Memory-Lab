"""Complete observed/fresh frozen confirmation with streaming history evidence."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.mixture_confirmation_data import (registered_inputs, source_seal, observed_inputs,
                                               MODELS, OLD_MODELS, NEW_MODELS)
from scripts.mixture_confirmation_runtime import (base, frozen_model, normal_contexts, history_record,
    compare_history, measure_context, observed_fit_replay, archive_exact, write_history, legacy_controls)
from scripts.readout_adapter_data import load_sources
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash
from sas_memory.training_data import data_fingerprint


def execute(args):
    p, packet, data, lock = registered_inputs()
    seal = source_seal()
    bp = json.loads((ROOT / 'configs/readout_adapter_training_plan.json').read_text())
    if args.output.exists() or args.smoke == bool(args.smoke_audit):
        raise ValueError('New observed-fit smoke or independently gated full run required')
    gate = None
    if args.smoke_audit:
        from scripts.analyze_mixture_confirmation import audit_run
        gate = json.loads(args.smoke_audit.read_text())
        if gate != audit_run(Path(gate['source_run']), True) or gate['status'] != 'audited_seven_model_observed_fit_replay':
            raise ValueError('All42observed-fit replays must pass independent audit first')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    m = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(), 'smoke': args.smoke,
         'protocol_sha256': packet['protocol_sha256'], 'source_sha256': seal['source_sha256'],
         'packet_manifest_sha256': streaming_hash(Path(p['dataset']) / 'manifest.json'),
         'training_lock_sha256': streaming_hash(Path(p['dataset']) / 'training-lock.json'),
         'model_order': list(MODELS), 'cohort_order': ['smoke'] if args.smoke else ['observed', 'fresh'],
         'cohorts': {}, 'training': False, 'oracle': False, 'donor': False,
         'normal_query_independent_histories': True,
         'smoke_gate': None if gate is None else {'path': str(args.smoke_audit), 'sha256': streaming_hash(args.smoke_audit)}}
    for name in seal['source_sha256']:
        target = args.output / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write_json(args.output / 'plan.json', p)
    write_json(args.output / 'training-lock.json', json.loads((Path(p['dataset']) / 'training-lock.json').read_text()))
    path = args.output / 'manifest.json'
    write_json(path, m)
    try:
        runtime = base.TextRuntime(bp['model_path'], 'it')
        hooks = base.hook_state(runtime.model)
        m.update(weights_before=base.weights_fingerprint(runtime.model), python=sys.executable,
                 packages={n: importlib.metadata.version(n) for n in bp['packages']},
                 gpu=torch.cuda.get_device_name(0))
        if (m['weights_before']['sha256'] != bp['native_loaded_weights_sha256']
            or not m['weights_before']['all_parameters_frozen'] or not m['weights_before']['no_parameter_gradients']
            or m['packages'] != bp['packages']
            or streaming_hash(Path(bp['installed_modeling_source'])) != bp['installed_modeling_source_sha256']):
            raise ValueError('Existing frozen native runtime changed')
        _, archive = observed_inputs()
        _, old, _, old_registry, old_queries, _ = load_sources()
        with torch.inference_mode():
            m['native_legacy_before'] = legacy_controls(runtime, bp, 'native')
            for cohort in m['cohort_order']:
                destination = args.output / cohort
                destination.mkdir()
                states = {}
                m['cohorts'][cohort] = {'models': states}
                if cohort == 'observed':
                    for name in OLD_MODELS:
                        source = Path(p['observed_run']) / (name + '.jsonl')
                        archive_exact(source, destination / source.name, archive['models'][name]['results_sha256'])
                        states[name] = {'archived': True, 'readouts': 1200,
                            'results_sha256': streaming_hash(destination / source.name),
                            'source_model_sha256': data_fingerprint(archive['models'][name]),
                            'source_run': p['observed_run']}
                    selected = NEW_MODELS
                else:
                    selected = MODELS
                fixed_reference = archive['models']['fixed_gdn']['histories'] if cohort == 'observed' else {}
                for name in selected:
                    state = {'archived': False, 'readouts': 0, 'history_contexts': 0, 'source_guard_checks': {}}
                    states[name] = state
                    output = destination / (name + '.jsonl')
                    history = destination / (name + '.histories.jsonl')
                    with frozen_model(runtime, name, bp, lock) as (_, adapter, descriptor):
                        state['model'] = descriptor
                        with output.open('x', buffering=1) as handle, history.open('x', buffering=1) as history_handle:
                            if cohort == 'smoke':
                                rows, cid, record, checks = observed_fit_replay(runtime, name, adapter,
                                    p, bp, old, old_registry, old_queries)
                                write_history(history_handle, cid, record)
                                for row in rows:
                                    handle.write(json.dumps(row) + '\n')
                                state.update(readouts=6, history_contexts=1, source_guard_checks={cid: checks})
                            else:
                                inputs = data[cohort]
                                seen = []
                                for task, context, source, guard in normal_contexts(runtime, adapter, inputs):
                                    cid = context['id']
                                    record = history_record(guard)
                                    if name not in ('native', 'fixed_gdn'):
                                        compare_history(record, fixed_reference[cid])
                                    elif name == 'fixed_gdn':
                                        fixed_reference[cid] = record
                                    write_history(history_handle, cid, record)
                                    state['history_contexts'] += 1
                                    for qid in context['query_ids']:
                                        q = inputs[task + '_queries'][qid]
                                        value = measure_context(runtime, adapter, context, source, q)
                                        guard.check()
                                        handle.write(json.dumps({'id': qid, 'task': task, 'model': name,
                                            'context_id': cid, 'registration_sha256': data_fingerprint(q),
                                            'normal_history_sha256': record['sha256'], **value}) + '\n')
                                        state['readouts'] += 1
                                        seen.append(qid)
                                    state['source_guard_checks'][cid] = guard.checks
                                    if state['history_contexts'] % 16 == 0:
                                        write_json(path, m)
                                        print(json.dumps({'event': 'mixture_confirmation_progress', 'cohort': cohort,
                                            'model': name, 'readouts': state['readouts'], 'histories': state['history_contexts']}), flush=True)
                                if seen != inputs['query_order'] or state['history_contexts'] != 200:
                                    raise ValueError('All1200questions and200histories required for every executed model')
                    state['results_sha256'] = streaming_hash(output)
                    state['history_records_sha256'] = streaming_hash(history)
                    state['hooks_restored'] = base.hook_state(runtime.model) == hooks
                    if not state['hooks_restored']:
                        raise ValueError('A frozen model leaked inference hooks')
                    state['native_legacy_after'] = legacy_controls(runtime, bp, 'native')
                    write_json(path, m)
                    print(json.dumps({'event': 'completed_mixture_confirmation_model', 'cohort': cohort,
                                      'model': name, 'readouts': state['readouts']}), flush=True)
        m['weights_after'] = base.weights_fingerprint(runtime.model)
        m['hooks_restored'] = base.hook_state(runtime.model) == hooks
        if m['weights_before'] != m['weights_after'] or not m['hooks_restored']:
            raise ValueError('Frozen native weights or hooks changed')
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
