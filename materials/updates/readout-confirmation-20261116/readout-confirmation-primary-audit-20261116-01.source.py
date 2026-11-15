"""Independent complete replay/confirmation coverage, grading and frozen-state audit."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.readout_confirmation_data import registered_inputs, source_seal, MODELS
from scripts.readout_confirmation_runtime import compare_history
from scripts.readout_adapter_data import load_sources
from scripts.run_controller_trial import write_json
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash


def grade(row, answer):
    match = re.fullmatch(r'\s*(-?\d+)\s*', row['generated'])
    correct = bool(match and match[1] == answer)
    if (row['answer'] != answer or row['greedy_exact'] != correct
        or not math.isfinite(row['answer_nll'])):
        raise ValueError('Raw exact-answer grading or finite NLL differs')
    return correct


def complete_rows(rows, order, model):
    if [r['id'] for r in rows] != order or any(r['model'] != model for r in rows):
        raise ValueError('No missing, duplicated, reordered or cross-model physical questions')


def audit_run(run, allow_smoke=False):
    run = Path(run)
    p, packet, inputs, lock = registered_inputs()
    seal = source_seal()
    m = json.loads((run / 'manifest.json').read_text())
    trained = json.loads((Path(p['training_run']) / 'manifest.json').read_text())
    if (m['status'] != 'completed' or m['smoke'] and not allow_smoke
        or m['source_sha256'] != seal['source_sha256'] or len(m['source_sha256']) != 289
        or m['protocol_sha256'] != packet['protocol_sha256']
        or m['packet_manifest_sha256'] != streaming_hash(Path(p['dataset']) / 'manifest.json')
        or m['training_lock_sha256'] != streaming_hash(Path(p['dataset']) / 'training-lock.json')
        or json.loads((run / 'training-lock.json').read_text()) != lock
        or json.loads((run / 'plan.json').read_text()) != p or m['model_order'] != list(MODELS)
        or list(m['models']) != list(MODELS) or m['weights_before'] != m['weights_after']
        or m['weights_before'] != trained['weights_before'] or m['packages'] != trained['packages']
        or m['python'] != trained['python'] or not m['hooks_restored']
        or any(m[k] for k in ('training', 'oracle', 'donor')) or not m['normal_query_independent_histories']
        or not datetime.fromisoformat(packet['registered_utc']) < datetime.fromisoformat(m['started_utc']) < datetime.fromisoformat(m['completed_utc'])):
        raise ValueError('Complete frozen289source confirmation run required')
    for name, digest in m['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / 'sources' / name) != digest:
            raise ValueError('Confirmation source bytes changed')
    train_plan = json.loads((ROOT / p['parent_training_protocol']).read_text())
    old_refs = json.loads((Path(train_plan['source_packet']) / 'references.json').read_text())
    expected_legacy = [{'id': key, 'readout': old_refs['legacy_full']['native'][key]}
                       for key in old_refs['legacy_ids']]
    if m['native_legacy_before'] != expected_legacy:
        raise ValueError('All existing native before-anchors required')
    old = old_registry = old_queries = None
    if m['smoke']:
        _, old, _, old_registry, old_queries, _ = load_sources()
        first = old_registry['contexts'][0]
        order = first['query_ids']
    else:
        order = inputs['query_order']
        gate = m['smoke_gate']
        g = json.loads(Path(gate['path']).read_text())
        if (streaming_hash(Path(gate['path'])) != gate['sha256']
            or g != audit_run(Path(g['source_run']), True)):
            raise ValueError('Prior independent observed-fit replay audit changed')
    all_contexts = {c['id']: c for c in inputs['set_registry']['contexts'] + inputs['copy_contexts']}
    all_queries = {**inputs['set_queries'], **inputs['copy_queries']}
    output = {}
    for name in MODELS:
        state = m['models'][name]
        description = state['model']
        path = run / f'{name}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        complete_rows(rows, order, name)
        if (streaming_hash(path) != state['results_sha256'] or state['readouts'] != len(order)
            or not state['hooks_restored'] or state['native_legacy_after'] != expected_legacy
            or description['name'] != name):
            raise ValueError('Complete raw data and native restore anchors required')
        if name != 'native':
            for key in ('fixed_gdn', 'fixed_gdn_after'):
                if (not description[key]['all_frozen_and_gradient_free']
                    or description[key]['state_sha256'] != train_plan['fixed_gdn_state_sha256']):
                    raise ValueError('Existing fixed-GDN state or gradients changed')
        if name in ('query', 'gate', 'query_gate'):
            scope = description['scope']
            weight = Path(p['training_run']) / name / 'step-000432/controller.safetensors'
            if (description['new_readout_parameters'] != 131072
                or description['checkpoint_sha256'] != lock['training']['files_sha256'][str(weight)]
                or description['state_sha256'] != lock['training']['final_states'][name]
                or description['state_sha256_after'] != description['state_sha256']
                or not scope['prefix_rows_bit_preserved'] or not scope['inactive_halves_bit_preserved']
                or scope['prefix_calls'] < 1 or scope['active_calls'] < 1):
                raise ValueError('Exact final frozen readout and question-only scope required')
        elif description['new_readout_parameters'] != 0:
            raise ValueError('A baseline acquired new readout capacity')
        if m['smoke']:
            model = 'native' if name == 'native' else 'gdn_all_tokens'
            source = Path(train_plan['source_run']) / f'{model}.jsonl'
            if name in ('query', 'gate', 'query_gate'):
                source = Path(p['training_run']) / name / 'final-fit.jsonl'
            refs = {r['id']: r for r in (json.loads(line) for line in source.read_text().splitlines())}
            if set(state['histories']) != {first['id']}:
                raise ValueError('Only registered observed first history belongs to smoke')
            original = old['models'][model]['prefixes'][first['prefix_key']]
            compare_history(state['histories'][first['id']],
                            {k: original[k] for k in state['histories'][first['id']]})
            for row in rows:
                if row['task'] != 'observed_fit' or any(row[k] != refs[row['id']][k]
                    for k in ('answer', 'answer_nll', 'greedy_exact', 'generated', 'query_tokens')):
                    raise ValueError('Final fit model does not reproduce every original readout')
                grade(row, old_queries[row['id']]['answer'])
            output[name] = {'observed_readouts_exact': 6}
        else:
            if set(state['histories']) != set(all_contexts) or set(state['source_guard_checks']) != set(all_contexts):
                raise ValueError('Every200normal question-independent histories required')
            if any(n < 7 for n in state['source_guard_checks'].values()):
                raise ValueError('Every source must be guarded before/after all six questions')
            if name in ('query', 'gate', 'query_gate'):
                for cid, record in state['histories'].items():
                    compare_history(record, m['models']['fixed_gdn']['histories'][cid])
            counts = {'set': 0, 'copy': 0}
            by_id = {}
            for row in rows:
                q = all_queries[row['id']]
                task = 'set' if row['id'] in inputs['set_queries'] else 'copy'
                cid = q['context_id']
                if (row['task'] != task or row['context_id'] != cid
                    or row['registration_sha256'] != data_fingerprint(q)
                    or row['normal_history_sha256'] != state['histories'][cid]['sha256']
                    or row['query_tokens'] != len(q['input_ids']) - q['ledger_tokens']):
                    raise ValueError('Raw readout used a different query or history')
                counts[task] += int(grade(row, q['answer']))
                by_id[row['id']] = row
            bindings = inputs['copy_registry']['bindings']
            output[name] = {'set_correct': counts['set'], 'set_questions': 864,
                'copy_physical_correct': counts['copy'], 'copy_physical_questions': 336,
                'copy_binding_correct': sum(by_id[b['physical_id']]['greedy_exact'] for b in bindings),
                'copy_bindings': 432, 'normal_histories': 200}
    return {'status': 'audited_observed_fit_replay_before_confirmation' if m['smoke'] else
            'completed_and_audited_frozen_readout_confirmation', 'source_run': str(run),
            'manifest_sha256': streaming_hash(run / 'manifest.json'),
            'packet_manifest_sha256': m['packet_manifest_sha256'], 'source_files_verified': 289,
            'physical_readouts_audited': 30 if m['smoke'] else 6000,
            'models': output, 'prior_failed_criteria_unchanged': p['old_failed_criteria'],
            'analysis_source_sha256': streaming_hash(Path(__file__)), 'completed_utc': m['completed_utc']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite independent confirmation audit')
    result = audit_run(args.run, args.allow_smoke)
    write_json(args.output, result)
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
