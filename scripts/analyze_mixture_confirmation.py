"""Audit every archived/executed answer and the preregistered fresh-only functional checks."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.mixture_confirmation_data import (registered_inputs, source_seal, observed_inputs, model_spec,
                                               MODELS, OLD_MODELS, NEW_MODELS, REGIMES)
from scripts.mixture_confirmation_runtime import read_histories, compare_history
from scripts.analyze_readout_confirmation import grade, complete_rows
from scripts.readout_adapter_data import load_sources
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash
from sas_memory.training_data import data_fingerprint


def verify_history_coverage(path, context_ids, checks):
    records = read_histories(path)
    if set(records) != set(context_ids) or set(checks) != set(context_ids) or any(v < 7 for v in checks.values()):
        raise ValueError('Complete normal history coverage and source guards required')
    return records


def functional_checks(summary, protected, joint_sets):
    mix, replay, old_gate, fixed = (summary[k] for k in ('gate_set_copy_mix', 'gate_set_replay', 'gate', 'fixed_gdn'))
    wins = sum(mix['copy_joint_by_seed'][s] > replay['copy_joint_by_seed'][s] for s in mix['copy_joint_by_seed'])
    by_depth = all(mix['copy_joint_by_depth'][d] >= replay['copy_joint_by_depth'][d] for d in ('1', '2', '3'))
    result = {
        'copy_joint_gain': mix['copy_joint6_correct'] > replay['copy_joint6_correct'] and by_depth and wins >= 3,
        'set_retention': mix['set_correct'] >= replay['set_correct'] and mix['set_correct'] >= old_gate['set_correct']
                         and protected['fixed_gdn'].issubset(protected['gate_set_copy_mix']),
        'strong_copy_reference': mix['copy_joint6_correct'] > old_gate['copy_joint6_correct']
                         and mix['copy_joint6_correct'] >= fixed['copy_joint6_correct']
                         and joint_sets['fixed_gdn'].issubset(joint_sets['gate_set_copy_mix'])}
    return {'cohort': 'fresh_only', 'passed': result, 'all_passed': all(result.values()),
            'copy_base_stream_net_improvements': wins,
            'protected_regressions_vs_fixed': sorted(protected['fixed_gdn'] - protected['gate_set_copy_mix']),
            'copy_joint_regressions_vs_fixed': sorted(joint_sets['fixed_gdn'] - joint_sets['gate_set_copy_mix'])}


def cohort_metrics(inputs, all_rows):
    summary, protected, joint_sets = {}, {}, {}
    for name in MODELS:
        rows = all_rows[name]
        by_id = {r['id']: r for r in rows}
        sq, cq = inputs['set_queries'], inputs['copy_queries']
        if len(by_id) != 1200 or set(by_id) != set(sq) | set(cq):
            raise ValueError('Every physical question required for cohort metrics')
        good = {qid for qid, row in by_id.items() if row['greedy_exact']}
        protected[name] = {qid for qid, q in sq.items() if q['role'] == 'protected_current' and qid in good}
        bindings = inputs['copy_registry']['bindings']
        cases = {c['id']: c for c in inputs['copy_cases']}
        grouped = {}
        for b in bindings:
            if b['variant'] != 'before':
                grouped.setdefault(b['case_id'], []).append(b)
        joint = set()
        by_depth = {str(d): 0 for d in (1, 2, 3)}
        by_seed = {str(c['seed']): 0 for c in cases.values()}
        for cid, bs in grouped.items():
            if len(bs) != 6:
                raise ValueError('Both event orders and all three COPY roles required')
            if all(b['physical_id'] in good for b in bs):
                joint.add(cid)
                by_depth[str(cases[cid]['dependency_depth'])] += 1
                by_seed[str(cases[cid]['seed'])] += 1
        if len(grouped) != 48 or len(by_seed) != 8:
            raise ValueError('All48COPYunits andeight independent base streams required')
        joint_sets[name] = joint
        summary[name] = {'set_correct': len(good & set(sq)), 'set_questions': 864,
                         'set_role_correct': {role: sum(qid in good for qid, q in sq.items() if q['role'] == role)
                                              for role in ('target_current', 'target_initial', 'protected_current')},
                         'copy_physical_correct': len(good & set(cq)), 'copy_physical_questions': 336,
                         'copy_binding_correct': sum(b['physical_id'] in good for b in bindings), 'copy_bindings': 432,
                         'copy_joint6_correct': len(joint), 'copy_joint6_units': 48,
                         'copy_joint_by_depth': by_depth, 'copy_joint_by_seed': by_seed}
    return summary, protected, joint_sets


def audit_run(run, allow_smoke=False):
    run = Path(run)
    p, packet, data, lock = registered_inputs()
    seal = source_seal()
    _, archive = observed_inputs()
    m = json.loads((run / 'manifest.json').read_text())
    trained = json.loads((Path(p['mixture_training_run']) / 'manifest.json').read_text())
    if (m['status'] != 'completed' or m['smoke'] and not allow_smoke
        or m['source_sha256'] != seal['source_sha256'] or len(m['source_sha256']) != 303
        or m['protocol_sha256'] != packet['protocol_sha256']
        or m['packet_manifest_sha256'] != streaming_hash(Path(p['dataset']) / 'manifest.json')
        or m['training_lock_sha256'] != streaming_hash(Path(p['dataset']) / 'training-lock.json')
        or json.loads((run / 'training-lock.json').read_text()) != json.loads((Path(p['dataset']) / 'training-lock.json').read_text())
        or json.loads((run / 'plan.json').read_text()) != p
        or m['model_order'] != list(MODELS) or m['weights_before'] != m['weights_after']
        or m['weights_before'] != trained['weights_before'] or m['packages'] != trained['packages']
        or m['python'] != trained['python'] or not m['hooks_restored']
        or any(m[k] for k in ('training', 'oracle', 'donor')) or not m['normal_query_independent_histories']
        or not datetime.fromisoformat(packet['registered_utc']) < datetime.fromisoformat(m['started_utc']) < datetime.fromisoformat(m['completed_utc'])):
        raise ValueError('Complete303source frozen observed/fresh run required')
    for name, digest in m['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / 'sources' / name) != digest:
            raise ValueError('Measured confirmation source changed')
    bp = json.loads((ROOT / 'configs/readout_adapter_training_plan.json').read_text())
    legacy = json.loads((Path(bp['source_packet']) / 'references.json').read_text())
    expected_legacy = [{'id': qid, 'readout': legacy['legacy_full']['native'][qid]} for qid in legacy['legacy_ids']]
    if m['native_legacy_before'] != expected_legacy:
        raise ValueError('All exact native legacy before-anchors required')
    _, old, _, old_registry, old_queries, _ = load_sources()
    expected_cohorts = ['smoke'] if m['smoke'] else ['observed', 'fresh']
    if m['cohort_order'] != expected_cohorts or list(m['cohorts']) != expected_cohorts:
        raise ValueError('Separate complete observed and fresh cohorts required')
    outputs = {}
    fresh_checks = None
    for cohort in expected_cohorts:
        states = m['cohorts'][cohort]['models']
        if list(states) != list(MODELS):
            raise ValueError('All seven models required in every cohort')
        folder = run / cohort
        if m['smoke']:
            first = old_registry['contexts'][0]
            order, inputs = first['query_ids'], None
        else:
            inputs = data[cohort]
            order = inputs['query_order']
        all_rows = {}
        fixed_histories = archive['models']['fixed_gdn']['histories'] if cohort == 'observed' else None
        for name in MODELS:
            state = states[name]
            raw = folder / (name + '.jsonl')
            rows = [json.loads(line) for line in raw.read_text().splitlines()]
            complete_rows(rows, order, name)
            all_rows[name] = rows
            if streaming_hash(raw) != state['results_sha256'] or state['readouts'] != len(order):
                raise ValueError('Complete raw answer bytes and counts required')
            if state['archived']:
                if (cohort != 'observed' or name not in OLD_MODELS
                    or raw.read_bytes() != (Path(p['observed_run']) / raw.name).read_bytes()
                    or state['source_model_sha256'] != data_fingerprint(archive['models'][name])
                    or state['source_run'] != p['observed_run']):
                    raise ValueError('Every old baseline must be preserved exactly as archived')
                histories = archive['models'][name]['histories']
            else:
                description = state['model']
                spec = model_spec(name, lock)
                if description['name'] != name or not state['hooks_restored'] or state['native_legacy_after'] != expected_legacy:
                    raise ValueError('Every executed model must restore native hooks and anchors')
                if name != 'native':
                    for key in ('fixed_gdn', 'fixed_gdn_after'):
                        if (description[key]['state_sha256'] != bp['fixed_gdn_state_sha256']
                            or not description[key]['all_frozen_and_gradient_free']):
                            raise ValueError('Existing fixed-GDN weights or gradients changed')
                if description['new_readout_parameters'] != spec['parameters']:
                    raise ValueError('Declared fixed readout capacity changed')
                if spec['checkpoint']:
                    scope = description['scope']
                    if (description['checkpoint_sha256'] != spec['checkpoint_sha256']
                        or description['state_sha256'] != spec['state_sha256']
                        or description['state_sha256_after'] != spec['state_sha256']
                        or description['description']['mode'] != spec['mode']
                        or description['description']['rank'] != spec['rank']
                        or not scope['prefix_rows_bit_preserved'] or not scope['inactive_halves_bit_preserved']
                        or scope['prefix_calls'] < 1 or scope['active_calls'] < 1):
                        raise ValueError('Locked final432readout, scope and arithmetic required')
                hp = folder / (name + '.histories.jsonl')
                cids = {first['id']} if m['smoke'] else {c['id'] for c in inputs['set_registry']['contexts'] + inputs['copy_contexts']}
                if streaming_hash(hp) != state['history_records_sha256'] or state['history_contexts'] != len(cids):
                    raise ValueError('Every streamed history record must be retained')
                histories = verify_history_coverage(hp, cids, state['source_guard_checks'])
            if m['smoke']:
                key = 'native' if name == 'native' else 'gdn_all_tokens'
                source = Path(bp['source_run']) / (key + '.jsonl')
                if name in ('query', 'gate', 'query_gate'):
                    source = Path(p['original_training_run']) / name / 'final-fit.jsonl'
                elif name in NEW_MODELS:
                    source = Path(p['mixture_training_run']) / REGIMES[name] / 'final-fit.jsonl'
                refs = {r['id']: r for r in (json.loads(line) for line in source.read_text().splitlines())}
                expected = old['models'][key]['prefixes'][first['prefix_key']]
                compare_history(histories[first['id']], {k: expected[k] for k in histories[first['id']]})
                for row in rows:
                    if row['task'] != 'observed_fit' or any(row[k] != refs[row['id']][k]
                        for k in ('answer', 'answer_nll', 'greedy_exact', 'generated', 'query_tokens')):
                        raise ValueError('All final frozen models must exactly replay their fit records')
                    grade(row, old_queries[row['id']]['answer'])
            else:
                if name == 'fixed_gdn' and cohort == 'fresh':
                    fixed_histories = histories
                if name not in ('native', 'fixed_gdn'):
                    for cid, record in histories.items():
                        compare_history(record, fixed_histories[cid])
                queries = {**inputs['set_queries'], **inputs['copy_queries']}
                for row in rows:
                    q = queries[row['id']]
                    task = 'set' if row['id'] in inputs['set_queries'] else 'copy'
                    if (row['task'] != task or row['context_id'] != q['context_id']
                        or row['registration_sha256'] != data_fingerprint(q)
                        or row['normal_history_sha256'] != histories[q['context_id']]['sha256']
                        or row['query_tokens'] != len(q['input_ids']) - q['ledger_tokens']):
                        raise ValueError('An answer used a different literal question or normal history')
                    grade(row, q['answer'])
        if m['smoke']:
            outputs[cohort] = {name: {'observed_fit_replays_exact': 6} for name in MODELS}
        else:
            summary, protected, joint = cohort_metrics(inputs, all_rows)
            outputs[cohort] = summary
            if cohort == 'fresh':
                fresh_checks = functional_checks(summary, protected, joint)
    if not m['smoke']:
        gate = m['smoke_gate']
        g = json.loads(Path(gate['path']).read_text())
        if streaming_hash(Path(gate['path'])) != gate['sha256'] or g != audit_run(Path(g['source_run']), True):
            raise ValueError('Prior independent42replay smoke audit required')
    return {'status': 'audited_seven_model_observed_fit_replay' if m['smoke'] else
                     'completed_and_audited_observed_and_fresh_mixture_confirmation',
            'source_run': str(run), 'manifest_sha256': streaming_hash(run / 'manifest.json'),
            'source_files_verified': 303, 'new_physical_readouts_audited': 42 if m['smoke'] else 10800,
            'all_physical_rows_including_archived_audited': 42 if m['smoke'] else 16800,
            'cohorts': outputs, 'fresh_functional_checks': fresh_checks,
            'prior_failed_criteria_unchanged': p['prior_failed_criteria_unchanged'],
            'architecture_superiority_claim': False, 'analysis_source_sha256': streaming_hash(Path(__file__)),
            'completed_utc': m['completed_utc']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite independent confirmation audit')
    result = audit_run(args.run, args.allow_smoke)
    write_json(args.output, result)
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
