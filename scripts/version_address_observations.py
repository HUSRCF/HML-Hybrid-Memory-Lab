"""Post-audit descriptive grading; retain every readout, stream and layer."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.package_version_address import validate_audit


def observe(audit_path):
    audit = json.loads(audit_path.read_text())
    validate_audit(audit)
    run = Path(audit['source_run'])
    if streaming_hash(run / 'manifest.json') != audit['manifest_sha256']:
        raise ValueError('Audited manifest changed')
    manifest = json.loads((run / 'manifest.json').read_text())
    packet = Path(manifest['packet'])
    pm = json.loads((packet / 'manifest.json').read_text())
    if streaming_hash(packet / 'manifest.json') != audit['packet_manifest_sha256']:
        raise ValueError('Registered packet changed')
    for name in ('queries.json', 'cases.json'):
        if streaming_hash(packet / name) != pm['payload_sha256'][name]:
            raise ValueError('Registered input bytes changed')
    queries = json.loads((packet / 'queries.json').read_text())
    cases = {c['id']: c for c in json.loads((packet / 'cases.json').read_text())}
    classified = []
    totals, roles, streams, joints = (defaultdict(list) for _ in range(4))
    evidence = defaultdict(list)
    by_id = {}
    for model in ('native', 'gdn_all_tokens'):
        path = run / (model + '.jsonl')
        if streaming_hash(path) != manifest['models'][model]['results_sha256']:
            raise ValueError('Complete raw answer bytes changed')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 1728 or [r['id'] for r in rows] != list(queries):
            raise ValueError('Every registered literal question is required')
        for row in rows:
            q = queries[row['id']]
            case = cases[q['case_id']]
            events = case['prefix_events'] + next(e['tail'] for e in case['episodes'] if e['id'] == q['episode'])
            if any(e['kind'] != 'SET' for e in events):
                raise ValueError('This independent grader supports the registered SET task only')
            entity = case['protected'] if q['role'] == 'protected_current' else case['target']
            own = [(i, e) for i, e in enumerate(events) if e['target'] == entity]
            support, fact = own[0 if q['role'] == 'target_initial' else -1]
            if str(fact['value']) != q['answer'] or row['answer'] != q['answer']:
                raise ValueError('Independent SET truth differs')
            parsed = re.fullmatch(r'\s*(-?\d+)\s*', row['generated'])
            correct = bool(parsed and parsed[1] == q['answer'])
            if correct != row['greedy_exact']:
                raise ValueError('Independent exact grading differs')
            same = [i for i, e in own if parsed and e['value'] == int(parsed[1])]
            other = [i for i, e in enumerate(events) if parsed and e['target'] != entity and e['value'] == int(parsed[1])]
            category = ('correct' if correct else 'non_integer' if not parsed else
                        'noncanonical_correct_integer' if int(parsed[1]) == int(q['answer']) else
                        'same_account_other_version_value' if same else
                        'other_account_recorded_value' if other else 'unrecorded_integer')
            if any(o['output_category'] != category or o['support_record_index'] != support for o in row['address_rows']):
                raise ValueError('Independent output classification differs')
            identity = {'model': model, 'cohort': row['cohort'], 'id': row['id'], 'seed': q['seed'],
                        'naming': q['naming'], 'style': q['question_style'], 'role': q['role'], 'episode': q['episode']}
            classified.append({**identity, 'correct': correct, 'answer': q['answer'], 'generated': row['generated'],
                               'category': category, 'support_record_index': support,
                               'matching_own_version_indices': same, 'matching_other_account_indices': other})
            key = (model, row['cohort'], q['question_style'])
            totals[key].append(correct)
            roles[(*key, q['role'])].append(correct)
            streams[(*key, q['seed'])].append(correct)
            joints[(*key, q['context_id'])].append(correct)
            by_id[(model, row['id'])] = row
            for obs in row['address_rows']:
                evidence[(*key, q['role'], obs['layer'], obs['suffix_position'], correct)].append(obs)
    def grade_table(table, fields):
        return [dict(zip(fields, key), correct=sum(v), questions=len(v)) for key, v in sorted(table.items())]
    descriptive = []
    for model in ('native', 'gdn_all_tokens'):
        for cohort in ('reused', 'fresh'):
            for style in ('state_term', 'event_term'):
                for role in ('target_current', 'target_initial', 'protected_current'):
                    for layer in (3, 7, 11, 15, 19, 23):
                        for position in (11, 23):
                            for correct in (False, True):
                                values = evidence[(model, cohort, style, role, layer, position, correct)]
                                matched = [v for v in values if v['emitted_record_mass'] is not None]
                                counts = Counter(v['output_category'] for v in values)
                                descriptive.append({'model': model, 'cohort': cohort, 'style': style, 'role': role,
                                    'layer': layer, 'suffix_position': position, 'correct': correct, 'questions': len(values),
                                    'category_counts': dict(counts), 'emitted_value_matches_record': len(matched),
                                    'emitted_mass_greater_than_support': sum(v['emitted_record_mass'] > v['support_mass'] for v in matched),
                                    'emitted_mass_equal_to_support': sum(v['emitted_record_mass'] == v['support_mass'] for v in matched),
                                    'emitted_mass_less_than_support': sum(v['emitted_record_mass'] < v['support_mass'] for v in matched),
                                    'support_global_top_rank': sum(v['support_rank_competition'] == 1 for v in values)})
    wording = []
    for model in ('native', 'gdn_all_tokens'):
        for qid, q in queries.items():
            if q['question_style'] != 'state_term':
                continue
            a, b = by_id[(model, qid)], by_id[(model, q['paired_query_id'])]
            wording.append({'model': model, 'cohort': a['cohort'], 'state_id': qid,
                            'event_id': b['id'], 'seed': q['seed'], 'naming': q['naming'], 'role': q['role'],
                            'both_wrong': not a['greedy_exact'] and not b['greedy_exact'],
                            'same_literal_wrong_output': not a['greedy_exact'] and not b['greedy_exact'] and a['generated'] == b['generated'],
                            'gain': int(not a['greedy_exact'] and b['greedy_exact']),
                            'loss': int(a['greedy_exact'] and not b['greedy_exact'])})
    if len(classified) != 3456 or len(descriptive) != 576 or len(wording) != 1728:
        raise ValueError('All readouts and descriptive strata must remain present')
    if any(len(v) != 3 for v in joints.values()):
        raise ValueError('All three roles must share each normal cache')
    joint_counts = defaultdict(list)
    joint_six = defaultdict(list)
    for (model, cohort, style, context), values in joints.items():
        joint_counts[(model, cohort, style)].append(all(values))
        joint_six[(model, cohort, context)].extend(values)
    if any(len(v) != 6 for v in joint_six.values()):
        raise ValueError('All six same-cache answers required')
    six_counts = defaultdict(list)
    for (model, cohort, _), values in joint_six.items():
        six_counts[(model, cohort)].append(all(values))
    categories = [{'model': model, 'cohort': cohort, **dict(Counter(
        r['category'] for r in classified if r['model'] == model and r['cohort'] == cohort))}
        for model in ('native', 'gdn_all_tokens') for cohort in ('reused', 'fresh')]
    return {'status': 'all3456normal_answers_independently_regraded',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': streaming_hash(Path(__file__)),
            'audit_sha256': streaming_hash(audit_path), 'manifest_sha256': audit['manifest_sha256'],
            'readouts': len(classified), 'all_output_categories': classified, 'error_category_counts': categories,
            'style_totals': grade_table(totals, ('model', 'cohort', 'style')),
            'role_totals': grade_table(roles, ('model', 'cohort', 'style', 'role')),
            'per_stream_totals': grade_table(streams, ('model', 'cohort', 'style', 'seed')),
            'same_cache_joint3': grade_table(joint_counts, ('model', 'cohort', 'style')),
            'same_cache_joint6': grade_table(six_counts, ('model', 'cohort')),
            'all_layer_position_correctness_strata': descriptive, 'all_wording_pairs': wording,
            'interpretation': 'Descriptive, not new acceptance criteria. Both cohorts have eight underlying synthetic streams; names, histories and questions are repeated views. All roles/layers/positions/wordings/errors retained. Record-value matching does not identify internal retrieval, and pooled attention mass is not causal use. No architecture gain, capacity limit or unique GDN-group fault is established.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite observations')
    result = observe(args.audit)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: result[k] for k in ('status', 'readouts', 'style_totals', 'error_category_counts')}), flush=True)


if __name__ == '__main__':
    main()
