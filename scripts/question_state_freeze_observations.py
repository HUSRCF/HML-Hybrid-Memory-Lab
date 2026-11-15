"""Post-audit independent grading of all arms; subsets remain descriptive only."""
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

ARMS = ['normal'] + [f'freeze_group{i}' for i in range(6)] + ['freeze_all18']


def observe(audit_path):
    audit = json.loads(audit_path.read_text())
    run = Path(audit['source_run'])
    manifest = json.loads((run / 'manifest.json').read_text())
    if (audit['status'] != 'completed_and_audited_question_state_freeze'
        or audit['primary_readouts_audited'] != 6912 or audit['source_files_verified'] != 261
        or streaming_hash(run / 'manifest.json') != audit['manifest_sha256']):
        raise ValueError('Complete unchanged audited primary required')
    packet = Path(manifest['packet'])
    pm = json.loads((packet / 'manifest.json').read_text())
    for name in ('queries.json', 'cases.json'):
        if streaming_hash(packet / name) != pm['payload_sha256'][name]:
            raise ValueError('Registered question/history bytes changed')
    queries = json.loads((packet / 'queries.json').read_text())
    cases = {c['id']: c for c in json.loads((packet / 'cases.json').read_text())}
    raw = {}
    grades = {}
    classified = []
    totals = defaultdict(lambda: [0, 0])
    role_totals = defaultdict(lambda: [0, 0])
    joint = defaultdict(list)
    for arm in ARMS:
        path = run / (arm + '.jsonl')
        if streaming_hash(path) != manifest['model_state']['file_sha256'][path.name]:
            raise ValueError('Per-arm raw file changed')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 864 or [r['id'] for r in rows] != list(queries):
            raise ValueError('All literal questions required in original order')
        raw[arm] = {r['id']: r for r in rows}
        for row in rows:
            q = queries[row['id']]
            case = cases[q['case_id']]
            episode = next(e for e in case['episodes'] if e['id'] == q['episode'])
            events = case['prefix_events'] + episode['tail']
            if any(e['kind'] != 'SET' for e in events):
                raise ValueError('Descriptive SET classification does not support other operations')
            entity = case['protected'] if q['role'] == 'protected_current' else case['target']
            own = [(i, e) for i, e in enumerate(events) if e['target'] == entity]
            truth = own[0 if q['role'] == 'target_initial' else -1][1]['value']
            if str(truth) != q['answer'] or row['answer'] != q['answer']:
                raise ValueError('Independently executed SET truth differs')
            match = re.fullmatch(r'\s*(-?\d+)\s*', row['generated'])
            grade = bool(match and match[1] == q['answer'])
            if grade != row['greedy_exact']:
                raise ValueError('Independent unconstrained exact grading differs')
            grades[(arm, q['id'])] = grade
            same = [i for i, e in own if match and e['value'] == int(match[1])]
            other = [i for i, e in enumerate(events) if match and e['target'] != entity and e['value'] == int(match[1])]
            category = ('correct' if grade else 'non_integer' if not match else
                        'same_account_other_version_value' if same else 'other_account_recorded_value' if other else 'unrecorded_integer')
            classified.append({'arm': arm, 'id': q['id'], 'seed': q['seed'], 'naming': q['naming'],
                               'question_style': q['question_style'], 'episode': q['episode'], 'role': q['role'],
                               'answer': q['answer'], 'generated': row['generated'], 'exact': grade,
                               'category': category, 'matching_queried_account_event_indices': same,
                               'matching_other_account_event_indices': other})
            for key, table in (((arm, q['question_style']), totals),
                               ((arm, q['question_style'], q['role']), role_totals)):
                table[key][0] += grade
                table[key][1] += 1
            joint[(arm, q['question_style'], q['context_id'])].append(grade)
    expected = {(r['arm'], r['id']): (r['gain'], r['loss'])
                for r in audit['metrics']['normal_to_intervention_pairs']}
    pairs = []
    for arm in ARMS[1:]:
        for qid, q in queries.items():
            base, value = grades[('normal', qid)], grades[(arm, qid)]
            gain, loss = int(not base and value), int(base and not value)
            if expected[(arm, qid)] != (gain, loss):
                raise ValueError('Independent normal/intervention grading differs')
            pairs.append({'arm': arm, 'id': qid, 'seed': q['seed'], 'naming': q['naming'],
                          'question_style': q['question_style'], 'episode': q['episode'], 'role': q['role'],
                          'gain': gain, 'loss': loss})
    original_errors = [qid for qid in queries if not grades[('normal', qid)]]
    if len(original_errors) != 50 or len(pairs) != 6048:
        raise ValueError('Every original error and all comparisons required')
    baseline_stable = [qid for qid, q in queries.items() if q['question_style'] == 'state_term'
                       and not grades[('normal', qid)] and not grades[('normal', q['paired_query_id'])]]
    if len(baseline_stable) != 17:
        raise ValueError('Descriptive previous stable subset must remain unchanged')
    stable = []
    for arm in ARMS:
        for qid in baseline_stable:
            other = queries[qid]['paired_query_id']
            stable.append({'arm': arm, 'state_id': qid, 'event_id': other,
                           'state_correct': grades[(arm, qid)], 'event_correct': grades[(arm, other)],
                           'state_generated': raw[arm][qid]['generated'], 'event_generated': raw[arm][other]['generated']})
    summary = []
    for arm in ARMS:
        for style in ('state_term', 'event_term'):
            c, n = totals[(arm, style)]
            selected = [r for r in pairs if r['arm'] == arm and r['question_style'] == style]
            summary.append({'arm': arm, 'style': style, 'correct': c, 'questions': n,
                            'gains': sum(r['gain'] for r in selected), 'losses': sum(r['loss'] for r in selected)})
    actual_joint = []
    combined = defaultdict(list)
    joint_counts = defaultdict(lambda: [0, 0])
    for (arm, style, cid), values in joint.items():
        if len(values) != 3:
            raise ValueError('All three roles must share a normal history source')
        joint_counts[(arm, style)][0] += all(values)
        joint_counts[(arm, style)][1] += 1
        combined[(arm, cid)].extend(values)
    for (arm, style), values in sorted(joint_counts.items()):
        actual_joint.append({'arm': arm, 'question_style': style, 'correct': values[0], 'contexts': values[1]})
    if actual_joint != audit['metrics']['same_cache_joint3']:
        raise ValueError('Independent joint grading differs')
    return {'status': 'all6912raw_answers_and6048pairs_independently_regraded',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': streaming_hash(Path(__file__)),
            'audit_sha256': streaming_hash(audit_path), 'manifest_sha256': audit['manifest_sha256'],
            'readouts': 6912, 'pairs': 6048, 'style_totals': summary,
            'role_totals': [{'arm': k[0], 'style': k[1], 'role': k[2], 'correct': v[0], 'questions': v[1]}
                            for k, v in sorted(role_totals.items())],
            'baseline_error_ids': original_errors,
            'all_original_errors_in_every_arm': [{'arm': arm, 'id': qid, 'correct': grades[(arm, qid)],
                                                'generated': raw[arm][qid]['generated']}
                                               for arm in ARMS for qid in original_errors],
            'value_attribution_all_readouts': classified,
            'error_category_counts': [{'arm': arm, **dict(Counter(r['category'] for r in classified if r['arm'] == arm))}
                                      for arm in ARMS],
            'same_cache_joint3': actual_joint,
            'same_cache_joint6': [{'arm': arm, 'correct': sum(all(v) for (a, _), v in combined.items() if a == arm),
                                   'contexts': 144} for arm in ARMS],
            'all_normal_intervention_pairs': pairs,
            'original17paired_failures_descriptive_only': stable,
            'interpretation': 'Post-audit descriptive regrading, not a new cohort or acceptance criterion. All864questions and all eight arms retained. Numeric value matching can be ambiguous and does not identify internal attention or state loss. Eight reused streams; no significance, unique fault source, architecture or multihop success claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite independent observations')
    result = observe(args.audit)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: result[k] for k in ('status', 'readouts', 'pairs', 'style_totals', 'error_category_counts')}), flush=True)


if __name__ == '__main__':
    main()
