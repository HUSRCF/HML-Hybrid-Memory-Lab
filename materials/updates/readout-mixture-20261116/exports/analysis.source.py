"""Report every final observed fit output; no confirmation or model selection."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv, hashlib, json, re, sys
ROOT = Path('/home/husrcf/Code/SAS'); sys.path.insert(0, str(ROOT))
BASE = Path('/media/990Pro/Code/SAS'); ART = BASE / 'artifacts/memory_update'
from scripts.analyze_readout_mixture_training import audit_run
from scripts.readout_mixture_data import registered_inputs
from scripts.analyze_readout_confirmation import grade

sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
run = BASE / 'runs/memory_update/readout-mixture-training-20261116-01'
audit_path = ART / 'readout-mixture-training-audit-20261116-01.json'
audit = json.loads(audit_path.read_text()); assert audit == audit_run(run)
p, packet, data = registered_inputs()
raw = {name: [json.loads(line) for line in (run / name / 'final-fit.jsonl').read_text().splitlines()]
       for name in ('set_replay', 'set_copy_mix')}
out = ART / 'readout-mixture-fit-exports-20261116-01'; assert not out.exists()
physical, logical, steps, pairs, joint = [], [], [], [], []
summary = {}
for name, values in raw.items():
    by_id = {r['id']: r for r in values}
    expected = {e['query_id'] for e in data['schedules'][name]}
    assert len(by_id) == len(values) and set(by_id) == expected
    own = []
    for value in values:
        qid = value['id']; e = data['examples'][qid]; q = data['queries'][qid]
        grade(value, e['answer'])
        r = {'regime': name, 'task': e['task'], 'id': qid, 'context_id': e['context_id'],
             'role': q.get('role', q.get('representative', {}).get('role', '')),
             'question': e['question'], 'gold': e['answer'], 'generated': value['generated'],
             'correct': int(value['greedy_exact']), 'answer_nll': value['answer_nll'],
             'teacher_nll_per_token': value['teacher_nll_per_token'],
             'teacher_numeric_tokens_correct': value['teacher_numeric_tokens_correct'],
             'known_question_tokens': len(e['known_ids']), 'teacher_input_tokens': len(e['teacher_ids'])}
        physical.append(r); own.append(r)
        if e['task'] == 'set':
            logical.append({**r, 'logical_id': 'set:' + qid, 'seed': q['seed'], 'naming': q['naming'],
                'question_style': q['question_style'], 'case_id': q['case_id'], 'base_case_id': 'set-s' + str(q['seed']),
                'dependency_depth': 0, 'condition': q['episode'], 'kind': q['kind'],
                'tail_records': q['tail_records'], 'active_entities': q['active_entities']})
    summary[name] = {task: {'correct': sum(r['correct'] for r in own if r['task'] == task),
                          'questions': sum(r['task'] == task for r in own)} for task in ('set', 'copy')}
    if name == 'set_copy_mix':
        groups = defaultdict(list)
        for b in data['copy_bindings']:
            r = next(r for r in own if r['id'] == b['physical_id'])
            logical.append({**r, 'logical_id': 'copy:' + b['case_id'] + ':' + b['variant'] + ':' + b['role'],
                'seed': b['seed'], 'naming': '', 'question_style': b['question_style'], 'case_id': b['case_id'],
                'base_case_id': b['base_case_id'], 'dependency_depth': b['dependency_depth'],
                'condition': b['variant'], 'kind': 'copy', 'tail_records': '', 'active_entities': ''})
            if b['variant'] != 'before':
                groups[b['case_id']].append((b, r))
        for cid, entries in groups.items():
            assert len(entries) == 6
            b = entries[0][0]
            joint.append({'regime': name, 'case_id': cid, 'seed': b['seed'], 'depth': b['dependency_depth'],
                          'question_style': b['question_style'], 'answers_correct': sum(r['correct'] for _, r in entries),
                          'answers_required': 6, 'joint_correct': int(all(r['correct'] for _, r in entries))})
    for line in (run / name / 'training.jsonl').read_text().splitlines():
        s = json.loads(line)
        steps.append({'regime': name, 'step': s['step'],
                      'teacher_loss_mean': sum(e['teacher_nll_per_token'] for e in s['examples']) / 4,
                      'gradient_norm': s['gradient_norm'], 'elapsed_seconds': s['elapsed_seconds']})
base = {r['id']: r for r in raw['set_replay']}
for row in raw['set_copy_mix']:
    if row['task'] == 'set':
        old = base[row['id']]
        pairs.append({'id': row['id'], 'role': data['queries'][row['id']]['role'],
                      'seed': data['queries'][row['id']]['seed'], 'gold': row['answer'],
                      'replay_generated': old['generated'], 'mixed_generated': row['generated'],
                      'replay_correct': int(old['greedy_exact']), 'mixed_correct': int(row['greedy_exact']),
                      'repair': int(not old['greedy_exact'] and row['greedy_exact']),
                      'regression': int(old['greedy_exact'] and not row['greedy_exact'])})
assert len(physical) == 2400 and len(logical) == 2592 and len(steps) == 864 and len(pairs) == 864 and len(joint) == 96
summary['set_copy_mix']['copy'].update(logical_correct=sum(r['correct'] for r in logical if r['task'] == 'copy'),
                                     logical_bindings=864, joint6_correct=sum(r['joint_correct'] for r in joint), joint6_units=96,
                                     joint_by_depth={str(d): sum(r['joint_correct'] for r in joint if r['depth'] == d) for d in (1, 2, 3)})
cells = defaultdict(list)
for r in logical:
    cells[tuple(r[k] for k in ('regime','task','question_style','dependency_depth','condition','role'))].append(r)
cell_rows = [{**dict(zip(('regime','task','question_style','dependency_depth','condition','role'), key)),
              'bindings': len(values), 'correct': sum(r['correct'] for r in values)} for key, values in cells.items()]
out.mkdir()
tables = {'physical-readouts.csv': physical, 'all-logical-fit-bindings.csv': logical,
          'all864-training-steps.csv': steps, 'paired-set-fit.csv': pairs,
          'all96-copy-joint6-fit.csv': joint, 'all-role-condition-cells.csv': cell_rows}
for name, values in tables.items():
    with (out / name).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    assert len(list(csv.DictReader((out / name).open()))) == len(values)
report = {'status': 'all2400_observed_final_fit_readouts_independently_regraded',
          'created_utc': datetime.now(timezone.utc).isoformat(), 'source_run': str(run),
          'training_audit_sha256': sha(audit_path), 'fit_not_confirmation': True, 'summary': summary,
          'set_mix_vs_replay_repairs': sum(r['repair'] for r in pairs),
          'set_mix_vs_replay_regressions': sum(r['regression'] for r in pairs),
          'table_rows': {name: len(rows) for name, rows in tables.items()},
          'new_model_inference': False, 'architecture_superiority_claim': False,
          'prior_failed_criteria_unchanged': audit['prior_failed_criteria_unchanged'], 'source_sha256': sha(Path(__file__))}
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
(out / 'analysis.source.py').write_bytes(Path(__file__).read_bytes())
(out / 'export-receipt.json').write_text(json.dumps({'status': report['status'], 'table_rows': report['table_rows'],
    'files_sha256': {f.name: sha(f) for f in sorted(out.iterdir())}}, indent=2) + '\n')
print(json.dumps(report), flush=True)
