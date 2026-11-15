"""Export complete audited confirmation into repository materials without rerunning a model."""
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, shutil

ROOT = Path('/home/husrcf/Code/SAS')
BASE = Path('/media/990Pro/Code/SAS')
ART = BASE / 'artifacts/memory_update'
DEST = ROOT / 'materials/updates/readout-confirmation-20261116'
EXPORT = ART / 'readout-confirmation-exports-20261116-01'
RUN = BASE / 'runs/memory_update/readout-confirmation-primary-20261116-01'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_csv(name):
    with (EXPORT / name).open(newline='') as f:
        return list(csv.DictReader(f))
report = json.loads((EXPORT / 'report.json').read_text())
audit_path = ART / 'readout-confirmation-primary-audit-20261116-01.json'
audit = json.loads(audit_path.read_text())
assert report['source_audit_sha256'] == sha(audit_path)
assert report['status'] == 'all6000physical_and6480logical_answers_independently_regraded'
pipeline = json.loads((ART / 'readout-confirmation-pipeline-20261116-01.json').read_text())
assert pipeline['status'] == 'completed_all6000_frozen_confirmation_readouts_and_independent_audit'
models = ('native', 'fixed_gdn', 'query', 'gate', 'query_gate')
physical = read_csv('physical-readouts.csv')
pairs = read_csv('all-model-pairs-physical.csv')
joint = read_csv('all-joint6-units.csv')
logical = read_csv('all-logical-bindings.csv')
assert len(physical) == 6000 and len(logical) == 6480
shutil.copytree(EXPORT, DEST / 'exports', dirs_exist_ok=False)
rawdest = DEST / 'raw'
rawdest.mkdir(exist_ok=False)
for model in models:
    shutil.copyfile(RUN / (model + '.jsonl'), rawdest / (model + '.jsonl'))
for f in sorted(ART.glob('readout-confirmation-*')):
    if f.is_file() and f.suffix in ('.json', '.py'):
        shutil.copyfile(f, DEST / f.name)
for f in sorted((BASE / 'logs/memory_update').glob('readout-confirmation-*.log')):
    if not any(x in f.name for x in ('package-', 'publish-', 'document-')):
        shutil.copyfile(f, DEST / f.name)
shutil.copyfile(RUN / 'manifest.json', DEST / 'primary-run-manifest.json')
shutil.copyfile(Path(__file__), DEST / 'complete-materials-export.source.py')

lines = [
    '# Frozen readout controls: complete new-stream confirmation', '',
    'All 6,000 registered physical answers and 6,480 logical bindings completed and passed independent audit. '
    'No prospective question was used for training, checkpoint selection, cohort replacement or early stopping. '
    'All three controls use the final step-432 checkpoints fixed before constructing either new cohort.', '',
    '| Frozen model | SET /864 | COPY physical /336 | COPY six-answer units /48 |',
    '| --- | ---: | ---: | ---: |']
for model in models:
    s, c = report['summary'][model]['set'], report['summary'][model]['copy']
    lines.append(f"| {model} | {s['physical_correct']} | {c['physical_correct']} | {c['joint6_correct']} |")
lines.extend(['',
    'All three readout controls improve SET net accuracy on every one of the eight base streams. '
    'Query and joint each repair 47 baseline errors while adding two protected-account errors; gate repairs 43 while adding one other error and preserves all 288 protected answers. '
    'COPY aggregate accuracy changes little, but its six-answer joint metric declines from 14/48 to 9/48, 11/48 and 10/48. '
    'Query has zero joint repairs and five regressions; gate has one repair and four regressions; joint has zero repairs and four regressions. '
    'Thus improved SET version reading does not preserve all prior successes or solve relational transfer. '
    'Both Query and output-gate training help SET, so these results do not uniquely identify a Query-address mechanism.', '',
    'COPY physical questions are deduplicated: its 432 logical role/order/depth/style bindings map to 336 actual answers per model. '
    'Each COPY six-answer unit requires both control/treatment event orders and all current/initial/protected questions correct. '
    'All 48 units and the before-update questions remain in the raw exports; the joint metric excludes the before variant.', '',
    '## Every repair and regression against the fixed GDN baseline', '',
    '| Readout control | Task | Repairs | New errors | Independent base-stream net gains / losses / ties |',
    '| --- | --- | ---: | ---: | --- |'])
for model in models[2:]:
    for task in ('set', 'copy'):
        pp = [r for r in pairs if r['model_left'] == 'fixed_gdn' and r['model_right'] == model and r['task'] == task]
        seed_deltas = defaultdict(int)
        for r in pp:
            seed_deltas[r['seed']] += int(r['gain']) - int(r['regression'])
        values = list(seed_deltas.values())
        assert len(values) == 8
        counts = f"{sum(x > 0 for x in values)} / {sum(x < 0 for x in values)} / {sum(x == 0 for x in values)}"
        lines.append(f"| {model} | {task} | {sum(int(r['gain']) for r in pp)} | {sum(int(r['regression']) for r in pp)} | {counts} |")
lines.extend(['',
    'These repair/regression counts use physical answers. The complete [physical pairs](exports/all-model-pairs-physical.csv) '
    'and [logical pairs](exports/all-model-pairs-logical.csv) retain all ten model comparisons, not only comparisons to fixed GDN.', '',
    '## SET role and version errors', '',
    '| Model | Current /288 | Initial /288 | Protected /288 | Wrong answer matches another seen value of requested account |',
    '| --- | ---: | ---: | ---: | ---: |'])
for model in models:
    values=[]
    for role in ('target_current', 'target_initial', 'protected_current'):
        rr=[r for r in physical if r['model']==model and r['task']=='set' and r['role']==role]
        assert len(rr)==288
        values.append(sum(int(r['correct']) for r in rr))
    other=report['summary'][model]['set']['error_value_counts']['matches_requested_entity_other_seen_value']
    lines.append(f'| {model} | {values[0]} | {values[1]} | {values[2]} | {other} |')
lines.extend(['',
    'The final column describes an output-value match, not proof of attending to a particular record or losing a stored version. '
    'All malformed, other-entity, ambiguous and unrecorded-value outputs remain in '
    '[the complete answers](exports/physical-readouts.csv). COPY can propagate identical values across accounts; '
    'matching entities are explicitly retained rather than treating a numeric match as unique attribution.', '',
    '## COPY joint results by dependency depth', '',
    '| Model | Depth 1 /16 | Depth 2 /16 | Depth 3 /16 | Protected physical /112 |',
    '| --- | ---: | ---: | ---: | ---: |'])
for model in models:
    values=[]
    for depth in (1,2,3):
        jj=[r for r in joint if r['model']==model and r['task']=='copy' and int(r['depth'])==depth]
        assert len(jj)==16
        values.append(sum(int(r['joint_correct']) for r in jj))
    pp=[r for r in physical if r['model']==model and r['task']=='copy' and r['role']=='protected_current']
    assert len(pp)==112
    lines.append(f"| {model} | {values[0]} | {values[1]} | {values[2]} | {sum(int(r['correct']) for r in pp)} |")
lines.extend(['',
    '![All frozen models and COPY dependency depths](exports/confirmation-summary.png)', '',
    '## Design and interpretation limits', '',
    '- Protocol `configs/readout_adapter_confirmation_plan.json` was registered at `3e44ab1` before implementation or prospective construction. '
    'The training protocol already fixed these cohort seeds. Training locks pin all saved checkpoints, optimizer/RNG states, fit logs and audits before new inputs are generated.',
    '- Eight independent base streams per task; names, question styles, dependency depths, event orders and record loads are paired views. '
    'There is one training initialization. Counts of questions or six-answer units are not independent sample sizes. '
    'Fresh SET seeds remain within the same generator and question-template families used for adaptation; this is not evidence of out-of-domain SET generalization.',
    '- SET seeds 88000–88007 cover both account names, all nine normal histories, two one-token-matched question wordings and all three roles. '
    'COPY seeds 89000–89007 retain depths 1/2/3, both styles, alternating 8/64 distractors and all variants/roles. '
    'The actual 208-file prior-input inventory has no seed, ID, full-prefix or full-prompt overlap; the cohort was not resampled.',
    '- Before confirmation, 30 observed-fit replays were exactly reproduced. All 372 CPU regression tests passed before sealing 289 source files. '
    'All 282 predecessor sources remain unchanged.',
    '- No Oracle, donor forward, recurrent rollback, gate intervention, sparse reader or further training is used. '
    'Each question forks an isolated normally built history. All 200 history records per learned readout match fixed GDN across all 48 tensor values, layouts, metadata and prefix logits; all source caches are guarded.',
    '- Each readout adds 131,072 parameters at layer 15 on the existing frozen GDN adapter. Query-only, output-gate-only and joint controls all use matched question/answer-only training and deployment. '
    'They are simple readout baselines, not a shared four-layer semantic controller.',
    '- Longest SET history: 7,519 tokens; longest COPY: 1,389 tokens. These results do not establish 100K-context behavior. '
    'The SET active-entity field denotes the nominal 2/16-account distractor pool; related tails additionally update the queried target.',
    '- COPY evaluates transfer/retention of SET-only readout adaptation. SET success does not demonstrate multi-hop success, and an error cannot by itself identify a unique GDN or Attention cause.',
    '- SET six-answer exports require two question wordings and three roles on one history. This differs from COPY two-order six-answer units and does not revise earlier registered joint-success criteria.',
    '- All three earlier failed scientific criteria remain false in their original scope. A new architecture advantage, unique Query-address explanation, independent-controller superiority comparison and shared-block benefit remain unestablished.', '',
    '## Complete artifacts', '',
    '- [Independent primary audit](readout-confirmation-primary-audit-20261116-01.json) and [complete run manifest](primary-run-manifest.json).',
    '- [All 6,000 physical answers](exports/physical-readouts.csv), [6,480 logical bindings](exports/all-logical-bindings.csv), '
    '[all 1,080 role-condition cells](exports/all-role-condition-cells.csv), [all 80 base-stream rows](exports/all-base-streams.csv).',
    '- [All six-answer units](exports/all-joint6-units.csv), [all joint model pairs](exports/all-joint6-model-pairs.csv), '
    '[all model/task contrasts](exports/all-model-task-contrasts.csv).',
    '- [Wordings](exports/all-wording-pairs.csv), [current/initial](exports/all-current-initial-pairs.csv), '
    '[renaming](exports/all-account-renaming-pairs.csv), [tail lengths](exports/all-tail-length-pairs.csv), '
    '[nominal account pools](exports/all-active-account-pairs.csv), [update relevance](exports/all-update-relevance-pairs.csv), '
    '[before/after normal updates](exports/all-before-after-pairs.csv).',
    '- All five raw JSONL files, complete source/test/input receipts and complete logs are retained in this folder. '
    'The external archive additionally includes literal inputs, sealed sources and trained checkpoint dependencies. '
    'Native 2B pretrained weights and the existing Conda environment remain separate dependencies.', '',
    'Full backup packaging/publication is the next operational step. The broader hybrid-memory research goal remains active.', ''])
(DEST/'README.md').write_text('\n'.join(lines))

progress_path=ROOT/'configs/experiment_progress.json'
progress=json.loads(progress_path.read_text())
entry=progress['readout_confirmation']
entry.update(status='complete6000_independently_audited_and_all_answers_exported',
             completed_utc=datetime.now(timezone.utc).isoformat(),
             independent_audit_sha256=sha(audit_path),physical_readouts=6000,logical_bindings=6480,
             complete_export=str(EXPORT),complete_export_receipt_sha256=sha(EXPORT/'export-receipt.json'),
             complete_summary=report['summary'],architecture_superiority_claim=False,
             next_action='Publish and roundtrip-verify the complete backup;interpret SET improvements and COPY retention against all frozen simple baselines before registering any further method. No tuning or resampling of this completed cohort.')
progress['latest_completed_phase']='readout_confirmation'
progress_path.write_text(json.dumps(progress,indent=2)+'\n')
print(json.dumps({'status':entry['status'],'destination':str(DEST),'new_inference':False}),flush=True)
