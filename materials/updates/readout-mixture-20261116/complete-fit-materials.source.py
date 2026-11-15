"""Commit complete observed fits and terminal provenance while confirmation runs separately."""
from pathlib import Path
from datetime import datetime, timezone
import gzip, hashlib, json, shutil

ROOT = Path('/home/husrcf/Code/SAS'); BASE = Path('/media/990Pro/Code/SAS')
ART = BASE / 'artifacts/memory_update'
DEST = ROOT / 'materials/updates/readout-mixture-20261116'
CONF = ROOT / 'materials/updates/mixture-confirmation-20261116'
RUN = BASE / 'runs/memory_update/readout-mixture-training-20261116-01'
EXPORT = ART / 'readout-mixture-fit-exports-20261116-01'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
audit = json.loads((ART / 'readout-mixture-training-audit-20261116-01.json').read_text())
report = json.loads((EXPORT / 'report.json').read_text())
assert report['training_audit_sha256'] == sha(ART / 'readout-mixture-training-audit-20261116-01.json')
pipeline = json.loads((ART / 'readout-mixture-pipeline-20261116-01.json').read_text())
assert pipeline['status'] == 'completed_paired_training_and_independent_audit_future_cases_not_generated'
for name, digest in json.loads((EXPORT / 'export-receipt.json').read_text())['files_sha256'].items():
    assert sha(EXPORT / name) == digest
shutil.copytree(EXPORT, DEST / 'exports', dirs_exist_ok=False)
(DEST / 'raw').mkdir(exist_ok=False)
for regime in ('set_replay', 'set_copy_mix'):
    shutil.copyfile(RUN / regime / 'final-fit.jsonl', DEST / 'raw' / (regime + '.jsonl'))
    shutil.copyfile(RUN / regime / 'training.jsonl', DEST / 'raw' / (regime + '.training.jsonl'))
shutil.copyfile(RUN / 'manifest.json', DEST / 'training-run-manifest.json')
raw = (RUN / 'normal-history-references.json').read_bytes()
packed = gzip.compress(raw, compresslevel=6, mtime=0)
(DEST / 'training-normal-history-references.json.gz').write_bytes(packed)
assert gzip.decompress(packed) == raw
(DEST / 'training-reference-gzip-receipt.json').write_text(json.dumps({
    'status': 'full256normal_history_records_exact_bytes_preserved', 'original_sha256': hashlib.sha256(raw).hexdigest(),
    'original_bytes': len(raw), 'gzip_sha256': hashlib.sha256(packed).hexdigest(), 'gzip_bytes': len(packed),
    'audited_native_run_unchanged': True}, indent=2) + '\n')
for path in sorted(ART.glob('readout-mixture-*')):
    if path.is_file() and path.suffix in ('.json', '.py'):
        shutil.copyfile(path, DEST / path.name)
for path in sorted((BASE / 'logs/memory_update').glob('readout-mixture-*.log')):
    if not any(word in path.name for word in ('document-', 'package-', 'publish-')):
        shutil.copyfile(path, DEST / path.name)
shutil.copyfile(Path(__file__), DEST / 'complete-fit-materials.source.py')
text = (DEST / 'README.md').read_text()
text = text.replace('两组正式训练流程已启动。自动流程将完成两组各 432 步、全部 2,400 个不同拟合输入读出及完整训练审计。',
                    '两组各 432 步、全部 2,400 个不同拟合输入读出及完整训练审计均已完成。')
section = '''## 完整拟合结果（已观察数据）

| 固定最终检查点 | SET /864 | COPY 实际问题 /672 | COPY 逻辑绑定 /864 | COPY 六问联合 /96 |
| --- | ---: | ---: | ---: | ---: |
| SET 重放 | 853 | 未在本拟合阶段执行 | — | — |
| SET＋COPY | 857 | 619 | 811 | 52 |

两组共同的864个SET问题，混合组修复8问、退步4问；所有逐题回答均保留。混合组的COPY六问联合按深度为32/32、15/32、5/32。这里只评估各自训练中见过的输入：不把SET重放组没有执行COPY拟合评估当成错误或零分，不据此推断迁移或架构优势。

完整[实际回答](exports/physical-readouts.csv)、[全部逻辑绑定](exports/all-logical-fit-bindings.csv)、[SET逐题配对](exports/paired-set-fit.csv)、[96个COPY联合单元](exports/all96-copy-joint6-fit.csv)、[864次优化器更新](exports/all864-training-steps.csv)和全部条件单元已导出并独立复核。全部六个检查点及优化器/RNG状态保留在备份中；原始回答、训练日志、审计和256份正常历史摘要也全部保留。

新七模型确认另有[固定协议及进度](../mixture-confirmation-20261116/README.md)。该实验继续执行，不能用本拟合分数替代其结果；不选择检查点、不追加训练，也不改登记判据。

'''
assert '## 完整拟合结果（已观察数据）' not in text
text = text.replace('## 已固定的后续评估', section + '## 已固定的后续评估')
(DEST / 'README.md').write_text(text)

path = CONF / 'README.md'
text = path.read_text().replace('两组训练仍在进行，不读取中途分数选择模型。',
    '两组训练及全部拟合回答已完成独立审计；始终固定使用两组最终432步，不按拟合分数选择模型。')
text += '\n新增8项CPU契约测试已通过（25.280秒）；实现提交`ac2ed7e`。当前运行完整388项回归，随后封存303个科学源文件、重执行两套完整训练审计并锁定依赖，再构造已登记的新队列。流水线无自动重试、无准确率门槛；任一执行或完整性检查失败即保留并停止。\n'
path.write_text(text)
for source, name in ((Path('/tmp/run_mixture_confirmation_pipeline.py'), 'pending-confirmation-pipeline.source.py'),
                     (Path('/tmp/check_mixture_confirmation_inputs.py'), 'pending-independent-input-check.source.py')):
    shutil.copyfile(source, CONF / name)
progress_path = ROOT / 'configs/experiment_progress.json'
progress = json.loads(progress_path.read_text())
for name in ('configs/experiment_progress.json', 'README.md'):
    assert name not in pipeline['scientific_sources_at_test_start']
progress['current_phase'] = 'mixture_confirmation_full_regression_before_registration'
progress['last_completed_phase'] = progress['latest_completed_phase'] = 'readout_mixture_observed_fit'
progress['readout_mixture'].update(status=audit['status'], training_completed=True,
    training_audit=str(ART / 'readout-mixture-training-audit-20261116-01.json'),
    training_audit_sha256=sha(ART / 'readout-mixture-training-audit-20261116-01.json'),
    observed_fit_report=report, pipeline_terminal=True,
    next_action='Publish complete paired training backup; execute alreadyregistered fixed seven-model confirmation. Fit isnotgeneralization; no checkpoint selection or extensions.')
progress['readout_mixture']['training_progress_snapshot'] = {'captured_utc': datetime.now(timezone.utc).isoformat(),
    'actual_logged_optimizer_updates': {'set_replay': 432, 'set_copy_mix': 432}, 'terminal': True}
progress['mixture_confirmation'].update(status='implemented_full388_cpu_regression_running', implementation_started=True,
    cpu_tests_passed=8, cpu_tests_seconds=25.280, implementation_commit='ac2ed7e', pipeline_session=48521,
    pipeline_receipt=str(ART / 'mixture-confirmation-pipeline-20261116-01.json'),
    next_action='Continue specific no-retrypipeline48521; seal303sources after388tests, lock complete training then construct fixedfuture cohorts; all42observed replays before10800primary answers. Preserveall296predecessors.')
progress_path.write_text(json.dumps(progress, indent=2) + '\n')
path = ROOT / 'README.md'; text = path.read_text()
start, end = text.index('**Current: paired Gate'), text.index('**Latest completed: all 6,000')
text = text[:start] + '''**Current: paired Gate training and all 2,400 observed-fit answers independently audited.**
The [complete task-coverage control](materials/updates/readout-mixture-20261116/README.md)
retains both final432checkpoints: SET replay853/864; mixture SET857/864 and COPY619/672.
The common SET fit has eight repairs and four regressions. These are observed fits, not generalization.
The [seven-model confirmation](materials/updates/mixture-confirmation-20261116/README.md) was registered
before implementation or new cases. Eight added CPU contracts passed; the full388regression is running.
It will separately retain all archived development answers and the fixed new-stream comparison,
without Oracle, donor forwards, checkpoint selection or additional training. Shared-block superiority remains unestablished.

''' + text[end:]
text = text.replace('simple controls before a shared-block method. The next stage is not yet trained or generated.',
                    'simple controls before a shared-block method; the paired training is now complete, as reported above.')
path.write_text(text)
print(json.dumps({'status': 'complete_observed_fit_materials_exported', 'summary': report['summary']}), flush=True)
