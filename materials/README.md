# Research materials

研究当前已暂停，未启动新的训练或推理。

研究对象是现有小型混合语言模型中的递归状态更新与历史证据读取。最近的
实验使用 Qwen3.5-2B IT；不能将这些结果当作 Base 模型的实测结果，也尚未
实现显式的共享语义依赖控制器。

## Current evidence

- 新八条事件流上的三项总体科学判据未通过；此前同一开发数据上的通过结果
  保留其原有适用范围，不能替代新数据验证。
- 四个复用样本上的最终状态实验完成了 32 次精确重放和 20 次完整答案生成。
  改变或恢复最后三层 GDN 的递归、卷积缓存会改变后续 logits 和正确答案损失，
  但这 20 次生成的完整答案均与各自未干预参考相同。
- 恢复全部状态后，缓存及后续每一步 logits 均精确恢复。112 份实际 logits
  张量已保存，全套 183 项测试通过。
- 这些是条件干预诊断，不能证明原始事实写入损坏，也不能证明新架构有效。
- 下一轮未见事件流实验已登记：保留八种原有基线，并增加同一共享检查点的
  问题阶段、前缀阶段启用对照。新数据、运行代码和推理尚未开始。

## Browsable files

- `reports/`: original audit reports, CSV tables, figures, exported sources and
  provenance files; failed and null results are retained.
- `logs/`: original setup, test and experiment execution logs.
- `registrations/`: original dataset registration manifests.
- `experiment_manifests/`: original run manifests, including failed and partial runs.
- `archive_manifest.json`: SHA-256 and size of every archived material file and
  every release attachment.
- `code_snapshot_manifest.json`: exact code/configuration snapshot before Git initialization.

The chronological technical record remains in the root README. The authoritative
resume state is `configs/experiment_progress.json`; resume only after the user
explicitly resumes the research goal.

## Complete backup

Release `research-backup-2026-11-15` contains gzip tar archives of `datasets`,
`runs`, `artifacts`, `logs`, plus the code/configuration snapshot. The archives
retain original relative paths, byte contents and SHA-256 hashes. Large archives
are split into numbered parts; concatenate parts in the order listed by
`archive_manifest.json` before extracting.

The dataset archive includes all existing program-generated data. The run
archive includes controller checkpoints, optimizer states, raw logits and sealed
source snapshots. No existing experiment result was rerun or changed for backup.
Pretrained model weights and caches are excluded.

To restore on the original workstation, download the release assets, verify
their SHA-256 values, then extract each material archive into
`/media/990Pro/Code/SAS`. Archives use paths such as `runs/memory_update/...`.
The code archive extracts into a separate directory containing the project
files. Inspect the archive listing before restoring into an existing tree.

Historical manifests deliberately retain their original absolute paths and
hashes. On another machine, acquire the official model weights separately and
provide the recorded model paths or a deliberate path mapping. The recorded
ROCm/PyTorch/Transformers environment is a reproducibility constraint, not a
portable installer guarantee.
