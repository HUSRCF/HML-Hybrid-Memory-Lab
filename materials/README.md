# Research materials

Latest: [fresh recurrent-state / question-Query intervention](updates/fresh-query-20261115/README.md).
All 3,264 readouts, 20 cells, 68 episode cells and 1,536 actual same-source replacements
are independently audited. Normal Queries restore GDN-adapted new-value Full reading
from 2/32 to 32/32 in an old-state recipient; the reverse swap yields 3/32.
A fixed-prefix answer regresses, and normal Full already scores 191/192.
This is a fresh eight-stream mechanism diagnostic; no new architecture claim or training.

[Complete byte-verified fresh-query backup](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/fresh-question-query-2026-11-15).

[Previous byte-verified crossover backup](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/post-tail-path-crossover-2026-11-15).

本目录记录 2026-11-15 暂停备份时点的材料。完整上传与校验完成后，用户已要求
继续实验；后续状态以 `configs/experiment_progress.json` 为准，Release 快照保持不变。

研究对象是现有小型混合语言模型中的递归状态更新与历史证据读取。最近的
实验使用 Qwen3.5-2B IT；不能将这些结果当作 Base 模型的实测结果，也尚未
实现显式的共享语义依赖控制器。

## Complete normal-encoding then subsequent-update controls

[完整诊断](updates/post-encoding-20261115/README.md)的8160项正式读出与170条件均已独立审计，
259项测试、两轮相反顺序各510项smoke、680份源缓存及206份源码通过核对。正常编码的共享前缀
固定，仅干预后续更新。Native当前值接64条无关更新后16/16降至8/16，Oracle仍16/16；停止Delta
更新会使新值读取全部失败。已有GDN适配器正常更新时，两档无关尾部都48/48。
当前不能据此主张联合架构必要；下一步隔离尾部KV编码与递归状态对Query的影响。未新增训练、模型或环境；旧失败门槛不变。

全部原始运行与登记材料已发布到[逐字节核验的增量Release](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/post-encoding-updates-2026-11-15)，三个附件均经实际下载核验。

## Complete whole-record budget controls

[完整记录预算对照](updates/update-budget-20261115/README.md)已完成13440次新读出及独立审计，
另保留6720项历史参考。全部60条件、100组配对比较及分层均导出。新控制的40个配置条件联合
均0/48；固定Oracle读取时，只有偶然保留原支持SET的情况正确。匹配写入数量本身不够，但
控制也会影响事实首次编码，不能直接归因于后续更新损坏。COPY选择已由类型预算决定。
241项测试、两轮720次smoke及240项完整缓存检查通过；197份测量源码与冻结参数不变。
后续先从正常编码的相同前缀分叉，只干预后续更新，未追加训练或扩模。

完整增量数据与原始运行已发布到[逐字节核验的Release](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/update-budget-controls-2026-11-15)。

## Complete gate-component decomposition

[前轮衰减与Delta分解](updates/gate-components-20261115/README.md)的6720次新读出和6720项历史参考
仍保留其原始结论与限制。关闭Delta配合Oracle读取时五配置均48/48；仅关闭衰减配合Oracle
为15、32、35、33、30/48。完整读取下单门干预损害其他角色。234项测试及原生BF16数值参考
修正保留，首次理想FP32校验失败不改写。本轮预算对照见上，所有旧科学失败门槛保持原判。

## Previous complete fixed-input recurrent-hold controls

[固定原始输入的递归更新阻断诊断](updates/oracle-write-hold-20261115/README.md)
完成6720次正式读出及独立审计，并通过229项测试、两轮各360次的反向顺序运行
及120项完整历史缓存检查。Native原始为1/48，单独更新阻断为0/48、单独Oracle读取
为13/48、二者同时为48/48；单独更新阻断在全部五个配置中均为0/48。
原Shared仅Oracle读取已达48/48，二者同时为47/48，完整结果保留该退化。
本轮输入与位置固定，依赖标签具有后见信息；不能直接部署，也不证明共享架构必要。
全部旧失败门槛保持原判，未追加训练或扩模。

## Previous complete text competence controls

[全部6720次文本交叉对照](updates/text-competence-factorial-20261115/README.md)及独立审计完成，223项测试通过。
Native在完整历史为1/48，保留原始COPY的支持文本为48/48，完整历史COPY编译为SET时为21/48。
这些是特权能力控制：支持文本改变了前缀计算、Token与位置，不能唯一定位GDN损伤或关系组合。
所有20个配置×条件及相关退化都保留，支持条件144种实际输入与原336任务ID分别报告；旧失败门槛不变。

## Previous completed evidence-access phase

[冻结历史状态的完整证据访问诊断](updates/trained-timing-evidence-access-20261115/README.md)
已完成5040次正式读出、独立审计和215项测试。Oracle支持事件掩码使原Shared/all
从14/48升至48/48；Native历史状态完全保留的question配置从15/48升至46/48，
GDN-only也达到46/48。这是复用8条事件流的解释性诊断，不证明新架构有效；
所有旧失败门槛保持原判。nonslice在五个配置上均为0/336，具有强干预和语法混淆。

此前[匹配训练/部署确认](updates/trained-shared-timing-20261115/README.md)的
5376次读出也已完整备份，三项研究门槛未通过。完整模型检查点另见对应Release。

## Evidence at the initial backup

以下记录备份时点的结果，恢复后的新批次见
[未见事件流时段验证](updates/unseen-shared-timing-20261115/README.md)。

- 备份前首批八条事件流上的三项总体科学判据未通过；此前同一开发数据上的通过结果
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
