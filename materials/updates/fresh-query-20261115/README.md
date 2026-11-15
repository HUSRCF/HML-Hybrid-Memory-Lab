# 新事件流上的递归状态与问题 Query：完整诊断

全部 3,264 项正式读出、1,536 项同来源真实替换重放、1,536 次问题专用捕获和 9,216 个实际 Query 张量通过独立审计。284 项测试、两轮各 204 项读出的相反模型顺序检查通过；222 份测量源码及冻结参数保持原样。8 条新事件流、40 个上下文、240 个问题，所有基线失败均保留。没有新增训练、环境或模型下载。

[完整 20 条件及 68 个阶段条件](exports/report.md)、[3,264 项正式读出](exports/primary.csv)、[1,536 项独立列出的真实重放](exports/same-source-replays.csv)、[178 项配对对照](exports/contrasts.csv)、[296 项分层结果](exports/strata.csv)、[768 项逐问题 NLL 交互](exports/nll-interactions.csv)。[完整图](exports/all20cells.png)的全部标签、分母和页脚已视觉检查，另有 SVG。

[独立审计](fresh-query-primary-audit-20261115-01.json)、[完整描述性提取](fresh-query-observations-20261115-01.json)、[相反顺序精确一致检查](fresh-query-smoke-equivalence-20261115-01.json)、[封存回执](registration-manifest.json)。供体指纹、实际替换调用与源缓存指纹分别见 [donor-queries.csv](exports/donor-queries.csv)、[replacement-calls.csv](exports/replacement-calls.csv)、[cache-fields.csv](exports/cache-fields.csv)。

## 干预含义

N 是正常编码前缀及尾部的缓存。H 从 N 分叉，只把全部 18 个递归矩阵恢复成同一模型正常前缀的矩阵，卷积状态、全部 Attention KV 与尾部后位置保持 N。没有在尾部关闭门控。H 没有真实的最后 Token logits，不能借用 N 的 logits。

标签第一位是接收方状态，第二位是问题 Query 来源。NN/HH 是未替换基线，NH/HN 只在一次已知问题后缀 forward 中替换全部 6 个 Attention 层实际 post-normalization/post-RoPE Query。后续答案评分和自由生成直通原模型。供体捕获只接受已知问题文本，不做 gold 评分或生成；Full 与 Oracle 分别捕获、分别使用。Oracle 依赖真实支持标注，只是能力控制。

同来源替换确实复制供体张量，全部 1,536 项与未替换五字段结果及读取概率精确一致。接收方非 Query 参数与源缓存受到实际张量保护，所有层的调用及位置均核对；不是绕过替换的恒等控制。

## 全部尾部后条件

每项 Full/Oracle 总体分母 192，新值分母 32；新值是 8 流 × 2 尾部长度 × 2 文体的重复问题，不是 32 条独立流。

| 模型 | 状态 / Query | Full 总体 | Full 新值 | Oracle 总体 | Oracle 新值 |
|---|---|---:|---:|---:|---:|
| native | NN | 159/192 | 27/32 | 191/192 | 32/32 |
| native | NH | 152/192 | 14/32 | 191/192 | 31/32 |
| native | HN | 161/192 | 26/32 | 192/192 | 32/32 |
| native | HH | 151/192 | 12/32 | 192/192 | 32/32 |
| GDN adapter | NN | 191/192 | 32/32 | 192/192 | 32/32 |
| GDN adapter | NH | 163/192 | 3/32 | 192/192 | 32/32 |
| GDN adapter | HN | 191/192 | 32/32 | 192/192 | 32/32 |
| GDN adapter | HH | 162/192 | 2/32 | 192/192 | 32/32 |

接尾部前，native Full 为 38/48，GDN adapter 为 48/48；两者 Oracle 均 48/48。这些基线失败没有用于筛选样本。

GDN 适配器保持正常尾部 KV、恢复旧递归矩阵后，新值由 32/32 降到 2/32。向该旧状态接收方换入正常 Query（HN），新值恢复 32/32；反过来给正常状态换入旧状态 Query（NH），新值降到 3/32。这是在新流上重复出现的、对 Query 路径具有方向性的干预效果。

但 HN 对 HH 的总体修复为 30 次增益、1 次退化；30 次增益均为新值，退化来自固定前缀证据。因此不能说“恢复且完全无损”。HN 与 NN 虽然都是 191/192，却有 1 次增益、1 次退化，并非所有回答相同。正常旧事实读取已很强，不能把交叉干预修复当成新架构超过强基线。

原生模型 HN 对 HH 为 15 次增益、5 次退化，其中新值 14 次增益、0 次退化，固定前缀证据 1 次增益、5 次退化。总体由 151/192 变为 161/192；正常 NN 为 159/192。原生结果较混合，不能只报告 GDN 适配器的有利结果。

GDN 适配器 Oracle 下四条件都 192/192；正常 Full 原本也有 191/192。全部 6 层的 N/H Query 指纹在每个固定问题、模型及 Reader 中都不同；指纹不同本身不说明哪层导致修复。NLL 与自由答案、相关新值与固定旧证据均单独报告。

## 结论边界与下一步

本任务中，旧递归状态与正常尾部 KV 的读取不匹配，可以通过问题阶段 Query 的交叉替换大幅改变。它反对直接把“恢复旧矩阵”当作通用保护策略，也说明读出变化不等同于事实已被抹除。

供体 Query 包含上游 Attention 和递归计算，交叉激活可能偏离自然分布；修复不唯一证明语义地址机制。没有中介百分比、显著性、创新性或新共享架构有效性声明；所有早先失败判据保持原判。

下一步先登记并定位单层 Query 的修复及反向损害，覆盖全部六层、两个固定模型、全部问题和两种方向；不按本轮成功样本选子集。它是复用队列的机制实验，不是另一轮新数据确认。暂不训练新控制器或扩模。

## 推理前登记与复现

协议 [fresh_question_query_mediation_plan.json](../../../configs/fresh_question_query_mediation_plan.json) SHA256 `b99755922d42236d6077c3a0d16bcc3e8cadc61ad34d093b8f2946432bac791d`。登记包 manifest SHA256 `78c42d760f1d0108e8a79adea427f8638bca5be88be3b8933574632ba1d82f60`，正式 manifest SHA256 `a9059e6309feae40f69c72ccacbb544875bd73cd65d366027de6ff0c9e44ce17`。

使用现有 Conda base、Qwen3.5-2B IT 本地权重及原有 GDN 检查点；不是 Base 模型结果。前缀 211 Token、尾部 112/896 Token。封存前独立重建全部输入与 122 份前序输入，模型构造、权重加载调用和 GPU 张量分配均为零。两项单测的 InferenceMode 设置问题已在封存前更正，最初失败日志与全部 284 项通过日志一起保留。

运行入口 `scripts/run_fresh_query_pipeline.sh`。完整原始数据和源码快照在增量备份中，不上传预训练权重或重复环境。恢复需要前序 Release 链、固定适配器、本地模型及记录的 ROCm/Transformers 后端。`donor-queries.csv` 保存实际字节指纹与布局，完整 Query 张量未另存副本；复现实际张量需要同环境重跑。
