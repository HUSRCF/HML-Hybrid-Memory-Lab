# 完整记录写入预算控制：增量恢复

本轮计划发布 tag `update-budget-controls-2026-11-15`。附件为 `update-budget-20261115.tar.gz`、`file_manifest.json` 和 `local_verification.json`；发布前逐文件解压核验，上传后实际下载三个附件核验字节。最终发布凭据另存于 Git。

压缩包的 `code/` 对应仓库，`storage/` 对应 `/media/990Pro/Code/SAS`。先核验外层大小和 SHA256，再在空目录解压并核验清单全部路径、长度和 SHA256；不要覆盖已有原始测量或失败记录。

这是增量快照。首先恢复[门控分解快照](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/gate-components-factorial-2026-11-15)，沿其恢复说明取得全部较早依赖。它保留首次失败的理想 FP32 内核校验、纯数值 BF16 探测、FLA 源码与原始参考补充登记，本轮没有更改它们。

本轮包含新数据集、197 份正式运行源码、两个相反加载顺序的 smoke、独立审计、240 项完整前缀检查、正式运行、逐题结果、CSV/图表及日志。每个正式配置条件文件含 336 项，40 文件合计 13440 次新物理读出；6720 项历史参考单独标记。相同事件流的多次读出不构成更多独立样本。

预训练模型、Conda 环境、编译缓存和此前已发布的训练检查点不重复打包。四个冻结适配配置继续使用[匹配训练快照](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/matched-shared-training-timing-2026-11-15)中的最终 step432 检查点，恢复到协议指定的原始绝对路径。

继续复用 Conda base：Torch `2.9.1+rocm7.2.0.lw.git7e1940d4`、Transformers `5.14.1`、FLA `0.6.0`，纯文本 IT、BF16、eager。加载后的原生权重 SHA256 为 `2fcdfb2c15878ad12794dcff913b5d43fe0787850a4b4d86185087bc9dcd6158`。适配器 FP64 仿射累加／FP32 输出仅是固定推理设置。

依赖恢复完整且输出路径尚不存在时，可重做审计：

```bash
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
python scripts/analyze_update_budget.py \
  /media/990Pro/Code/SAS/runs/memory_update/update-budget-controls-20261115-01 \
  --output /tmp/update-budget-restored-audit.json
```

审计核对协议、197 份运行源码、模型/检查点指纹、掩码重建、旧端点、两轮 smoke、实际门控/Attention 以及每个原始结果。恢复后的新运行必须使用不同的空输出目录；已完成的测量源码和旧科学失败判定保持原样。

本轮按 Oracle 提供的类型/数量匹配完整记录，仍有后见预算信息；不是一个对所有问题共用的在线状态。读取掩码始终来自原支持记录，不跟随随机或位置写入掩码。全部六个 Attention 层保留，beta=0 同时停止 Delta 擦除和插入，原生衰减继续执行。
