# 本轮增量快照恢复

本轮Release tag为`gate-components-factorial-2026-11-15`。附件包含gzip tar、逐文件`file_manifest.json`及外层`local_verification.json`；发布前逐项解压比对字节，上传后实际下载三个附件再次比对。最终发布凭据单独提交到Git。

快照中的`code/`对应仓库路径，`storage/`对应`/media/990Pro/Code/SAS`。先检查外层压缩包大小与SHA256，再按清单检查解压后的每个文件。不要直接覆盖已有实验结果；先在临时目录核验，原始路径恢复只在空的恢复位置进行。

这是增量备份，需要先恢复[前轮递归更新阻断快照](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/oracle-write-hold-factorial-2026-11-15)，并沿其恢复说明取得更早的注册数据和审计依赖。模型与适配器的原始绝对路径是注册协议的一部分；路径不匹配会触发验证失败。

预训练模型、Conda环境、编译缓存不包含在本轮压缩包中。四个既有适配配置使用此前发布的最终step432检查点，见[匹配训练快照](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/matched-shared-training-timing-2026-11-15)及其依赖。不要用其他步数或重新训练的权重替代。

运行环境继续复用Conda base：Torch `2.9.1+rocm7.2.0.lw.git7e1940d4`、Transformers `5.14.1`、FLA `0.6.0`。原生IT模型加载权重SHA256固定为`2fcdfb2c15878ad12794dcff913b5d43fe0787850a4b4d86185087bc9dcd6158`。冻结适配器的FP64累加/FP32输出是本系列推理精度配置，不是训练设置。

原始首次smoke的理想FP32内核校验失败、该版本188份源码、纯数值探测张量、FLA源码、补充登记和后续完整运行均保留。成功执行的门控包装不改变原生Kernel。恢复后不要修改前轮已完成测量的180份源码，也不要把失败运行目录作为新输出复用。

以下命令只在全部依赖恢复且没有同名新输出时使用：

```bash
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
python scripts/analyze_gate_components.py \
  /media/990Pro/Code/SAS/runs/memory_update/gate-components-factorial-20261115-01 \
  --output /tmp/gate-components-restored-audit.json
```

审计会检查188份运行源码、额外内核校验登记、模型/检查点指纹、所有历史参考和旧端点、门控/Attention证明、两轮smoke及逐题结果。正式新读出6720次；历史参考6720次单独标记，统一表13440项不得当作新增独立样本。旧科学失败门槛保持原判。
