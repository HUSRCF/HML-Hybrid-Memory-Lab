# 本轮快照的内容与恢复路径

本轮快照的三个附件为`oracle-write-hold-20261115.tar.gz`、`file_manifest.json`、`local_verification.json`。后者记录压缩包的完整字节数和SHA256；清单逐项记录归档成员的路径、字节数和SHA256。打包时程序逐项读取解压成员核对，发布时实际下载三个远端附件并再次核对。

归档顶层为`code/`、`storage/`及嵌入的`file_manifest.json`。先解压到一个空目录；原工作站对应关系如下。

| 归档路径 | 原工作站路径 |
| --- | --- |
| `code/` | `/home/husrcf/Code/SAS/` |
| `storage/` | `/media/990Pro/Code/SAS/` |

`storage/`保存本轮注册数据、全部主运行和两轮smoke的原始读出、源码快照、控制证明、日志及派生结果，并保留直接前驱的完整运行与注册数据。`code/`保存测量依赖、协议、审计器、导出和发布工具及本轮文档。

原始绝对路径和哈希保持不变。在另一台机器恢复时，需要明确映射这些路径，并使用记录的模型版本、已发布适配器检查点及环境。既有Conda base的说明见仓库README；本轮没有新建环境。

预训练权重、环境、编译缓存和已发布的训练检查点不重复打包。此前备份及检查点可从以下Release沿清单恢复。

- [上一轮完整文本控制及全部原始证据](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/text-competence-factorial-2026-11-15)
- [冻结前缀证据访问实验](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/trained-timing-evidence-access-2026-11-15)
- [匹配训练的最终检查点](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/matched-shared-training-timing-2026-11-15)
- [初始完整研究备份与其他既有检查点](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/research-backup-2026-11-15)

已冻结的运行和数据目录保持原名。重新执行会要求新的输出目录，便于同时保留原始证据与重放结果。
