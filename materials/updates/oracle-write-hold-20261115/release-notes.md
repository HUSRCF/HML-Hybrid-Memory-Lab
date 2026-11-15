Complete fixed-input recurrent-hold diagnostic: all 6,720 primary readouts, 8,640 related bindings and 3,360 exact prior full/oracle replays independently audited. All five fixed configurations and all four registered conditions are included. No training or model scaleup.

Joint cases correct /48:

| Configuration | Full | Hold | Oracle reader | Hold + oracle |
| --- | ---: | ---: | ---: | ---: |
| native | 1 | 0 | 13 | 48 |
| shared | 14 | 0 | 48 | 47 |
| shared_question_trained_all | 15 | 0 | 47 | 48 |
| shared_question_trained_question | 15 | 0 | 46 | 48 |
| gdn_all_tokens | 16 | 0 | 46 | 48 |

Native improves to 48/48 only under the combined privileged intervention here. Hold alone gives 0/48 in every configuration and harms historical/unrelated answers. Shared/all falls from 48/48 with the oracle reader to 47/48 with both; that regression is retained.

Validation: 229 tests; two byte-identical opposite-order 360-readout smokes; 60 actual cached-state identity controls; 120 no-op controls; 240 Attention probability controls; 120 independent complete-cache/logits/history-KV preservation checks; all 180 measurement sources and frozen native/adapter weights retained. Exported 6,720 main, 1,920 paired and 1,680 factorial CSV rows were checked directly against raw results.

The same original input tokens and positions are used. Support labels depend on the later question, so the update policy is a hindsight oracle. Convolution and GDN readout continue, and downstream contextualized KV may change. Eight reused source streams are correlated; no natural state-damage, online efficiency, shared-controller superiority or new architecture claim. All earlier failed criteria remain failed.

Assets contain the complete current run, both smokes, registrations, sealed sources, direct predecessor evidence, control proofs, raw readouts, audit/export sources, tables, figures and logs. Pretrained weights/environments/caches and previously published checkpoints are excluded. The archive has code/ and storage/ prefixes; file_manifest.json inventories every payload and local_verification.json gives archive SHA256 and size.

[Complete report and limitations](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/tree/main/materials/updates/oracle-write-hold-20261115) · [Restore paths and checkpoint releases](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/blob/main/materials/updates/oracle-write-hold-20261115/restore.md)
