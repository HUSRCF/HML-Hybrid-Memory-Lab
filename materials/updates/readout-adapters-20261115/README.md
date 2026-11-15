# Frozen-GDN learned readout controls

All three fixed432step runs,all5184final-fit answers and the independent282source training audit are complete. Every6912baseline+adapted fit answer was independently regraded. These are observed fit results; prospective confirmation is a separate registered phase.

| Model | Correct /1728 | Baseline errors repaired | New regressions |
| --- | ---: | ---: | ---: |
| Fixed GDN |1633|0|0|
| Query |1720|92|5|
| Output gate |1715|84|2|
| Query + output gate |1727|95|1|

The joint control repairs all95baseline errors but introduces one other-entity value error. Query-only retains eight errors;gate-only retains13. Every raw answer,error category and regression is retained in [readouts.csv](exports/readouts.csv) and [paired-vs-fixed-gdn.csv](exports/paired-vs-fixed-gdn.csv). All conditions appear in [all-role-condition-cells.csv](exports/all-role-condition-cells.csv);all1296steps appear in [all-training-steps.csv](exports/all-training-steps.csv). [Report](exports/report.json) and [figure](exports/fit-summary.png) are complete observed-fit summaries.

Each new arm has131072parameters at Attention layer15. Query-only and output-gate-only use rank32;the joint arm uses two independent rank16branches. Native2B weights and the existing696192parameter GDN adapter stay frozen. New adapters act only after the known fact boundary,during both question and answer,with identical training and deployment scope. All fact histories retain normal updates and full Attention. All1728previously observed questions are explicitly fit data,including all95baseline errors. There is no checkpoint selection,early stopping or budget extension;only the final432weights are evaluated.

The three teacher losses at final fit are0.00625479(Query),0.00463121(gate) and0.00090610(joint),averaged over four numeric-answer/newline tokens. Free numeric answers use the normal cached greedy inference path. Teacher loss and free-generation accuracy are separate measurements.

Gate-only also fits strongly,so these results do not uniquely identify Query addressing as the sole deficient computation. Improving observed SET examples does not establish generalization,COPY competence orhybrid-block architecture superiority. One training initialization was used. All prior failed scientific criteria remain unchanged.

The original protocol was registered before implementation at `configs/readout_adapter_training_plan.json`. A source-only amendment moved the new module outside the historical `sas_memory/*.py` inventory without changing old sources;the first full regression failure is preserved. All360tests then passed and277sources were sealed. The first native smoke failed on its first backward before any optimizer update: native recurrent cache writeback overwrote a tensor saved by autograd. `configs/readout_adapter_cache_commit_addendum.json` registered isolated copy-on-write commits only on disposable teacher-cache instances;no installed model/FLA function,normal source cache ordeployment cache changed. All364tests passed before the corrected282source seal.

Corrected real2B scratch safety and independent audit passed for each arm: exact zero-adapter replay,effective gradients through both low-rank factors,all native/fixed parameters frozen,normal prefixes unchanged,and native no-gradient versus corrected gradient-enabled logits,loss and all48final cache tensors bit-identical both before and after an actual throwaway update. Scratch weights were discarded before the fixed training schedule. Every288normal history was checked against the prior strong-GDN cache bank during training and again at final fit.

The source seals,passing and failed logs,independent input receipts,complete training manifest and original+correction audits are preserved alongside these results. The confirmation protocol,code and372test source seal are in `../readout-confirmation-20261116`:fixed88000SET and89000COPY seeds,allfive models and6000physical questions. The inference code was sealed before these complete fit outcomes were read. Neither fit scores nor future outcomes select checkpoints,arms,seeds orconditions.

The sole joint-fit regression is the protected account U query `What is the last balance assigned to account U?` in seed82001/original/related-64-load16:gold142,prediction665. The matching normal fact cache is unchanged;this is a readout regression,not evidence that the new adapter corrupted fact-prefix writes.

Complete trained checkpoints,optimizer/RNG states,the existing fixed-GDN weights and full fit artifacts are backed up in [the fit Release](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/readout-adapter-fit-2026-11-16). All1553archive members were decompressed and byte-checked,331committed source/material files matched the release target,and all three remote assets were downloaded and byte-verified. Archive SHA256: `e4d5a8059781f5fe3c0b1139ad5eebc1e34c66ee4d716dddd9aa7fdbaed3316d`. The initial backup captured its own still-open operation log;the published repack excludes active packaging logs,with no scientific payload change.
