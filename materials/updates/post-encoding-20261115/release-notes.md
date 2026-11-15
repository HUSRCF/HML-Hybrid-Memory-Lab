# Normally encoded prefix and subsequent updates

Complete frozen Qwen3.5-2B IT diagnostic: 8 fresh streams, 240 unique task questions, 8,160 primary readouts and all 170 conditions. Prefix encoding is held fixed before subsequent-update interventions; all questions share the same source memory.

All 259 tests and two opposite-order 510-readout operational smokes passed. The independent audit covers 680 source caches, 640 tail conditions, 320 additional no-op readouts, 240 legacy operational readouts per run, all 206 measurement sources and frozen weights/hooks. All raw readouts, source snapshots, registrations, logs, controls, 280 paired contrasts, 2,210 strata and 800 before/after strata are included.

Native current-value performance falls from 16/16 to 8/16 after 64 unrelated updates, with oracle reading still 16/16. Suppressing Delta or both gates gives 0/640 new-value control readouts under both readers, while oracle fixed-prefix evidence remains 1,600/1,600. These are repeated diagnostic cells over 8 streams. An existing GDN adapter already gets 48/48 on both unrelated tail lengths with normal updates. These results do not establish the need for a new joint architecture; tail KV, Query and convolution remain possible mediators.

The archive preserves raw readouts and cache fingerprints/layouts/check receipts, not 680 full cache tensor dumps. Recomputing those checks requires the matching model/backend. There is no new training, model, environment, significance claim or scaleup trigger; previous failed scientific criteria stay unchanged.

This is an incremental snapshot. Restore the [previous complete budget-control release](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/update-budget-controls-2026-11-15) and its predecessor chain, plus the earlier [fixed adapter checkpoints](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/matched-shared-training-timing-2026-11-15). Pretrained LLM weights, environments and compiler/download caches remain local.
