#!/usr/bin/env bash
# Operational orchestration; measured code lives in the230source sealed packet.
set -euo pipefail
cd /home/husrcf/Code/SAS
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
SAS_LAYER_STORAGE=/media/990Pro/Code/SAS
SAS_LAYER_ARTIFACTS="$SAS_LAYER_STORAGE/artifacts/memory_update"
SAS_LAYER_LOGS="$SAS_LAYER_STORAGE/logs/memory_update"
SAS_LAYER_PRIMARY="$SAS_LAYER_STORAGE/runs/memory_update/query-layer-primary-20261115-01"
SAS_LAYER_EQUIVALENCE="$SAS_LAYER_ARTIFACTS/query-layer-smoke-equivalence-20261115-01.json"
for SAS_LAYER_TARGET in "$SAS_LAYER_PRIMARY" "$SAS_LAYER_EQUIVALENCE" \
    "$SAS_LAYER_ARTIFACTS/query-layer-primary-audit-20261115-01.json" \
    "$SAS_LAYER_ARTIFACTS/query-layer-smoke-forward-audit-20261115-01.json" \
    "$SAS_LAYER_ARTIFACTS/query-layer-smoke-reverse-audit-20261115-01.json" \
    "$SAS_LAYER_STORAGE/runs/memory_update/query-layer-smoke-forward-20261115-01" \
    "$SAS_LAYER_STORAGE/runs/memory_update/query-layer-smoke-reverse-20261115-01"; do
    if [[ -e "$SAS_LAYER_TARGET" ]]; then
        echo "Preserve existing output: $SAS_LAYER_TARGET" >&2
        exit 1
    fi
done
for SAS_LAYER_DIRECTION in forward reverse; do
    SAS_LAYER_ORDER_ARGS=()
    if [[ "$SAS_LAYER_DIRECTION" == reverse ]]; then SAS_LAYER_ORDER_ARGS=(--reverse-model-order); fi
    SAS_LAYER_RUN="$SAS_LAYER_STORAGE/runs/memory_update/query-layer-smoke-$SAS_LAYER_DIRECTION-20261115-01"
    SAS_LAYER_AUDIT="$SAS_LAYER_ARTIFACTS/query-layer-smoke-$SAS_LAYER_DIRECTION-audit-20261115-01.json"
    python -B scripts/run_query_layer.py --smoke "${SAS_LAYER_ORDER_ARGS[@]}" --output "$SAS_LAYER_RUN" \
        > "$SAS_LAYER_LOGS/query-layer-smoke-$SAS_LAYER_DIRECTION-20261115-01.log" 2>&1
    python -B scripts/analyze_query_layer.py "$SAS_LAYER_RUN" --allow-smoke --output "$SAS_LAYER_AUDIT" \
        > "$SAS_LAYER_LOGS/query-layer-smoke-$SAS_LAYER_DIRECTION-audit-20261115-01.log" 2>&1
    echo "Independently audited $SAS_LAYER_DIRECTION single-layer operational smoke."
done
python -B scripts/analyze_query_layer.py --equivalence \
    "$SAS_LAYER_ARTIFACTS/query-layer-smoke-forward-audit-20261115-01.json" \
    "$SAS_LAYER_ARTIFACTS/query-layer-smoke-reverse-audit-20261115-01.json" \
    --output "$SAS_LAYER_EQUIVALENCE" > "$SAS_LAYER_LOGS/query-layer-smoke-equivalence-20261115-01.log" 2>&1
echo 'Exact opposite-order equivalence passed; starting4608registered single-layer readouts.'
python -B scripts/run_query_layer.py --output "$SAS_LAYER_PRIMARY" --smoke-equivalence "$SAS_LAYER_EQUIVALENCE" \
    > "$SAS_LAYER_LOGS/query-layer-primary-20261115-01.log" 2>&1
python -B scripts/analyze_query_layer.py "$SAS_LAYER_PRIMARY" \
    --output "$SAS_LAYER_ARTIFACTS/query-layer-primary-audit-20261115-01.json" \
    > "$SAS_LAYER_LOGS/query-layer-primary-audit-20261115-01.log" 2>&1
echo 'All4608new readouts,1536reference replays,384single-layer self controls and768captures independently audited.'
