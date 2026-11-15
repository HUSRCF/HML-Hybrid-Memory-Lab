#!/usr/bin/env bash
# Resume after audit-only correction; preserve the exact completed forward run.
set -euo pipefail
cd /home/husrcf/Code/SAS
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
SAS_LAYER_STORAGE=/media/990Pro/Code/SAS
SAS_LAYER_ARTIFACTS="$SAS_LAYER_STORAGE/artifacts/memory_update"
SAS_LAYER_LOGS="$SAS_LAYER_STORAGE/logs/memory_update"
SAS_LAYER_FORWARD="$SAS_LAYER_STORAGE/runs/memory_update/query-layer-smoke-forward-20261115-01"
SAS_LAYER_REVERSE="$SAS_LAYER_STORAGE/runs/memory_update/query-layer-smoke-reverse-20261115-01"
SAS_LAYER_PRIMARY="$SAS_LAYER_STORAGE/runs/memory_update/query-layer-primary-20261115-01"
SAS_LAYER_FORWARD_AUDIT="$SAS_LAYER_ARTIFACTS/query-layer-smoke-forward-audit-corrected-20261115-01.json"
SAS_LAYER_REVERSE_AUDIT="$SAS_LAYER_ARTIFACTS/query-layer-smoke-reverse-audit-corrected-20261115-01.json"
SAS_LAYER_EQUIVALENCE="$SAS_LAYER_ARTIFACTS/query-layer-smoke-equivalence-corrected-20261115-01.json"
SAS_LAYER_PRIMARY_AUDIT="$SAS_LAYER_ARTIFACTS/query-layer-primary-audit-corrected-20261115-01.json"
for SAS_LAYER_TARGET in "$SAS_LAYER_REVERSE" "$SAS_LAYER_PRIMARY" "$SAS_LAYER_FORWARD_AUDIT" \
    "$SAS_LAYER_REVERSE_AUDIT" "$SAS_LAYER_EQUIVALENCE" "$SAS_LAYER_PRIMARY_AUDIT"; do
    if [[ -e "$SAS_LAYER_TARGET" ]]; then
        echo "Preserve existing output: $SAS_LAYER_TARGET" >&2
        exit 1
    fi
done
python -B scripts/analyze_query_layer_corrected.py "$SAS_LAYER_FORWARD" --allow-smoke --output "$SAS_LAYER_FORWARD_AUDIT" \
    > "$SAS_LAYER_LOGS/query-layer-smoke-forward-audit-corrected-20261115-01.log" 2>&1
echo 'Existing forward288readouts independently audited; no model rerun.'
python -B scripts/run_query_layer_corrected.py --smoke --reverse-model-order --output "$SAS_LAYER_REVERSE" \
    > "$SAS_LAYER_LOGS/query-layer-smoke-reverse-20261115-01.log" 2>&1
python -B scripts/analyze_query_layer_corrected.py "$SAS_LAYER_REVERSE" --allow-smoke --output "$SAS_LAYER_REVERSE_AUDIT" \
    > "$SAS_LAYER_LOGS/query-layer-smoke-reverse-audit-corrected-20261115-01.log" 2>&1
python -B scripts/analyze_query_layer_corrected.py --equivalence "$SAS_LAYER_FORWARD_AUDIT" "$SAS_LAYER_REVERSE_AUDIT" \
    --output "$SAS_LAYER_EQUIVALENCE" > "$SAS_LAYER_LOGS/query-layer-smoke-equivalence-corrected-20261115-01.log" 2>&1
echo 'Exact opposite-order equivalence passed; starting4608registered single-layer readouts.'
python -B scripts/run_query_layer_corrected.py --output "$SAS_LAYER_PRIMARY" --smoke-equivalence "$SAS_LAYER_EQUIVALENCE" \
    > "$SAS_LAYER_LOGS/query-layer-primary-20261115-01.log" 2>&1
python -B scripts/analyze_query_layer_corrected.py "$SAS_LAYER_PRIMARY" --output "$SAS_LAYER_PRIMARY_AUDIT" \
    > "$SAS_LAYER_LOGS/query-layer-primary-audit-corrected-20261115-01.log" 2>&1
echo 'All4608new readouts,1536reference replays,384single-layer self controls and768captures independently audited.'
