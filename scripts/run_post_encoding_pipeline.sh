#!/usr/bin/env bash
# Operational driver only; scientific sources are sealed in the registered206files.
set -euo pipefail
cd /home/husrcf/Code/SAS
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
SAS_POST_STORAGE=/media/990Pro/Code/SAS
SAS_POST_ARTIFACTS="$SAS_POST_STORAGE/artifacts/memory_update"
SAS_POST_LOGS="$SAS_POST_STORAGE/logs/memory_update"
SAS_POST_FORWARD="$SAS_POST_STORAGE/runs/memory_update/post-encoding-smoke-forward-20261115-01"
SAS_POST_REVERSE="$SAS_POST_STORAGE/runs/memory_update/post-encoding-smoke-reverse-20261115-01"
SAS_POST_PRIMARY="$SAS_POST_STORAGE/runs/memory_update/post-encoding-updates-20261115-01"
SAS_POST_FORWARD_AUDIT="$SAS_POST_ARTIFACTS/post-encoding-smoke-forward-audit-20261115-01.json"
SAS_POST_REVERSE_AUDIT="$SAS_POST_ARTIFACTS/post-encoding-smoke-reverse-audit-20261115-01.json"
SAS_POST_EQUIVALENCE="$SAS_POST_ARTIFACTS/post-encoding-smoke-equivalence-20261115-01.json"
for SAS_POST_TARGET in "$SAS_POST_FORWARD" "$SAS_POST_REVERSE" "$SAS_POST_PRIMARY" \
    "$SAS_POST_FORWARD_AUDIT" "$SAS_POST_REVERSE_AUDIT" "$SAS_POST_EQUIVALENCE"; do
    if [[ -e "$SAS_POST_TARGET" ]]; then
        echo "Preserve existing output: $SAS_POST_TARGET" >&2
        exit 1
    fi
done
for SAS_POST_DIRECTION in forward reverse; do
    SAS_POST_REVERSE_ARGS=()
    if [[ "$SAS_POST_DIRECTION" == reverse ]]; then SAS_POST_REVERSE_ARGS=(--reverse-model-order); fi
    SAS_POST_RUN="$SAS_POST_STORAGE/runs/memory_update/post-encoding-smoke-$SAS_POST_DIRECTION-20261115-01"
    SAS_POST_AUDIT="$SAS_POST_ARTIFACTS/post-encoding-smoke-$SAS_POST_DIRECTION-audit-20261115-01.json"
    python -B scripts/run_post_encoding.py --smoke "${SAS_POST_REVERSE_ARGS[@]}" --output "$SAS_POST_RUN" \
        > "$SAS_POST_LOGS/post-encoding-smoke-$SAS_POST_DIRECTION-20261115-01.log" 2>&1
    python -B scripts/analyze_post_encoding.py "$SAS_POST_RUN" --allow-smoke --output "$SAS_POST_AUDIT" \
        > "$SAS_POST_LOGS/post-encoding-smoke-$SAS_POST_DIRECTION-audit-20261115-01.log" 2>&1
    echo "Audited $SAS_POST_DIRECTION operational smoke."
done
python -B scripts/analyze_post_encoding.py --equivalence "$SAS_POST_FORWARD_AUDIT" "$SAS_POST_REVERSE_AUDIT" \
    --output "$SAS_POST_EQUIVALENCE" > "$SAS_POST_LOGS/post-encoding-smoke-equivalence-20261115-01.log" 2>&1
echo 'Exact opposite-order smoke equivalence passed; starting the registered primary.'
python -B scripts/run_post_encoding.py --output "$SAS_POST_PRIMARY" --smoke-equivalence "$SAS_POST_EQUIVALENCE" \
    > "$SAS_POST_LOGS/post-encoding-updates-20261115-01.log" 2>&1
python -B scripts/analyze_post_encoding.py "$SAS_POST_PRIMARY" \
    --output "$SAS_POST_ARTIFACTS/post-encoding-primary-audit-20261115-01.json" \
    > "$SAS_POST_LOGS/post-encoding-primary-audit-20261115-01.log" 2>&1
echo 'All8160primary readouts completed and independently audited.'
