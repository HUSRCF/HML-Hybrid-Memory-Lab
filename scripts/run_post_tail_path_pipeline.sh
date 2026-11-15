#!/usr/bin/env bash
# Operational orchestration; all measurement sources are sealed in the214file packet.
set -euo pipefail
cd /home/husrcf/Code/SAS
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
SAS_CROSS_STORAGE=/media/990Pro/Code/SAS
SAS_CROSS_ARTIFACTS="$SAS_CROSS_STORAGE/artifacts/memory_update"
SAS_CROSS_LOGS="$SAS_CROSS_STORAGE/logs/memory_update"
SAS_CROSS_PRIMARY="$SAS_CROSS_STORAGE/runs/memory_update/post-tail-path-primary-20261115-01"
SAS_CROSS_FORWARD_AUDIT="$SAS_CROSS_ARTIFACTS/post-tail-path-smoke-forward-audit-20261115-01.json"
SAS_CROSS_REVERSE_AUDIT="$SAS_CROSS_ARTIFACTS/post-tail-path-smoke-reverse-audit-20261115-01.json"
SAS_CROSS_EQUIVALENCE="$SAS_CROSS_ARTIFACTS/post-tail-path-smoke-equivalence-20261115-01.json"
SAS_CROSS_PRIMARY_AUDIT="$SAS_CROSS_ARTIFACTS/post-tail-path-primary-audit-20261115-01.json"
for SAS_CROSS_TARGET in "$SAS_CROSS_PRIMARY" "$SAS_CROSS_FORWARD_AUDIT" "$SAS_CROSS_REVERSE_AUDIT" \
    "$SAS_CROSS_EQUIVALENCE" "$SAS_CROSS_PRIMARY_AUDIT" \
    "$SAS_CROSS_STORAGE/runs/memory_update/post-tail-path-smoke-forward-20261115-01" \
    "$SAS_CROSS_STORAGE/runs/memory_update/post-tail-path-smoke-reverse-20261115-01"; do
    if [[ -e "$SAS_CROSS_TARGET" ]]; then
        echo "Preserve existing output: $SAS_CROSS_TARGET" >&2
        exit 1
    fi
done
for SAS_CROSS_DIRECTION in forward reverse; do
    SAS_CROSS_REVERSE_ARGS=()
    if [[ "$SAS_CROSS_DIRECTION" == reverse ]]; then SAS_CROSS_REVERSE_ARGS=(--reverse-model-order); fi
    SAS_CROSS_RUN="$SAS_CROSS_STORAGE/runs/memory_update/post-tail-path-smoke-$SAS_CROSS_DIRECTION-20261115-01"
    SAS_CROSS_AUDIT="$SAS_CROSS_ARTIFACTS/post-tail-path-smoke-$SAS_CROSS_DIRECTION-audit-20261115-01.json"
    python -B scripts/run_post_tail_path.py --smoke "${SAS_CROSS_REVERSE_ARGS[@]}" --output "$SAS_CROSS_RUN" \
        > "$SAS_CROSS_LOGS/post-tail-path-smoke-$SAS_CROSS_DIRECTION-20261115-01.log" 2>&1
    python -B scripts/analyze_post_tail_path.py "$SAS_CROSS_RUN" --allow-smoke --output "$SAS_CROSS_AUDIT" \
        > "$SAS_CROSS_LOGS/post-tail-path-smoke-$SAS_CROSS_DIRECTION-audit-20261115-01.log" 2>&1
    echo "Audited $SAS_CROSS_DIRECTION operational smoke."
done
python -B scripts/analyze_post_tail_path.py --equivalence "$SAS_CROSS_FORWARD_AUDIT" "$SAS_CROSS_REVERSE_AUDIT" \
    --output "$SAS_CROSS_EQUIVALENCE" > "$SAS_CROSS_LOGS/post-tail-path-smoke-equivalence-20261115-01.log" 2>&1
echo 'Exact opposite-order equivalence passed; starting4608registered new readouts.'
python -B scripts/run_post_tail_path.py --output "$SAS_CROSS_PRIMARY" --smoke-equivalence "$SAS_CROSS_EQUIVALENCE" \
    > "$SAS_CROSS_LOGS/post-tail-path-primary-20261115-01.log" 2>&1
python -B scripts/analyze_post_tail_path.py "$SAS_CROSS_PRIMARY" --output "$SAS_CROSS_PRIMARY_AUDIT" \
    > "$SAS_CROSS_LOGS/post-tail-path-primary-audit-20261115-01.log" 2>&1
echo 'All4608new readouts and1536endpoint replays completed and independently audited.'
