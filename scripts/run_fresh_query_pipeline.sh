#!/usr/bin/env bash
# Operational orchestration; all measurement sources are sealed in the222file packet.
set -euo pipefail
cd /home/husrcf/Code/SAS
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
SAS_QUERY_STORAGE=/media/990Pro/Code/SAS
SAS_QUERY_ARTIFACTS="$SAS_QUERY_STORAGE/artifacts/memory_update"
SAS_QUERY_LOGS="$SAS_QUERY_STORAGE/logs/memory_update"
SAS_QUERY_PRIMARY="$SAS_QUERY_STORAGE/runs/memory_update/fresh-query-primary-20261115-01"
SAS_QUERY_FORWARD_AUDIT="$SAS_QUERY_ARTIFACTS/fresh-query-smoke-forward-audit-20261115-01.json"
SAS_QUERY_REVERSE_AUDIT="$SAS_QUERY_ARTIFACTS/fresh-query-smoke-reverse-audit-20261115-01.json"
SAS_QUERY_EQUIVALENCE="$SAS_QUERY_ARTIFACTS/fresh-query-smoke-equivalence-20261115-01.json"
SAS_QUERY_PRIMARY_AUDIT="$SAS_QUERY_ARTIFACTS/fresh-query-primary-audit-20261115-01.json"
for SAS_QUERY_TARGET in "$SAS_QUERY_PRIMARY" "$SAS_QUERY_FORWARD_AUDIT" "$SAS_QUERY_REVERSE_AUDIT" \
    "$SAS_QUERY_EQUIVALENCE" "$SAS_QUERY_PRIMARY_AUDIT" \
    "$SAS_QUERY_STORAGE/runs/memory_update/fresh-query-smoke-forward-20261115-01" \
    "$SAS_QUERY_STORAGE/runs/memory_update/fresh-query-smoke-reverse-20261115-01"; do
    if [[ -e "$SAS_QUERY_TARGET" ]]; then
        echo "Preserve existing output: $SAS_QUERY_TARGET" >&2
        exit 1
    fi
done
for SAS_QUERY_DIRECTION in forward reverse; do
    SAS_QUERY_REVERSE_ARGS=()
    if [[ "$SAS_QUERY_DIRECTION" == reverse ]]; then SAS_QUERY_REVERSE_ARGS=(--reverse-model-order); fi
    SAS_QUERY_RUN="$SAS_QUERY_STORAGE/runs/memory_update/fresh-query-smoke-$SAS_QUERY_DIRECTION-20261115-01"
    SAS_QUERY_AUDIT="$SAS_QUERY_ARTIFACTS/fresh-query-smoke-$SAS_QUERY_DIRECTION-audit-20261115-01.json"
    python -B scripts/run_fresh_query.py --smoke "${SAS_QUERY_REVERSE_ARGS[@]}" --output "$SAS_QUERY_RUN" \
        > "$SAS_QUERY_LOGS/fresh-query-smoke-$SAS_QUERY_DIRECTION-20261115-01.log" 2>&1
    python -B scripts/analyze_fresh_query.py "$SAS_QUERY_RUN" --allow-smoke --output "$SAS_QUERY_AUDIT" \
        > "$SAS_QUERY_LOGS/fresh-query-smoke-$SAS_QUERY_DIRECTION-audit-20261115-01.log" 2>&1
    echo "Audited $SAS_QUERY_DIRECTION operational smoke."
done
python -B scripts/analyze_fresh_query.py --equivalence "$SAS_QUERY_FORWARD_AUDIT" "$SAS_QUERY_REVERSE_AUDIT" \
    --output "$SAS_QUERY_EQUIVALENCE" > "$SAS_QUERY_LOGS/fresh-query-smoke-equivalence-20261115-01.log" 2>&1
echo 'Exact opposite-order equivalence passed; starting3264registered fresh readouts.'
python -B scripts/run_fresh_query.py --output "$SAS_QUERY_PRIMARY" --smoke-equivalence "$SAS_QUERY_EQUIVALENCE" \
    > "$SAS_QUERY_LOGS/fresh-query-primary-20261115-01.log" 2>&1
python -B scripts/analyze_fresh_query.py "$SAS_QUERY_PRIMARY" --output "$SAS_QUERY_PRIMARY_AUDIT" \
    > "$SAS_QUERY_LOGS/fresh-query-primary-audit-20261115-01.log" 2>&1
echo 'All3264fresh readouts,1536same-source replays and1536question-only captures completed and independently audited.'
