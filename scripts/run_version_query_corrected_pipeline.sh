#!/usr/bin/env bash
# Both exact model orders precede all1728registered new Full readouts.
set -euo pipefail
cd /home/husrcf/Code/SAS
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
SAS_VERSION_ROOT=/media/990Pro/Code/SAS
SAS_VERSION_ARTIFACTS="$SAS_VERSION_ROOT/artifacts/memory_update"
SAS_VERSION_LOGS="$SAS_VERSION_ROOT/logs/memory_update"
SAS_VERSION_FORWARD="$SAS_VERSION_ROOT/runs/memory_update/version-query-smoke-forward-corrected-20261115-01"
SAS_VERSION_REVERSE="$SAS_VERSION_ROOT/runs/memory_update/version-query-smoke-reverse-corrected-20261115-01"
SAS_VERSION_PRIMARY="$SAS_VERSION_ROOT/runs/memory_update/version-query-primary-corrected-20261115-01"
SAS_VERSION_EQ="$SAS_VERSION_ARTIFACTS/version-query-smoke-equivalence-corrected-20261115-01.json"
for SAS_VERSION_TARGET in "$SAS_VERSION_FORWARD" "$SAS_VERSION_REVERSE" "$SAS_VERSION_PRIMARY" "$SAS_VERSION_EQ" \
    "$SAS_VERSION_ARTIFACTS/version-query-smoke-forward-audit-corrected-20261115-01.json" \
    "$SAS_VERSION_ARTIFACTS/version-query-smoke-reverse-audit-corrected-20261115-01.json" \
    "$SAS_VERSION_ARTIFACTS/version-query-primary-audit-corrected-20261115-01.json"; do
    if [[ -e "$SAS_VERSION_TARGET" ]]; then
        echo "Preserve existing output: $SAS_VERSION_TARGET" >&2
        exit 1
    fi
done
for SAS_VERSION_SUFFIX in smoke-forward smoke-forward-audit smoke-reverse smoke-reverse-audit smoke-equivalence primary primary-audit; do
    if [[ -e "$SAS_VERSION_LOGS/version-query-$SAS_VERSION_SUFFIX-corrected-20261115-01.log" ]]; then
        echo "Preserve existing log: $SAS_VERSION_SUFFIX" >&2
        exit 1
    fi
done
python -B scripts/run_version_query_corrected.py --smoke --output "$SAS_VERSION_FORWARD" \
    > "$SAS_VERSION_LOGS/version-query-smoke-forward-corrected-20261115-01.log" 2>&1
python -B scripts/analyze_version_query_corrected.py --run "$SAS_VERSION_FORWARD" --allow-smoke \
    --output "$SAS_VERSION_ARTIFACTS/version-query-smoke-forward-audit-corrected-20261115-01.json" \
    > "$SAS_VERSION_LOGS/version-query-smoke-forward-audit-corrected-20261115-01.log" 2>&1
echo 'Forward108new readouts and1728archived references independently audited; starting reverse order.'
python -B scripts/run_version_query_corrected.py --smoke --reverse-model-order --output "$SAS_VERSION_REVERSE" \
    > "$SAS_VERSION_LOGS/version-query-smoke-reverse-corrected-20261115-01.log" 2>&1
python -B scripts/analyze_version_query_corrected.py --run "$SAS_VERSION_REVERSE" --allow-smoke \
    --output "$SAS_VERSION_ARTIFACTS/version-query-smoke-reverse-audit-corrected-20261115-01.json" \
    > "$SAS_VERSION_LOGS/version-query-smoke-reverse-audit-corrected-20261115-01.log" 2>&1
python -B scripts/analyze_version_query_corrected.py --forward "$SAS_VERSION_FORWARD" --reverse "$SAS_VERSION_REVERSE" --output "$SAS_VERSION_EQ" \
    > "$SAS_VERSION_LOGS/version-query-smoke-equivalence-corrected-20261115-01.log" 2>&1
echo 'Exact opposite-order equivalence passed; starting all1728registered paired-question readouts.'
python -B scripts/run_version_query_corrected.py --output "$SAS_VERSION_PRIMARY" --smoke-equivalence "$SAS_VERSION_EQ" \
    > "$SAS_VERSION_LOGS/version-query-primary-corrected-20261115-01.log" 2>&1
python -B scripts/analyze_version_query_corrected.py --run "$SAS_VERSION_PRIMARY" \
    --output "$SAS_VERSION_ARTIFACTS/version-query-primary-audit-corrected-20261115-01.json" \
    > "$SAS_VERSION_LOGS/version-query-primary-audit-corrected-20261115-01.log" 2>&1
echo 'All1728newreadouts and1728labelled archived references independently audited; no new training.'
