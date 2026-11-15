#!/usr/bin/env bash
# All registered cells run only after exact, independently audited opposite orders.
set -euo pipefail
cd /home/husrcf/Code/SAS
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
SAS_NATURAL_ROOT=/media/990Pro/Code/SAS
SAS_NATURAL_ARTIFACTS="$SAS_NATURAL_ROOT/artifacts/memory_update"
SAS_NATURAL_LOGS="$SAS_NATURAL_ROOT/logs/memory_update"
SAS_NATURAL_FORWARD="$SAS_NATURAL_ROOT/runs/memory_update/natural-load-smoke-forward-20261115-01"
SAS_NATURAL_REVERSE="$SAS_NATURAL_ROOT/runs/memory_update/natural-load-smoke-reverse-20261115-01"
SAS_NATURAL_PRIMARY="$SAS_NATURAL_ROOT/runs/memory_update/natural-load-primary-20261115-01"
SAS_NATURAL_EQ="$SAS_NATURAL_ARTIFACTS/natural-load-smoke-equivalence-20261115-01.json"
for SAS_NATURAL_TARGET in "$SAS_NATURAL_FORWARD" "$SAS_NATURAL_REVERSE" "$SAS_NATURAL_PRIMARY" "$SAS_NATURAL_EQ" \
    "$SAS_NATURAL_ARTIFACTS/natural-load-smoke-forward-audit-20261115-01.json" \
    "$SAS_NATURAL_ARTIFACTS/natural-load-smoke-reverse-audit-20261115-01.json" \
    "$SAS_NATURAL_ARTIFACTS/natural-load-primary-audit-20261115-01.json"; do
    if [[ -e "$SAS_NATURAL_TARGET" ]]; then
        echo "Preserve existing output: $SAS_NATURAL_TARGET" >&2
        exit 1
    fi
done
python -B scripts/run_natural_load.py --smoke --output "$SAS_NATURAL_FORWARD" \
    > "$SAS_NATURAL_LOGS/natural-load-smoke-forward-20261115-01.log" 2>&1
python -B scripts/analyze_natural_load.py --run "$SAS_NATURAL_FORWARD" --allow-smoke \
    --output "$SAS_NATURAL_ARTIFACTS/natural-load-smoke-forward-audit-20261115-01.json" \
    > "$SAS_NATURAL_LOGS/natural-load-smoke-forward-audit-20261115-01.log" 2>&1
echo 'Forward108readouts independently audited; starting reverse order.'
python -B scripts/run_natural_load.py --smoke --reverse-model-order --output "$SAS_NATURAL_REVERSE" \
    > "$SAS_NATURAL_LOGS/natural-load-smoke-reverse-20261115-01.log" 2>&1
python -B scripts/analyze_natural_load.py --run "$SAS_NATURAL_REVERSE" --allow-smoke \
    --output "$SAS_NATURAL_ARTIFACTS/natural-load-smoke-reverse-audit-20261115-01.json" \
    > "$SAS_NATURAL_LOGS/natural-load-smoke-reverse-audit-20261115-01.log" 2>&1
python -B scripts/analyze_natural_load.py --forward "$SAS_NATURAL_FORWARD" --reverse "$SAS_NATURAL_REVERSE" --output "$SAS_NATURAL_EQ" \
    > "$SAS_NATURAL_LOGS/natural-load-smoke-equivalence-20261115-01.log" 2>&1
echo 'Exact opposite-order equivalence passed; starting all3456registered readouts.'
python -B scripts/run_natural_load.py --output "$SAS_NATURAL_PRIMARY" --smoke-equivalence "$SAS_NATURAL_EQ" \
    > "$SAS_NATURAL_LOGS/natural-load-primary-20261115-01.log" 2>&1
python -B scripts/analyze_natural_load.py --run "$SAS_NATURAL_PRIMARY" \
    --output "$SAS_NATURAL_ARTIFACTS/natural-load-primary-audit-20261115-01.json" \
    > "$SAS_NATURAL_LOGS/natural-load-primary-audit-20261115-01.log" 2>&1
echo 'All3456normal-update readouts independently audited; no new training.'
