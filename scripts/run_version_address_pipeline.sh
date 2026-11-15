#!/usr/bin/env bash
# Two exact passive216-readout model-order controls precede3456normal readouts.
set -euo pipefail
cd /home/husrcf/Code/SAS
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
SAS_VA_ROOT=/media/990Pro/Code/SAS
SAS_VA_ARTIFACTS="$SAS_VA_ROOT/artifacts/memory_update"
SAS_VA_LOGS="$SAS_VA_ROOT/logs/memory_update"
SAS_VA_FORWARD="$SAS_VA_ROOT/runs/memory_update/version-address-smoke-forward-20261115-01"
SAS_VA_REVERSE="$SAS_VA_ROOT/runs/memory_update/version-address-smoke-reverse-20261115-01"
SAS_VA_PRIMARY="$SAS_VA_ROOT/runs/memory_update/version-address-primary-20261115-01"
SAS_VA_EQ="$SAS_VA_ARTIFACTS/version-address-smoke-equivalence-20261115-01.json"
for SAS_VA_TARGET in "$SAS_VA_FORWARD" "$SAS_VA_REVERSE" "$SAS_VA_PRIMARY" "$SAS_VA_EQ" \
    "$SAS_VA_ARTIFACTS/version-address-smoke-forward-audit-20261115-01.json" \
    "$SAS_VA_ARTIFACTS/version-address-smoke-reverse-audit-20261115-01.json" \
    "$SAS_VA_ARTIFACTS/version-address-primary-audit-20261115-01.json"; do
    if [[ -e "$SAS_VA_TARGET" ]]; then
        echo "Preserve existing output: $SAS_VA_TARGET" >&2
        exit 1
    fi
done
for SAS_VA_SUFFIX in smoke-forward smoke-forward-audit smoke-reverse smoke-reverse-audit smoke-equivalence primary primary-audit; do
    if [[ -e "$SAS_VA_LOGS/version-address-$SAS_VA_SUFFIX-20261115-01.log" ]]; then
        echo "Preserve existing log: $SAS_VA_SUFFIX" >&2
        exit 1
    fi
done
python -B scripts/run_version_address.py --smoke --output "$SAS_VA_FORWARD" \
    > "$SAS_VA_LOGS/version-address-smoke-forward-20261115-01.log" 2>&1
python -B scripts/analyze_version_address.py --run "$SAS_VA_FORWARD" --allow-smoke \
    --output "$SAS_VA_ARTIFACTS/version-address-smoke-forward-audit-20261115-01.json" \
    > "$SAS_VA_LOGS/version-address-smoke-forward-audit-20261115-01.log" 2>&1
echo 'Forward216readouts and raw FP32 record masses independently audited; reverse model order.'
python -B scripts/run_version_address.py --smoke --reverse-model-order --output "$SAS_VA_REVERSE" \
    > "$SAS_VA_LOGS/version-address-smoke-reverse-20261115-01.log" 2>&1
python -B scripts/analyze_version_address.py --run "$SAS_VA_REVERSE" --allow-smoke \
    --output "$SAS_VA_ARTIFACTS/version-address-smoke-reverse-audit-20261115-01.json" \
    > "$SAS_VA_LOGS/version-address-smoke-reverse-audit-20261115-01.log" 2>&1
python -B scripts/analyze_version_address.py --forward "$SAS_VA_FORWARD" --reverse "$SAS_VA_REVERSE" \
    --output "$SAS_VA_EQ" > "$SAS_VA_LOGS/version-address-smoke-equivalence-20261115-01.log" 2>&1
echo 'All36raw FP32 files and both model readouts exactly match across orders; start3456readouts.'
python -B scripts/run_version_address.py --output "$SAS_VA_PRIMARY" --smoke-equivalence "$SAS_VA_EQ" \
    > "$SAS_VA_LOGS/version-address-primary-20261115-01.log" 2>&1
python -B scripts/analyze_version_address.py --run "$SAS_VA_PRIMARY" \
    --output "$SAS_VA_ARTIFACTS/version-address-primary-audit-20261115-01.json" \
    > "$SAS_VA_LOGS/version-address-primary-audit-20261115-01.log" 2>&1
echo 'All3456readouts,1728old exact replays and all576raw FP32 files independently audited.'
