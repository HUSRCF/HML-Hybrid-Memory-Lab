#!/usr/bin/env bash
# Exact per-arm opposite-order controls precede all6912registered readouts.
set -euo pipefail
cd /home/husrcf/Code/SAS
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
SAS_QSF_ROOT=/media/990Pro/Code/SAS
SAS_QSF_ARTIFACTS="$SAS_QSF_ROOT/artifacts/memory_update"
SAS_QSF_LOGS="$SAS_QSF_ROOT/logs/memory_update"
SAS_QSF_FORWARD="$SAS_QSF_ROOT/runs/memory_update/question-state-freeze-smoke-forward-20261115-01"
SAS_QSF_REVERSE="$SAS_QSF_ROOT/runs/memory_update/question-state-freeze-smoke-reverse-20261115-01"
SAS_QSF_PRIMARY="$SAS_QSF_ROOT/runs/memory_update/question-state-freeze-primary-20261115-01"
SAS_QSF_EQ="$SAS_QSF_ARTIFACTS/question-state-freeze-smoke-equivalence-20261115-01.json"
for SAS_QSF_TARGET in "$SAS_QSF_FORWARD" "$SAS_QSF_REVERSE" "$SAS_QSF_PRIMARY" "$SAS_QSF_EQ" \
    "$SAS_QSF_ARTIFACTS/question-state-freeze-smoke-forward-audit-20261115-01.json" \
    "$SAS_QSF_ARTIFACTS/question-state-freeze-smoke-reverse-audit-20261115-01.json" \
    "$SAS_QSF_ARTIFACTS/question-state-freeze-primary-audit-20261115-01.json"; do
    if [[ -e "$SAS_QSF_TARGET" ]]; then
        echo "Preserve existing output: $SAS_QSF_TARGET" >&2
        exit 1
    fi
done
for SAS_QSF_SUFFIX in smoke-forward smoke-forward-audit smoke-reverse smoke-reverse-audit smoke-equivalence primary primary-audit; do
    if [[ -e "$SAS_QSF_LOGS/question-state-freeze-$SAS_QSF_SUFFIX-20261115-01.log" ]]; then
        echo "Preserve existing log: $SAS_QSF_SUFFIX" >&2
        exit 1
    fi
done
python -B scripts/run_question_state_freeze.py --smoke --output "$SAS_QSF_FORWARD" \
    > "$SAS_QSF_LOGS/question-state-freeze-smoke-forward-20261115-01.log" 2>&1
python -B scripts/analyze_question_state_freeze.py --run "$SAS_QSF_FORWARD" --allow-smoke \
    --output "$SAS_QSF_ARTIFACTS/question-state-freeze-smoke-forward-audit-20261115-01.json" \
    > "$SAS_QSF_LOGS/question-state-freeze-smoke-forward-audit-20261115-01.log" 2>&1
echo 'Forward432readouts and864archived references independently audited; reversing arm order.'
python -B scripts/run_question_state_freeze.py --smoke --reverse-arm-order --output "$SAS_QSF_REVERSE" \
    > "$SAS_QSF_LOGS/question-state-freeze-smoke-reverse-20261115-01.log" 2>&1
python -B scripts/analyze_question_state_freeze.py --run "$SAS_QSF_REVERSE" --allow-smoke \
    --output "$SAS_QSF_ARTIFACTS/question-state-freeze-smoke-reverse-audit-20261115-01.json" \
    > "$SAS_QSF_LOGS/question-state-freeze-smoke-reverse-audit-20261115-01.log" 2>&1
python -B scripts/analyze_question_state_freeze.py --forward "$SAS_QSF_FORWARD" --reverse "$SAS_QSF_REVERSE" --output "$SAS_QSF_EQ" \
    > "$SAS_QSF_LOGS/question-state-freeze-smoke-equivalence-20261115-01.log" 2>&1
echo 'All8per-arm raw files and normal caches exactly match across orders; starting6912readouts.'
python -B scripts/run_question_state_freeze.py --output "$SAS_QSF_PRIMARY" --smoke-equivalence "$SAS_QSF_EQ" \
    > "$SAS_QSF_LOGS/question-state-freeze-primary-20261115-01.log" 2>&1
python -B scripts/analyze_question_state_freeze.py --run "$SAS_QSF_PRIMARY" \
    --output "$SAS_QSF_ARTIFACTS/question-state-freeze-primary-audit-20261115-01.json" \
    > "$SAS_QSF_LOGS/question-state-freeze-primary-audit-20261115-01.log" 2>&1
echo 'All6912readouts,864normal replays and864archived references independently audited; no training.'
