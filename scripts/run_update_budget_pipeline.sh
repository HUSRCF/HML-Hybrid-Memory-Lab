#!/usr/bin/env bash
# Whole-record budget controls; all previous scientific sources remain sealed.
set -eo pipefail
cd /home/husrcf/Code/SAS
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
GATE_RUNS=/media/990Pro/Code/SAS/runs/memory_update
GATE_ARTIFACTS=/media/990Pro/Code/SAS/artifacts/memory_update
GATE_LOGS=/media/990Pro/Code/SAS/logs/memory_update
python scripts/run_update_budget.py --smoke --output "$GATE_RUNS/update-budget-smoke-forward-20261115-01" > "$GATE_LOGS/update-budget-smoke-forward-20261115-01.log" 2>&1
python scripts/analyze_update_budget.py "$GATE_RUNS/update-budget-smoke-forward-20261115-01" --allow-smoke --output "$GATE_ARTIFACTS/update-budget-smoke-forward-audit-20261115-01.json" > "$GATE_LOGS/update-budget-smoke-forward-audit-20261115-01.log" 2>&1
python scripts/run_update_budget.py --smoke --reverse-model-order --output "$GATE_RUNS/update-budget-smoke-reverse-20261115-01" > "$GATE_LOGS/update-budget-smoke-reverse-20261115-01.log" 2>&1
python scripts/analyze_update_budget.py "$GATE_RUNS/update-budget-smoke-reverse-20261115-01" --allow-smoke --output "$GATE_ARTIFACTS/update-budget-smoke-reverse-audit-20261115-01.json" > "$GATE_LOGS/update-budget-smoke-reverse-audit-20261115-01.log" 2>&1
python scripts/analyze_update_budget.py --equivalence "$GATE_ARTIFACTS/update-budget-smoke-forward-audit-20261115-01.json" "$GATE_ARTIFACTS/update-budget-smoke-reverse-audit-20261115-01.json" --output "$GATE_ARTIFACTS/update-budget-smoke-equivalence-20261115-01.json" > "$GATE_LOGS/update-budget-smoke-equivalence-20261115-01.log" 2>&1
python scripts/verify_update_budget_prefix.py --reference-audit "$GATE_ARTIFACTS/update-budget-smoke-forward-audit-20261115-01.json" --output "$GATE_ARTIFACTS/update-budget-prefix-verification-20261115-01.json" > "$GATE_LOGS/update-budget-prefix-verification-20261115-01.log" 2>&1
python scripts/run_update_budget.py --output "$GATE_RUNS/update-budget-controls-20261115-01" --smoke-equivalence "$GATE_ARTIFACTS/update-budget-smoke-equivalence-20261115-01.json" > "$GATE_LOGS/update-budget-primary-20261115-01.log" 2>&1
python scripts/analyze_update_budget.py "$GATE_RUNS/update-budget-controls-20261115-01" --output "$GATE_ARTIFACTS/update-budget-primary-audit-20261115-01.json" > "$GATE_LOGS/update-budget-primary-audit-20261115-01.log" 2>&1
echo 'update-budget pipeline completed; interpretation and publication still required'
