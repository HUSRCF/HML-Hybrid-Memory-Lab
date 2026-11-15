#!/usr/bin/env bash
# Wait for the existing measurement pipeline, then export every audited cell.
set -eo pipefail
cd /home/husrcf/Code/SAS
if [[ $# -ne 1 || ! $1 =~ ^[0-9]+$ ]]; then
  echo 'usage: bash scripts/run_update_budget_exports.sh EXISTING_PIPELINE_PID' >&2
  exit 2
fi
BUDGET_PIPELINE_PID=$1
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
BUDGET_ARTIFACTS=/media/990Pro/Code/SAS/artifacts/memory_update
BUDGET_LOGS=/media/990Pro/Code/SAS/logs/memory_update
while kill -0 "$BUDGET_PIPELINE_PID" 2>/dev/null; do
  sleep 15
done
python - <<'PY'
from pathlib import Path
import json
p=Path('/media/990Pro/Code/SAS/artifacts/memory_update/update-budget-primary-audit-20261115-01.json')
a=json.loads(p.read_text())
if a['status']!='completed_and_audited_update_budget' or a['new_physical_readouts_audited']!=13440:
    raise ValueError('Existing pipeline did not complete the full audited measurement; preserve logs and stop')
PY
python scripts/export_update_budget.py --audit "$BUDGET_ARTIFACTS/update-budget-primary-audit-20261115-01.json" --prefix "$BUDGET_ARTIFACTS/update-budget-figure-20261115-01" > "$BUDGET_LOGS/update-budget-export-20261115-01.log" 2>&1
python scripts/report_update_budget.py --audit "$BUDGET_ARTIFACTS/update-budget-primary-audit-20261115-01.json" --exported "$BUDGET_ARTIFACTS/update-budget-figure-20261115-01.data.json" --output "$BUDGET_ARTIFACTS/update-budget-report-20261115-01" > "$BUDGET_LOGS/update-budget-report-20261115-01.log" 2>&1
echo 'all audited budget results exported; visual review and interpretation still required'
