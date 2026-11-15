#!/usr/bin/env bash
# New operational runs after preserving failed strict-ideal smoke 01.
set -eo pipefail
cd /home/husrcf/Code/SAS
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
GATE_RUNS=/media/990Pro/Code/SAS/runs/memory_update
GATE_ARTIFACTS=/media/990Pro/Code/SAS/artifacts/memory_update
GATE_LOGS=/media/990Pro/Code/SAS/logs/memory_update
python scripts/run_gate_components.py --smoke --output "$GATE_RUNS/gate-components-smoke-forward-20261115-02" > "$GATE_LOGS/gate-components-smoke-forward-20261115-02.log" 2>&1
python scripts/analyze_gate_components.py "$GATE_RUNS/gate-components-smoke-forward-20261115-02" --allow-smoke --output "$GATE_ARTIFACTS/gate-components-smoke-forward-audit-20261115-01.json" > "$GATE_LOGS/gate-components-smoke-forward-audit-20261115-01.log" 2>&1
python scripts/run_gate_components.py --smoke --reverse-model-order --output "$GATE_RUNS/gate-components-smoke-reverse-20261115-02" > "$GATE_LOGS/gate-components-smoke-reverse-20261115-02.log" 2>&1
python scripts/analyze_gate_components.py "$GATE_RUNS/gate-components-smoke-reverse-20261115-02" --allow-smoke --output "$GATE_ARTIFACTS/gate-components-smoke-reverse-audit-20261115-01.json" > "$GATE_LOGS/gate-components-smoke-reverse-audit-20261115-01.log" 2>&1
python scripts/analyze_gate_components.py --equivalence "$GATE_ARTIFACTS/gate-components-smoke-forward-audit-20261115-01.json" "$GATE_ARTIFACTS/gate-components-smoke-reverse-audit-20261115-01.json" --output "$GATE_ARTIFACTS/gate-components-smoke-equivalence-20261115-01.json" > "$GATE_LOGS/gate-components-smoke-equivalence-20261115-01.log" 2>&1
python scripts/verify_gate_component_prefix.py --reference-audit "$GATE_ARTIFACTS/gate-components-smoke-forward-audit-20261115-01.json" --output "$GATE_ARTIFACTS/gate-components-prefix-verification-20261115-01.json" > "$GATE_LOGS/gate-components-prefix-verification-20261115-01.log" 2>&1
python scripts/run_gate_components.py --output "$GATE_RUNS/gate-components-factorial-20261115-01" --smoke-equivalence "$GATE_ARTIFACTS/gate-components-smoke-equivalence-20261115-01.json" > "$GATE_LOGS/gate-components-primary-20261115-01.log" 2>&1
python scripts/analyze_gate_components.py "$GATE_RUNS/gate-components-factorial-20261115-01" --output "$GATE_ARTIFACTS/gate-components-primary-audit-20261115-01.json" > "$GATE_LOGS/gate-components-primary-audit-20261115-01.log" 2>&1
echo 'gate-components pipeline completed; interpretation and publication still required'
