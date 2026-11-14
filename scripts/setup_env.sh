#!/usr/bin/env bash
set -euo pipefail
SAS_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAS_STORAGE_ROOT="${SAS_STORAGE_ROOT:-/media/990Pro/Code/SAS}"
SAS_PYTHON=/home/husrcf/anaconda3/bin/python
mkdir -p "$SAS_STORAGE_ROOT/artifacts/environment" "$SAS_STORAGE_ROOT/data" "$SAS_STORAGE_ROOT/runs"
if [[ ! -x "$SAS_PYTHON" ]]; then
    echo "Missing existing Conda base Python." >&2
    exit 1
fi
uv pip install --python "$SAS_PYTHON" --no-deps --no-cache \
    --constraint "$SAS_PROJECT_ROOT/environment/host-constraints.txt" \
    --requirement "$SAS_PROJECT_ROOT/requirements.txt"
"$SAS_PYTHON" "$SAS_PROJECT_ROOT/scripts/check_dependencies.py"
"$SAS_PYTHON" - "$SAS_PROJECT_ROOT" <<'PY'
from pathlib import Path
import importlib.metadata as md
import sys
root = Path(sys.argv[1])
names = ('fla-core', 'flash-linear-attention', 'datasets', 'xxhash', 'multiprocess')
(root/'environment/project-packages.lock.txt').write_text(''.join(f'{name}=={md.version(name)}\n' for name in names))
PY
