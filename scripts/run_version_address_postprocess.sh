#!/usr/bin/env bash
# Run only after the primary audit; preserve all existing outputs on rerun.
set -euo pipefail
cd /home/husrcf/Code/SAS
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
SAS_VA_STORE=/media/990Pro/Code/SAS
SAS_VA_ARTIFACTS="$SAS_VA_STORE/artifacts/memory_update"
SAS_VA_LOGS="$SAS_VA_STORE/logs/memory_update"
SAS_VA_AUDIT="$SAS_VA_ARTIFACTS/version-address-primary-audit-20261115-01.json"
SAS_VA_OBSERVATIONS="$SAS_VA_ARTIFACTS/version-address-observations-20261115-01.json"
SAS_VA_EXPORT="$SAS_VA_STORE/artifacts/memory_update/version-address-exports-20261115-01"
SAS_VA_MATERIALS=materials/updates/version-address-20261115/exports
for SAS_VA_OUTPUT in "$SAS_VA_OBSERVATIONS" "$SAS_VA_EXPORT" "$SAS_VA_MATERIALS" \
    "$SAS_VA_LOGS/version-address-observations-20261115-01.log" \
    "$SAS_VA_LOGS/version-address-export-20261115-01.log"; do
    if [[ -e "$SAS_VA_OUTPUT" ]]; then
        echo "Preserve existing output: $SAS_VA_OUTPUT" >&2
        exit 1
    fi
done
python -B scripts/version_address_observations.py "$SAS_VA_AUDIT" \
    --output "$SAS_VA_OBSERVATIONS" \
    > "$SAS_VA_LOGS/version-address-observations-20261115-01.log" 2>&1
python -B scripts/export_version_address.py --audit "$SAS_VA_AUDIT" \
    --output "$SAS_VA_EXPORT" --materials "$SAS_VA_MATERIALS" \
    > "$SAS_VA_LOGS/version-address-export-20261115-01.log" 2>&1
echo 'All3456answers independently regraded; complete tables and plots exported.'
