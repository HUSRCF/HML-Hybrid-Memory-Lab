"""Bind fresh-seed providers to an isolated copy of the sealed native engine."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.locality_replication import (PROTOCOL, RUNTIME_SOURCES, evaluated_members,
                                           load_engine, read_plan, registered_dataset)
from sas_memory.write_locality_data import streaming_hash


def checked_smoke(path, dataset_manifest):
    from scripts.analyze_write_locality_replication import audit_run
    report = json.loads(path.read_text())
    if report["status"] != "audited_smoke" or report["records_audited"] != 12:
        raise ValueError("Independent actual-model replica smoke audit required")
    if audit_run(Path(report["source_run"]), allow_smoke=True) != report or report["dataset_manifest_sha256"] != dataset_manifest:
        raise ValueError("Replica smoke audit/source/data changed")
    return {"report_path": str(path), "report_sha256": streaming_hash(path),
            "manifest_sha256": report["manifest_sha256"], "records_audited": report["records_audited"]}


def configured_engine():
    engine = load_engine("scripts/run_write_locality_load_screen.py", "_sas_replica_native_engine")
    # All model execution, weight hashing, unrestricted decoding, observer
    # checks and raw-row writing remain the discovery engine's implementation.
    # Only explicit study providers change, in this isolated module namespace.
    engine.PROTOCOL = PROTOCOL
    engine.DATA_SOURCES = RUNTIME_SOURCES
    engine.read_protocol = read_plan
    engine.registered_dataset = registered_dataset
    engine.evaluated_members = evaluated_members
    engine.checked_smoke = checked_smoke
    return engine


if __name__ == "__main__":
    configured_engine().main()
