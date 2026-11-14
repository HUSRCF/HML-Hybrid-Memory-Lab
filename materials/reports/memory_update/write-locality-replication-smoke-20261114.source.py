"""Reconstruct and audit the true new replica identities with the sealed engine."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.locality_replication import (PROTOCOL, RUNTIME_SOURCES, discovery_reference,
    evaluated_members, load_engine, primary_recurrence, read_plan, registered_dataset, summarize)
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json


def configured_engine():
    engine = load_engine("scripts/analyze_write_locality_load_screen.py", "_sas_replica_audit_engine")
    engine.PROTOCOL = PROTOCOL
    engine.DATA_SOURCES = RUNTIME_SOURCES
    engine.read_protocol = read_plan
    engine.registered_dataset = registered_dataset
    engine.evaluated_members = evaluated_members
    engine.summarize = summarize
    return engine


def audit_run(path, allow_smoke=False):
    report = configured_engine().audit_run(path, allow_smoke)
    plan = read_plan()
    reference = discovery_reference(plan)
    manifest = json.loads((path / "manifest.json").read_text())
    source_manifest = json.loads((Path(reference["source_run"]) / "manifest.json").read_text())
    if report["weights_sha256"] != reference["weights_sha256"] or report["packages"] != source_manifest["packages"]:
        raise ValueError("Replica loaded weights/runtime packages differ from discovery")
    required = {"scripts/run_write_locality_replication.py", "scripts/analyze_write_locality_replication.py",
                "sas_memory/locality_replication.py"}
    if not required <= manifest["source_sha256"].keys():
        raise ValueError("Actual replica study bindings are not archived")
    report.update(analysis_source_sha256=streaming_hash(Path(__file__)),
                  audit_engine_source_sha256=streaming_hash(ROOT / "scripts/analyze_write_locality_load_screen.py"),
                  provider_source_sha256=streaming_hash(ROOT / "sas_memory/locality_replication.py"),
                  primary_recurrence=None if manifest["smoke"] else primary_recurrence(report["metrics"], plan))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    report = audit_run(args.run, args.allow_smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status": report["status"], "records_audited": report["records_audited"],
                      "correct": report["metrics"]["correct"], "primary_recurrence": report["primary_recurrence"]}))


if __name__ == "__main__":
    main()
