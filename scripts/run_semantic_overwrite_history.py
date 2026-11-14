"""Bind the sealed native execute function to counterbalanced cases and rendering."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.overwrite_history_data import (DATA_SOURCES, DESTINATION, PROTOCOL, evaluated_members,
    identity, prompt_for, read_plan, registered_inputs, safety_samples)
from scripts.run_controller_trial import write_json
from scripts.run_semantic_history_no_repeat import RUNTIME_SOURCES as SEALED_RUNTIME_SOURCES

ENGINE = ROOT / "scripts/run_semantic_history_no_repeat.py"
RUNTIME_SOURCES = tuple(sorted(set(SEALED_RUNTIME_SOURCES) | set(DATA_SOURCES) | {
    "scripts/run_semantic_overwrite_history.py", "scripts/analyze_semantic_overwrite_history.py",
    "scripts/overwrite_history_metrics.py",
}))


def engine_binding():
    return {"path": str(ENGINE), "sha256": streaming_hash(ENGINE), "bound_protocol": str(PROTOCOL),
        "bound_data_functions": {name: "scripts.overwrite_history_data." + name
            for name in ("evaluated_members", "identity", "prompt_for", "safety_samples")},
        "source_set_sha256": data_fingerprint(RUNTIME_SOURCES),
        "method": "private module instance; replace only protocol, source set and four data functions; execute bytecode unchanged"}


def bound_engine():
    spec = importlib.util.spec_from_file_location("_sas_counterbalanced_native_engine", ENGINE)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.PROTOCOL, engine.RUNTIME_SOURCES = PROTOCOL, RUNTIME_SOURCES
    engine.evaluated_members, engine.identity = evaluated_members, identity
    engine.prompt_for, engine.safety_samples = prompt_for, safety_samples
    return engine


def checked_smoke(path, dataset_digest):
    from scripts.analyze_semantic_overwrite_history import audit_run
    report = json.loads(path.read_text())
    if (report["status"] != "audited_smoke" or report["records_audited"] != 24 or
        report["dataset_manifest_sha256"] != dataset_digest or audit_run(Path(report["source_run"]), allow_smoke=True) != report):
        raise ValueError("Complete independently audited24-query counterbalanced smoke required")
    return {"path": str(path), "sha256": streaming_hash(path), "manifest_sha256": report["manifest_sha256"]}


def execute(args, plan, registered, data, registry, gate):
    engine = bound_engine()
    engine.execute(args, plan, registered, data, registry, gate)
    path = args.output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["execution_engine"] = engine_binding()
    write_json(path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DESTINATION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-audit", type=Path)
    args = parser.parse_args()
    if bool(args.smoke_audit) == args.smoke or args.output.exists():
        raise ValueError("Choose a new operational smoke or provide its complete independent audit")
    plan = read_plan()
    registered, data, registry = registered_inputs(args.dataset)
    gate = None if args.smoke else checked_smoke(args.smoke_audit, streaming_hash(args.dataset / "manifest.json"))
    execute(args, plan, registered, data, registry, gate)


if __name__ == "__main__":
    main()
