"""Bind the sealed transfer execution engine to the registered five-model panel."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.locality_panel_data import DATA_SOURCES, DESTINATION, MODELS, PROTOCOL, read_plan, registered_inputs
from scripts.run_controller_trial import write_json
from scripts.run_write_locality_adapter_transfer import RUNTIME_SOURCES as SEALED_RUNTIME_SOURCES

ENGINE = ROOT / "scripts/run_write_locality_adapter_transfer.py"
RUNTIME_SOURCES = tuple(sorted(set(SEALED_RUNTIME_SOURCES) | set(DATA_SOURCES) | {
    "scripts/run_write_locality_control_panel.py", "scripts/analyze_write_locality_control_panel.py",
    "scripts/locality_panel_metrics.py",
}))


def engine_binding():
    return {"path": str(ENGINE), "sha256": streaming_hash(ENGINE), "bound_models": list(MODELS),
            "bound_protocol": str(PROTOCOL), "source_set_sha256": data_fingerprint(RUNTIME_SOURCES),
            "method": "private module instance; replace only MODELS, PROTOCOL and RUNTIME_SOURCES; execute unchanged"}


def bound_engine():
    spec = importlib.util.spec_from_file_location("_sas_fixed_panel_transfer_engine", ENGINE)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.MODELS, engine.PROTOCOL, engine.RUNTIME_SOURCES = MODELS, PROTOCOL, RUNTIME_SOURCES
    return engine


def checked_smoke(path, dataset_digest):
    from scripts.analyze_write_locality_control_panel import audit_run
    report = json.loads(path.read_text())
    if (report["status"] != "audited_smoke" or report["physical_readouts_audited"] != 120 or
        report["dataset_manifest_sha256"] != dataset_digest or
        audit_run(Path(report["source_run"]), allow_smoke=True) != report):
        raise ValueError("Complete independently audited120-query panel smoke required")
    return {"path": str(path), "sha256": streaming_hash(path), "manifest_sha256": report["manifest_sha256"]}


def execute(args, plan, registered, contexts, registry, gate):
    engine = bound_engine()
    engine.execute(args, plan, registered, contexts, registry, gate)
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
        raise ValueError("Choose a new operational smoke or provide its completed independent audit")
    plan = read_plan()
    registered, contexts, registry = registered_inputs(args.dataset)
    gate = None if args.smoke else checked_smoke(args.smoke_audit, streaming_hash(args.dataset / "manifest.json"))
    execute(args, plan, registered, contexts, registry, gate)


if __name__ == "__main__":
    main()
