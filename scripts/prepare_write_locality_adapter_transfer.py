"""Register all natural long-prefix endpoints for two fixed GDN checkpoints."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.locality_event_data import registered_inputs as event_inputs
from scripts.locality_transfer_data import (DATA_SOURCES, DESTINATION, PROTOCOL, build_registry,
    checked_references, read_plan, safety_samples, selected_inputs)
from scripts.run_controller_trial import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    if args.output != DESTINATION or args.output.exists():
        raise ValueError("Use the new registered destination; existing registrations remain immutable")
    plan = read_plan()
    references = checked_references(plan, reexecute_event=True)
    print(json.dumps({"event": "complete_native_event_audit_reexecuted", "readouts": 1200,
                      "old_training_audits": "pinned and source/artifact checked; not reexecuted"}), flush=True)
    event, contexts, event_registry = event_inputs(Path(plan["event_dataset"]))
    registry = build_registry(plan, contexts, event_registry, references, tokenizer_runtime(event["model_path"]))
    smoke = selected_inputs(registry, True)
    smoke_samples = safety_samples(*smoke)
    full_samples = safety_samples(*selected_inputs(registry, False))
    if len(smoke[0]) != 6 or len(smoke[1]) != 28 or len(smoke[2]) != 24 or len(full_samples[0]) != 20:
        raise ValueError("Operational/full cohort safety coverage changed")
    manifest = {"status": "registered_before_inference", "registered_utc": datetime.now(timezone.utc).isoformat(),
                "training": False, "model_inference": False, "model_path": event["model_path"],
                "protocol_sha256": streaming_hash(PROTOCOL), "event_dataset_manifest_sha256": plan["event_dataset_manifest_sha256"],
                "references": references, "complete_event_audit_reexecuted_at_registration": True,
                "registry_sha256": data_fingerprint(registry), "counts": plan["counts"],
                "source_sha256": {name: streaming_hash(ROOT / name) for name in DATA_SOURCES},
                "smoke": {"pairs": 6, "bindings_per_model": 28, "physical_queries_per_model": 24,
                          "prefix_samples": len(smoke_samples[0]), "attention_restore_samples_per_model": len(smoke_samples[1])},
                "full_safety": {"prefix_samples": len(full_samples[0]), "attention_restore_samples_per_model": len(full_samples[1])},
                "interpretation": plan["interpretation"]}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("plan", plan), ("registry", registry), ("manifest", manifest)):
        write_json(args.output / (name + ".json"), value)
    for name in DATA_SOURCES:
        target = args.output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    print(json.dumps({"status": manifest["status"], "registry_sha256": manifest["registry_sha256"],
                      "counts": manifest["counts"], "smoke": manifest["smoke"], "full_safety": manifest["full_safety"]}), flush=True)


if __name__ == "__main__":
    main()
