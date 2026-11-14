"""Freeze the additional five-checkpoint panel without duplicating models or data."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.locality_panel_data import (DATA_SOURCES, DESTINATION, PROTOCOL, checked_references, read_plan)
from scripts.locality_transfer_data import safety_samples, selected_inputs
from scripts.run_controller_trial import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    if args.output != DESTINATION or args.output.exists():
        raise ValueError("Use only the new registered destination; preserve existing artifacts")
    plan = read_plan()
    original, _, registry, refs = checked_references(plan, reexecute_transfer=True)
    smoke, full = selected_inputs(registry, True), selected_inputs(registry, False)
    if [len(x) for x in smoke] != [6, 28, 24] or [len(x) for x in full] != [96, 448, 384]:
        raise ValueError("Panel did not retain the identical complete natural cohort")
    manifest = {"status": "registered_before_inference", "registered_utc": datetime.now(timezone.utc).isoformat(),
                "training": False, "model_inference": False, "model_path": original["model_path"],
                "protocol_sha256": streaming_hash(PROTOCOL), "registry_sha256": data_fingerprint(registry),
                "source_dataset": plan["source_dataset"],
                "source_dataset_manifest_sha256": plan["source_dataset_manifest_sha256"],
                "references": refs, "counts": plan["counts"],
                "complete_prior_transfer_audit_reexecuted_at_registration": True,
                "source_sha256": {name: streaming_hash(ROOT / name) for name in DATA_SOURCES},
                "smoke": {"pairs": 6, "physical_queries_per_model": 24, "bindings_per_model": 28,
                          "prefix_samples": len(safety_samples(*smoke)[0]), "attention_restore_samples_per_model": len(safety_samples(*smoke)[1])},
                "full_safety": {"prefix_samples": len(safety_samples(*full)[0]), "attention_restore_samples_per_model": len(safety_samples(*full)[1])},
                "interpretation": plan["interpretation"]}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "plan.json", plan)
    for name in DATA_SOURCES:
        target = args.output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    print(json.dumps({"status": manifest["status"], "registry_sha256": manifest["registry_sha256"],
                      "complete_prior_transfer_audit_reexecuted": True, "counts": manifest["counts"]}), flush=True)


if __name__ == "__main__":
    main()
