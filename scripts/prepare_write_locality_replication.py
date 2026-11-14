"""Freeze the registered fresh-seed replica using the unchanged event recipe."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.locality_replication import (DATA_SOURCES, PROTOCOL, discovery_reference,
                                           gate_descriptor, read_plan)
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import build_data, streaming_hash, token_registry, validate_data
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.run_controller_trial import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = read_plan()
    destination = args.output or Path(plan["data_source"])
    if destination != Path(plan["data_source"]) or destination.exists():
        raise ValueError("Use the registered new dataset destination; existing registrations are immutable")
    reference = discovery_reference(plan, reexecute=True)
    print(json.dumps({"event": "complete_discovery_audit_reexecuted", "records": reference["records_audited"]}), flush=True)
    model = Path(json.loads((ROOT / plan["model_config"]).read_text())[plan["model_key"]])
    data = build_data(plan)
    counts = validate_data(data, plan)
    registry = token_registry(data, plan, tokenizer_runtime(str(model)))
    if {row["prompt_tokens"] for row in registry["validation"]} != {2865}:
        raise ValueError("Replica inputs differ from the registered2865-token design")
    checkpoints = sorted(model.glob("*.safetensors"))
    if not checkpoints:
        raise ValueError("Existing local checkpoint required")
    manifest = {"status": "registered_before_inference", "registered_utc": datetime.now(timezone.utc).isoformat(),
                "model_inference": False, "training": False, "reader_policy": "full", "model_path": str(model),
                "protocol_sha256": streaming_hash(PROTOCOL), "counts": counts,
                "data_sha256": data_fingerprint(data), "registry_sha256": data_fingerprint(registry),
                "evaluation_keys_sha256": data_fingerprint([[row["case_id"], row["variant"], row["role"]] for row in registry["validation"]]),
                "reference_gate": gate_descriptor(plan), "complete_discovery_audit_reexecuted_at_registration": True,
                "discovery_weights_sha256": reference["weights_sha256"],
                "source_sha256": {relative: streaming_hash(ROOT / relative) for relative in DATA_SOURCES},
                "tokenizer_sha256": {file.name: streaming_hash(file) for file in sorted(model.iterdir()) if file.is_file() and
                                     ("token" in file.name or file.name in ("config.json", "generation_config.json") or file.suffix == ".jinja")},
                "checkpoint_sha256": {file.name: streaming_hash(file) for file in checkpoints}, "interpretation": plan["purpose"]}
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in (("plan", plan), ("data", data), ("registry", registry), ("manifest", manifest)):
        write_json(destination / f"{name}.json", value)
    (destination / "discovery_audit.json").write_bytes(Path(plan["discovery_audit"]).read_bytes())
    for relative in DATA_SOURCES:
        target = destination / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    print(json.dumps({"status": manifest["status"], "counts": counts, "seeds": plan["development_grid"]["base_seeds"],
                      "prompt_tokens": [2865], "data_sha256": manifest["data_sha256"], "registry_sha256": manifest["registry_sha256"]}))


if __name__ == "__main__":
    main()
