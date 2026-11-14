"""Freeze executable natural history pairs and actual tokens before inference."""

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
from scripts.history_no_repeat_data import (DATA_SOURCES, DESTINATION, PROTOCOL, build_data, evaluated_members,
    identity, model_artifacts, read_plan, reference_gate, safety_samples, token_registry, validate_data)
from scripts.run_controller_trial import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    if args.output != DESTINATION or args.output.exists():
        raise ValueError("Use the new registered destination; never overwrite prior evidence")
    plan = read_plan()
    gate = reference_gate(plan, reexecute=True)
    print(json.dumps({"event": "complete1920_query_control_panel_audit_reexecuted"}), flush=True)
    model = json.loads((ROOT / plan["model_config"]).read_text())[plan["model_key"]]
    if model != gate["model_path"]:
        raise ValueError("Native model path changed from the audited reference")
    data = build_data(plan)
    counts = validate_data(data, plan)
    registry = token_registry(data, plan, tokenizer_runtime(model))
    tokenizer, checkpoint = model_artifacts(model)
    smoke = evaluated_members(data, True)
    if len(smoke) != 24 or len(safety_samples(smoke)) != 8 or len(safety_samples(data["validation"])) != 16:
        raise ValueError("Operational/full safety coverage differs from registered protocol")
    manifest = {"status": "registered_before_inference", "registered_utc": datetime.now(timezone.utc).isoformat(),
        "model_inference": False, "training": False, "model_path": model, "reader_policy": "full",
        "protocol_sha256": streaming_hash(PROTOCOL), "counts": counts, "reference_gate": gate,
        "complete_control_panel_audit_reexecuted_at_registration": True,
        "data_sha256": data_fingerprint(data), "registry_sha256": data_fingerprint(registry),
        "evaluation_keys_sha256": data_fingerprint([identity(row) for row in data["validation"]]),
        "source_sha256": {name: streaming_hash(ROOT / name) for name in DATA_SOURCES},
        "tokenizer_sha256": tokenizer, "checkpoint_sha256": checkpoint,
        "smoke_queries": 24, "smoke_history_and_replay_samples": 8, "full_history_and_replay_samples": 16,
        "numeric_evidence_scope": "Standalone three-digit SET value fields, audited by actual offsets; E-prefixed account identifiers are separate lexical fields.",
        "interpretation": plan["purpose"]}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("plan", plan), ("data", data), ("registry", registry), ("manifest", manifest)):
        write_json(args.output / (name + ".json"), value)
    for name in DATA_SOURCES:
        destination = args.output / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    print(json.dumps({"status": manifest["status"], "counts": counts,
        "prompt_tokens": sorted({row["prompt_tokens"] for row in registry["validation"]}),
        "registry_sha256": manifest["registry_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
