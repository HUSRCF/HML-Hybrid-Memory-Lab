"""Freeze compact factorial cases, exact source tokens and reference evidence."""

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
from scripts.overwrite_history_data import (DATA_SOURCES, DESTINATION, PROTOCOL, build_data,
    evaluated_members, identity, model_artifacts, read_plan, reference_gate, safety_samples, token_registry, validate_data)
from scripts.run_controller_trial import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    if args.output != DESTINATION or args.output.exists():
        raise ValueError("Use only the new registered destination; preserve all previous evidence")
    plan = read_plan()
    gate = reference_gate(plan, reexecute=True)
    print(json.dumps({"event": "complete_prior384_query_audit_reexecuted", "prior_history_primary_passed": False}), flush=True)
    model = json.loads((ROOT / plan["model_config"]).read_text())[plan["model_key"]]
    if model != gate["model_path"]:
        raise ValueError("Native model path changed")
    data = build_data(plan)
    counts = validate_data(data, plan)
    registry = token_registry(data, plan, tokenizer_runtime(model))
    tokenizer, checkpoint = model_artifacts(model)
    smoke = evaluated_members(data, True)
    if len(smoke) != 24 or len(safety_samples(smoke)) != 8 or len(safety_samples(evaluated_members(data))) != 16:
        raise ValueError("Prescribed current-value safety coverage changed")
    manifest = {"status": "registered_before_inference", "registered_utc": datetime.now(timezone.utc).isoformat(),
        "training": False, "model_inference": False, "reader_policy": "full", "model_path": model,
        "protocol_sha256": streaming_hash(PROTOCOL), "counts": counts, "reference_gate": gate,
        "complete_reference_audit_reexecuted_at_registration": True, "data_sha256": data_fingerprint(data),
        "registry_sha256": data_fingerprint(registry),
        "evaluation_keys_sha256": data_fingerprint([identity(row) for row in evaluated_members(data)]),
        "source_sha256": {name: streaming_hash(ROOT / name) for name in DATA_SOURCES},
        "tokenizer_sha256": tokenizer, "checkpoint_sha256": checkpoint,
        "smoke_queries": 24, "smoke_access_and_replay_samples": 8, "full_access_and_replay_samples": 16,
        "data_layout": "Store32case templates once plus384query selectors; expand record3 early values only in the audited renderer.",
        "interpretation": plan["purpose"]}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("plan", plan), ("data", data), ("registry", registry), ("manifest", manifest)):
        write_json(args.output / (name + ".json"), value)
    for name in DATA_SOURCES:
        target = args.output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    print(json.dumps({"status": manifest["status"], "counts": counts,
        "prompt_tokens": sorted({r["prompt_tokens"] for r in registry["validation"]}),
        "registry_sha256": manifest["registry_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
