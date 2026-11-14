"""Freeze exact existing cases, actual token registry and seven fixed checkpoints."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.copy_transfer_data import (DATA_SOURCES, DESTINATION, PROTOCOL, build_registry,
    model_artifacts, read_plan, reference_gate, safety_samples, selected_inputs, validate_cases)
from scripts.run_controller_trial import write_json


def main():
    if DESTINATION.exists():
        raise ValueError("Preserve the existing registration; use this protocol only once")
    plan = read_plan()
    refs = reference_gate(plan, reexecute=True)
    print(json.dumps({"event": "prior_native384_audit_reexecuted_and_seven_checkpoints_verified"}), flush=True)
    cases = json.loads(Path(plan["source_cases"]).read_text())
    validate_cases(cases, plan)
    registry = build_registry(cases, tokenizer_runtime(plan["model_path"]))
    smoke, bindings = selected_inputs(registry, True)
    if len(smoke) != 18 or len(bindings) != 18 or len(safety_samples(registry)) != 6:
        raise ValueError("Operational smoke/sample coverage changed")
    tokenizer, checkpoint = model_artifacts(plan["model_path"])
    manifest = {"status": "registered_before_inference", "registered_utc": datetime.now(timezone.utc).isoformat(),
        "training": False, "model_inference": False, "model_path": plan["model_path"], "counts": plan["counts"],
        "protocol_sha256": streaming_hash(PROTOCOL), "cases_sha256": data_fingerprint(cases),
        "registry_sha256": data_fingerprint(registry), "references": refs,
        "complete_previous_native_audit_reexecuted": True, "tokenizer_sha256": tokenizer, "checkpoint_sha256": checkpoint,
        "source_sha256": {name: streaming_hash(ROOT / name) for name in DATA_SOURCES}}
    DESTINATION.mkdir(parents=True, exist_ok=False)
    for name, value in (("plan", plan), ("cases", cases), ("registry", registry), ("manifest", manifest)):
        write_json(DESTINATION / (name + ".json"), value)
    for name in DATA_SOURCES:
        target = DESTINATION / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    print(json.dumps({"status": manifest["status"], "counts": plan["counts"],
        "prompt_tokens": sorted({q["prompt_tokens"] for q in registry["physical"]}),
        "registry_sha256": manifest["registry_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
