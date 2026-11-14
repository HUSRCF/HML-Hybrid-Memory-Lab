"""Freeze the registered fixed-length locality data before native inference."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import (DATA_SOURCES, PROTOCOL, build_data, read_protocol,
                                           reference_gate, streaming_hash, token_registry, validate_data)
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.run_controller_trial import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read_protocol()
    gate = reference_gate(plan)
    model = Path(json.loads((ROOT / plan["model_config"]).read_text())[plan["model_key"]])
    data = build_data(plan)
    counts = validate_data(data, plan)
    registry = token_registry(data, plan, tokenizer_runtime(str(model)))
    checkpoints = sorted(model.glob("*.safetensors"))
    if not checkpoints:
        raise ValueError("Locality screen requires the existing local checkpoint")
    manifest = {"status": "registered_before_inference", "registered_utc": datetime.now(timezone.utc).isoformat(),
                "model_inference": False, "training": False, "model_path": str(model), "reader_policy": "full",
                "protocol_sha256": streaming_hash(PROTOCOL), "counts": counts,
                "reference_gate": gate,
                "data_sha256": data_fingerprint(data), "registry_sha256": data_fingerprint(registry),
                "evaluation_keys_sha256": data_fingerprint([[row["case_id"], row["variant"], row["role"]] for row in registry["validation"]]),
                "source_sha256": {relative: streaming_hash(ROOT / relative) for relative in DATA_SOURCES},
                "tokenizer_sha256": {path.name: streaming_hash(path) for path in sorted(model.iterdir()) if path.is_file() and
                                     ("token" in path.name or path.name == "config.json" or path.suffix == ".jinja")},
                "checkpoint_sha256": {path.name: streaming_hash(path) for path in checkpoints},
                "interpretation": plan["purpose"]}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("plan", plan), ("data", data), ("registry", registry), ("manifest", manifest)):
        write_json(args.output / f"{name}.json", value)
    for relative in DATA_SOURCES:
        destination = args.output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    print(json.dumps({"status": manifest["status"], "counts": counts,
                      "prompt_tokens": sorted({row["prompt_tokens"] for row in registry["validation"]}),
                      "data_sha256": manifest["data_sha256"], "registry_sha256": manifest["registry_sha256"]}))


if __name__ == "__main__":
    main()
