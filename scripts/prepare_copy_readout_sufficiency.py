"""Freeze oracle and equal-budget non-slice masks without new task data."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.copy_readout_data import DATA_SOURCES, DESTINATION, PROTOCOL, build_masks, checked_source, read_plan, sample_ids, selected_inputs
from scripts.run_controller_trial import write_json


def main():
    if DESTINATION.exists():
        raise ValueError("Preserve existing readout registration")
    plan = read_plan()
    _, cases, registry, _, refs = checked_source(plan, reexecute=True)
    print(json.dumps({"event": "complete_source2688_query_audit_reexecuted"}), flush=True)
    masks = build_masks(cases, registry, tokenizer_runtime(plan["model_path"]))
    smoke, bindings = selected_inputs(registry, True)
    samples = sample_ids(cases, registry)
    if len(smoke) != 18 or len(bindings) != 18 or not set(samples) <= {q["id"] for q in smoke}:
        raise ValueError("Prescribed operational/all-role safety coverage changed")
    manifest = {"status": "registered_before_inference", "registered_utc": datetime.now(timezone.utc).isoformat(),
        "training": False, "model_inference": False, "counts": plan["counts"], "protocol_sha256": streaming_hash(PROTOCOL),
        "references": refs, "complete_source2688_query_audit_reexecuted": True,
        "mask_registry_sha256": data_fingerprint(masks), "sample_query_ids": samples,
        "source_sha256": {n: streaming_hash(ROOT / n) for n in DATA_SOURCES}}
    DESTINATION.mkdir(parents=True, exist_ok=False)
    for name, value in (("plan", plan), ("masks", masks), ("manifest", manifest)):
        write_json(DESTINATION / (name + ".json"), value)
    for name in DATA_SOURCES:
        destination = DESTINATION / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    print(json.dumps({"status": manifest["status"], "counts": plan["counts"], "masks": len(masks),
        "record_token_budgets": sorted({m["retained_record_tokens"] for m in masks.values()}),
        "mask_registry_sha256": manifest["mask_registry_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
