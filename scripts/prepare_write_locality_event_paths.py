"""Register complete gated cache-source inputs and both A/C commit companions."""

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
from scripts.locality_event_data import (
    DATA_SOURCES, DESTINATION, PROTOCOLS, REPLICA_AUDIT, checked_references,
    load_contexts, make_registry, read_protocols,
)
from scripts.run_controller_trial import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--replica-audit", type=Path, default=REPLICA_AUDIT)
    args = parser.parse_args()
    if args.output.exists() or args.output != DESTINATION:
        raise ValueError("Use the new registered destination; existing source registrations are immutable")
    protocols = read_protocols()
    references, recurrence = checked_references(args.replica_audit, reexecute=True)
    print(json.dumps({"event": "both_complete_reference_audits_reexecuted", "records": 1152,
                      "primary_recurrence": recurrence}), flush=True)
    contexts = load_contexts(references, protocols)
    model_path = contexts["replica"]["manifest"]["model_path"]
    registry = make_registry(contexts, tokenizer_runtime(model_path))
    manifest = {"status": "registered_before_inference", "registered_utc": datetime.now(timezone.utc).isoformat(),
                "model_inference": False, "training": False, "adapters": False, "model_path": model_path,
                "replica_audit_path": str(args.replica_audit), "references": references,
                "primary_recurrence": recurrence, "complete_reference_audits_reexecuted_at_registration": True,
                "source_sha256": {name: streaming_hash(ROOT / name) for name in DATA_SOURCES},
                "protocol_sha256": {name: streaming_hash(ROOT / name) for name in PROTOCOLS},
                "registry_sha256": data_fingerprint(registry), "counts": registry["counts"],
                "source_boundary": "All source caches must stop after the162 ledger records and exclude every question/answer token.",
                "interpretation": "Frozen diagnosis, not a repair method. Mixed-cache effects and oracle recovery do not prove irreversible recurrent memory damage or its absence."}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("protocols", protocols), ("registry", registry), ("manifest", manifest)):
        write_json(args.output / (name + ".json"), value)
    for cohort, reference in references.items():
        (args.output / (cohort + "_audit.json")).write_bytes(Path(reference["audit_path"]).read_bytes())
    for name in DATA_SOURCES:
        target = args.output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    print(json.dumps({"status": manifest["status"], "registry_sha256": manifest["registry_sha256"],
                      "counts": manifest["counts"]}), flush=True)


if __name__ == "__main__":
    main()
