"""Register boundary-only record-KV masks before running any ablated inference."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.counterbalance_data import tokenizer_runtime, file_digest
from sas_memory.record_kv_ablation import build_record_registry, reference_protocol
from sas_memory.training_data import data_fingerprint
from scripts.run_controller_trial import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/record_kv_ablation_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    reference, original, reference_plan, cases, sources, result_hashes = reference_protocol(plan, ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, str(ROOT / "scripts/analyze_readout_counterbalance.py"), str(reference),
                    "--output", str(args.output / "reference.audit.json")], check=True)
    audit = json.loads((args.output / "reference.audit.json").read_text())
    if audit["status"] != "completed" or audit["manifest_sha256"] != file_digest(reference / "manifest.json"):
        raise ValueError("Reference audit does not match the completed run")
    model_path = original["model_path"]
    registry = build_record_registry(tokenizer_runtime(model_path), cases)
    for name, value in (("plan", plan), ("reference_plan", reference_plan), ("cases", cases), ("registry", registry)):
        write_json(args.output / f"{name}.json", value)
    (args.output / "reference_manifest.json").write_bytes((reference / "manifest.json").read_bytes())
    source_paths = sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__).resolve(), ROOT / "scripts/analyze_readout_counterbalance.py"]
    manifest = {"status": "validated_before_inference", "registered_utc": datetime.now(timezone.utc).isoformat(),
                "plan_sha256": data_fingerprint(plan), "cases_sha256": data_fingerprint(cases),
                "registry_sha256": data_fingerprint(registry), "reference_manifest_sha256": file_digest(reference / "manifest.json"),
                "reference_audit_sha256": file_digest(args.output / "reference.audit.json"),
                "reference_result_sha256": result_hashes, "checkpoint_sources": sources, "model_path": model_path,
                "tokenizer_sha256": {name: file_digest(Path(model_path) / name) for name in ("tokenizer.json", "tokenizer_config.json")},
                "contexts": len(registry), "prelude_tokens": sorted({row["prelude_tokens"] for row in registry}),
                "source_sha256": {str(path.relative_to(ROOT)): file_digest(path) for path in source_paths}}
    for relative in manifest["source_sha256"]:
        destination = args.output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "contexts", "prelude_tokens", "registry_sha256")}))


if __name__ == "__main__":
    main()
