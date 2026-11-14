"""Independently reconstruct the data and audit all native locality observations."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import (DATA_SOURCES, PROTOCOL, evaluated_members, identity, read_protocol,
                                           registered_dataset, streaming_hash)
from sas_memory.write_locality_metrics import summarize
from scripts.analyze_controller_trial import load_rows
from scripts.run_controller_trial import write_json


def audit_run(path, allow_smoke=False):
    manifest = json.loads((path / "manifest.json").read_text())
    plan = read_protocol()
    dataset = Path(manifest["dataset"])
    registered, data, registry = registered_dataset(dataset)
    if (manifest["status"] != "completed" or manifest["training"] or manifest["adapters"] or manifest["reader_policy"] != "full" or
        manifest["dataset_manifest_sha256"] != streaming_hash(dataset / "manifest.json") or
        manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or json.loads((path / "plan.json").read_text()) != plan or
        manifest["reference_gate"] != registered["reference_gate"] or manifest["model_path"] != registered["model_path"] or
        manifest["decoding"] != {"maximum_new_tokens": 12, "candidate_restriction": False, "gold_answer_nll_tokens": 3}):
        raise ValueError("Incomplete, changed or non-native locality run")
    if manifest["smoke"] and not allow_smoke:
        raise ValueError("Operational safety sample is not a complete locality screen")
    for stamp in (plan["registered_utc"], registered["registered_utc"]):
        if datetime.fromisoformat(stamp) >= datetime.fromisoformat(manifest["started_utc"]):
            raise ValueError("Locality protocol/data were not frozen before inference")
    required_sources = set(DATA_SOURCES) | {"sas_memory/write_locality_metrics.py", "scripts/run_write_locality_load_screen.py",
        "scripts/analyze_write_locality_load_screen.py", "scripts/check_record_attention_weights.py", "scripts/run_oracle_kv_readout.py"}
    if not required_sources <= manifest["source_sha256"].keys():
        raise ValueError("Essential locality runtime/audit source coverage is missing")
    for relative, digest in manifest["source_sha256"].items():
        if streaming_hash(path / "sources" / relative) != digest or streaming_hash(ROOT / relative) != digest:
            raise ValueError("Actual/archived locality runtime or analysis source changed")
    configs = {file.name: streaming_hash(file) for file in Path(manifest["model_path"]).glob("generation_config.json")}
    if manifest["generation_config_sha256"] != configs:
        raise ValueError("Native generation config changed")
    weights = manifest["weights_before"]
    if (weights != manifest["weights_after"] or not weights["all_parameters_frozen"] or
        not weights["no_parameter_gradients"] or not manifest["hooks_restored"] or
        len(weights["sha256"]) != 64 or weights["state_tensors"] < 24 or weights["state_bytes"] < 1_000_000_000):
        raise ValueError("Actual loaded model weight identity/freeze/hook safety failed")
    if not manifest["smoke"]:
        gate = manifest["smoke_gate"]
        smoke = json.loads(Path(gate["report_path"]).read_text())
        if (streaming_hash(gate["report_path"]) != gate["report_sha256"] or smoke["status"] != "audited_smoke" or
            smoke["records_audited"] != 12 or smoke["manifest_sha256"] != gate["manifest_sha256"] or
            smoke["dataset_manifest_sha256"] != manifest["dataset_manifest_sha256"] or
            smoke["weights_sha256"] != weights["sha256"] or smoke["packages"] != manifest["packages"]):
            raise ValueError("Registered operational smoke gate changed")
    elif manifest["smoke_gate"] is not None:
        raise ValueError("Operational smoke unexpectedly uses a prior smoke gate")
    members = evaluated_members(data, manifest["smoke"])
    if manifest["evaluated_keys"] != [identity(example) for example in members]:
        raise ValueError("Native locality cohort/order changed")
    expected = {tuple(identity(example)): example["query"]["answer"] for example in members}
    result_path = path / "validation.native.jsonl"
    rows = load_rows(result_path, expected)
    if streaming_hash(result_path) != manifest["result_sha256"]:
        raise ValueError("Native locality raw result file changed")
    indexed = {(row["case_id"], row["variant"], row["role"]): row for row in registry["validation"]}
    for row, example in zip(rows, members):
        key = tuple(identity(example))
        registration = indexed[key]
        if ((row["case_id"], row["variant"], row["role"]) != key or row["registration"] != registration or
            row["reader_policy"] != "full" or row["query_tokens"] != registration["prompt_tokens"] - registration["ledger_tokens"] or
            any(row[field] != example["case"][field] for field in ("base_stream_id", "base_case_id", "seed", "active_accounts", "value_version"))):
            raise ValueError("Native locality observation identity/input metadata changed")
    keys = sorted(identity(example) for example in members)
    if [check["key"] for check in manifest["safety_checks"]] != [keys[0], keys[-1]]:
        raise ValueError("Prescribed first/last native observation safety checks missing")
    for check in manifest["safety_checks"]:
        if not check["native_replay_exact"] or set(check["full_attention_weights"]) != {str(index) for index in (3, 7, 11, 15, 19, 23)}:
            raise ValueError("Native replay or full-history layer coverage failed")
        for stat in check["full_attention_weights"].values():
            if stat["prefill_calls"] != 1 or stat["cached_calls"] < 1 or stat["max_record_mass"] <= 0:
                raise ValueError("Actual historical Attention observation failed")
    metrics = ({"examples": len(rows), "correct": sum(row["greedy_exact"] for row in rows), "screen_signal_evaluated": False}
               if manifest["smoke"] else summarize(rows))
    return {"status": "audited_smoke" if manifest["smoke"] else "completed_and_audited", "source_run": str(path),
            "manifest_sha256": streaming_hash(path / "manifest.json"), "records_audited": len(rows),
            "dataset_manifest_sha256": manifest["dataset_manifest_sha256"], "data_sha256": registered["data_sha256"],
            "registry_sha256": registered["registry_sha256"], "protocol_sha256": manifest["protocol_sha256"],
            "weights_sha256": weights["sha256"], "packages": manifest["packages"],
            "metrics": metrics, "reference_gate": registered["reference_gate"], "interpretation": plan["purpose"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    report = audit_run(args.run, args.allow_smoke)
    report["analysis_source_sha256"] = streaming_hash(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status": report["status"], "records_audited": report["records_audited"],
                      "correct": report["metrics"]["correct"],
                      "screen_signal": report["metrics"].get("screen_signal", {}).get("present")}))


if __name__ == "__main__":
    main()
