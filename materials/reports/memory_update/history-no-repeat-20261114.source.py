"""Independently reconstruct and audit every no-repeat natural-world readout."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.history_no_repeat_data import PROTOCOL, evaluated_members, identity, read_plan, registered_inputs, safety_samples
from scripts.history_no_repeat_metrics import summarize
from scripts.run_controller_trial import write_json
from scripts.run_semantic_history_no_repeat import RUNTIME_SOURCES


def audit_rows(rows, members, registrations):
    if [[r["case_id"], r["variant"], r["role"]] for r in rows] != [identity(row) for row in members]:
        raise ValueError("Missing, repeated, filtered or reordered natural-world questions")
    for row, example in zip(rows, members):
        case, query = example["case"], example["query"]
        registration = registrations[tuple(identity(example))]
        match = re.fullmatch(r"\s*(-?\d+)\s*", row["generated"])
        if (row["answer"] != query["answer"] or row["greedy_exact"] != bool(match and match.group(1) == query["answer"]) or
            not math.isfinite(row["answer_nll"]) or row["answer_nll"] < 0 or
            row["prefill"] != "full" or row["reader_policy"] != "full" or
            row["registration_sha256"] != data_fingerprint(registration) or
            row["query_tokens"] != len(registration["question_ids"]) or
            any(row[field] != case[field] for field in ("cohort", "base_stream_id", "base_case_id", "seed", "active_accounts", "value_version"))):
            raise ValueError("Own-world truth, exact scoring or actual source registration changed")


def audit_run(path, allow_smoke=False):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    plan = read_plan()
    if manifest["status"] != "completed" or (manifest["smoke"] and not allow_smoke):
        raise ValueError("Completed native diagnostic required; smoke is operational only")
    registered, data, registry = registered_inputs(Path(manifest["dataset"]))
    members = evaluated_members(data, manifest["smoke"])
    samples = safety_samples(members)
    sources = set(RUNTIME_SOURCES) | {str(p.relative_to(ROOT)) for p in (ROOT / "sas_memory").glob("*.py")}
    weights = manifest["weights_after"]
    if (manifest["training"] or manifest["adapters"] or manifest["reader_policy"] != "full" or
        manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or json.loads((path / "plan.json").read_text()) != plan or
        manifest["dataset_manifest_sha256"] != streaming_hash(Path(manifest["dataset"]) / "manifest.json") or
        manifest["registry_sha256"] != registered["registry_sha256"] or manifest["reference_gate"] != registered["reference_gate"] or
        manifest["model_path"] != registered["model_path"] or manifest["packages"] != registered["reference_gate"]["packages"] or
        manifest["decoding"] != {"maximum_new_tokens": 12, "candidate_restriction": False, "gold_answer_nll_tokens": 3} or
        manifest["evaluated_keys"] != [identity(row) for row in members] or manifest["safety_sample_keys"] != samples or
        manifest["completed_readouts"] != len(members) or set(manifest["source_sha256"]) != sources or
        manifest["weights_before"] != weights or weights["sha256"] != plan["native_loaded_weights_sha256"] or
        not weights["all_parameters_frozen"] or not weights["no_parameter_gradients"] or not manifest["hooks_restored"] or
        not (datetime.fromisoformat(plan["registered_utc"]) < datetime.fromisoformat(registered["registered_utc"]) <
             datetime.fromisoformat(manifest["started_utc"]) < datetime.fromisoformat(manifest["completed_utc"]))):
        raise ValueError("Frozen native scope, full source coverage, runtime or registration changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / "sources" / name) != digest:
            raise ValueError("Actual/archived native diagnostic implementation changed")
    configs = {p.name: streaming_hash(p) for p in Path(manifest["model_path"]).glob("generation_config.json")}
    if manifest["generation_config_sha256"] != configs:
        raise ValueError("Native generation configuration changed")
    if manifest["smoke"]:
        if manifest["smoke_gate"] is not None:
            raise ValueError("Operational smoke unexpectedly claims an earlier gate")
    else:
        gate = manifest["smoke_gate"]
        gate_path = Path(gate["path"])
        report = json.loads(gate_path.read_text())
        if (streaming_hash(gate_path) != gate["sha256"] or report["status"] != "audited_smoke" or report["records_audited"] != 24 or
            report["manifest_sha256"] != gate["manifest_sha256"] or report["dataset_manifest_sha256"] != manifest["dataset_manifest_sha256"] or
            report["weights_sha256"] != weights["sha256"] or audit_run(Path(report["source_run"]), allow_smoke=True) != report):
            raise ValueError("Complete independent native24-query operational smoke gate changed")
    result_path = path / "validation.native.jsonl"
    if streaming_hash(result_path) != manifest["result_sha256"]:
        raise ValueError("Actual recorded native observations changed")
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    registrations = {(r["case_id"], r["variant"], r["role"]): r for r in registry["validation"]}
    audit_rows(rows, members, registrations)
    indexed = {(r["case_id"], r["variant"], r["role"]): r for r in rows}
    if [r["key"] for r in manifest["safety_checks"]] != samples:
        raise ValueError("Prescribed historical access and replay sample coverage incomplete")
    for check in manifest["safety_checks"]:
        row = indexed[tuple(check["key"])]
        if (not check["native_replay_exact"] or not check["hooks_restored"] or
            check["native_replay"] != {k: row[k] for k in ("answer", "answer_nll", "greedy_exact", "generated", "query_tokens")} or
            set(check["full_attention_weights"]) != {str(i) for i in (3, 7, 11, 15, 19, 23)}):
            raise ValueError("Native historical replay or layer coverage differs")
        if any(stat["prefill_calls"] != 1 or stat["cached_calls"] < 1 or stat["query_rows"] < 1 or
               not math.isfinite(stat["max_record_mass"]) or stat["max_record_mass"] <= 0 or stat["max_record_probability"] <= 0
               for stat in check["full_attention_weights"].values()):
            raise ValueError("Actual full-prefill/cached historical Attention access failed")
    metrics = ({"examples": len(rows), "correct": sum(r["greedy_exact"] for r in rows), "primary_signal_evaluated": False}
               if manifest["smoke"] else summarize(rows, data))
    return {"status": "audited_smoke" if manifest["smoke"] else "completed_and_audited", "source_run": str(path),
        "manifest_sha256": streaming_hash(path / "manifest.json"), "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "data_sha256": registered["data_sha256"], "registry_sha256": registered["registry_sha256"],
        "protocol_sha256": manifest["protocol_sha256"], "weights_sha256": weights["sha256"], "packages": manifest["packages"],
        "records_audited": len(rows), "pairs_audited": len(members) // 6, "actual_attention_checks": len(samples),
        "native_exact_replays": len(samples), "metrics": metrics, "reference_gate": registered["reference_gate"],
        "interpretation": plan["purpose"], "analysis_source_sha256": streaming_hash(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    report = audit_run(args.run, args.allow_smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status": report["status"], "records_audited": report["records_audited"],
        "actual_attention_checks": report["actual_attention_checks"], "primary_signal": report["metrics"].get("primary_signal")}))


if __name__ == "__main__":
    main()
