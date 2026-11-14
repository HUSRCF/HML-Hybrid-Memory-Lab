"""Independently reconstruct the five-checkpoint panel and its operational gates."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.locality_panel_data import MODELS, PROTOCOL, read_plan, registered_inputs
from scripts.locality_panel_metrics import summarize
from scripts.locality_transfer_data import safety_samples, selected_inputs
from scripts.analyze_write_locality_adapter_transfer import audit_rows
from scripts.run_controller_trial import write_json
from scripts.run_write_locality_control_panel import RUNTIME_SOURCES, engine_binding


def audit_run(path, allow_smoke=False):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    plan = read_plan()
    if manifest["status"] != "completed" or (manifest["smoke"] and not allow_smoke):
        raise ValueError("Completed full panel required; smoke is operational only")
    registered, _, registry = registered_inputs(Path(manifest["dataset"]))
    pairs, bindings, physical = selected_inputs(registry, manifest["smoke"])
    prefix_keys, query_samples = safety_samples(pairs, bindings, physical)
    queries = {q["id"]: q for q in physical}
    required_sources = set(RUNTIME_SOURCES) | {str(p.relative_to(ROOT)) for p in (ROOT / "sas_memory").glob("*.py")}
    weights = manifest["weights_after"]
    if (manifest["training"] or manifest["new_adapters"] or manifest["reader_policy"] != "full" or
        manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or json.loads((path / "plan.json").read_text()) != plan or
        manifest["registry_sha256"] != registered["registry_sha256"] or
        manifest["dataset_manifest_sha256"] != streaming_hash(Path(manifest["dataset"]) / "manifest.json") or
        manifest["selected_pair_ids"] != [p["id"] for p in pairs] or manifest["selected_physical_ids"] != list(queries) or
        manifest["physical_queries_per_model"] != len(physical) or manifest["reported_bindings_per_model"] != len(bindings) or
        manifest["prefix_sample_keys"] != prefix_keys or manifest["attention_restore_query_ids"] != query_samples or
        manifest["completed_models"] != list(MODELS) or list(manifest["models"]) != list(MODELS) or
        set(manifest["source_sha256"]) != required_sources or manifest["generation"] != plan["generation"] or
        manifest["execution_engine"] != engine_binding() or
        manifest["model_path"] != registered["model_path"] or manifest["packages"] != registered["references"]["event"]["packages"] or
        manifest["weights_before"] != weights or weights["sha256"] != plan["native_loaded_weights_sha256"] or
        not weights["all_parameters_frozen"] or not weights["no_parameter_gradients"] or not manifest["hooks_restored"] or
        not (datetime.fromisoformat(plan["registered_utc"]) < datetime.fromisoformat(registered["registered_utc"]) <
             datetime.fromisoformat(manifest["started_utc"]) < datetime.fromisoformat(manifest["completed_utc"]))):
        raise ValueError("Panel execution binding, frozen scope, complete coverage or runtime changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / "sources" / name) != digest:
            raise ValueError("Actual/archived panel execution or analysis implementation changed")
    if manifest["smoke"]:
        if manifest["smoke_gate"] is not None:
            raise ValueError("Operational smoke incorrectly claims an earlier gate")
    else:
        gate = manifest["smoke_gate"]
        gate_path = Path(gate["path"])
        report = json.loads(gate_path.read_text())
        if (streaming_hash(gate_path) != gate["sha256"] or report["status"] != "audited_smoke" or
            report["physical_readouts_audited"] != 120 or gate["manifest_sha256"] != report["manifest_sha256"] or
            report["dataset_manifest_sha256"] != manifest["dataset_manifest_sha256"] or
            audit_run(Path(report["source_run"]), allow_smoke=True) != report):
            raise ValueError("Complete independent120-query operational smoke gate changed")
    rows_by_model = {}
    for name in MODELS:
        spec, state = plan["models"][name], manifest["models"][name]
        training = json.loads((Path(spec["training_run"]) / "manifest.json").read_text())
        if (state["description"] != training["arms"][spec["training_arm"]]["description"] or
            state["checkpoint_sha256"] != spec["checkpoint_sha256"] or state["token_policy"] != spec["token_policy"] or
            state["adapter_parameters"] != spec["parameters"] or
            state["adapter_state_sha256_before"] != state["adapter_state_sha256_after"] or
            state["adapter_state_sha256_after"] != registered["references"]["training"][name]["checkpoint_state_sha256"] or
            not state["all_adapter_parameters_frozen"] or not state["hooks_restored"] or
            state["completed_readouts"] != len(physical) or
            streaming_hash(path / (name + ".jsonl")) != state["results_sha256"]):
            raise ValueError("Actual panel checkpoint/projection scope, freeze or completion changed")
        expected_prefixes = prefix_keys if spec["token_policy"] == "question" else []
        if [item["source_key"] for item in state["prefix_checks"]] != expected_prefixes:
            raise ValueError("Question-phase native prefix sample coverage incomplete")
        for item in state["prefix_checks"]:
            query = next(q for q in physical if q["source_key"] == item["source_key"])
            if (item["ledger_tokens"] != query["ledger_tokens"] or
                not item["complete_KV_recurrent_conv_bookkeeping_exact"] or not item["last_prefix_logits_exact"] or
                not re.fullmatch(r"[0-9a-f]{64}", item["prefix_sha256"])):
                raise ValueError("Complete native ledger memory, metadata or logits differ")
        observed = {item["id"]: item for item in state["full_attention_checks"]}
        if len(observed) != len(state["full_attention_checks"]) or observed.keys() != set(query_samples):
            raise ValueError("Actual historical Attention observation coverage incomplete")
        for item in observed.values():
            if set(item["stats"]) != {str(i) for i in (3, 7, 11, 15, 19, 23)} or any(
                stat["prefill_calls"] != 1 or stat["cached_calls"] < 1 or stat["query_rows"] < 1 or
                not math.isfinite(stat["max_record_mass"]) or stat["max_record_mass"] <= 0 or stat["max_record_probability"] <= 0
                for stat in item["stats"].values()):
                raise ValueError("Actual six-layer full-prefill/cached historical access failed")
        restored = {item["id"]: item for item in state["native_restore_checks"]}
        if len(restored) != len(state["native_restore_checks"]) or restored.keys() != set(query_samples) or any(
            not item["exact"] or item["native_full_readout"] != queries[key]["native_full_readout"] for key, item in restored.items()):
            raise ValueError("Detached native readouts failed exact full-prefill replay")
        rows = [json.loads(line) for line in (path / (name + ".jsonl")).read_text().splitlines()]
        audit_rows(rows, physical, name, spec)
        rows_by_model[name] = rows
    return {"status": "audited_smoke" if manifest["smoke"] else "completed_and_audited", "source_run": str(path),
            "manifest_sha256": streaming_hash(path / "manifest.json"), "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
            "registry_sha256": registered["registry_sha256"], "weights_sha256": weights["sha256"],
            "physical_readouts_audited": len(physical) * len(MODELS), "reported_bindings_audited": len(bindings) * len(MODELS),
            "pairs_per_model_audited": len(pairs), "prefix_identity_checks": len(prefix_keys),
            "actual_attention_checks": len(query_samples) * len(MODELS), "detached_native_replays": len(query_samples) * len(MODELS),
            "execution_engine": manifest["execution_engine"],
            "adapter_state_sha256": {name: manifest["models"][name]["adapter_state_sha256_after"] for name in MODELS},
            "teacher_train_fit": {name: plan["models"][name]["teacher_train_fit"] for name in MODELS},
            "metrics": summarize(registry, rows_by_model, manifest["smoke"]),
            "interpretation": plan["interpretation"], "analysis_source_sha256": streaming_hash(Path(__file__))}


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
    print(json.dumps({key: report[key] for key in ("status", "physical_readouts_audited", "reported_bindings_audited", "prefix_identity_checks")}))


if __name__ == "__main__":
    main()
