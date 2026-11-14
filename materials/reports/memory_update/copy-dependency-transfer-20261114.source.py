"""Independently reconstruct every existing COPY input and fixed-model comparison."""

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
from scripts.analyze_write_locality_adapter_transfer import audit_rows
from scripts.copy_transfer_data import MODELS, PROTOCOL, read_plan, registered_inputs, safety_samples, selected_inputs
from scripts.copy_transfer_metrics import summarize
from scripts.run_controller_trial import write_json
from scripts.run_copy_dependency_transfer import RUNTIME_SOURCES


def audit_run(path, allow_smoke=False):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "completed" or (manifest["smoke"] and not allow_smoke):
        raise ValueError("Completed COPY transfer required; smoke is operational only")
    plan = read_plan()
    registered, cases, registry = registered_inputs(Path(manifest["dataset"]))
    physical, bindings = selected_inputs(registry, manifest["smoke"])
    sample_ids = safety_samples(registry)
    queries = {q["id"]: q for q in physical}
    source_names = set(RUNTIME_SOURCES) | {str(p.relative_to(ROOT)) for p in (ROOT / "sas_memory").glob("*.py")}
    weights = manifest["weights_after"]
    if (manifest["training"] or manifest["new_adapters"] or manifest["reader_policy"] != "full" or
        manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or json.loads((path / "plan.json").read_text()) != plan or
        manifest["dataset_manifest_sha256"] != streaming_hash(Path(manifest["dataset"]) / "manifest.json") or
        manifest["registry_sha256"] != registered["registry_sha256"] or manifest["references"] != registered["references"] or
        manifest["selected_physical_ids"] != list(queries) or manifest["bindings_per_model"] != len(bindings) or
        manifest["safety_sample_ids"] != sample_ids or manifest["completed_models"] != list(MODELS) or
        list(manifest["models"]) != list(MODELS) or set(manifest["source_sha256"]) != source_names or
        manifest["generation"] != plan["generation"] or manifest["model_path"] != plan["model_path"] or
        manifest["packages"] != plan["packages"] or manifest["weights_before"] != weights or
        weights["sha256"] != plan["native_loaded_weights_sha256"] or not weights["all_parameters_frozen"] or
        not weights["no_parameter_gradients"] or not manifest["hooks_restored"] or
        not (datetime.fromisoformat(plan["registered_utc"]) < datetime.fromisoformat(registered["registered_utc"]) <
             datetime.fromisoformat(manifest["started_utc"]) < datetime.fromisoformat(manifest["completed_utc"]))):
        raise ValueError("Complete COPY scope, model freeze, runtime or pre-inference registration changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / "sources" / name) != digest:
            raise ValueError("Actual/archived COPY execution implementation changed")
    configs = {p.name: streaming_hash(p) for p in Path(plan["model_path"]).glob("generation_config.json")}
    if manifest["generation_config_sha256"] != configs:
        raise ValueError("Actual native generation configuration changed")
    if manifest["smoke"]:
        if manifest["smoke_gate"] is not None:
            raise ValueError("Smoke unexpectedly claims an earlier gate")
    else:
        gate = manifest["smoke_gate"]
        gate_path = Path(gate["path"])
        report = json.loads(gate_path.read_text())
        if (streaming_hash(gate_path) != gate["sha256"] or report["manifest_sha256"] != gate["manifest_sha256"] or
            report["status"] != "audited_smoke" or report["physical_readouts_audited"] != 144 or
            report["dataset_manifest_sha256"] != manifest["dataset_manifest_sha256"] or
            audit_run(Path(report["source_run"]), True) != report):
            raise ValueError("Complete independent144-query COPY smoke gate changed")
    all_rows, native = {}, {}
    for name in MODELS:
        spec, state = plan["models"][name], manifest["models"][name]
        if (state["completed_readouts"] != len(physical) or not state["hooks_restored"] or
            streaming_hash(path / (name + ".jsonl")) != state["results_sha256"] or
            state["token_policy"] != (spec["token_policy"] if spec else "none")):
            raise ValueError("Actual fixed model completion or result file changed")
        if spec:
            training = json.loads((Path(spec["training_run"]) / "manifest.json").read_text())
            if (state["description"] != training["arms"][spec["training_arm"]]["description"] or
                state["adapter_parameters"] != spec["parameters"] or state["checkpoint_sha256"] != spec["checkpoint_sha256"] or
                state["adapter_state_sha256_before"] != state["adapter_state_sha256_after"] or
                state["adapter_state_sha256_after"] != registered["references"]["training"][name]["checkpoint_state_sha256"] or
                not state["all_adapter_parameters_frozen"]):
                raise ValueError("Actual fixed adapter tensors, scope or gradients changed")
        elif any(k.startswith(("adapter_", "checkpoint_")) for k in state):
            raise ValueError("Native arm unexpectedly contains adapter tensors")
        rows = [json.loads(line) for line in (path / (name + ".jsonl")).read_text().splitlines()]
        audit_rows(rows, physical, name, spec or {"token_policy": "none"})
        all_rows[name] = rows
        if name == "native":
            native = {r["id"]: {k: r[k] for k in ("answer", "answer_nll", "greedy_exact", "generated", "query_tokens")} for r in rows}
        prefixes = [queries[q]["source_key"] for q in sample_ids] if spec and spec["token_policy"] == "question" else []
        if [p["source_key"] for p in state["prefix_checks"]] != prefixes:
            raise ValueError("All prescribed native-prefix checks are required")
        for check in state["prefix_checks"]:
            q = next(q for q in physical if q["source_key"] == check["source_key"])
            if (check["ledger_tokens"] != q["ledger_tokens"] or not check["complete_KV_recurrent_conv_bookkeeping_exact"] or
                not check["last_prefix_logits_exact"] or not re.fullmatch(r"[0-9a-f]{64}", check["prefix_sha256"])):
                raise ValueError("Complete native-prefix state, convolution cache or logits differ")
        if [c["id"] for c in state["attention_checks"]] != sample_ids or [c["id"] for c in state["native_restore_checks"]] != sample_ids:
            raise ValueError("Full historical-access or native replay sample coverage changed")
        for check in state["attention_checks"]:
            if set(check["stats"]) != {str(i) for i in (3, 7, 11, 15, 19, 23)} or any(
                s["prefill_calls"] != 1 or s["cached_calls"] < 1 or s["query_rows"] < 1 or
                not math.isfinite(s["max_record_mass"]) or s["max_record_mass"] <= 0 or s["max_record_probability"] <= 0
                for s in check["stats"].values()):
                raise ValueError("Actual six-layer full/cached historical access failed")
        if any(not c["exact"] or c["native_readout"] != native[c["id"]] for c in state["native_restore_checks"]):
            raise ValueError("Detached native readout failed exact replay")
    metrics = ({"primary_evaluated": False, "physical_queries_per_model": len(physical)} if manifest["smoke"]
               else summarize(all_rows, cases, registry, plan))
    return {"status": "audited_smoke" if manifest["smoke"] else "completed_and_audited", "source_run": str(path),
        "manifest_sha256": streaming_hash(path / "manifest.json"), "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "registry_sha256": registered["registry_sha256"], "weights_sha256": weights["sha256"],
        "physical_readouts_audited": len(physical) * len(MODELS), "reported_bindings_audited": len(bindings) * len(MODELS),
        "actual_attention_checks": 6 * len(MODELS), "detached_native_replays": 6 * len(MODELS), "native_prefix_identity_checks": 12,
        "references": registered["references"], "metrics": metrics, "interpretation": plan["scope"],
        "analysis_source_sha256": streaming_hash(Path(__file__))}


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
    print(json.dumps({k: report[k] for k in ("status", "physical_readouts_audited", "reported_bindings_audited", "actual_attention_checks")}), flush=True)


if __name__ == "__main__":
    main()
