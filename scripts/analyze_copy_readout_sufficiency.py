"""Independent all-input grading, mask/cached-access and frozen-memory audit."""

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
from scripts.copy_readout_data import MODELS, PROTOCOL, REGIMES, read_plan, registered_inputs, sample_ids, safety_samples, selected_inputs
from scripts.copy_readout_metrics import summarize
from scripts.copy_readout_runtime import check_weight_stats, readout
from scripts.run_controller_trial import write_json
from scripts.run_copy_readout_sufficiency import RUNTIME_SOURCES


def audit_rows(rows, physical, masks, model, regime, policy):
    if [r["id"] for r in rows] != [q["id"] for q in physical]:
        raise ValueError("Complete ordered physical readouts required")
    for row, q in zip(rows, physical):
        match = re.fullmatch(r"\s*(-?\d+)\s*", row["generated"])
        if (row["model"] != model or row["regime"] != regime or row["token_policy"] != policy or row["prefill"] != "full" or
            row["registration_sha256"] != data_fingerprint(q) or row["mask_registration_sha256"] != data_fingerprint(masks[q["id"]]) or
            row["keep_bits_sha256"] != data_fingerprint(masks[q["id"]]["keep_bits"][regime]) or
            row["answer"] != q["answer"] or row["query_tokens"] != q["question_tokens"] or
            row["greedy_exact"] != bool(match and match.group(1) == q["answer"]) or not math.isfinite(row["answer_nll"]) or row["answer_nll"] < 0 or
            row["hook_calls"]["prefill"] != 1 or row["hook_calls"]["cached"] < 1):
            raise ValueError("Own-world grading, exact mask, boundary or actual cached access changed")


def audit_proofs(proofs, sample_keys, full_reference, indexed_rows):
    if [p["id"] for p in proofs] != sample_keys:
        raise ValueError("Complete prescribed all-role preservation samples required")
    for p in proofs:
        if (not p["unmasked_reference_exact"] or p["unmasked_readout"] != readout(full_reference[p["id"]]) or
            not p["all_history_cache_logits_readout_exact"] or p["all_history_hook_calls"]["prefill"] != 2 or
            p["all_history_hook_calls"]["cached"] < 1 or not p["hooks_restored"] or
            list(p["regimes"]) != list(REGIMES) or list(p["unmasked_weights"]) != list(REGIMES)):
            raise ValueError("Actual full/all-history controls or hook restoration failed")
        hashes = set()
        for r in REGIMES:
            check_weight_stats(p["unmasked_weights"][r], False)
            s = p["regimes"][r]
            if (not s["complete_prefix_memory_exact"] or not s["prefix_logits_exact"] or s["historical_kv_fields_exact"] != 12 or
                s["prefix_sha256"] != s["masked_prefix_sha256"] or not re.fullmatch(r"[0-9a-f]{64}", s["prefix_sha256"]) or
                s["proof_hook_calls"] != {"prefill": 2, "cached": 0} or
                s["readout"] != readout(indexed_rows[r][p["id"]]) or s["measure_hook_calls"] != indexed_rows[r][p["id"]]["hook_calls"]):
                raise ValueError("Complete recurrent/conv/KV/bookkeeping preservation or observed readout changed")
            check_weight_stats(s["attention_stats"], True)
            hashes.add(s["prefix_sha256"])
        if len(hashes) != 1:
            raise ValueError("The two interventions used different original prefix states")


def audit_run(path, allow_smoke=False):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "completed" or (manifest["smoke"] and not allow_smoke):
        raise ValueError("Complete readout evaluation required; smoke is operational only")
    plan = read_plan()
    registered, source_plan, cases, registry, masks, full_rows = registered_inputs(Path(manifest["dataset"]))
    physical, bindings = selected_inputs(registry, manifest["smoke"])
    samples, prefix_samples = sample_ids(cases, registry), safety_samples(registry)
    queries = {q["id"]: q for q in physical}
    sources = set(RUNTIME_SOURCES) | {str(p.relative_to(ROOT)) for p in (ROOT / "sas_memory").glob("*.py")}
    weights = manifest["weights_after"]
    if (manifest["training"] or manifest["new_adapters"] or manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or
        manifest["dataset_manifest_sha256"] != streaming_hash(Path(manifest["dataset"]) / "manifest.json") or
        manifest["mask_registry_sha256"] != registered["mask_registry_sha256"] or manifest["references"] != registered["references"] or
        json.loads((path / "plan.json").read_text()) != plan or manifest["model_path"] != plan["model_path"] or manifest["packages"] != plan["packages"] or
        manifest["selected_physical_ids"] != list(queries) or manifest["bindings_per_model_regime"] != len(bindings) or
        manifest["sample_query_ids"] != samples or manifest["native_prefix_sample_ids"] != prefix_samples or
        manifest["generation"] != plan["generation"] or manifest["completed_models"] != list(MODELS) or list(manifest["models"]) != list(MODELS) or
        set(manifest["source_sha256"]) != sources or manifest["weights_before"] != weights or
        weights["sha256"] != plan["native_loaded_weights_sha256"] or not weights["all_parameters_frozen"] or not weights["no_parameter_gradients"] or
        not manifest["hooks_restored"] or not (datetime.fromisoformat(plan["registered_utc"]) < datetime.fromisoformat(registered["registered_utc"]) <
        datetime.fromisoformat(manifest["started_utc"]) < datetime.fromisoformat(manifest["completed_utc"]))):
        raise ValueError("Frozen readout scope, full source coverage, native runtime or registration changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / "sources" / name) != digest:
            raise ValueError("Actual/archived readout execution source changed")
    if manifest["generation_config_sha256"] != {p.name: streaming_hash(p) for p in Path(plan["model_path"]).glob("generation_config.json")}:
        raise ValueError("Actual native generation config changed")
    if manifest["smoke"]:
        if manifest["smoke_gate"] is not None:
            raise ValueError("Operational smoke has an unexpected earlier gate")
    else:
        gate = manifest["smoke_gate"]
        gate_path = Path(gate["path"])
        report = json.loads(gate_path.read_text())
        if (streaming_hash(gate_path) != gate["sha256"] or report["manifest_sha256"] != gate["manifest_sha256"] or
            report["status"] != "audited_smoke" or report["new_physical_readouts_audited"] != 288 or
            report["dataset_manifest_sha256"] != manifest["dataset_manifest_sha256"] or audit_run(Path(report["source_run"]), True) != report):
            raise ValueError("Complete independent288-query smoke gate changed")
    refs = {m: {r["id"]: r for r in full_rows[m]} for m in MODELS}
    source_manifest = json.loads((Path(plan["source_run"]) / "manifest.json").read_text())
    all_rows = {}
    for name in MODELS:
        state, spec = manifest["models"][name], source_plan["models"][name]
        if (state["completed_queries_per_regime"] != len(physical) or not state["hooks_restored"] or
            state["token_policy"] != (spec["token_policy"] if spec else "none") or list(state["results_sha256"]) != list(REGIMES)):
            raise ValueError("Actual fixed model/regime coverage changed")
        if spec:
            training = json.loads((Path(spec["training_run"]) / "manifest.json").read_text())
            if (state["description"] != training["arms"][spec["training_arm"]]["description"] or state["adapter_parameters"] != spec["parameters"] or
                state["checkpoint_sha256"] != spec["checkpoint_sha256"] or state["adapter_state_sha256_before"] != state["adapter_state_sha256_after"] or
                state["adapter_state_sha256_after"] != registered["references"]["training"][name]["checkpoint_state_sha256"] or not state["all_adapter_parameters_frozen"]):
                raise ValueError("Actual adapter tensors, allocation or gradients changed")
        elif any(k.startswith(("adapter_", "checkpoint_")) for k in state):
            raise ValueError("Native arm unexpectedly has adapter tensors")
        all_rows[name] = {}
        for r in REGIMES:
            result_path = path / f"{name}.{r}.jsonl"
            if streaming_hash(result_path) != state["results_sha256"][r]:
                raise ValueError("Actual recorded readout observations changed")
            rows = [json.loads(line) for line in result_path.read_text().splitlines()]
            audit_rows(rows, physical, masks, name, r, state["token_policy"])
            all_rows[name][r] = rows
        indexed = {r: {row["id"]: row for row in rows} for r, rows in all_rows[name].items()}
        audit_proofs(state["proofs"], samples, refs[name], indexed)
        prefixes = [queries[i]["source_key"] for i in prefix_samples] if spec and spec["token_policy"] == "question" else []
        if [p["source_key"] for p in state["prefix_checks"]] != prefixes:
            raise ValueError("Question-only native-prefix sample coverage incomplete")
        for p in state["prefix_checks"]:
            q = next(q for q in physical if q["source_key"] == p["source_key"])
            if p["ledger_tokens"] != q["ledger_tokens"] or not p["complete_KV_recurrent_conv_bookkeeping_exact"] or not p["last_prefix_logits_exact"] or not re.fullmatch(r"[0-9a-f]{64}", p["prefix_sha256"]):
                raise ValueError("Complete question-only native prefix or logits changed")
        if prefixes:
            old = {p["source_key"]: p["prefix_sha256"] for p in source_manifest["models"][name]["prefix_checks"]}
            if any(p["regimes"]["oracle"]["prefix_sha256"] != old[queries[p["id"]]["source_key"]] for p in state["proofs"]):
                raise ValueError("Question-only original prefix differs from the sealed full-history source")
        if [p["id"] for p in state["native_restore_checks"]] != samples or any(
            not p["exact"] or p["readout"] != readout(refs["native"][p["id"]]) for p in state["native_restore_checks"]):
            raise ValueError("Detached original native full-history readouts failed exact replay")
    metrics = ({"primary_evaluated": False} if manifest["smoke"] else summarize(all_rows, full_rows, cases, registry, source_plan, plan))
    return {"status": "audited_smoke" if manifest["smoke"] else "completed_and_audited", "source_run": str(path),
        "manifest_sha256": streaming_hash(path / "manifest.json"), "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "mask_registry_sha256": registered["mask_registry_sha256"], "weights_sha256": weights["sha256"], "references": registered["references"],
        "new_physical_readouts_audited": len(physical) * 16, "new_related_bindings_audited": len(bindings) * 16,
        "full_reference_queries_per_model": 336, "all_history_exact_controls": 80, "actual_masked_attention_checks": 160,
        "complete_prefix_and_historical_KV_proofs": 160, "detached_native_exact_replays": 80, "question_only_native_prefix_checks": 12,
        "metrics": metrics, "interpretation": plan["scope"], "analysis_source_sha256": streaming_hash(Path(__file__))}


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
    print(json.dumps({k: report[k] for k in ("status", "new_physical_readouts_audited", "actual_masked_attention_checks", "complete_prefix_and_historical_KV_proofs")}), flush=True)


if __name__ == "__main__":
    main()
