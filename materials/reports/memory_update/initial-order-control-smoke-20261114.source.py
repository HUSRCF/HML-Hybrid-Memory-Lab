"""Audit frozen canonical-to-permuted semantic readouts without retraining."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from sas_memory.semantic_metrics import summarize_semantics
from sas_memory.training_data import data_fingerprint
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.analyze_controller_trial import load_rows
from scripts.analyze_paired_semantic_controllers import audit_run as audit_training
from scripts.analyze_paired_semantic_probe import identity, registered_dataset
from scripts.compare_paired_semantic_controllers import functional_units, paired_counts
from scripts.prepare_initial_order_control import build_cases, token_registry
from scripts.run_controller_trial import write_json


MODELS = ["native", "shared", "independent", "local", "local_question"]


def registered_inputs(plan):
    path = Path(plan["data_source"])
    manifest = json.loads((path / "manifest.json").read_text())
    base_path = Path(manifest["base_data"])
    base, data, original_registry = registered_dataset(base_path)
    if (manifest["status"] != "registered_before_inference" or manifest["model_inference"] or manifest["training"] or
        manifest["base_data_manifest_sha256"] != file_hash(base_path / "manifest.json") or manifest["base_data_sha256"] != base["data_sha256"] or
        plan["models"] != MODELS or plan["training"] or plan["reader_policy"] != "full" or
        plan["data_manifest_sha256"] != file_hash(path / "manifest.json") or plan["examples_per_model"] != len(data["validation"]) or
        manifest["model_path"] != base["model_path"] or manifest["condition"] != plan["condition"]):
        raise ValueError("Initial-order registration/base/model changed")
    for relative, digest in manifest["source_sha256"].items():
        if file_hash(path / "sources" / relative) != digest or file_hash(ROOT / relative) != digest:
            raise ValueError("Registered order-control source changed")
    cases, permutations = build_cases(data)
    registry = token_registry(tokenizer_runtime(base["model_path"]), data, cases, original_registry)
    for name, value in (("cases", cases), ("permutations", permutations), ("registry", registry)):
        if json.loads((path / f"{name}.json").read_text()) != value or data_fingerprint(value) != manifest[f"{name}_sha256"]:
            raise ValueError("Order-control truth/permutation/tokens changed")
        if name in ("cases", "registry") and plan[f"{name}_sha256"] != manifest[f"{name}_sha256"]:
            raise ValueError("Order-control plan/data changed")
    return manifest, base, data, cases, registry


def audited_references(plan):
    run = Path(plan["training_reference_run"])
    audit = audit_training(run)
    training_plan = json.loads((run / "plan.json").read_text())
    if (audit["status"] != "completed_and_audited" or set(audit["arms"]) != set(MODELS[1:]) or
        file_hash(run / "plan.json") != plan["training_plan_sha256"] or
        training_plan["native_reference_run"] != plan["native_reference_run"] or training_plan["group_index"] != plan["group_index"]):
        raise ValueError("Original matched trial is incomplete or changed")
    for name in MODELS[1:]:
        spec = plan["checkpoint_sources"][name]
        if (any(spec[key] != training_plan["arms"][name][key] for key in ("mode", "rank", "token_policy", "parameters")) or
            Path(spec["weights"]) != run / name / "step-000432/controller.safetensors" or file_hash(spec["weights"]) != spec["weights_sha256"]):
            raise ValueError("Pinned frozen checkpoint changed")
    return audit


def members(data, smoke):
    if not smoke:
        return data["validation"]
    group = max(example["case"]["base_case_id"] for example in data["validation"] if example["case"]["family"] == "copy")
    return [example for example in data["validation"] if example["case"]["base_case_id"] == group and example["variant"] == "control" and example["query"]["role"] == "target_current"]


def raw_metrics(rows, smoke):
    if smoke:
        return {"examples": len(rows), "correct": sum(row["greedy_exact"] for row in rows), "operational_smoke_only": True}
    return summarize_semantics(rows)


def canonical_rows(plan, expected):
    paths = {"native": Path(plan["native_reference_run"]) / "validation.native.jsonl"}
    paths.update({name: Path(plan["training_reference_run"]) / name / "step-000432/evaluation/validation.jsonl" for name in MODELS[1:]})
    return {name: [row for row in load_rows(path, {(r["case_id"], r["variant"], r["role"]): r["answer"] for r in map(json.loads, path.read_text().splitlines())})
                   if (row["case_id"], row["variant"], row["role"]) in expected] for name, path in paths.items()}


def audit_run(path, allow_smoke=False):
    manifest = json.loads((path / "manifest.json").read_text())
    plan = json.loads((path / "plan.json").read_text())
    registered, base, data, cases, registry = registered_inputs(plan)
    references = audited_references(plan)
    if (manifest["status"] != "completed" or manifest["training"] or manifest["completed_models"] != MODELS or
        manifest["plan_sha256"] != data_fingerprint(plan) or manifest["training_reference_audit"] != references or
        manifest["model_path"] != base["model_path"] or not manifest["frozen_parameters_and_no_gradients"]):
        raise ValueError("Incomplete/changed frozen order-control run")
    if manifest["smoke"] and not allow_smoke:
        raise ValueError("Operational smoke is not full semantic evidence")
    for stamp in (plan["registered_utc"], registered["registered_utc"]):
        if datetime.fromisoformat(stamp) >= datetime.fromisoformat(manifest["started_utc"]):
            raise ValueError("Condition was not registered before inference")
    native_manifest = json.loads((Path(plan["native_reference_run"]) / "manifest.json").read_text())
    if manifest["packages"] != native_manifest["packages"]:
        raise ValueError("Runtime package versions changed")
    for relative, digest in manifest["source_sha256"].items():
        if file_hash(path / "sources" / relative) != digest or file_hash(ROOT / relative) != digest:
            raise ValueError("Actual/archived order-control source changed")
    selected = members(data, manifest["smoke"])
    expected = {tuple(identity(example)): example["query"]["answer"] for example in selected}
    sample_keys = [sorted(expected)[0], sorted(expected)[-1]]
    indexed = {(row["case_id"], row["variant"], row["role"]): row for row in registry}
    evaluated, metrics = {}, {}
    for name in MODELS:
        state = manifest["models"][name]
        output = path / f"{name}.jsonl"
        if file_hash(output) != state["results_sha256"] or not state["weights_unchanged"] or not state["native_restore_exact"]:
            raise ValueError("Frozen model/output/restoration changed")
        if name != "native" and (state["checkpoint"] != plan["checkpoint_sources"][name] or state["state_sha256_before"] != state["state_sha256_after"]):
            raise ValueError("Loaded checkpoint or evaluated adapter state changed")
        rows = load_rows(output, expected)
        if [tuple((row["case_id"], row["variant"], row["role"])) for row in rows] != list(expected):
            raise ValueError("Order-control evaluation order changed")
        for row, example in zip(rows, selected):
            key = tuple(identity(example))
            if (row["registration"] != indexed[key] or row["condition"] != plan["condition"] or row["reader_policy"] != "full" or
                any(row[field] != example["case"][field] for field in ("family", "base_case_id", "value_version", "seed", "distractor_updates"))):
                raise ValueError("Actual order-control prompt/metadata changed")
        if [tuple(check["key"]) for check in state["full_attention_checks"]] != sample_keys:
            raise ValueError("Missing prescribed full-history controls")
        for check in state["full_attention_checks"]:
            if set(check["weights"]) != {str(index) for index in (3, 7, 11, 15, 19, 23)}:
                raise ValueError("Missing observed full-attention layer")
            for stats in check["weights"].values():
                if stats["prefill_calls"] != 1 or stats["cached_calls"] < 1 or stats["max_record_mass"] <= 0:
                    raise ValueError("Full-history access was not observed")
        prefix = state["pre_question_prefix_checks"]
        if name == "local_question":
            if len(prefix) != 2 or any(not check["kv_recurrent_conv_and_logits_exact"] for check in prefix):
                raise ValueError("Question-only prefix identity failed")
        elif prefix:
            raise ValueError("Unexpected prefix identity claim")
        metrics[name] = raw_metrics(rows, manifest["smoke"])
        if metrics[name] != state["metrics"]:
            raise ValueError("Summary differs from raw free generations")
        evaluated[name] = rows
    report = {"status": "audited_operational_smoke" if manifest["smoke"] else "completed_and_audited", "source_run": str(path),
              "manifest_sha256": file_hash(path / "manifest.json"), "models": metrics, "records_audited": len(selected) * len(MODELS),
              "interpretation": plan["interpretation"]}
    canonical = canonical_rows(plan, expected)
    if not manifest["smoke"]:
        original_units = {name: functional_units(rows) for name, rows in canonical.items()}
        permuted_units = {name: functional_units(rows) for name, rows in evaluated.items()}
        report["canonical_to_permuted"] = {name: {metric: paired_counts(original_units[name][metric], permuted_units[name][metric]) for metric in original_units[name]} for name in MODELS}
        report["permuted_functional_units"] = {name: {metric: {"correct": sum(values.values()), "units": len(values)} for metric, values in units.items()} for name, units in permuted_units.items()}
        report["shared_vs_controls"] = {name: {metric: paired_counts(permuted_units[name][metric], permuted_units["shared"][metric]) for metric in permuted_units[name]} for name in MODELS if name != "shared"}
        report["family_role_counts"] = {name: [{"family": family, "role": role, "correct": sum(row["greedy_exact"] for row in rows if row["family"] == family and row["role"] == role), "examples": sum(row["family"] == family and row["role"] == role for row in rows)} for family in sorted({row["family"] for row in rows}) for role in ("target_current", "protected_current", "target_initial")] for name, rows in evaluated.items()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    report = audit_run(args.run, args.allow_smoke)
    report["analysis_source_sha256"] = file_hash(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status": report["status"], "records": report["records_audited"], "correct": {name: value["correct"] for name, value in report["models"].items()}}))


if __name__ == "__main__":
    main()
