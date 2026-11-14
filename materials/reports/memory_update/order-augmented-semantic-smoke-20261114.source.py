"""Audit same-budget order augmentation against both observed development orders."""

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
from scripts.analyze_controller_trial import load_rows
from scripts.analyze_initial_order_control import audit_run as audit_prior_order, registered_inputs
from scripts.analyze_paired_semantic_controllers import audit_checkpoint, audit_evaluation, audit_run as audit_prior_training, audit_safety
from scripts.analyze_paired_semantic_probe import audit_run as audit_native, identity
from scripts.prepare_order_augmented_semantics import registered_augmented_dataset
from scripts.run_controller_trial import write_json


ARMS = ["shared", "independent", "local", "local_question"]


def order_inputs(plan):
    source = Path(plan["order_control_plan"])
    if file_hash(source) != plan["order_control_plan_sha256"]:
        raise ValueError("Fixed development order-control plan changed")
    order_plan = json.loads(source.read_text())
    registration, base, data, cases, registry = registered_inputs(order_plan)
    if base["data_sha256"] != plan["base_data_sha256"]:
        raise ValueError("Augmentation and order control use different underlying facts")
    examples = [{"case": cases[example["case"]["id"]], "variant": example["variant"], "query": example["query"]} for example in data["validation"]]
    return examples, registry


def reference_audits(plan):
    paths = {"native": "native_reference_run", "training": "training_reference_run", "order": "order_reference_run"}
    for label, field in paths.items():
        if file_hash(Path(plan[field]) / "manifest.json") != plan[f"{label}_reference_manifest_sha256"]:
            raise ValueError("Original development reference manifest changed")
    native = audit_native(Path(plan["native_reference_run"]))
    training = audit_prior_training(Path(plan["training_reference_run"]))
    order = audit_prior_order(Path(plan["order_reference_run"]))
    if any(audit["status"] != "completed_and_audited" for audit in (native, training, order)):
        raise ValueError("Original semantic/order references must complete independent audit first")
    original_plan = json.loads((Path(plan["training_reference_run"]) / "plan.json").read_text())
    if file_hash(Path(plan["training_reference_run"]) / "plan.json") != plan["original_training_plan_sha256"]:
        raise ValueError("Original training protocol changed")
    matched = ("backbone", "group_index", "initialization_seed", "reader_policy", "adapt_gdn_keys", "arms", "unique_training_examples",
               "gradient_accumulation", "maximum_optimizer_steps", "checkpoint_steps", "evaluation_steps", "learning_rate", "weight_decay",
               "max_gradient_norm", "max_input_tokens", "optimized_input_token_budgets", "supervised_answer_token_budgets", "objective")
    if any(plan[key] != original_plan[key] for key in matched):
        raise ValueError("Order augmentation changed a matched training/model/loss budget")
    original_manifest = json.loads((Path(plan["training_reference_run"]) / "manifest.json").read_text())
    if plan["initial_state_sha256"] != {arm: original_manifest["arms"][arm]["initial_state_sha256"] for arm in ARMS}:
        raise ValueError("Paired initialization differs from original training")
    if native["metrics"]["correct"] == native["metrics"]["examples"]:
        raise ValueError("No native task error to repair")
    return {"native": native, "training": training, "order": order}


def audit_order_evaluation(path, arm, plan):
    examples, registry = order_inputs(plan)
    checkpoint = path / arm / f"step-{plan['maximum_optimizer_steps']:06d}"
    evaluation = checkpoint / "initial_order_evaluation"
    summary = json.loads((evaluation / "summary.json").read_text())
    record = json.loads((checkpoint / "checkpoint.json").read_text())
    if (summary["status"] != "completed" or summary["arm"] != arm or summary["step"] != plan["maximum_optimizer_steps"] or
        summary["condition"] != "initial_record_derangement" or not summary["adapter_unchanged"] or not summary["training_rng_unchanged"] or
        summary["evaluation_state_sha256"] != record["state_sha256"] or
        summary["results_sha256"] != file_hash(evaluation / "validation.jsonl")):
        raise ValueError("Final frozen augmented-order evaluation identity failed")
    expected = {tuple(identity(example)): example["query"]["answer"] for example in examples}
    rows = load_rows(evaluation / "validation.jsonl", expected)
    for row, example, registered in zip(rows, examples, registry):
        if ([row["case_id"], row["variant"], row["role"]] != identity(example) or row["registration"] != registered or
            row["condition"] != summary["condition"] or row["reader_policy"] != "full" or
            any(row[field] != example["case"][field] for field in ("base_case_id", "family", "seed", "value_version", "distractor_updates"))):
            raise ValueError("Augmented-order evaluated inputs/metadata changed")
    keys = sorted(expected)
    if [tuple(check["key"]) for check in summary["full_attention_checks"]] != [keys[0], keys[-1]]:
        raise ValueError("Missing prescribed permuted full-attention samples")
    for check in summary["full_attention_checks"]:
        if set(check["weights"]) != {str(index) for index in (3, 7, 11, 15, 19, 23)}:
            raise ValueError("Missing permuted full-attention layer")
        for stats in check["weights"].values():
            if stats["prefill_calls"] != 1 or stats["cached_calls"] < 1 or stats["max_record_mass"] <= 0:
                raise ValueError("No observed actual full historical access")
    checks = summary["pre_question_prefix_checks"]
    if plan["arms"][arm]["token_policy"] == "question":
        if len(checks) != 2 or any(not check["kv_recurrent_conv_and_logits_exact"] for check in checks):
            raise ValueError("Augmented question-only permuted prefix identity failed")
    elif checks:
        raise ValueError("Unexpected all-token prefix identity claim")
    metrics = summarize_semantics(rows)
    if metrics != summary["semantic_metrics"]:
        raise ValueError("Augmented-order summary differs from raw free generations")
    return {"metrics": metrics, "evaluation_summary_sha256": file_hash(evaluation / "summary.json")}


def audit_run(path, allow_smoke=False):
    manifest = json.loads((path / "manifest.json").read_text())
    plan = json.loads((path / "plan.json").read_text())
    registered, data, registry = registered_augmented_dataset(Path(plan["data_source"]))
    references = reference_audits(plan)
    if (manifest["status"] != "completed" or manifest["completed_arms"] != ARMS or manifest["reader_policy"] != "full" or
        manifest["plan_sha256"] != data_fingerprint(plan) or manifest["native_reference"] != references["native"] or
        json.loads((path / "reference_audits.json").read_text()) != references or
        plan["profile"] != "ordinary_initial_order_augmentation" or
        manifest["model_path"] != registered["model_path"] or manifest["data_manifest_sha256"] != file_hash(Path(plan["data_source"]) / "manifest.json")):
        raise ValueError("Incomplete or changed order-augmentation trial")
    if manifest["smoke"] and not allow_smoke:
        raise ValueError("One-update safety smoke is not scheduled training")
    for stamp in (registered["registered_utc"], plan["registered_utc"]):
        if datetime.fromisoformat(stamp) >= datetime.fromisoformat(manifest["started_utc"]):
            raise ValueError("Augmentation data/protocol was not registered before training")
    native_manifest = json.loads((Path(plan["native_reference_run"]) / "manifest.json").read_text())
    if manifest["packages"] != native_manifest["packages"]:
        raise ValueError("Matched runtime packages changed")
    for relative, digest in manifest["source_sha256"].items():
        if file_hash(path / "sources" / relative) != digest or file_hash(ROOT / relative) != digest:
            raise ValueError("Actual/archived augmented training source changed")
    for key in ("data_sha256", "registry_sha256", "tokenized_input_answer_sha256"):
        if plan[key] != registered[key]:
            raise ValueError("Augmentation plan/data changed")
    encoded = json.loads((Path(plan["data_source"]) / "tokenized_train.json").read_text())
    if encoded != json.loads((path / "tokenized_train.json").read_text()):
        raise ValueError("Actual optimized training tokens changed")
    models = {}
    for arm in ARMS:
        audit_safety(json.loads((path / arm / "safety_check.json").read_text()), plan["arms"][arm])
        if manifest["smoke"]:
            models[arm] = {"parameters": plan["arms"][arm]["parameters"], "status": "safety_passed"}
            continue
        state = manifest["arms"][arm]
        if (state["completed_checkpoints"] != plan["checkpoint_steps"] or not state["native_restore_exact"] or
            state["initial_state_sha256"] != plan["initial_state_sha256"][arm] or
            len((path / arm / "training.jsonl").read_text().splitlines()) != plan["maximum_optimizer_steps"]):
            raise ValueError("Augmented training initialization/schedule/native restoration failed")
        for step in plan["checkpoint_steps"]:
            audit_checkpoint(path, arm, step, plan, encoded)
        canonical = audit_evaluation(path, arm, plan, data, encoded, registry)
        order = audit_order_evaluation(path, arm, plan)
        if state["extra_validation"] != order:
            raise ValueError("Saved augmented-order evaluation descriptor changed")
        models[arm] = {"parameters": plan["arms"][arm]["parameters"], "checkpoints_audited": len(plan["checkpoint_steps"]),
                       "teacher_train_fit": canonical["teacher_train_fit"], "canonical": canonical, "permuted": order}
    return {"status": "audited_smoke" if manifest["smoke"] else "completed_and_audited", "source_run": str(path),
            "manifest_sha256": file_hash(path / "manifest.json"), "arms": models,
            "validation_records_audited": 0 if manifest["smoke"] else len(ARMS) * 2 * len(data["validation"]),
            "teacher_train_records_audited": 0 if manifest["smoke"] else len(ARMS) * len(data["train"]),
            "initial_state_matches_reference": not manifest["smoke"], "interpretation": plan["interpretation"]}


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
    print(json.dumps({"status": report["status"], "arms": {arm: {condition: value[condition]["metrics"]["correct"] for condition in ("canonical", "permuted")} if "canonical" in value else value for arm, value in report["arms"].items()}}))


if __name__ == "__main__":
    main()
