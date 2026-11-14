"""Audit matched shared/independent/local semantic-update feasibility trials."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash, microbatch_indices
from sas_memory.semantic_metrics import summarize_semantics
from sas_memory.training_data import data_fingerprint
from scripts.analyze_controller_trial import load_rows
from scripts.analyze_paired_semantic_probe import audit_run as audit_native, identity, registered_dataset
from scripts.run_controller_trial import write_json


def audit_safety(check, arm):
    if (check["status"] != "passed" or check["initial_logit_max_error"] != 0 or
        check["trainable_parameters"] != arm["parameters"] or not check["backbone_gradients_absent"] or
        check["mode"] != arm["mode"] or check["rank"] != arm["rank"] or
        check["token_policy"] != arm["token_policy"] or check["reader_policy"] != "full"):
        raise ValueError("Semantic controller safety/parameter allocation failed")
    for field in ("output_gradient_norms", "encoder_gradient_norms_after_one_step"):
        if not check[field] or any(value is None or not math.isfinite(value) or value <= 0 for value in check[field].values()):
            raise ValueError("Missing finite controller gradients")
    if arm["token_policy"] == "question" and check["pre_question_cache_and_logits_exact_after_update"] is not True:
        raise ValueError("Question-only one-update prefix identity failed")


def audit_checkpoint(path, arm, step, plan, encoded):
    import torch
    from safetensors import safe_open
    checkpoint = path / arm / f"step-{step:06d}"
    record = json.loads((checkpoint / "checkpoint.json").read_text())
    spec = plan["arms"][arm]
    if (record["arm"] != arm or record["step"] != step or record["policy"] != spec["token_policy"] or
        record["plan_sha256"] != data_fingerprint(plan) or
        file_hash(checkpoint / "controller.safetensors") != record["controller_sha256"] or
        file_hash(checkpoint / "optimizer.pt") != record["optimizer_sha256"]):
        raise ValueError("Semantic checkpoint identity changed")
    log = (path / arm / "training.jsonl").read_bytes()[:record["training_log_prefix_bytes"]]
    if hashlib.sha256(log).hexdigest() != record["training_log_prefix_sha256"]:
        raise ValueError("Semantic training-log prefix changed")
    rows = [json.loads(line) for line in log.splitlines()]
    if [row["step"] for row in rows] != list(range(1, step + 1)):
        raise ValueError("Incomplete semantic optimizer steps")
    input_tokens = answer_tokens = 0
    for number, row in enumerate(rows, 1):
        indices = microbatch_indices(number, len(encoded), plan["gradient_accumulation"])
        expected = [{"training_index": index, "case_id": encoded[index]["case_id"],
                     "variant": encoded[index]["variant"], "role": encoded[index]["role"]} for index in indices]
        if row["microbatches"] != expected or any(not math.isfinite(row[key]) or row[key] < 0 for key in ("loss", "gradient_norm")):
            raise ValueError("Semantic training order/loss/gradient changed")
        input_tokens += sum(len(encoded[index]["input_ids"]) for index in indices)
        answer_tokens += sum(len(encoded[index]["answer_ids"]) for index in indices)
    if (input_tokens != plan["optimized_input_token_budgets"][str(step)] or input_tokens != record["optimized_input_tokens"] or
        answer_tokens != plan["supervised_answer_token_budgets"][str(step)] or answer_tokens != record["supervised_answer_tokens"]):
        raise ValueError("Semantic optimization token budget changed")
    saved = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer, names = saved["optimizer"], record["optimizer_parameter_names"]
    if len(optimizer["param_groups"]) != 1:
        raise ValueError("Unexpected optimizer groups")
    group = optimizer["param_groups"][0]
    if (len(group["params"]) != len(names) or group["lr"] != plan["learning_rate"] or group["weight_decay"] != plan["weight_decay"]):
        raise ValueError("Semantic optimizer settings changed")
    with safe_open(checkpoint / "controller.safetensors", framework="pt", device="cpu") as weights:
        if set(weights.keys()) != set(names) or sum(weights.get_tensor(name).numel() for name in names) != spec["parameters"]:
            raise ValueError("Semantic adapter checkpoint shapes/count changed")
        state_digest = hashlib.sha256()
        for name in sorted(names):
            tensor = weights.get_tensor(name)
            state_digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
            state_digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        if state_digest.hexdigest() != record["state_sha256"]:
            raise ValueError("Semantic adapter state fingerprint changed")
        for index, name in zip(group["params"], names):
            tensor, moment = weights.get_tensor(name), optimizer["state"][index]
            if (not torch.isfinite(tensor).all() or moment["step"].item() != step or
                any(moment[key].shape != tensor.shape or not torch.isfinite(moment[key]).all() for key in ("exp_avg", "exp_avg_sq"))):
                raise ValueError("Invalid semantic weights or AdamW state")
    return record


def audit_evaluation(path, arm, plan, data, encoded, registry):
    checkpoint = path / arm / f"step-{plan['maximum_optimizer_steps']:06d}"
    evaluation = checkpoint / "evaluation"
    summary = json.loads((evaluation / "summary.json").read_text())
    if (summary["status"] != "completed" or summary["arm"] != arm or summary["step"] != plan["maximum_optimizer_steps"] or
        not summary["adapter_unchanged"] or not summary["training_rng_unchanged"]):
        raise ValueError("Semantic evaluation model/RNG identity failed")
    for name, digest in summary["file_sha256"].items():
        if file_hash(evaluation / name) != digest:
            raise ValueError("Semantic evaluation output changed")
    if plan["arms"][arm]["token_policy"] == "question":
        if len(summary["pre_question_prefix_checks"]) != 2 or any(not check["kv_recurrent_conv_and_logits_exact"] for check in summary["pre_question_prefix_checks"]):
            raise ValueError("Final question-only short/long prefix checks failed")
    elif summary["pre_question_prefix_checks"]:
        raise ValueError("Unexpected identity claim for all-token controller")
    sample_keys = sorted(identity(example) for example in data["validation"])
    if [check["key"] for check in summary["full_attention_checks"]] != [sample_keys[0], sample_keys[-1]]:
        raise ValueError("Missing final full-attention checks")
    for check in summary["full_attention_checks"]:
        if set(check["weights"]) != {str(index) for index in (3, 7, 11, 15, 19, 23)}:
            raise ValueError("Missing semantic full-attention layer")
        for stats in check["weights"].values():
            if stats["prefill_calls"] != 1 or stats["cached_calls"] < 1 or stats["max_record_mass"] <= 0:
                raise ValueError("Full-history semantic access failed")
    teacher = [json.loads(line) for line in (evaluation / "teacher_train.jsonl").read_text().splitlines()]
    if len(teacher) != len(encoded):
        raise ValueError("Incomplete final teacher-forced training fit")
    for index, (row, item) in enumerate(zip(teacher, encoded)):
        if (row["training_index"] != index or
            any(row[key] != item[key] for key in ("case_id", "variant", "role")) or
            row["answer_ids"] != item["answer_ids"] or len(row["teacher_prediction_ids"]) != len(item["answer_ids"]) or
            not math.isfinite(row["teacher_loss_including_newline"]) or row["teacher_loss_including_newline"] < 0):
            raise ValueError("Final teacher-forced fit identities changed")
    expected = {tuple(identity(example)): example["query"]["answer"] for example in data["validation"]}
    rows = load_rows(evaluation / "validation.jsonl", expected)
    for row, example, registration in zip(rows, data["validation"], registry["validation"]):
        if ([row["case_id"], row["variant"], row["role"]] != identity(example) or row["registration"] != registration or
            row["reader_policy"] != "full" or
            any(row[key] != example["case"][key] for key in ("base_case_id", "value_version", "family", "seed", "distractor_updates"))):
            raise ValueError("Final semantic validation identity changed")
    metrics = summarize_semantics(rows)
    if metrics != summary["semantic_metrics"]:
        raise ValueError("Final semantic summary disagrees with raw outputs")
    fit = {"examples": len(teacher), "teacher_loss_including_newline": sum(row["teacher_loss_including_newline"] for row in teacher) / len(teacher),
           "teacher_numeric_tokens_correct": sum(sum(left == right for left, right in zip(row["answer_ids"][:3], row["teacher_prediction_ids"][:3])) for row in teacher),
           "teacher_numeric_tokens": 3 * len(teacher),
           "teacher_newline_correct": sum(row["answer_ids"][3] == row["teacher_prediction_ids"][3] for row in teacher)}
    if fit != summary["teacher_train_fit"]:
        raise ValueError("Teacher-forced summary disagrees with raw observations")
    return {"metrics": metrics, "teacher_train_fit": fit, "evaluation_summary_sha256": file_hash(evaluation / "summary.json")}


def audit_run(path, allow_smoke=False):
    manifest = json.loads((path / "manifest.json").read_text())
    plan = json.loads((path / "plan.json").read_text())
    arms = list(plan["arms"])
    if (manifest["status"] != "completed" or manifest["completed_arms"] != arms or
        manifest["plan_sha256"] != data_fingerprint(plan) or manifest["reader_policy"] != "full"):
        raise ValueError("Incomplete semantic controller trial")
    if manifest["smoke"] and not allow_smoke:
        raise ValueError("Semantic safety smoke is not completed training")
    if datetime.fromisoformat(plan["registered_utc"]) >= datetime.fromisoformat(manifest["started_utc"]):
        raise ValueError("Semantic training plan was not fixed before training")
    registered, data, registry = registered_dataset(Path(plan["data_source"]))
    native = audit_native(Path(plan["native_reference_run"]))
    if native != manifest["native_reference"]:
        raise ValueError("Original semantic native reference changed")
    native_manifest = json.loads((Path(plan["native_reference_run"]) / "manifest.json").read_text())
    if manifest["model_path"] != registered["model_path"] or manifest["packages"] != native_manifest["packages"]:
        raise ValueError("Semantic controller model/packages differ from native reference")
    for key in ("data_sha256", "registry_sha256", "tokenized_input_answer_sha256"):
        if plan[key] != registered[key]:
            raise ValueError("Semantic training registration changed")
    for relative, digest in manifest["source_sha256"].items():
        if file_hash(path / "sources" / relative) != digest:
            raise ValueError("Archived semantic controller source changed")
    for relative in ("sas_memory/controllers.py", "sas_memory/runtime.py", "sas_memory/tasks.py", "scripts/run_controller_trial.py"):
        if file_hash(ROOT / relative) != manifest["source_sha256"][relative]:
            raise ValueError("Actual semantic controller/runtime changed")
    encoded = json.loads((Path(plan["data_source"]) / "tokenized_train.json").read_text())
    if encoded != json.loads((path / "tokenized_train.json").read_text()) or manifest["data_manifest_sha256"] != file_hash(Path(plan["data_source"]) / "manifest.json"):
        raise ValueError("Optimized semantic inputs/registration changed")
    models = {}
    for arm, spec in plan["arms"].items():
        check = json.loads((path / arm / "safety_check.json").read_text())
        audit_safety(check, spec)
        if manifest["smoke"]:
            models[arm] = {"parameters": spec["parameters"], "status": "safety_passed"}
            continue
        state = manifest["arms"][arm]
        if state["completed_checkpoints"] != plan["checkpoint_steps"] or not state["native_restore_exact"]:
            raise ValueError("Semantic checkpoint schedule/native restoration failed")
        if len((path / arm / "training.jsonl").read_text().splitlines()) != plan["maximum_optimizer_steps"]:
            raise ValueError("Extra/missing semantic optimizer steps")
        records = [audit_checkpoint(path, arm, step, plan, encoded) for step in plan["checkpoint_steps"]]
        models[arm] = {"parameters": spec["parameters"], "checkpoints_audited": len(records),
                       **audit_evaluation(path, arm, plan, data, encoded, registry)}
    return {"status": "audited_smoke" if manifest["smoke"] else "completed_and_audited", "source_run": str(path),
            "manifest_sha256": file_hash(path / "manifest.json"), "arms": models,
            "validation_records_audited": 0 if manifest["smoke"] else len(arms) * len(data["validation"]),
            "teacher_train_records_audited": 0 if manifest["smoke"] else len(arms) * len(data["train"]),
            "native_reference_metrics": native["metrics"], "interpretation": plan["interpretation"]}


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
    print(json.dumps({"status": report["status"], "arms": {arm: {key: value["metrics"][key] for key in ("correct", "examples", "joint_roles")} if "metrics" in value else value for arm, value in report["arms"].items()}}))


if __name__ == "__main__":
    main()
