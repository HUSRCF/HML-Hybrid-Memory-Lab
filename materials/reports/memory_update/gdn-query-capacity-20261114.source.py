"""Audit matched query-capacity safety checks and fixed-budget paired readouts."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash, microbatch_indices
from sas_memory.training_data import data_fingerprint
from scripts.analyze_blocked_reader_adaptation import audit_safety, tokenizer_runtime
from scripts.analyze_controller_trial import load_rows
from scripts.analyze_single_fact_probe import family_metrics, registered_dataset
from scripts.run_controller_trial import encode_example, write_json


def audit_sources(path, manifest):
    for relative, digest in manifest["source_sha256"].items():
        if file_hash(path / "sources" / relative) != digest:
            raise ValueError("Archived capacity-control source changed")


def audit_smoke(path, plan=None):
    manifest = json.loads((path / "manifest.json").read_text())
    saved_plan = json.loads((path / "plan.json").read_text())
    if plan is not None and saved_plan != plan:
        raise ValueError("Safety gate belongs to a different capacity plan")
    plan = saved_plan
    if (manifest["status"] != "completed" or manifest["stage"] != "safety_smoke_only" or
        manifest["completed_arms"] != ["local", "gdn_query"] or manifest["training_schedule_started"] or
        not manifest["backbone_frozen_no_gradients"] or manifest["plan_sha256"] != data_fingerprint(plan)):
        raise ValueError("Incomplete capacity-control safety gate")
    if datetime.fromisoformat(plan["revised_utc"]) >= datetime.fromisoformat(manifest["started_utc"]):
        raise ValueError("Safety gate ran before plan revision")
    audit_sources(path, manifest)
    arms = {}
    for arm, expected in (("local", plan["baseline_trainable_parameters"]), ("gdn_query", plan["expected_trainable_parameters"])):
        check = json.loads((path / f"{arm}.safety.json").read_text())
        audit_safety(check, "question", expected)
        if arm == "gdn_query":
            if check["layers"] != plan["layers"] or check["rank"] != plan["rank"]:
                raise ValueError("Query target layers/rank changed")
            for key, value in check["direct_projection_checks_after_update"].items():
                if not all(value.values()) or check["packed_projection_shapes"][key] != [6144, 2048]:
                    raise ValueError("Actual query/control projection dimensions or locality failed")
        arms[arm] = {"parameters": check["trainable_parameters"], "safety_sha256": file_hash(path / f"{arm}.safety.json")}
    return {"status": "completed_and_audited", "source_run": str(path), "manifest_sha256": file_hash(path / "manifest.json"),
            "source_sha256": manifest["source_sha256"], "packages": manifest["packages"], "model_path": manifest["model_path"],
            "arms": arms, "interpretation": "Safety/one-update checks only, not schedule training or functional recovery."}


def audit_checkpoint(path, arm, step, plan, data, encoded):
    import torch
    from safetensors import safe_open
    checkpoint = path / arm / f"step-{step:06d}"
    record = json.loads((checkpoint / "checkpoint.json").read_text())
    if (record["step"] != step or record["arm"] != arm or record["policy"] != "question" or
        record["plan_sha256"] != data_fingerprint(plan) or
        file_hash(checkpoint / "controller.safetensors") != record["controller_sha256"] or
        file_hash(checkpoint / "optimizer.pt") != record["optimizer_sha256"]):
        raise ValueError("Checkpoint protocol/weight/optimizer identity changed")
    log_bytes = (path / arm / "training.jsonl").read_bytes()[:record["training_log_prefix_bytes"]]
    import hashlib
    if hashlib.sha256(log_bytes).hexdigest() != record["training_log_prefix_sha256"]:
        raise ValueError("Checkpoint training-log prefix changed")
    logs = [json.loads(line) for line in log_bytes.splitlines()]
    if [row["step"] for row in logs] != list(range(1, step + 1)):
        raise ValueError("Training log is incomplete")
    input_tokens = answer_tokens = 0
    for index, row in enumerate(logs, 1):
        if any(not math.isfinite(row[key]) or row[key] < 0 for key in ("loss", "gradient_norm")):
            raise ValueError("Invalid loss or gradient")
        indices = microbatch_indices(index, len(encoded), plan["gradient_accumulation"])
        expected = [{"training_index": item, "case_id": encoded[item]["case_id"],
                     "keep_sha256": encoded[item]["reader_mask"]["keep_sha256"],
                     "reader_calls": {"prefill": 1, "cached": 0}} for item in indices]
        if row["microbatches"] != expected:
            raise ValueError("Paired training order/mask changed")
        input_tokens += sum(len(encoded[item]["input_ids"]) for item in indices)
        answer_tokens += sum(len(encoded[item]["answer_ids"]) for item in indices)
    if (input_tokens != plan["optimized_input_token_budgets"][str(step)] or
        record["optimized_input_tokens"] != input_tokens or record["supervised_answer_tokens"] != answer_tokens):
        raise ValueError("Checkpoint optimized token budget changed")
    saved = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer = saved["optimizer"]
    names = record["optimizer_parameter_names"]
    groups = optimizer["param_groups"]
    if len(groups) != 1 or len(groups[0]["params"]) != len(names):
        raise ValueError("Optimizer parameter layout changed")
    settings = groups[0]
    if settings["lr"] != plan["learning_rate"] or settings["weight_decay"] != plan["weight_decay"]:
        raise ValueError("Optimizer settings changed")
    expected_count = plan["baseline_trainable_parameters"] if arm == "local" else plan["expected_trainable_parameters"]
    with safe_open(checkpoint / "controller.safetensors", framework="pt", device="cpu") as weights:
        if set(weights.keys()) != set(names) or sum(weights.get_tensor(name).numel() for name in names) != expected_count:
            raise ValueError("Adapter checkpoint shape/count changed")
        for index, name in zip(settings["params"], names):
            weight, moment = weights.get_tensor(name), optimizer["state"][index]
            if (not torch.isfinite(weight).all() or moment["step"].item() != step or
                any(moment[key].shape != weight.shape or not torch.isfinite(moment[key]).all() for key in ("exp_avg", "exp_avg_sq"))):
                raise ValueError("Invalid adapter/AdamW snapshot")
    summary_path = checkpoint / "evaluation/summary.json"
    summary = json.loads(summary_path.read_text())
    if (summary["status"] != "completed" or summary["arm"] != arm or summary["step"] != step or
        not summary["adapter_unchanged"] or not summary["training_rng_unchanged"] or
        not summary["pre_question_prefix_checks"] or
        any(not check["kv_recurrent_conv_and_logits_exact"] for check in summary["pre_question_prefix_checks"])):
        raise ValueError("Checkpoint evaluation changed model/RNG/prefix")
    for relative, digest in summary["file_sha256"].items():
        if file_hash(checkpoint / "evaluation" / relative) != digest:
            raise ValueError("Evaluation result file changed")
    probability = summary["probability_checks"]
    if set(probability["teacher_forced"]) != {str(i) for i in (3, 7, 11, 15, 19, 23)}:
        raise ValueError("Missing actual probability observations")
    for layer, teacher in probability["teacher_forced"].items():
        for stat in (teacher, probability["cached_answer"][layer]):
            if stat["prefill_calls"] != 1 or stat["max_record_probability"] != 0 or stat["max_record_mass"] != 0:
                raise ValueError("Actual blocked record probability is nonzero")
        if probability["cached_answer"][layer]["cached_calls"] < 1 or probability["unmasked_control"][layer]["max_record_mass"] <= 0:
            raise ValueError("Cached mask/nonvacuous control failed")
    models = {}
    for split in ("train", "validation"):
        members = data[split]
        expected = {(example["case"]["id"], example["variant"], example["query"]["role"]): example["query"]["answer"] for example in members}
        rows = load_rows(checkpoint / "evaluation" / f"{split}.jsonl", expected)
        if len(rows) != len(members):
            raise ValueError("Incomplete paired evaluation")
        for index, (row, example) in enumerate(zip(rows, members)):
            if (row["training_index"] != index or row["case_id"] != example["case"]["id"] or
                any(row[key] != example["case"][key] for key in ("base_case_id", "value_version", "family")) or
                row["readout_calls"]["prefill"] != 1 or row["readout_calls"]["cached"] < 1):
                raise ValueError("Wrong paired evaluation identity/reader")
            item = encoded[index] if split == "train" else encode_example(tokenizer_runtime_cache(plan), example, "records_blocked")
            if row["reader_mask"] != item["reader_mask"] or row["supervised_ids"] != item["answer_ids"]:
                raise ValueError("Evaluation mask/labels changed")
            if (len(row["teacher_prediction_ids"]) != len(row["supervised_ids"]) or
                not math.isfinite(row["teacher_loss_including_newline"]) or row["teacher_loss_including_newline"] < 0):
                raise ValueError("Invalid teacher-forced observation")
        models[split] = family_metrics(rows)
    return {"step": step, "arm": arm, "optimized_input_tokens": input_tokens,
            "supervised_answer_tokens": answer_tokens, "splits": models,
            "checkpoint_manifest_sha256": file_hash(checkpoint / "checkpoint.json")}


_TOKEN_RUNTIME = {}


def tokenizer_runtime_cache(plan):
    key = plan["data_source"]
    if key not in _TOKEN_RUNTIME:
        manifest = json.loads((Path(key) / "manifest.json").read_text())
        _TOKEN_RUNTIME[key] = tokenizer_runtime(manifest["model_path"])
    return _TOKEN_RUNTIME[key]


def audit_training(path):
    manifest = json.loads((path / "manifest.json").read_text())
    plan = json.loads((path / "plan.json").read_text())
    if (manifest["status"] != "completed" or manifest["completed_arms"] != ["local", "gdn_query"] or
        manifest["plan_sha256"] != data_fingerprint(plan) or manifest["evaluated_splits"] != ["train", "validation"]):
        raise ValueError("Incomplete capacity trial")
    if datetime.fromisoformat(plan["revised_utc"]) >= datetime.fromisoformat(manifest["started_utc"]):
        raise ValueError("Plan was not fixed before training")
    audit_sources(path, manifest)
    safety = audit_smoke(Path(manifest["safety_gate"]["source_run"]), plan)
    if safety != manifest["safety_gate"] or safety["packages"] != manifest["packages"] or safety["model_path"] != manifest["model_path"]:
        raise ValueError("Original real-model safety gate or training runtime changed")
    from scripts.analyze_single_fact_probe import audit_run as audit_native
    native = audit_native(Path(manifest["native_gate"]["source_run"]))
    if native != manifest["native_gate"]:
        raise ValueError("Original native positive control changed")
    registered, data, _ = registered_dataset(Path(plan["data_source"]))
    if (registered["data_sha256"] != plan["data_sha256"] or manifest["data_sha256"] != plan["data_sha256"] or
        manifest["tokenized_input_answer_sha256"] != plan["tokenized_input_answer_sha256"]):
        raise ValueError("Capacity trial data changed")
    encoded = json.loads((Path(plan["data_source"]) / "tokenized_train.json").read_text())
    if encoded != json.loads((path / "tokenized_train.json").read_text()):
        raise ValueError("Actual optimized tokens changed")
    for relative in ("sas_memory/controllers.py", "sas_memory/gdn_query_control.py", "sas_memory/runtime.py", "sas_memory/kv_readout.py"):
        if file_hash(ROOT / relative) != manifest["source_sha256"][relative]:
            raise ValueError("Runtime changed since capacity trial")
    models = {}
    for arm in ("local", "gdn_query"):
        state = manifest["arms"][arm]
        if not state["native_restore_exact"] or state["completed_checkpoints"] != plan["checkpoint_steps"]:
            raise ValueError("Native restore or fixed evaluation schedule failed")
        logs = (path / arm / "training.jsonl").read_text().splitlines()
        if len(logs) != plan["checkpoint_steps"][-1]:
            raise ValueError("Extra/missing optimization steps")
        models[arm] = [audit_checkpoint(path, arm, step, plan, data, encoded) for step in plan["checkpoint_steps"]]
    return {"status": "completed_and_audited", "source_run": str(path), "manifest_sha256": file_hash(path / "manifest.json"),
            "native_and_safety_gates_reaudited": True,
            "arms": models, "records_audited": sum(len(data[split]) for split in ("train", "validation")) * 2 * len(plan["checkpoint_steps"]),
            "interpretation": plan["interpretation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = audit_smoke(args.run) if args.smoke else audit_training(args.run)
    report["analysis_source_sha256"] = file_hash(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: report[key] for key in ("status", "records_audited") if key in report}), flush=True)


if __name__ == "__main__":
    main()
