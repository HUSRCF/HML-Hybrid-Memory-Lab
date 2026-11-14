"""Audit registered blocked-reader training and functional held-out readouts."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from scripts.analyze_controller_trial import aggregate, load_rows
from scripts.run_controller_trial import encode_example, prompt_for, validate_adaptation_plan, write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenizer_runtime(model_path):
    from transformers import AutoTokenizer
    from sas_memory.runtime import TextRuntime
    runtime = TextRuntime.__new__(TextRuntime)
    runtime.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    marker = "SAS_UNIQUE_CONTENT_MARKER"
    template = runtime.tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    runtime.prompt_start, runtime.prompt_end = template.split(marker)
    return runtime


def audit_safety(check, policy, parameters):
    if (check["status"] != "passed" or check["initial_logit_max_error"] != 0 or
        check["trainable_parameters"] != parameters or not check["backbone_gradients_absent"] or
        check["token_policy"] != policy or check["reader_policy"] != "records_blocked"):
        raise ValueError("Safety or parameter-budget mismatch")
    if policy == "question" and check["pre_question_cache_and_logits_exact_after_update"] is not True:
        raise ValueError("Question-only prefix identity not verified")
    for key in ("output_gradient_norms", "encoder_gradient_norms_after_one_step"):
        if not check[key] or any(v is None or not math.isfinite(v) or v <= 0 for v in check[key].values()):
            raise ValueError("Missing finite nonzero adapter gradients")
    probability = check["reader_probability_checks_after_update"]
    if set(probability["teacher_forced"]) != {str(i) for i in (3, 7, 11, 15, 19, 23)}:
        raise ValueError("Missing full-attention layers")
    for layer, teacher in probability["teacher_forced"].items():
        cached = probability["cached_answer"][layer]
        for stat in (teacher, cached):
            if stat["max_record_probability"] != 0 or stat["max_record_mass"] != 0 or stat["prefill_calls"] != 1:
                raise ValueError("Blocked probability is nonzero or was not observed")
        if cached["cached_calls"] < 1 or probability["unmasked_control"][layer]["max_record_mass"] <= 0:
            raise ValueError("Missing cached observation or nonvacuous probability control")


def audit_run(path, expected_plan=None):
    from safetensors import safe_open
    from sas_memory.record_kv_ablation import record_blocked_mask
    manifest = json.loads((path / "manifest.json").read_text())
    plan = json.loads((path / "adaptation_plan.json").read_text())
    config = manifest["config"]
    if expected_plan is not None and plan != expected_plan:
        raise ValueError("Registered adaptation plan changed")
    if (manifest["status"] != "completed" or manifest["stage"] != "training" or
        manifest["modes"] != [plan["mode"]] or manifest["completed_modes"] != manifest["modes"]):
        raise ValueError("Training incomplete, smoke-only, or wrong mode")
    if manifest["adaptation_plan_sha256"] != data_fingerprint(plan) or manifest["reader_policy"] != "records_blocked":
        raise ValueError("Plan or reader policy changed")
    if datetime.fromisoformat(plan["registered_utc"]) >= datetime.fromisoformat(manifest["started_utc"]):
        raise ValueError("Plan was not registered before training")
    for relative, expected in manifest["source_sha256"].items():
        if digest(path / "sources" / relative) != expected:
            raise ValueError(f"Source snapshot changed: {relative}")
    data = json.loads((path / "data.json").read_text())
    validate_adaptation_plan(config, data, plan)
    if data_fingerprint(data) != manifest["data_sha256"]:
        raise ValueError("Training data fingerprint changed")
    runtime = tokenizer_runtime(manifest["model_path"])
    encoded = [encode_example(runtime, example, "records_blocked") for example in data["train"]]
    saved = json.loads((path / "tokenized_train.json").read_text())
    if encoded != saved:
        raise ValueError("Tokenized inputs, labels, boundaries or training masks changed")
    tokens_hash = data_fingerprint([{key: item[key] for key in ("input_ids", "answer_ids")} for item in encoded])
    if tokens_hash != plan["tokenized_input_answer_sha256"] or tokens_hash != manifest["tokenized_input_answer_sha256"]:
        raise ValueError("Original optimized inputs/answers differ")
    if (len(encoded) != plan["training_examples"] or
        sum(len(item["answer_ids"]) for item in encoded) != plan["supervised_answer_tokens"] or
        manifest["supervised_answer_tokens_per_mode"] != plan["supervised_answer_tokens"] or
        sum(len(item["input_ids"]) for item in encoded) != manifest["optimized_input_tokens_per_mode"]):
        raise ValueError("Training token budget changed")
    registry = {"train": [{key: item[key] for key in ("case_id", "variant", "role", "reader_mask")} for item in encoded]}
    expected_rows = {}
    for split in ("validation", "test"):
        registry[split], expected_rows[split] = [], {}
        for case in data[split]:
            for variant, content in case["variants"].items():
                for query in content["queries"]:
                    key = (case["id"], variant, query["role"])
                    expected_rows[split][key] = query["answer"]
                    ledger, prompt = prompt_for(runtime, case, variant, query)
                    registry[split].append({"case_id": case["id"], "variant": variant, "role": query["role"],
                                            "reader_mask": record_blocked_mask(runtime, ledger, prompt)})
    if registry != json.loads((path / "reader_registry.json").read_text()) or data_fingerprint(registry) != manifest["reader_registry_sha256"]:
        raise ValueError("Registered masks differ from recomputed boundary-only masks")
    mode = plan["mode"]
    check = json.loads((path / "safety_checks.json").read_text())[mode]
    audit_safety(check, config["token_policy"], plan["expected_trainable_parameters"])
    logs = [json.loads(line) for line in (path / mode / "training.jsonl").read_text().splitlines()]
    if [row["step"] for row in logs] != list(range(1, plan["steps"] + 1)):
        raise ValueError("Optimizer steps differ from registered budget")
    for row in logs:
        if (not math.isfinite(row["loss"]) or not math.isfinite(row["gradient_norm"]) or
            row["reader_policy"] != "records_blocked" or
            row["reader_calls"] != [{"prefill": 1, "cached": 0}] * plan["gradient_accumulation"]):
            raise ValueError("Training loss/gradient invalid or a microbatch was not masked")
    checkpoint = path / mode / "controller.safetensors"
    with safe_open(checkpoint, framework="pt", device="cpu") as weights:
        parameters = sum(math.prod(weights.get_slice(key).get_shape()) for key in weights.keys())
    if parameters != plan["expected_trainable_parameters"]:
        raise ValueError("Saved checkpoint parameter count differs")
    all_rows, summary = {}, {}
    for split in ("validation", "test"):
        masks = {(row["case_id"], row["variant"], row["role"]): row["reader_mask"] for row in registry[split]}
        all_rows[split], summary[split] = {}, {}
        for label, result_path in (("native", path / f"native_{split}.jsonl"), ("adapted", path / mode / f"{split}.jsonl")):
            rows = load_rows(result_path, expected_rows[split])
            for row in rows:
                key = (row["case_id"], row["variant"], row["role"])
                if (row["reader_policy"] != "records_blocked" or row["reader_mask"] != masks[key] or
                    row["readout_calls"]["prefill"] != 1 or row["readout_calls"]["cached"] < 1):
                    raise ValueError("Evaluation mask/input changed or cached answer access unrestricted")
            all_rows[split][label] = rows
            summary[split][label] = aggregate(rows)
    return {"path": str(path), "manifest_sha256": digest(path / "manifest.json"),
            "checkpoint_sha256": digest(checkpoint), "token_policy": config["token_policy"],
            "initial_state_sha256": json.loads((path / mode / "controller.json").read_text())["initial_state_sha256"],
            "data_sha256": manifest["data_sha256"], "tokenized_input_answer_sha256": tokens_hash,
            "reader_registry_sha256": manifest["reader_registry_sha256"], "parameters": parameters,
            "steps": len(logs), "training_loss_first": logs[0]["loss"], "training_loss_last": logs[-1]["loss"],
            "metrics": summary}, all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs=2, type=Path)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/blocked_reader_adaptation_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    arms, rows = [], []
    for path in args.runs:
        arm, result = audit_run(path, plan)
        arms.append(arm)
        rows.append(result)
    if {arm["token_policy"] for arm in arms} != {"question", "all"}:
        raise ValueError("Both registered token-policy arms required")
    for key in ("initial_state_sha256", "data_sha256", "tokenized_input_answer_sha256", "reader_registry_sha256", "parameters", "steps"):
        if arms[0][key] != arms[1][key]:
            raise ValueError(f"Arms differ in {key}")
    if any(rows[0][split]["native"] != rows[1][split]["native"] for split in ("validation", "test")):
        raise ValueError("Native blocked-reader baselines differ")
    report = {"status": "completed_and_audited", "analysis_source_sha256": digest(Path(__file__)), "arms": arms,
              "interpretation": "One fixed initialization and reused diagnostic cases. Question-only success is constructive recoverability through adapted computation; all-token gains change both recurrent and attention prefix paths. Failure does not establish absent information, state capacity limits or GDN write damage. No semantic shared-controller architecture claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    for arm in arms:
        print(json.dumps({"policy": arm["token_policy"], "test": arm["metrics"]["test"]["adapted"]}))


if __name__ == "__main__":
    main()
