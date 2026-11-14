"""Run registered 64/256/1024-step calibration on fixed training/validation data."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import (assert_rng_equal, file_hash, load_checkpoint, microbatch_indices,
                                    restore_log, rng_state, save_checkpoint)
from sas_memory.training_data import data_fingerprint
from scripts.analyze_blocked_reader_adaptation import audit_run, audit_safety
from scripts.run_controller_confirmation import verify_native_prefix
from scripts.run_controller_trial import (encode_example, evaluate, prompt_for, reader_context, safety_check,
                                           state_fingerprint, supervised_loss, verify_blocked_reader, write_json)
from scripts.run_oracle_kv_readout import measure


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def fit_training(runtime, controller, policy, data, encoded, path):
    import torch
    rows = []
    with torch.inference_mode(), path.open("w", buffering=1) as output:
        for index, (item, example) in enumerate(zip(encoded, data)):
            calls = []
            loss, logits = supervised_loss(runtime, item, controller, policy, "records_blocked", calls)
            ledger, prompt = prompt_for(runtime, example["case"], example["variant"], example["query"])
            numeric_ids = runtime.continuation_ids(prompt, example["query"]["answer"])
            if (len(numeric_ids) != 3 or item["answer_ids"][:3] != numeric_ids or
                runtime.tokenizer.decode(item["answer_ids"][3:]) != "\n"):
                raise ValueError("Numeric/newline token split differs from the fixed training protocol")
            with reader_context(runtime, "records_blocked", item["reader_mask"]) as reader:
                actual = measure(runtime, ledger, example["query"])
            if calls != [{"prefill": 1, "cached": 0}] or reader.calls["prefill"] != 1 or reader.calls["cached"] < 1:
                raise ValueError("Fit evaluation missed a blocked-reader forward")
            row = {"training_index": index, "case_id": item["case_id"], "variant": item["variant"], "role": item["role"],
                   "family": item["family"], "supervised_ids": item["answer_ids"],
                   "teacher_prediction_ids": logits.argmax(-1)[0].tolist(), "teacher_loss_including_newline": loss.item(),
                   "reader_mask": item["reader_mask"], "readout_calls": dict(reader.calls), **actual}
            rows.append(row)
            output.write(json.dumps(row) + "\n")
    return {"examples": len(rows), "free_greedy_correct": sum(row["greedy_exact"] for row in rows),
            "teacher_loss_including_newline": mean(row["teacher_loss_including_newline"] for row in rows),
            "gold_answer_nll_excluding_newline": mean(row["answer_nll"] for row in rows),
            "teacher_digit_correct": sum(sum(a == b for a, b in zip(row["supervised_ids"][:3], row["teacher_prediction_ids"][:3])) for row in rows),
            "teacher_digit_tokens": 3 * len(rows),
            "teacher_newline_correct": sum(row["supervised_ids"][3] == row["teacher_prediction_ids"][3] for row in rows),
            "prediction_counts": dict(Counter(row["generated"].strip() for row in rows))}


def evaluate_checkpoint(runtime, controller, policy, data, encoded, checkpoint, reference, step):
    evaluation = checkpoint / "evaluation"
    if evaluation.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        evaluation.rename(checkpoint / f"interrupted-evaluation-{stamp}")
    evaluation.mkdir()
    before_rng, before_state = rng_state(controller), state_fingerprint(controller)
    fit = fit_training(runtime, controller, policy, data["train"], encoded, evaluation / "train.jsonl")
    validation = evaluate(runtime, data["validation"], evaluation / "validation.jsonl", controller, policy, "records_blocked")
    if step == 64:
        actual = (evaluation / "validation.jsonl").read_bytes()
        expected = (reference / "local/validation.jsonl").read_bytes()
        if actual != expected:
            raise ValueError("Step-64 validation does not exactly replay the reference")
    prefix_checks = []
    if policy == "question":
        ledgers = []
        for case in data["validation"]:
            query = case["variants"]["control"]["queries"][0]
            ledger, _ = prompt_for(runtime, case, "control", query)
            ledgers.append((len(runtime.ids(ledger)), case["id"], ledger))
        for length, case_id, ledger in (min(ledgers), max(ledgers)):
            prefix_checks.append({"case_id": case_id, "ledger_tokens": length, **verify_native_prefix(runtime, controller, ledger)})
    controller.set_token_policy(policy, encoded[0]["ledger_tokens"])
    probability_checks = verify_blocked_reader(runtime, encoded[0])
    if state_fingerprint(controller) != before_state:
        raise ValueError("Evaluation modified the adapter")
    assert_rng_equal(before_rng, rng_state(controller))
    summary = {"status": "completed", "step": step, "policy": policy, "train": fit, "validation": validation,
               "pre_question_prefix_checks": prefix_checks, "probability_checks": probability_checks,
               "adapter_unchanged": True, "training_rng_unchanged": True,
               "file_sha256": {name: file_hash(evaluation / name) for name in ("train.jsonl", "validation.jsonl", "validation.summary.json")}}
    atomic_json(evaluation / "summary.json", summary)
    return summary


def evaluation_complete(checkpoint):
    path = checkpoint / "evaluation/summary.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text())
    if summary["status"] != "completed":
        return False
    record = json.loads((checkpoint / "checkpoint.json").read_text())
    if (file_hash(checkpoint / "controller.safetensors") != record["controller_sha256"] or
        file_hash(checkpoint / "optimizer.pt") != record["optimizer_sha256"] or
        summary["step"] != record["step"] or summary["policy"] != record["policy"] or
        not summary["adapter_unchanged"] or not summary["training_rng_unchanged"]):
        raise ValueError("Committed checkpoint/evaluation identity changed")
    for relative, digest in summary["file_sha256"].items():
        if file_hash(path.parent / relative) != digest:
            raise ValueError("Committed checkpoint evaluation changed")
    return True


def verify_step64(controller, rows, reference, path):
    import torch
    from safetensors.torch import load_file
    original = load_file(str(reference / "local/controller.safetensors"), device="cpu")
    logs = [json.loads(line) for line in (reference / "local/training.jsonl").read_text().splitlines()]
    weights_exact = set(original) == set(controller.state_dict()) and all(
        torch.equal(value.detach().cpu(), original[key]) for key, value in controller.state_dict().items())
    result = {"weights_exact": weights_exact, "losses_exact": len(rows) == len(logs) == 64 and all(a["loss"] == b["loss"] for a, b in zip(rows, logs)),
              "gradient_norms_exact": len(rows) == len(logs) == 64 and all(a["gradient_norm"] == b["gradient_norm"] for a, b in zip(rows, logs)),
              "reference_checkpoint_sha256": file_hash(reference / "local/controller.safetensors")}
    write_json(path, result)
    if not all(result[key] for key in ("weights_exact", "losses_exact", "gradient_norms_exact")):
        from safetensors.torch import save_file
        save_file({key: value.detach().cpu().contiguous() for key, value in controller.state_dict().items()}, str(path.with_suffix(".mismatch.safetensors")))
        raise ValueError(f"Step-64 exact replay failed: {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/blocked_reader_budget_calibration_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Validate data, initialization and real-model gradient/mask safety only")
    parser.add_argument("--resume", action="store_true", help="Resume a terminal/interrupted run from its last committed optimizer state")
    args = parser.parse_args()
    if args.smoke and args.resume:
        raise ValueError("Smoke and resume cannot be combined")
    plan = json.loads(args.plan.read_text())
    if (plan["reader_policy"] != "records_blocked" or plan["checkpoint_steps"][0] != 64 or
        plan["checkpoint_steps"] != sorted(set(plan["checkpoint_steps"])) or
        plan["checkpoint_steps"][-1] != plan["maximum_optimizer_steps"]):
        raise ValueError("Invalid registered calibration schedule")
    policies = [arm["token_policy"] for arm in plan["arms"].values()]
    if policies != ["question", "all"]:
        raise ValueError("Both matched token-policy arms are required")
    references = {policy: Path(plan["reference_step64"][policy]) for policy in policies}
    sources = {policy: audit_run(path)[0] for policy, path in references.items()}
    for policy, source in sources.items():
        if source["token_policy"] != policy or any(source[key] != plan[key] for key in ("initial_state_sha256", "data_sha256", "tokenized_input_answer_sha256")):
            raise ValueError("Reference differs from the registered initialization/data")
        if source["parameters"] != plan["expected_trainable_parameters"] or source["steps"] != 64:
            raise ValueError("Reference parameter/step budget changed")
    data = json.loads((Path(plan["data_source"]) / "data.json").read_text())
    if data_fingerprint(data) != plan["data_sha256"] or len(data["train"]) != plan["unique_training_examples"]:
        raise ValueError("Fixed original data changed")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with (args.output / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, plan, policies, references, sources, data)


def execute(args, plan, policies, references, sources, data):
    import torch
    from sas_memory.controllers import GroupController
    from sas_memory.record_kv_ablation import record_blocked_mask
    from sas_memory.runtime import TextRuntime
    torch.set_num_threads(4)
    manifest_path = args.output / "manifest.json"
    source_paths = sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__).resolve(),
        ROOT / "scripts/run_controller_trial.py", ROOT / "scripts/check_record_attention_weights.py",
        ROOT / "scripts/run_controller_confirmation.py", ROOT / "scripts/run_oracle_kv_readout.py",
        ROOT / "scripts/analyze_blocked_reader_adaptation.py", ROOT / "scripts/analyze_controller_trial.py"]
    fingerprints = {str(path.relative_to(ROOT)): file_hash(path) for path in source_paths}
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if (manifest["status"] not in ("running", "failed") or manifest["smoke"] or
            manifest["plan_sha256"] != data_fingerprint(plan) or manifest["checkpoint_sources"] != sources or
            manifest["source_sha256"] != fingerprints):
            raise ValueError("Run is complete, smoke-only, or has incompatible plan/source/checkpoints")
        for relative, digest in fingerprints.items():
            if file_hash(args.output / "sources" / relative) != digest:
                raise ValueError("Resume source snapshot changed")
        manifest.setdefault("resume_history", []).append({"utc": datetime.now(timezone.utc).isoformat(), "previous_status": manifest["status"]})
        manifest.pop("error", None)
        manifest["status"] = "running"
    else:
        manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "smoke": args.smoke,
                    "plan_sha256": data_fingerprint(plan), "checkpoint_sources": sources, "evaluated_splits": ["train", "validation"],
                    "source_sha256": fingerprints, "arms": {}, "completed_policies": []}
        write_json(args.output / "plan.json", plan)
        write_json(args.output / "data.json", data)
        for relative in fingerprints:
            snapshot = args.output / "sources" / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((ROOT / relative).read_bytes())
    manifest["pid"] = os.getpid()
    atomic_json(manifest_path, manifest)
    try:
        model_path = json.loads((ROOT / "configs/models.json").read_text())[plan["backbone"]]
        runtime = TextRuntime(model_path, plan["backbone"])
        packages = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")}
        if any(packages != json.loads((path / "manifest.json").read_text())["packages"] for path in references.values()):
            raise ValueError("Packages differ from the step-64 reference")
        manifest.update(model_path=model_path, python=sys.executable, packages=packages, gpu=torch.cuda.get_device_name(0))
        encoded = [encode_example(runtime, example, "records_blocked") for example in data["train"]]
        if any(encoded != json.loads((path / "tokenized_train.json").read_text()) for path in references.values()):
            raise ValueError("Training inputs/answers/reader masks differ from the original")
        if data_fingerprint([{key: item[key] for key in ("input_ids", "answer_ids")} for item in encoded]) != plan["tokenized_input_answer_sha256"]:
            raise ValueError("Optimized token identities changed")
        if sum(len(item["answer_ids"]) for item in encoded) != plan["supervised_answer_tokens_per_cycle"]:
            raise ValueError("Supervised token budget changed")
        for step in plan["checkpoint_steps"]:
            indices = [index for s in range(1, step + 1) for index in microbatch_indices(s, len(encoded), plan["gradient_accumulation"])]
            if (sum(len(encoded[index]["input_ids"]) for index in indices) != plan["optimized_input_token_budgets"][str(step)] or
                sum(len(encoded[index]["answer_ids"]) for index in indices) != plan["supervised_answer_token_budgets"][str(step)]):
                raise ValueError("Registered cumulative token budgets do not match the cyclic schedule")
        registry = {"train": [item["reader_mask"] for item in encoded], "validation": []}
        for case in data["validation"]:
            for variant, content in case["variants"].items():
                for query in content["queries"]:
                    ledger, prompt = prompt_for(runtime, case, variant, query)
                    registry["validation"].append({"case_id": case["id"], "variant": variant, "role": query["role"],
                                                   "reader_mask": record_blocked_mask(runtime, ledger, prompt)})
        if args.resume:
            if (encoded != json.loads((args.output / "tokenized_train.json").read_text()) or
                registry != json.loads((args.output / "reader_registry.json").read_text())):
                raise ValueError("Saved inputs/registry changed before resume")
        else:
            write_json(args.output / "tokenized_train.json", encoded)
            write_json(args.output / "reader_registry.json", registry)
        manifest["reader_registry_sha256"] = data_fingerprint(registry)
        manifest["data_sha256"] = data_fingerprint(data)
        manifest["tokenized_input_answer_sha256"] = plan["tokenized_input_answer_sha256"]
        atomic_json(manifest_path, manifest)
        with torch.inference_mode():
            _, native_logits = supervised_loss(runtime, encoded[0], reader_policy="records_blocked")
            native_logits = native_logits.clone()
        for policy in policies:
            if policy in manifest["completed_policies"]:
                if not evaluation_complete(args.output / policy / f"step-{plan['maximum_optimizer_steps']:06d}"):
                    raise ValueError("Completed arm is missing its final evaluation")
                continue
            reference_config = json.loads((references[policy] / "manifest.json").read_text())["config"]
            for key in ("learning_rate", "weight_decay", "max_gradient_norm", "group_index", "initialization_seed", "gradient_accumulation"):
                if reference_config[key] != plan[key]:
                    raise ValueError(f"Calibration changed reference {key}")
            check = safety_check(runtime, encoded[0], plan["mode"], plan["rank"], plan["group_index"],
                                 plan["initialization_seed"], reference_config["adapt_gdn_keys"], policy, "records_blocked")
            audit_safety(check, policy, plan["expected_trainable_parameters"])
            policy_path = args.output / policy
            policy_path.mkdir(exist_ok=args.resume)
            write_json(policy_path / "safety_check.json", check)
            torch.manual_seed(plan["initialization_seed"])
            controller = GroupController(runtime.model, plan["group_index"], plan["rank"], plan["mode"], adapt_gdn_keys=reference_config["adapt_gdn_keys"])
            if state_fingerprint(controller) != plan["initial_state_sha256"]:
                controller.close()
                raise ValueError("Zero-output initialization differs from the registered reference")
            optimizer = torch.optim.AdamW(controller.parameters(), lr=plan["learning_rate"], weight_decay=plan["weight_decay"])
            log_path = policy_path / "training.jsonl"
            start_step, rows = 0, []
            manifest["arms"].setdefault(policy, {"initial_state_sha256": plan["initial_state_sha256"], "completed_checkpoints": []})
            atomic_json(manifest_path, manifest)
            try:
                if not args.smoke:
                    snapshots = sorted(policy_path.glob("step-*/checkpoint.json"))
                    if snapshots:
                        checkpoint = snapshots[-1].parent
                        saved = load_checkpoint(checkpoint, controller, optimizer)
                        if (saved["plan_sha256"] != manifest["plan_sha256"] or saved["policy"] != policy or
                            saved["step"] not in plan["checkpoint_steps"] or state_fingerprint(controller) != saved["state_sha256"]):
                            raise ValueError("Resume checkpoint belongs to another protocol")
                        retained = restore_log(log_path, saved)
                        if retained:
                            manifest["resume_history"][-1].setdefault("retained_log_tails", []).append(retained)
                        rows = [json.loads(line) for line in log_path.read_text().splitlines()]
                        start_step = saved["step"]
                        if not evaluation_complete(checkpoint):
                            evaluate_checkpoint(runtime, controller, policy, data, encoded, checkpoint, references[policy], start_step)
                        if start_step not in manifest["arms"][policy]["completed_checkpoints"]:
                            manifest["arms"][policy]["completed_checkpoints"].append(start_step)
                    elif log_path.exists() and log_path.stat().st_size:
                        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                        log_path.rename(policy_path / f"interrupted-before-first-checkpoint-{stamp}.jsonl")
                        manifest["resume_history"][-1]["restarted_before_first_committed_step"] = True
                    with log_path.open("a", buffering=1) as log:
                        for step in range(start_step + 1, plan["maximum_optimizer_steps"] + 1):
                            started = time.monotonic()
                            optimizer.zero_grad(set_to_none=True)
                            losses, micros = [], []
                            for index in microbatch_indices(step, len(encoded), plan["gradient_accumulation"]):
                                item, calls = encoded[index], []
                                loss, _ = supervised_loss(runtime, item, controller, policy, "records_blocked", calls)
                                if not torch.isfinite(loss) or calls != [{"prefill": 1, "cached": 0}]:
                                    raise ValueError("Nonfinite loss or unmasked training forward")
                                (loss / plan["gradient_accumulation"]).backward()
                                losses.append(loss.item())
                                micros.append({"training_index": index, "case_id": item["case_id"], "variant": item["variant"], "role": item["role"],
                                               "keep_sha256": item["reader_mask"]["keep_sha256"], "reader_calls": calls[0]})
                            gradient = torch.nn.utils.clip_grad_norm_(controller.parameters(), plan["max_gradient_norm"], error_if_nonfinite=True)
                            optimizer.step()
                            if any(p.grad is not None or p.requires_grad for p in runtime.model.parameters()):
                                raise ValueError("Backbone unexpectedly received gradients")
                            row = {"step": step, "loss": mean(losses), "gradient_norm": gradient.item(), "microbatches": micros,
                                   "optimization_seconds": time.monotonic() - started}
                            log.write(json.dumps(row) + "\n")
                            rows.append(row)
                            if step % 64 == 0:
                                print(json.dumps({"event": "training", "policy": policy, "step": step, "loss": row["loss"]}), flush=True)
                            if step in plan["checkpoint_steps"]:
                                log.flush()
                                if step == 64:
                                    verify_step64(controller, rows, references[policy], policy_path / "step64_replay.json")
                                checkpoint = save_checkpoint(policy_path, step, controller, optimizer, log_path,
                                    {"policy": policy, "plan_sha256": manifest["plan_sha256"], "state_sha256": state_fingerprint(controller),
                                     "optimizer_parameter_names": [name for name, _ in controller.named_parameters()],
                                     "step64_replay_sha256": file_hash(policy_path / "step64_replay.json"),
                                     "optimized_input_tokens": sum(len(encoded[index]["input_ids"]) for s in range(1, step + 1) for index in microbatch_indices(s, len(encoded), plan["gradient_accumulation"])),
                                     "supervised_answer_tokens": sum(len(encoded[index]["answer_ids"]) for s in range(1, step + 1) for index in microbatch_indices(s, len(encoded), plan["gradient_accumulation"]))})
                                summary = evaluate_checkpoint(runtime, controller, policy, data, encoded, checkpoint, references[policy], step)
                                manifest["arms"][policy]["completed_checkpoints"].append(step)
                                atomic_json(manifest_path, manifest)
                                print(json.dumps({"event": "checkpoint_evaluated", "policy": policy, "step": step,
                                                  "train_correct": summary["train"]["free_greedy_correct"], "train_count": summary["train"]["examples"]}), flush=True)
                manifest["completed_policies"].append(policy)
            finally:
                controller.close()
                del controller, optimizer
                gc.collect()
                torch.cuda.empty_cache()
            with torch.inference_mode():
                _, restored = supervised_loss(runtime, encoded[0], reader_policy="records_blocked")
            if not torch.equal(restored, native_logits):
                raise ValueError("Detaching adapter did not restore native blocked logits")
            manifest["arms"][policy]["native_restore_exact"] = True
            atomic_json(manifest_path, manifest)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
