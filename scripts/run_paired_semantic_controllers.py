"""One-pass matched shared/independent/local adaptation on semantic updates."""

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import (assert_rng_equal, file_hash, load_checkpoint, microbatch_indices,
                                    restore_log, rng_state, save_checkpoint)
from sas_memory.semantic_metrics import summarize_semantics
from sas_memory.tasks import INSTRUCTIONS
from sas_memory.training_data import data_fingerprint
from scripts.analyze_paired_semantic_controllers import audit_checkpoint, audit_evaluation, audit_safety
from scripts.analyze_paired_semantic_probe import audit_run as audit_native, identity, registered_dataset
from scripts.check_record_attention_weights import ObservedRecordWeights
from scripts.run_blocked_reader_budget_calibration import atomic_json
from scripts.run_controller_confirmation import verify_native_prefix
from scripts.run_controller_trial import encode_example, prompt_for, safety_check, state_fingerprint, supervised_loss, write_json
from scripts.run_oracle_kv_readout import measure


def teacher_fit(runtime, controller, policy, encoded, path):
    import torch
    rows = []
    with torch.inference_mode(), path.open("w", buffering=1) as output:
        for index, item in enumerate(encoded):
            if len(item["answer_ids"]) != 4 or runtime.tokenizer.decode(item["answer_ids"][3:]) != "\n":
                raise ValueError("Training answer/newline token split changed")
            loss, logits = supervised_loss(runtime, item, controller, policy, "full")
            row = {"training_index": index, **{key: item[key] for key in ("case_id", "variant", "role")},
                   "answer_ids": item["answer_ids"], "teacher_prediction_ids": logits.argmax(-1)[0].tolist(),
                   "teacher_loss_including_newline": loss.item()}
            output.write(json.dumps(row) + "\n")
            rows.append(row)
    return {"examples": len(rows), "teacher_loss_including_newline": sum(row["teacher_loss_including_newline"] for row in rows) / len(rows),
            "teacher_numeric_tokens_correct": sum(sum(left == right for left, right in zip(row["answer_ids"][:3], row["teacher_prediction_ids"][:3])) for row in rows),
            "teacher_numeric_tokens": len(rows) * 3,
            "teacher_newline_correct": sum(row["answer_ids"][3] == row["teacher_prediction_ids"][3] for row in rows)}


def evaluate_final(runtime, controller, arm, policy, plan, data, encoded, registry, checkpoint):
    evaluation = checkpoint / "evaluation"
    if evaluation.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        evaluation.rename(checkpoint / f"interrupted-evaluation-{stamp}")
    evaluation.mkdir()
    before_rng, before_state = rng_state(controller), state_fingerprint(controller)
    fit = teacher_fit(runtime, controller, policy, encoded, evaluation / "teacher_train.jsonl")
    keys = sorted(identity(example) for example in data["validation"])
    sample_keys = {tuple(keys[0]), tuple(keys[-1])}
    rows, full_checks = [], []
    with (evaluation / "validation.jsonl").open("w", buffering=1) as output:
        for index, (example, registration) in enumerate(zip(data["validation"], registry["validation"])):
            case, query = example["case"], example["query"]
            ledger, prompt = prompt_for(runtime, case, example["variant"], query)
            if data_fingerprint(runtime.ids(prompt)) != registration["prompt_ids_sha256"]:
                raise ValueError("Actual semantic validation prompt changed")
            controller.set_token_policy(policy, registration["ledger_tokens"])
            if tuple(identity(example)) in sample_keys:
                with ObservedRecordWeights(runtime.model, len(runtime.ids(runtime.prompt_start + INSTRUCTIONS)), registration["ledger_tokens"]) as observer:
                    result = measure(runtime, ledger, query)
                for stats in observer.stats.values():
                    if stats["prefill_calls"] != 1 or stats["cached_calls"] < 1 or stats["max_record_mass"] <= 0:
                        raise ValueError("Final semantic full-history access failed")
                full_checks.append({"key": identity(example), "weights": observer.stats})
            else:
                result = measure(runtime, ledger, query)
            row = {"case_id": case["id"], "variant": example["variant"], "role": query["role"],
                   **{key: case[key] for key in ("base_case_id", "value_version", "family", "seed", "distractor_updates")},
                   "reader_policy": "full", "registration": registration, **result}
            rows.append(row)
            output.write(json.dumps(row) + "\n")
            if (index + 1) % 108 == 0:
                print(json.dumps({"event": "validation", "arm": arm, "examples": index + 1}), flush=True)
    prefix_checks = []
    if policy == "question":
        ledgers = []
        for example in data["validation"]:
            ledger, _ = prompt_for(runtime, example["case"], example["variant"], example["query"])
            ledgers.append((len(runtime.ids(ledger)), example["case"]["id"], example["variant"], ledger))
        for length, case_id, variant, ledger in (min(ledgers), max(ledgers)):
            prefix_checks.append({"case_id": case_id, "variant": variant, "ledger_tokens": length, **verify_native_prefix(runtime, controller, ledger)})
    if state_fingerprint(controller) != before_state:
        raise ValueError("Semantic evaluation changed adapter weights")
    assert_rng_equal(before_rng, rng_state(controller))
    summary = {"status": "completed", "arm": arm, "step": plan["maximum_optimizer_steps"], "teacher_train_fit": fit,
               "semantic_metrics": summarize_semantics(rows), "pre_question_prefix_checks": prefix_checks,
               "full_attention_checks": sorted(full_checks, key=lambda check: check["key"]),
               "adapter_unchanged": True, "training_rng_unchanged": True,
               "file_sha256": {name: file_hash(evaluation / name) for name in ("teacher_train.jsonl", "validation.jsonl")}}
    atomic_json(evaluation / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/paired_semantic_controller_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="All four real-model one-update safety checks, without schedule training")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.resume:
        raise ValueError("Smoke and resume cannot be combined")
    plan = json.loads(args.plan.read_text())
    if (list(plan["arms"]) != ["shared", "independent", "local", "local_question"] or
        plan["reader_policy"] != "full" or plan["checkpoint_steps"] != [144, 288, 432] or
        plan["evaluation_steps"] != [432] or plan["maximum_optimizer_steps"] != 432 or
        plan["gradient_accumulation"] != 2 or not plan["adapt_gdn_keys"]):
        raise ValueError("Unregistered semantic controller protocol")
    registered, data, registry = registered_dataset(Path(plan["data_source"]))
    native = audit_native(Path(plan["native_reference_run"]))
    native_plan = json.loads((Path(plan["native_reference_run"]) / "plan.json").read_text())
    if native["metrics"]["correct"] == native["metrics"]["examples"]:
        raise ValueError("Native semantic validation is perfect; adaptation gate is not triggered")
    for key in ("data_sha256", "registry_sha256", "tokenized_input_answer_sha256"):
        if registered[key] != plan[key] or (key in native_plan and native_plan[key] != plan[key]):
            raise ValueError("Semantic controller and native registrations differ")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with (args.output / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, plan, registered, data, registry, native)


def execute(args, plan, registered, data, registry, native, *, extra_sources=(), extra_validation=None, extra_validation_audit=None):
    import torch
    from sas_memory.controllers import GroupController
    from sas_memory.runtime import TextRuntime
    torch.set_num_threads(4)
    sources = sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__).resolve(),
        ROOT / "scripts/analyze_paired_semantic_controllers.py", ROOT / "scripts/analyze_paired_semantic_probe.py",
        ROOT / "scripts/prepare_paired_semantic_updates.py", ROOT / "scripts/run_controller_trial.py",
        ROOT / "scripts/run_controller_confirmation.py", ROOT / "scripts/run_oracle_kv_readout.py",
        ROOT / "scripts/check_record_attention_weights.py", ROOT / "scripts/run_blocked_reader_budget_calibration.py",
        ROOT / "scripts/analyze_blocked_reader_adaptation.py", ROOT / "scripts/analyze_controller_trial.py"] + list(extra_sources)
    fingerprints = {str(path.relative_to(ROOT)): file_hash(path) for path in sources}
    manifest_path = args.output / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if (manifest["status"] not in ("running", "failed") or manifest["smoke"] or
            manifest["source_sha256"] != fingerprints or manifest["plan_sha256"] != data_fingerprint(plan) or manifest["native_reference"] != native):
            raise ValueError("Resume protocol/source/reference changed, or run is complete")
        for relative, digest in fingerprints.items():
            if file_hash(args.output / "sources" / relative) != digest:
                raise ValueError("Archived semantic controller source changed before resume")
        manifest.setdefault("resume_history", []).append({"utc": datetime.now(timezone.utc).isoformat(), "previous_status": manifest["status"]})
        manifest.update(status="running")
        manifest.pop("error", None)
    else:
        manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "smoke": args.smoke,
                    "reader_policy": "full", "source_sha256": fingerprints, "plan_sha256": data_fingerprint(plan),
                    "data_manifest_sha256": file_hash(Path(plan["data_source"]) / "manifest.json"),
                    "native_reference": native, "arms": {}, "completed_arms": []}
        write_json(args.output / "plan.json", plan)
        for relative in fingerprints:
            destination = args.output / "sources" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / relative).read_bytes())
    manifest["pid"] = os.getpid()
    atomic_json(manifest_path, manifest)
    try:
        runtime = TextRuntime(registered["model_path"], "it")
        packages = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")}
        if packages != json.loads((Path(plan["native_reference_run"]) / "manifest.json").read_text())["packages"]:
            raise ValueError("Training packages differ from native reference")
        manifest.update(model_path=registered["model_path"], packages=packages, python=sys.executable, gpu=torch.cuda.get_device_name(0))
        encoded = [encode_example(runtime, example, "full") for example in data["train"]]
        if (encoded != json.loads((Path(plan["data_source"]) / "tokenized_train.json").read_text()) or
            len(encoded) != plan["unique_training_examples"] or len(encoded) != plan["maximum_optimizer_steps"] * plan["gradient_accumulation"] or
            max(len(item["input_ids"]) for item in encoded) != plan["max_input_tokens"]):
            raise ValueError("Actual semantic optimized inputs/count/length changed")
        token_path = args.output / "tokenized_train.json"
        if args.resume:
            if encoded != json.loads(token_path.read_text()):
                raise ValueError("Saved semantic optimized tokens changed")
        else:
            write_json(token_path, encoded)
        safety_index = next(index for index, example in enumerate(data["train"]) if example["case"]["family"] == "copy" and example["variant"] == "control" and example["query"]["role"] == "target_current")
        manifest["safety_training_index"] = safety_index
        with torch.inference_mode():
            _, native_logits = supervised_loss(runtime, encoded[safety_index], reader_policy="full")
            native_logits = native_logits.clone()
        for arm, spec in plan["arms"].items():
            if arm in manifest["completed_arms"]:
                if not args.smoke:
                    audit_checkpoint(args.output, arm, plan["maximum_optimizer_steps"], plan, encoded)
                    audit_evaluation(args.output, arm, plan, data, encoded, registry)
                    if extra_validation_audit is not None:
                        extra_validation_audit(args.output, arm, plan)
                continue
            arm_path = args.output / arm
            arm_path.mkdir(exist_ok=args.resume)
            check = safety_check(runtime, encoded[safety_index], spec["mode"], spec["rank"], plan["group_index"],
                                 plan["initialization_seed"], True, spec["token_policy"], "full")
            audit_safety(check, spec)
            write_json(arm_path / "safety_check.json", check)
            if args.smoke:
                manifest["completed_arms"].append(arm)
                atomic_json(manifest_path, manifest)
                print(json.dumps({"event": "safety_passed", "arm": arm, "parameters": check["trainable_parameters"]}), flush=True)
                continue
            torch.manual_seed(plan["initialization_seed"])
            controller = GroupController(runtime.model, plan["group_index"], spec["rank"], spec["mode"], adapt_gdn_keys=True)
            optimizer = torch.optim.AdamW(controller.parameters(), lr=plan["learning_rate"], weight_decay=plan["weight_decay"])
            state = manifest["arms"].setdefault(arm, {"initial_state_sha256": state_fingerprint(controller), "description": controller.description(), "completed_checkpoints": []})
            log_path, start_step = arm_path / "training.jsonl", 0
            try:
                if arm in plan.get("initial_state_sha256", {}) and state["initial_state_sha256"] != plan["initial_state_sha256"][arm]:
                    raise ValueError("Registered paired initialization differs from the reference controller")
                if state["initial_state_sha256"] != state_fingerprint(controller):
                    raise ValueError("Fresh semantic initialization changed")
                with torch.inference_mode():
                    _, zero_logits = supervised_loss(runtime, encoded[safety_index], controller, spec["token_policy"], "full")
                if not torch.equal(zero_logits, native_logits):
                    raise ValueError("Fresh semantic controller is not native-identical")
                snapshots = sorted(arm_path.glob("step-*/checkpoint.json"))
                if snapshots:
                    if not args.resume:
                        raise ValueError("Unexpected checkpoint in fresh semantic run")
                    record = load_checkpoint(snapshots[-1].parent, controller, optimizer)
                    if record["arm"] != arm or record["plan_sha256"] != manifest["plan_sha256"] or state_fingerprint(controller) != record["state_sha256"]:
                        raise ValueError("Semantic resume checkpoint belongs to another protocol")
                    retained = restore_log(log_path, record)
                    if retained:
                        manifest["resume_history"][-1].setdefault("retained_log_tails", []).append(retained)
                    start_step = record["step"]
                    audit_checkpoint(args.output, arm, start_step, plan, encoded)
                    if start_step not in state["completed_checkpoints"]:
                        state["completed_checkpoints"].append(start_step)
                    if start_step == plan["maximum_optimizer_steps"]:
                        if not (snapshots[-1].parent / "evaluation/summary.json").exists():
                            evaluate_final(runtime, controller, arm, spec["token_policy"], plan, data, encoded, registry, snapshots[-1].parent)
                        audit_evaluation(args.output, arm, plan, data, encoded, registry)
                elif log_path.exists() and log_path.stat().st_size:
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    log_path.rename(arm_path / f"interrupted-before-first-checkpoint-{stamp}.jsonl")
                atomic_json(manifest_path, manifest)
                with log_path.open("a", buffering=1) as log:
                    for step in range(start_step + 1, plan["maximum_optimizer_steps"] + 1):
                        started = time.monotonic()
                        optimizer.zero_grad(set_to_none=True)
                        losses, micros = [], []
                        for index in microbatch_indices(step, len(encoded), plan["gradient_accumulation"]):
                            item = encoded[index]
                            loss, _ = supervised_loss(runtime, item, controller, spec["token_policy"], "full")
                            if not torch.isfinite(loss):
                                raise ValueError("Nonfinite semantic training loss")
                            (loss / plan["gradient_accumulation"]).backward()
                            losses.append(loss.item())
                            micros.append({"training_index": index, **{key: item[key] for key in ("case_id", "variant", "role")}})
                        gradient = torch.nn.utils.clip_grad_norm_(controller.parameters(), plan["max_gradient_norm"], error_if_nonfinite=True)
                        optimizer.step()
                        if any(parameter.requires_grad or parameter.grad is not None for parameter in runtime.model.parameters()):
                            raise ValueError("Semantic training changed frozen-backbone gradients")
                        log.write(json.dumps({"step": step, "loss": sum(losses) / len(losses), "gradient_norm": gradient.item(),
                            "microbatches": micros, "optimization_seconds": time.monotonic() - started}) + "\n")
                        if step % 48 == 0:
                            print(json.dumps({"event": "training", "arm": arm, "step": step, "loss": sum(losses) / len(losses)}), flush=True)
                        if step in plan["checkpoint_steps"]:
                            log.flush()
                            checkpoint = save_checkpoint(arm_path, step, controller, optimizer, log_path,
                                {"arm": arm, "policy": spec["token_policy"], "plan_sha256": manifest["plan_sha256"],
                                 "state_sha256": state_fingerprint(controller), "optimizer_parameter_names": [name for name, _ in controller.named_parameters()],
                                 "optimized_input_tokens": plan["optimized_input_token_budgets"][str(step)],
                                 "supervised_answer_tokens": plan["supervised_answer_token_budgets"][str(step)]})
                            audit_rng = rng_state(controller)
                            audit_checkpoint(args.output, arm, step, plan, encoded)
                            assert_rng_equal(audit_rng, rng_state(controller))
                            state["completed_checkpoints"].append(step)
                            atomic_json(manifest_path, manifest)
                            if step in plan["evaluation_steps"]:
                                summary = evaluate_final(runtime, controller, arm, spec["token_policy"], plan, data, encoded, registry, checkpoint)
                                audit_evaluation(args.output, arm, plan, data, encoded, registry)
                                print(json.dumps({"event": "final_evaluated", "arm": arm, "correct": summary["semantic_metrics"]["correct"], "examples": len(data["validation"])}), flush=True)
                if extra_validation is not None:
                    extra_rng = rng_state(controller)
                    state["extra_validation"] = extra_validation(runtime, controller, arm, spec["token_policy"], plan, arm_path / f"step-{plan['maximum_optimizer_steps']:06d}")
                    if extra_validation_audit is not None:
                        extra_validation_audit(args.output, arm, plan)
                    assert_rng_equal(extra_rng, rng_state(controller))
            finally:
                controller.close()
                del controller, optimizer
                gc.collect()
                torch.cuda.empty_cache()
            with torch.inference_mode():
                _, restored = supervised_loss(runtime, encoded[safety_index], reader_policy="full")
            if not torch.equal(restored, native_logits):
                raise ValueError("Detaching semantic controller did not restore native logits")
            state["native_restore_exact"] = True
            manifest["completed_arms"].append(arm)
            atomic_json(manifest_path, manifest)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
