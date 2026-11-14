"""Matched fresh query/local controls on registered four-value fact families."""

import argparse
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
from scripts.analyze_gdn_query_capacity import audit_checkpoint, audit_smoke
from scripts.analyze_single_fact_probe import family_metrics
from scripts.run_blocked_reader_budget_calibration import atomic_json, fit_training
from scripts.run_controller_confirmation import verify_native_prefix
from scripts.run_controller_trial import (encode_example, prompt_for, state_fingerprint,
                                           supervised_loss, verify_blocked_reader, write_json)
from scripts.run_gdn_query_capacity_smoke import native_gate


def evaluate_checkpoint(runtime, controller, arm, data, encoded, checkpoint, step):
    evaluation = checkpoint / "evaluation"
    if evaluation.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        evaluation.rename(checkpoint / f"interrupted-evaluation-{stamp}")
    evaluation.mkdir()
    before_rng, before_state = rng_state(controller), state_fingerprint(controller)
    metrics = {}
    for split in ("train", "validation"):
        path = evaluation / f"{split}.jsonl"
        teacher = fit_training(runtime, controller, "question", data[split], encoded[split], path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != len(data[split]):
            raise ValueError("Incomplete family evaluation")
        for row, example in zip(rows, data[split]):
            row.update({key: example["case"][key] for key in ("base_case_id", "value_version")})
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        metrics[split] = {"teacher": teacher, "families": family_metrics(rows)}
    ledgers = []
    for example in data["validation"]:
        ledger, _ = prompt_for(runtime, example["case"], example["variant"], example["query"])
        ledgers.append((len(runtime.ids(ledger)), example["case"]["id"], ledger))
    prefix_checks = [{"case_id": case_id, "ledger_tokens": length, **verify_native_prefix(runtime, controller, ledger)}
                     for length, case_id, ledger in (min(ledgers), max(ledgers))]
    controller.set_token_policy("question", encoded["train"][0]["ledger_tokens"])
    probability_checks = verify_blocked_reader(runtime, encoded["train"][0])
    if state_fingerprint(controller) != before_state:
        raise ValueError("Evaluation changed the controller")
    assert_rng_equal(before_rng, rng_state(controller))
    summary = {"status": "completed", "step": step, "arm": arm, "metrics": metrics,
               "pre_question_prefix_checks": prefix_checks, "probability_checks": probability_checks,
               "adapter_unchanged": True, "training_rng_unchanged": True,
               "file_sha256": {f"{split}.jsonl": file_hash(evaluation / f"{split}.jsonl") for split in metrics}}
    atomic_json(evaluation / "summary.json", summary)
    return summary


def make_controller(runtime, plan, arm):
    import torch
    from sas_memory.controllers import GroupController
    from sas_memory.gdn_query_control import GDNQueryController
    torch.manual_seed(plan["initialization_seed"])
    if arm == "local":
        controller = GroupController(runtime.model, plan["group_index"], 22, "local", adapt_gdn_keys=True)
        expected = plan["baseline_trainable_parameters"]
    else:
        controller = GDNQueryController(runtime.model, plan["group_index"], plan["rank"])
        expected = plan["expected_trainable_parameters"]
    if controller.description()["trainable_parameters"] != expected:
        controller.close()
        raise ValueError("Actual controller capacity differs from registration")
    return controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/gdn_query_readout_control_plan.json")
    parser.add_argument("--native-run", type=Path, default=Path("/media/990Pro/Code/SAS/runs/memory_update/single-fact-native-probe-20261114-01"))
    parser.add_argument("--safety-run", type=Path, default=Path("/media/990Pro/Code/SAS/runs/memory_update/gdn-query-capacity-smoke-20261114-01"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if (plan["checkpoint_steps"] != [64, 256, 1024] or plan["gradient_accumulation"] != 2 or
        plan["initialization_seed"] != 1701 or plan["backbone"] != "it"):
        raise ValueError("Unregistered capacity-control schedule")
    registered, data, registry, native = native_gate(plan, args.native_run)
    safety = audit_smoke(args.safety_run, plan)
    for relative in ("sas_memory/controllers.py", "sas_memory/gdn_query_control.py", "sas_memory/runtime.py",
                     "sas_memory/kv_readout.py", "scripts/run_controller_trial.py"):
        if file_hash(ROOT / relative) != safety["source_sha256"][relative]:
            raise ValueError("Safety-tested controller/runtime changed")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with (args.output / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, plan, registered, data, native, safety)


def execute(args, plan, registered, data, native, safety):
    import torch
    from sas_memory.runtime import TextRuntime
    torch.set_num_threads(4)
    sources = sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__).resolve(),
        ROOT / "scripts/analyze_gdn_query_capacity.py", ROOT / "scripts/run_gdn_query_capacity_smoke.py",
        ROOT / "scripts/run_blocked_reader_budget_calibration.py", ROOT / "scripts/analyze_single_fact_probe.py",
        ROOT / "scripts/prepare_single_fact_control.py", ROOT / "scripts/run_controller_trial.py",
        ROOT / "scripts/run_controller_confirmation.py", ROOT / "scripts/run_oracle_kv_readout.py",
        ROOT / "scripts/check_record_attention_weights.py", ROOT / "scripts/analyze_blocked_reader_adaptation.py",
        ROOT / "scripts/analyze_controller_trial.py"]
    fingerprints = {str(path.relative_to(ROOT)): file_hash(path) for path in sources}
    manifest_path = args.output / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if (manifest["status"] not in ("running", "failed") or manifest["plan_sha256"] != data_fingerprint(plan) or
            manifest["source_sha256"] != fingerprints or manifest["safety_gate"] != safety or manifest["native_gate"] != native):
            raise ValueError("Resume sources/protocol/gates changed, or trial is complete")
        for relative, digest in fingerprints.items():
            if file_hash(args.output / "sources" / relative) != digest:
                raise ValueError("Archived source changed before resume")
        manifest.setdefault("resume_history", []).append({"utc": datetime.now(timezone.utc).isoformat(), "previous_status": manifest["status"]})
        manifest.pop("error", None)
        manifest["status"] = "running"
    else:
        manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                    "plan_sha256": data_fingerprint(plan), "source_sha256": fingerprints, "safety_gate": safety,
                    "native_gate": native, "arms": {}, "completed_arms": [], "evaluated_splits": ["train", "validation"],
                    "data_sha256": plan["data_sha256"], "tokenized_input_answer_sha256": plan["tokenized_input_answer_sha256"]}
        write_json(args.output / "plan.json", plan)
        for relative in fingerprints:
            snapshot = args.output / "sources" / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((ROOT / relative).read_bytes())
    manifest["pid"] = os.getpid()
    atomic_json(manifest_path, manifest)
    try:
        runtime = TextRuntime(registered["model_path"], "it")
        packages = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")}
        if packages != safety["packages"] or registered["model_path"] != safety["model_path"]:
            raise ValueError("Training runtime differs from the safety-tested runtime")
        manifest.update(model_path=registered["model_path"], python=sys.executable, packages=packages, gpu=torch.cuda.get_device_name(0))
        encoded = {split: [encode_example(runtime, example, "records_blocked") for example in data[split]] for split in ("train", "validation")}
        reference = json.loads((Path(plan["data_source"]) / "tokenized_train.json").read_text())
        if encoded["train"] != reference or len(reference) != plan["unique_training_examples"]:
            raise ValueError("Actual optimized inputs/answers/masks differ from registration")
        tokens_path = args.output / "tokenized_train.json"
        if args.resume:
            if encoded["train"] != json.loads(tokens_path.read_text()):
                raise ValueError("Saved optimized inputs changed")
        else:
            write_json(tokens_path, encoded["train"])
        for step in plan["checkpoint_steps"]:
            actual = sum(len(encoded["train"][index]["input_ids"]) for update in range(1, step + 1)
                         for index in microbatch_indices(update, len(reference), plan["gradient_accumulation"]))
            if actual != plan["optimized_input_token_budgets"][str(step)]:
                raise ValueError("Cumulative optimized input budget differs from registration")
        with torch.inference_mode():
            _, native_logits = supervised_loss(runtime, reference[0], reader_policy="records_blocked")
            native_logits = native_logits.clone()
        for arm in ("local", "gdn_query"):
            if arm in manifest["completed_arms"]:
                audit_checkpoint(args.output, arm, plan["checkpoint_steps"][-1], plan, data, reference)
                continue
            controller = make_controller(runtime, plan, arm)
            initial_state = state_fingerprint(controller)
            optimizer = torch.optim.AdamW(controller.parameters(), lr=plan["learning_rate"], weight_decay=plan["weight_decay"])
            arm_path = args.output / arm
            arm_path.mkdir(exist_ok=args.resume)
            log_path = arm_path / "training.jsonl"
            state = manifest["arms"].setdefault(arm, {"initial_state_sha256": initial_state, "description": controller.description(), "completed_checkpoints": []})
            if state["initial_state_sha256"] != initial_state:
                raise ValueError("Fresh zero-output initialization changed")
            start_step = 0
            try:
                with torch.inference_mode():
                    _, zero_logits = supervised_loss(runtime, reference[0], controller, "question", "records_blocked")
                if not torch.equal(zero_logits, native_logits):
                    raise ValueError("Fresh controller changed native logits")
                snapshots = sorted(arm_path.glob("step-*/checkpoint.json"))
                if snapshots:
                    if not args.resume:
                        raise ValueError("Unexpected prior checkpoint in fresh trial")
                    checkpoint = snapshots[-1].parent
                    record = load_checkpoint(checkpoint, controller, optimizer)
                    if (record["arm"] != arm or record["plan_sha256"] != manifest["plan_sha256"] or
                        state_fingerprint(controller) != record["state_sha256"]):
                        raise ValueError("Resume checkpoint belongs to another arm/protocol")
                    retained = restore_log(log_path, record)
                    if retained:
                        manifest["resume_history"][-1].setdefault("retained_log_tails", []).append(retained)
                    start_step = record["step"]
                    if start_step not in state["completed_checkpoints"]:
                        evaluate_checkpoint(runtime, controller, arm, data, encoded, checkpoint, start_step)
                        audit_checkpoint(args.output, arm, start_step, plan, data, reference)
                        state["completed_checkpoints"].append(start_step)
                elif log_path.exists() and log_path.stat().st_size:
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    log_path.rename(arm_path / f"interrupted-before-first-checkpoint-{stamp}.jsonl")
                atomic_json(manifest_path, manifest)
                with log_path.open("a", buffering=1) as log:
                    for step in range(start_step + 1, plan["checkpoint_steps"][-1] + 1):
                        started = time.monotonic()
                        optimizer.zero_grad(set_to_none=True)
                        losses, micros = [], []
                        for index in microbatch_indices(step, len(reference), plan["gradient_accumulation"]):
                            item, calls = reference[index], []
                            loss, _ = supervised_loss(runtime, item, controller, "question", "records_blocked", calls)
                            if not torch.isfinite(loss) or calls != [{"prefill": 1, "cached": 0}]:
                                raise ValueError("Nonfinite loss/unmasked training forward")
                            (loss / plan["gradient_accumulation"]).backward()
                            losses.append(loss.item())
                            micros.append({"training_index": index, "case_id": item["case_id"],
                                           "keep_sha256": item["reader_mask"]["keep_sha256"], "reader_calls": calls[0]})
                        gradient = torch.nn.utils.clip_grad_norm_(controller.parameters(), plan["max_gradient_norm"], error_if_nonfinite=True)
                        optimizer.step()
                        if any(parameter.requires_grad or parameter.grad is not None for parameter in runtime.model.parameters()):
                            raise ValueError("Frozen backbone received a gradient")
                        row = {"step": step, "loss": mean(losses), "gradient_norm": gradient.item(), "microbatches": micros,
                               "optimization_seconds": time.monotonic() - started}
                        log.write(json.dumps(row) + "\n")
                        if step % 64 == 0:
                            print(json.dumps({"event": "training", "arm": arm, "step": step, "loss": row["loss"]}), flush=True)
                        if step in plan["checkpoint_steps"]:
                            log.flush()
                            indices = [index for update in range(1, step + 1) for index in microbatch_indices(update, len(reference), plan["gradient_accumulation"])]
                            checkpoint = save_checkpoint(arm_path, step, controller, optimizer, log_path,
                                {"arm": arm, "policy": "question", "plan_sha256": manifest["plan_sha256"],
                                 "state_sha256": state_fingerprint(controller),
                                 "optimizer_parameter_names": [name for name, _ in controller.named_parameters()],
                                 "optimized_input_tokens": sum(len(reference[index]["input_ids"]) for index in indices),
                                 "supervised_answer_tokens": sum(len(reference[index]["answer_ids"]) for index in indices)})
                            summary = evaluate_checkpoint(runtime, controller, arm, data, encoded, checkpoint, step)
                            before_audit_rng = rng_state(controller)
                            audit_checkpoint(args.output, arm, step, plan, data, reference)
                            assert_rng_equal(before_audit_rng, rng_state(controller))
                            state["completed_checkpoints"].append(step)
                            atomic_json(manifest_path, manifest)
                            print(json.dumps({"event": "checkpoint_evaluated", "arm": arm, "step": step,
                                "metrics": {split: {key: summary["metrics"][split]["families"][key] for key in ("correct", "examples", "all_four_correct", "families")}
                                            for split in ("train", "validation")}}), flush=True)
            finally:
                controller.close()
                del controller, optimizer
                gc.collect()
                torch.cuda.empty_cache()
            with torch.inference_mode():
                _, restored = supervised_loss(runtime, reference[0], reader_policy="records_blocked")
            if not torch.equal(restored, native_logits):
                raise ValueError("Detaching the trained control did not restore native logits")
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
