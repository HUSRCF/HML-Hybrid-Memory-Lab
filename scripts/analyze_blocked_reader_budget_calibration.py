"""Audit cyclic optimization budgets, optimizer snapshots and train/validation readouts."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash, microbatch_indices
from sas_memory.record_kv_ablation import record_blocked_mask
from sas_memory.training_data import data_fingerprint
from scripts.analyze_blocked_reader_adaptation import audit_run, audit_safety, tokenizer_runtime
from scripts.analyze_controller_trial import aggregate, load_rows
from scripts.run_controller_trial import encode_example, prompt_for, state_fingerprint, write_json


def fit_summary(path, data, encoded, runtime):
    expected = {(item["case_id"], item["variant"], item["role"]): example["query"]["answer"] for item, example in zip(encoded, data)}
    rows = load_rows(path, expected)
    if len(rows) != len(encoded):
        raise ValueError("Training fit is incomplete")
    for index, (row, item, example) in enumerate(zip(rows, encoded, data)):
        if (row["training_index"] != index or any(row[key] != item[key] for key in ("case_id", "variant", "role", "family")) or
            row["reader_mask"] != item["reader_mask"] or row["supervised_ids"] != item["answer_ids"] or
            len(row["teacher_prediction_ids"]) != len(item["answer_ids"]) or
            row["readout_calls"]["prefill"] != 1 or row["readout_calls"]["cached"] < 1 or
            any(type(value) is not int or value < 0 for value in row["teacher_prediction_ids"]) or
            not math.isfinite(row["teacher_loss_including_newline"]) or row["teacher_loss_including_newline"] < 0):
            raise ValueError("Training-fit identities, masks, teacher labels or cached calls changed")
        if runtime.tokenizer.decode(row["supervised_ids"][:3]) != example["query"]["answer"] or runtime.tokenizer.decode(row["supervised_ids"][3:]) != "\n":
            raise ValueError("Numeric/newline split invalid")
    return {"examples": len(rows), "free_greedy_correct": sum(row["greedy_exact"] for row in rows),
            "teacher_loss_including_newline": mean(row["teacher_loss_including_newline"] for row in rows),
            "gold_answer_nll_excluding_newline": mean(row["answer_nll"] for row in rows),
            "teacher_digit_correct": sum(sum(a == b for a, b in zip(row["supervised_ids"][:3], row["teacher_prediction_ids"][:3])) for row in rows),
            "teacher_digit_tokens": 3 * len(rows),
            "teacher_newline_correct": sum(row["supervised_ids"][3] == row["teacher_prediction_ids"][3] for row in rows),
            "prediction_counts": dict(Counter(row["generated"].strip() for row in rows))}


def audit_calibration(path, allow_smoke=False):
    import torch
    from safetensors.torch import load_file
    manifest = json.loads((path / "manifest.json").read_text())
    plan = json.loads((path / "plan.json").read_text())
    if manifest["status"] != "completed" or manifest["completed_policies"] != ["question", "all"]:
        raise ValueError("Calibration has not completed both arms")
    if manifest["smoke"] and not allow_smoke:
        raise ValueError("Smoke cannot establish optimization/fit results")
    if manifest["evaluated_splits"] != ["train", "validation"] or list(path.rglob("test.jsonl")):
        raise ValueError("Unexpected test inference in budget calibration")
    if manifest["plan_sha256"] != data_fingerprint(plan) or datetime.fromisoformat(plan["registered_utc"]) >= datetime.fromisoformat(manifest["started_utc"]):
        raise ValueError("Plan changed or was registered after training")
    for relative, digest in manifest["source_sha256"].items():
        if file_hash(path / "sources" / relative) != digest:
            raise ValueError(f"Source snapshot changed: {relative}")
    for relative in ("sas_memory/runtime.py", "sas_memory/controllers.py", "sas_memory/calibration.py", "sas_memory/kv_readout.py", "scripts/run_controller_trial.py"):
        if file_hash(ROOT / relative) != manifest["source_sha256"][relative]:
            raise ValueError("Actual training/runtime implementation differs")
    sources = {policy: audit_run(Path(reference))[0] for policy, reference in plan["reference_step64"].items()}
    if sources != manifest["checkpoint_sources"]:
        raise ValueError("Original references changed")
    data = json.loads((path / "data.json").read_text())
    if data_fingerprint(data) != plan["data_sha256"] or manifest["data_sha256"] != plan["data_sha256"]:
        raise ValueError("Original fixed data changed")
    runtime = tokenizer_runtime(manifest["model_path"])
    encoded = [encode_example(runtime, example, "records_blocked") for example in data["train"]]
    if encoded != json.loads((path / "tokenized_train.json").read_text()):
        raise ValueError("Training tokens/masks do not reconstruct")
    if data_fingerprint([{key: item[key] for key in ("input_ids", "answer_ids")} for item in encoded]) != plan["tokenized_input_answer_sha256"]:
        raise ValueError("Original input/answer tokens changed")
    registry = {"train": [item["reader_mask"] for item in encoded], "validation": []}
    expected_validation, ledgers = {}, []
    for case in data["validation"]:
        for variant, content in case["variants"].items():
            for query in content["queries"]:
                ledger, prompt = prompt_for(runtime, case, variant, query)
                key = (case["id"], variant, query["role"])
                expected_validation[key] = query["answer"]
                registry["validation"].append({"case_id": case["id"], "variant": variant, "role": query["role"],
                                               "reader_mask": record_blocked_mask(runtime, ledger, prompt)})
        ledger, _ = prompt_for(runtime, case, "control", case["variants"]["control"]["queries"][0])
        ledgers.append((len(runtime.ids(ledger)), case["id"], ledger))
    if registry != json.loads((path / "reader_registry.json").read_text()) or data_fingerprint(registry) != manifest["reader_registry_sha256"]:
        raise ValueError("Boundary-only registry changed")
    validation_masks = {(row["case_id"], row["variant"], row["role"]): row["reader_mask"] for row in registry["validation"]}
    validation_cases = {case["id"]: case for case in data["validation"]}
    arms, rows_audited = {}, 0
    for policy in ("question", "all"):
        policy_path = path / policy
        check = json.loads((policy_path / "safety_check.json").read_text())
        audit_safety(check, policy, plan["expected_trainable_parameters"])
        if not manifest["arms"][policy]["native_restore_exact"] or manifest["arms"][policy]["initial_state_sha256"] != plan["initial_state_sha256"]:
            raise ValueError("Native restore or initialization check failed")
        if manifest["smoke"]:
            if list(policy_path.glob("step-*")) or (policy_path / "training.jsonl").exists():
                raise ValueError("Smoke unexpectedly performed scheduled training")
            arms[policy] = {"safety_passed": True}
            continue
        if manifest["arms"][policy]["completed_checkpoints"] != plan["checkpoint_steps"]:
            raise ValueError("A registered checkpoint was skipped")
        logs_path = policy_path / "training.jsonl"
        log_bytes = logs_path.read_bytes()
        logs = [json.loads(line) for line in log_bytes.splitlines()]
        if [row["step"] for row in logs] != list(range(1, plan["maximum_optimizer_steps"] + 1)):
            raise ValueError("Incomplete, duplicate, or extra optimizer steps")
        for row in logs:
            indices = microbatch_indices(row["step"], len(encoded), plan["gradient_accumulation"])
            if (not math.isfinite(row["loss"]) or row["loss"] < 0 or not math.isfinite(row["gradient_norm"]) or
                len(row["microbatches"]) != len(indices)):
                raise ValueError("Invalid optimizer step")
            for micro, index in zip(row["microbatches"], indices):
                item = encoded[index]
                if (micro["training_index"] != index or any(micro[key] != item[key] for key in ("case_id", "variant", "role")) or
                    micro["keep_sha256"] != item["reader_mask"]["keep_sha256"] or micro["reader_calls"] != {"prefill": 1, "cached": 0}):
                    raise ValueError("An optimized example was reordered, changed or not masked")
        reference = Path(plan["reference_step64"][policy])
        replay = json.loads((policy_path / "step64_replay.json").read_text())
        reference_logs = [json.loads(line) for line in (reference / "local/training.jsonl").read_text().splitlines()]
        if (not all(replay[key] for key in ("weights_exact", "losses_exact", "gradient_norms_exact")) or
            any(a[key] != b[key] for a, b in zip(logs[:64], reference_logs) for key in ("loss", "gradient_norm"))):
            raise ValueError("First 64 steps did not replay the reference")
        checkpoints = {}
        for step in plan["checkpoint_steps"]:
            checkpoint = policy_path / f"step-{step:06d}"
            record = json.loads((checkpoint / "checkpoint.json").read_text())
            if (record["step"] != step or record["policy"] != policy or record["plan_sha256"] != manifest["plan_sha256"] or
                record["step64_replay_sha256"] != file_hash(policy_path / "step64_replay.json") or
                record["controller_sha256"] != file_hash(checkpoint / "controller.safetensors") or
                record["optimizer_sha256"] != file_hash(checkpoint / "optimizer.pt")):
                raise ValueError("Checkpoint identity/fingerprints changed")
            prefix = log_bytes[:record["training_log_prefix_bytes"]]
            if len(prefix.splitlines()) != step or hashlib.sha256(prefix).hexdigest() != record["training_log_prefix_sha256"]:
                raise ValueError("Checkpoint training-log prefix changed")
            indices = [index for s in range(1, step + 1) for index in microbatch_indices(s, len(encoded), plan["gradient_accumulation"])]
            budgets = {"optimized_input_tokens": sum(len(encoded[index]["input_ids"]) for index in indices),
                       "supervised_answer_tokens": sum(len(encoded[index]["answer_ids"]) for index in indices)}
            if (any(record[key] != value for key, value in budgets.items()) or budgets["optimized_input_tokens"] != plan["optimized_input_token_budgets"][str(step)] or
                budgets["supervised_answer_tokens"] != plan["supervised_answer_token_budgets"][str(step)]):
                raise ValueError("Checkpoint cumulative token budgets changed")
            weights = load_file(str(checkpoint / "controller.safetensors"), device="cpu")
            if (sum(t.numel() for t in weights.values()) != plan["expected_trainable_parameters"] or
                state_fingerprint(SimpleNamespace(state_dict=lambda: weights)) != record["state_sha256"] or
                any(not torch.isfinite(value).all() for value in weights.values())):
                raise ValueError("Saved adapter parameters invalid")
            if step == 64:
                original = load_file(str(reference / "local/controller.safetensors"), device="cpu")
                if set(original) != set(weights) or any(not torch.equal(weights[key], original[key]) for key in weights):
                    raise ValueError("Step-64 actual weights differ from the original")
            saved = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
            groups = saved["optimizer"]["param_groups"]
            names = record["optimizer_parameter_names"]
            if (len(groups) != 1 or groups[0]["lr"] != plan["learning_rate"] or groups[0]["weight_decay"] != plan["weight_decay"] or
                len(names) != len(set(names)) or set(names) != set(weights) or len(groups[0]["params"]) != len(names)):
                raise ValueError("Saved optimizer target/parameter group mismatch")
            for index, name in zip(groups[0]["params"], names):
                state = saved["optimizer"]["state"][index]
                if state["step"].item() != step:
                    raise ValueError("Saved optimizer moment step disagrees with checkpoint")
                for key in ("exp_avg", "exp_avg_sq"):
                    if state[key].shape != weights[name].shape or not torch.isfinite(state[key]).all():
                        raise ValueError("Optimizer moment shapes/values invalid")
            if saved["rng"]["cpu"].dtype != torch.uint8 or not saved["rng"]["cuda"]:
                raise ValueError("Missing CPU/GPU RNG snapshot")
            evaluation = checkpoint / "evaluation"
            summary = json.loads((evaluation / "summary.json").read_text())
            if summary["status"] != "completed" or summary["step"] != step or summary["policy"] != policy or not summary["adapter_unchanged"] or not summary["training_rng_unchanged"]:
                raise ValueError("Checkpoint evaluation identity failed")
            for name, digest in summary["file_sha256"].items():
                if file_hash(evaluation / name) != digest:
                    raise ValueError("Checkpoint evaluation rows changed")
            fit = fit_summary(evaluation / "train.jsonl", data["train"], encoded, runtime)
            if fit != summary["train"]:
                raise ValueError("Training-fit summary disagrees with raw rows")
            validation = load_rows(evaluation / "validation.jsonl", expected_validation)
            for row in validation:
                key = (row["case_id"], row["variant"], row["role"])
                case = validation_cases[row["case_id"]]
                if (row["reader_policy"] != "records_blocked" or row["reader_mask"] != validation_masks[key] or
                    row["readout_calls"]["prefill"] != 1 or row["readout_calls"]["cached"] < 1 or
                    any(row[field] != case[field] for field in ("family", "seed", "distractor_updates"))):
                    raise ValueError("Validation query had changed input/mask or unrestricted cached access")
            if step == 64 and (evaluation / "validation.jsonl").read_bytes() != (reference / "local/validation.jsonl").read_bytes():
                raise ValueError("Step-64 validation replay changed")
            grouped = defaultdict(list)
            for row in validation:
                grouped[(row["family"], row["variant"], row["role"])].append(row)
            expected_grouped = [{"family": family, "variant": variant, "role": role, "count": len(members),
                                 "correct": sum(r["greedy_exact"] for r in members), "accuracy": mean(r["greedy_exact"] for r in members),
                                 "answer_nll": mean(r["answer_nll"] for r in members)} for (family, variant, role), members in sorted(grouped.items())]
            if summary["validation"] != {"rows": len(validation), "grouped": expected_grouped}:
                raise ValueError("Validation summary disagrees with raw rows")
            audit_safety({**check, "reader_probability_checks_after_update": summary["probability_checks"]}, policy, plan["expected_trainable_parameters"])
            if policy == "question":
                selected = [(length, case_id) for length, case_id, _ in (min(ledgers), max(ledgers))]
                if [(item["ledger_tokens"], item["case_id"]) for item in summary["pre_question_prefix_checks"]] != selected or not all(item["kv_recurrent_conv_and_logits_exact"] for item in summary["pre_question_prefix_checks"]):
                    raise ValueError("Final trained short/long question-only prefix identity not established")
            checkpoints[str(step)] = {"training_fit": fit, "validation": aggregate(validation), "budgets": budgets,
                                      "controller_sha256": record["controller_sha256"], "optimizer_sha256": record["optimizer_sha256"]}
            rows_audited += len(encoded) + len(validation)
        arms[policy] = {"checkpoints": checkpoints, "optimizer_steps": len(logs), "step64_exact_replay": True,
                        "retained_interruption_files": [str(p.relative_to(path)) for p in policy_path.glob("interrupted-*")]}
    return {"status": "audited_smoke" if manifest["smoke"] else "completed_and_audited", "source_run": str(path),
            "manifest_sha256": file_hash(path / "manifest.json"), "plan_sha256": manifest["plan_sha256"], "arms": arms,
            "records_audited": rows_audited, "evaluated_splits": manifest["evaluated_splits"],
            "interpretation": "One initialization, fixed repeated original training data and a reused validation split. Checkpoints and budgets were fixed before this run. No new test-cohort inference or checkpoint selection. Fitting these data does not establish semantic generalization or a new architecture; persistent failure leaves adapter/readout capacity and accessible information unresolved."}


def plot_report(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for policy, label, color in (("question", "Question-only", "#2681b5"), ("all", "All-token", "#d5634f")):
        checkpoints = report["arms"][policy]["checkpoints"]
        steps = [int(step) for step in checkpoints]
        fit = [checkpoints[str(step)]["training_fit"] for step in steps]
        joint = [next(item for item in checkpoints[str(step)]["validation"]["joint_all_three_correct"] if item["family"] == "copy") for step in steps]
        for ax, values in zip(axes, ([item["free_greedy_correct"] / item["examples"] for item in fit],
                                     [item["teacher_loss_including_newline"] for item in fit], [item["accuracy"] for item in joint])):
            ax.plot(steps, values, marker="o", label=label, color=color)
    for ax, title in zip(axes, ("Training greedy exact / 128", "Training teacher loss (incl. newline)", "Validation COPY joint / 8")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(steps, [str(step) for step in steps])
        ax.set_xlabel("Optimizer updates")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.2)
    axes[0].set_ylim(-.02, 1.05)
    axes[2].set_ylim(-.02, 1.05)
    axes[1].set_ylim(bottom=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .04))
    fig.text(.5, .015, "Same 128 training examples cycled; one initialization. Reused validation only; no architectural superiority claim.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .14, 1, 1))
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    report = audit_calibration(args.run, args.allow_smoke)
    report["analysis_source_sha256"] = file_hash(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    if args.plot and report["status"] == "completed_and_audited":
        plot_report(report, args.plot)
    if report["status"] == "completed_and_audited":
        for policy, arm in report["arms"].items():
            for step, checkpoint in arm["checkpoints"].items():
                print(json.dumps({"policy": policy, "step": step, "training_fit": checkpoint["training_fit"]["free_greedy_correct"], "validation": checkpoint["validation"]}))
    else:
        print(json.dumps({"status": report["status"], "arms": list(report["arms"])}))


if __name__ == "__main__":
    main()
