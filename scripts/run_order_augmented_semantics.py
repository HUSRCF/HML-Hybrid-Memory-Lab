"""Run the original matched training engine with ordinary order-augmented data."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import assert_rng_equal, file_hash, rng_state
from sas_memory.semantic_metrics import summarize_semantics
from sas_memory.tasks import INSTRUCTIONS
from sas_memory.training_data import data_fingerprint
from scripts.analyze_order_augmented_semantics import ARMS, audit_order_evaluation, order_inputs, reference_audits
from scripts.analyze_paired_semantic_probe import identity
from scripts.check_record_attention_weights import ObservedRecordWeights
from scripts.prepare_order_augmented_semantics import registered_augmented_dataset
from scripts.run_blocked_reader_budget_calibration import atomic_json
from scripts.run_controller_confirmation import verify_native_prefix
from scripts.run_controller_trial import prompt_for, state_fingerprint, write_json
from scripts.run_oracle_kv_readout import measure
from scripts.run_paired_semantic_controllers import execute


def evaluate_order(runtime, controller, arm, policy, plan, checkpoint):
    import torch
    path = checkpoint / "initial_order_evaluation"
    if (path / "summary.json").exists():
        return audit_order_evaluation(checkpoint.parents[1], arm, plan)
    if path.exists():
        path.rename(checkpoint / f"interrupted-order-evaluation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}")
    path.mkdir()
    examples, registry = order_inputs(plan)
    keys = sorted(tuple(identity(example)) for example in examples)
    sample_keys = {keys[0], keys[-1]}
    before_rng, before_state = rng_state(controller), state_fingerprint(controller)
    rows, observations = [], []
    with torch.inference_mode(), (path / "validation.jsonl").open("w", buffering=1) as output:
        for number, (example, registered) in enumerate(zip(examples, registry), 1):
            case, query = example["case"], example["query"]
            ledger, prompt = prompt_for(runtime, case, example["variant"], query)
            if data_fingerprint(runtime.ids(prompt)) != registered["prompt_ids_sha256"]:
                raise ValueError("Actual augmented-order prompt tokens changed")
            controller.set_token_policy(policy, registered["ledger_tokens"])
            if tuple(identity(example)) in sample_keys:
                with ObservedRecordWeights(runtime.model, len(runtime.ids(runtime.prompt_start + INSTRUCTIONS)), registered["ledger_tokens"]) as observer:
                    result = measure(runtime, ledger, query)
                if set(observer.stats) != {3, 7, 11, 15, 19, 23}:
                    raise ValueError("Missing actual full-attention layer")
                for stats in observer.stats.values():
                    if stats["prefill_calls"] != 1 or stats["cached_calls"] < 1 or stats["max_record_mass"] <= 0:
                        raise ValueError("No actual full historical access")
                observations.append({"key": identity(example), "weights": observer.stats})
            else:
                result = measure(runtime, ledger, query)
            row = {"case_id": case["id"], "variant": example["variant"], "role": query["role"],
                   **{key:case[key] for key in ("family", "base_case_id", "value_version", "seed", "distractor_updates")},
                   "registration": registered, "condition": "initial_record_derangement", "reader_policy": "full", **result}
            rows.append(row)
            output.write(json.dumps(row) + "\n")
            if number % 108 == 0:
                print(json.dumps({"event": "permuted_validation", "arm": arm, "examples": number}), flush=True)
        prefix_checks = []
        if policy == "question":
            ledgers = []
            for example in examples:
                ledger, _ = prompt_for(runtime, example["case"], example["variant"], example["query"])
                ledgers.append((len(runtime.ids(ledger)), example["case"]["id"], example["variant"], ledger))
            for length, case_id, variant, ledger in (min(ledgers), max(ledgers)):
                prefix_checks.append({"case_id": case_id, "variant": variant, "ledger_tokens": length, **verify_native_prefix(runtime, controller, ledger)})
    if state_fingerprint(controller) != before_state:
        raise ValueError("Frozen augmented-order evaluation changed adapter weights")
    assert_rng_equal(before_rng, rng_state(controller))
    summary = {"status": "completed", "arm": arm, "step": plan["maximum_optimizer_steps"], "condition": "initial_record_derangement",
               "evaluation_state_sha256": before_state, "adapter_unchanged": True, "training_rng_unchanged": True,
               "semantic_metrics": summarize_semantics(rows), "full_attention_checks": sorted(observations, key=lambda check:check["key"]),
               "pre_question_prefix_checks": prefix_checks, "results_sha256": file_hash(path / "validation.jsonl")}
    atomic_json(path / "summary.json", summary)
    print(json.dumps({"event": "permuted_evaluated", "arm": arm, "correct": summary["semantic_metrics"]["correct"], "examples": len(rows)}), flush=True)
    return audit_order_evaluation(checkpoint.parents[1], arm, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/order_augmented_semantic_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.resume:
        raise ValueError("Safety smoke cannot resume scheduled training")
    plan = json.loads(args.plan.read_text())
    if plan["profile"] != "ordinary_initial_order_augmentation" or list(plan["arms"]) != ARMS:
        raise ValueError("Unknown augmented training protocol")
    registered, data, registry = registered_augmented_dataset(Path(plan["data_source"]))
    for key in ("data_sha256", "registry_sha256", "tokenized_input_answer_sha256"):
        if registered[key] != plan[key]:
            raise ValueError("Augmented training data differ from registered protocol")
    references = reference_audits(plan)
    order_examples, _ = order_inputs(plan)
    if [identity(example) for example in data["validation"]] != [identity(example) for example in order_examples]:
        raise ValueError("Canonical/permuted development cohorts differ")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with (args.output / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reference_path = args.output / "reference_audits.json"
        if args.resume:
            if json.loads(reference_path.read_text()) != references:
                raise ValueError("Original development reference audits changed before resume")
        else:
            write_json(reference_path, references)
        extra_sources = [Path(__file__).resolve()] + [ROOT / "scripts" / name for name in (
            "prepare_order_augmented_semantics.py", "analyze_order_augmented_semantics.py", "analyze_initial_order_control.py",
            "prepare_initial_order_control.py", "compare_paired_semantic_controllers.py")]
        execute(args, plan, registered, data, registry, references["native"], extra_sources=extra_sources,
                extra_validation=evaluate_order, extra_validation_audit=audit_order_evaluation)


if __name__ == "__main__":
    main()
