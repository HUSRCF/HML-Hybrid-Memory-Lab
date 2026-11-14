"""Evaluate five pinned frozen models on registered independent-record permutations."""

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from sas_memory.tasks import INSTRUCTIONS
from sas_memory.training_data import data_fingerprint
from scripts.analyze_initial_order_control import MODELS, audited_references, members, raw_metrics, registered_inputs
from scripts.analyze_paired_semantic_probe import identity
from scripts.check_record_attention_weights import ObservedRecordWeights
from scripts.run_blocked_reader_budget_calibration import atomic_json
from scripts.run_controller_confirmation import verify_native_prefix
from scripts.run_controller_trial import prompt_for, state_fingerprint, write_json
from scripts.run_oracle_kv_readout import measure


def execute(args, plan, registered, base, data, cases, registry, reference):
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    from sas_memory.runtime import TextRuntime
    torch.set_num_threads(4)
    sources = sorted((ROOT / "sas_memory").glob("*.py")) + [ROOT / "scripts" / name for name in (
        "run_initial_order_control.py", "analyze_initial_order_control.py", "prepare_initial_order_control.py",
        "analyze_paired_semantic_controllers.py", "analyze_paired_semantic_probe.py", "prepare_paired_semantic_updates.py",
        "compare_paired_semantic_controllers.py", "analyze_blocked_reader_adaptation.py", "analyze_controller_trial.py",
        "run_controller_trial.py", "run_controller_confirmation.py", "run_oracle_kv_readout.py", "check_record_attention_weights.py",
        "run_blocked_reader_budget_calibration.py")]
    fingerprints = {str(path.relative_to(ROOT)): file_hash(path) for path in sources}
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "training": False,
                "smoke": args.smoke, "pid": os.getpid(), "plan_sha256": data_fingerprint(plan), "source_sha256": fingerprints,
                "training_reference_audit": reference, "completed_models": [], "models": {}}
    write_json(args.output / "plan.json", plan)
    for relative in fingerprints:
        destination = args.output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    atomic_json(args.output / "manifest.json", manifest)
    try:
        runtime = TextRuntime(base["model_path"], "it")
        packages = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")}
        native_manifest = json.loads((Path(plan["native_reference_run"]) / "manifest.json").read_text())
        if packages != native_manifest["packages"]:
            raise ValueError("Frozen order-control runtime packages changed")
        manifest.update(model_path=base["model_path"], packages=packages, python=sys.executable, gpu=torch.cuda.get_device_name(0))
        selected = members(data, args.smoke)
        indexed = {(row["case_id"], row["variant"], row["role"]): row for row in registry}
        keys = sorted(tuple(identity(example)) for example in selected)
        sample_keys = {keys[0], keys[-1]}
        native_results = {}
        for name in MODELS:
            controller = None
            state = {"full_attention_checks": [], "pre_question_prefix_checks": []}
            if name != "native":
                spec = plan["checkpoint_sources"][name]
                if file_hash(spec["weights"]) != spec["weights_sha256"]:
                    raise ValueError("Pinned weights changed before loading")
                controller = GroupController(runtime.model, plan["group_index"], spec["rank"], spec["mode"], adapt_gdn_keys=True)
                controller.load_state_dict(load_file(spec["weights"], device=str(next(controller.parameters()).device)), strict=True)
                if sum(parameter.numel() for parameter in controller.parameters()) != spec["parameters"]:
                    raise ValueError("Frozen controller allocation changed")
                controller.eval().requires_grad_(False)
                state.update(checkpoint=spec, state_sha256_before=state_fingerprint(controller))
            rows = []
            try:
                with torch.inference_mode(), (args.output / f"{name}.jsonl").open("w", buffering=1) as output:
                    for number, example in enumerate(selected, 1):
                        key = tuple(identity(example))
                        case, query = cases[example["case"]["id"]], example["query"]
                        registration = indexed[key]
                        ledger, prompt = prompt_for(runtime, case, example["variant"], query)
                        if data_fingerprint(runtime.ids(prompt)) != registration["prompt_ids_sha256"]:
                            raise ValueError("Actual permuted input tokens changed")
                        if controller is not None:
                            controller.set_token_policy(spec["token_policy"], registration["ledger_tokens"])
                        if key in sample_keys:
                            with ObservedRecordWeights(runtime.model, len(runtime.ids(runtime.prompt_start + INSTRUCTIONS)), registration["ledger_tokens"]) as observer:
                                result = measure(runtime, ledger, query)
                            if set(observer.stats) != {3, 7, 11, 15, 19, 23}:
                                raise ValueError("Missing observed full-attention layer")
                            for stats in observer.stats.values():
                                if stats["prefill_calls"] != 1 or stats["cached_calls"] < 1 or stats["max_record_mass"] <= 0:
                                    raise ValueError("No observed full-history access")
                            state["full_attention_checks"].append({"key": list(key), "weights": observer.stats})
                        else:
                            result = measure(runtime, ledger, query)
                        if name == "native" and key in sample_keys:
                            replay = measure(runtime, ledger, query)
                            if replay["generated"] != result["generated"] or replay["answer_nll"] != result["answer_nll"]:
                                raise ValueError("Native permuted replay is not exact")
                            native_results[key] = result
                        row = {"case_id": case["id"], "variant": example["variant"], "role": query["role"],
                               **{field: case[field] for field in ("base_case_id", "family", "seed", "value_version", "distractor_updates")},
                               "condition": plan["condition"], "reader_policy": "full", "registration": registration, **result}
                        rows.append(row)
                        output.write(json.dumps(row) + "\n")
                        if number % 108 == 0:
                            print(json.dumps({"event": "order_validation", "model": name, "examples": number}), flush=True)
                    if name == "local_question":
                        ledgers = []
                        for example in selected:
                            ledger, _ = prompt_for(runtime, cases[example["case"]["id"]], example["variant"], example["query"])
                            ledgers.append((len(runtime.ids(ledger)), example["case"]["id"], example["variant"], ledger))
                        for length, case_id, variant, ledger in (min(ledgers), max(ledgers)):
                            state["pre_question_prefix_checks"].append({"case_id": case_id, "variant": variant, "ledger_tokens": length, **verify_native_prefix(runtime, controller, ledger)})
                state["full_attention_checks"].sort(key=lambda check: check["key"])
                if controller is not None:
                    state["state_sha256_after"] = state_fingerprint(controller)
                    if state["state_sha256_before"] != state["state_sha256_after"]:
                        raise ValueError("Frozen evaluated weights changed")
                state["weights_unchanged"] = True
            finally:
                if controller is not None:
                    controller.close()
                    del controller
                gc.collect()
                torch.cuda.empty_cache()
            with torch.inference_mode():
                for key in sorted(sample_keys):
                    example = next(example for example in selected if tuple(identity(example)) == key)
                    ledger, _ = prompt_for(runtime, cases[example["case"]["id"]], example["variant"], example["query"])
                    restored = measure(runtime, ledger, example["query"])
                    if restored["generated"] != native_results[key]["generated"] or restored["answer_nll"] != native_results[key]["answer_nll"]:
                        raise ValueError("Native outputs did not restore after frozen controller detach")
            state.update(native_restore_exact=True, metrics=raw_metrics(rows, args.smoke), results_sha256=file_hash(args.output / f"{name}.jsonl"))
            manifest["models"][name] = state
            manifest["completed_models"].append(name)
            atomic_json(args.output / "manifest.json", manifest)
            print(json.dumps({"event": "model_completed", "model": name, "correct": state["metrics"]["correct"], "examples": len(rows)}), flush=True)
        if any(parameter.requires_grad or parameter.grad is not None for parameter in runtime.model.parameters()):
            raise ValueError("Frozen backbone gained trainable parameters or gradients")
        manifest.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(), frozen_parameters_and_no_gradients=True)
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        atomic_json(args.output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/initial_order_control_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    registered, base, data, cases, registry = registered_inputs(plan)
    reference = audited_references(plan)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, plan, registered, base, data, cases, registry, reference)


if __name__ == "__main__":
    main()
