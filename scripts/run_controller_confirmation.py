"""Evaluate fixed controller checkpoints on a locked executable-task cohort."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_controller_trial import evaluate, prompt_for, write_json
from sas_memory.training_data import build_eval_cases, data_fingerprint


def verify_native_prefix(runtime, controller, ledger):
    import torch
    from sas_memory.runtime import assert_cache_equal
    controller.set_token_policy("none")
    native_cache, native_logits, _ = runtime.advance(ledger)
    controller.set_token_policy("question", len(runtime.ids(ledger)))
    actual_cache, actual_logits, _ = runtime.advance(ledger)
    assert_cache_equal(native_cache, actual_cache)
    if not torch.equal(native_logits, actual_logits):
        raise AssertionError("Question-only checkpoint changed ledger logits")
    return {"kv_recurrent_conv_and_logits_exact": True}


def evaluate_oracle(runtime, cases, path, options, controller=None, token_policy="all"):
    from dataclasses import asdict
    from sas_memory.oracle import evidence_slice
    from sas_memory.tasks import INSTRUCTIONS, Operation, execute, render
    count, prefix_check = 0, None
    with path.open("w", buffering=1) as output:
        for case in cases:
            for name in options["variants"]:
                variant = case["variants"][name]
                events = [Operation(**event) for event in case["prefix"] + variant["tail"]]
                for query in variant["queries"]:
                    if query["role"] not in options["roles"]:
                        continue
                    selected = evidence_slice(events, query["entity"], query["version"])
                    if str(execute(selected)[query["version"]][query["entity"]]) != query["answer"]:
                        raise AssertionError("Oracle slice changed executable truth")
                    selected_events = [asdict(event) for event in selected]
                    ledger = runtime.prompt_start + INSTRUCTIONS + render(selected_events)
                    if count == 0 and controller is not None and token_policy == "question":
                        prefix_check = {"case_id": case["id"], "selected_event_count": len(selected),
                                        **verify_native_prefix(runtime, controller, ledger)}
                    if controller is not None:
                        controller.set_token_policy(token_policy, len(runtime.ids(ledger)))
                    result = runtime.probe(ledger, None, query["question"], query["answer"], [query["answer"]],
                                           generate=True, prefill="full", score_batch=False)
                    for field in ("candidate_prediction", "candidate_correct", "candidate_scores"):
                        result.pop(field)
                    row = {"case_id": case["id"], "family": case["family"], "seed": case["seed"],
                           "distractor_updates": case["distractor_updates"], "variant": name, "role": query["role"],
                           "oracle_events": selected_events, "original_event_count": len(events),
                           "selected_event_count": len(selected), **result}
                    output.write(json.dumps(row) + "\n")
                    count += 1
    return {"rows": count, "pre_question_cache_check": prefix_check}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/controller_locked_confirmation.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sources, used_seeds = {}, set()
    common_protocol, common_data = None, None
    for label in config["modes"]:
        spec = config.get("checkpoint_runs", {}).get(label, {"training_run": config.get("training_run"), "mode": label})
        source = Path(spec["training_run"])
        training = json.loads((source / "manifest.json").read_text())
        mode = spec["mode"]
        if training["status"] != "completed" or mode not in training["completed_modes"]:
            raise ValueError(f"Training checkpoint incomplete: {label}")
        protocol = training["config"]
        identity = (protocol["model"], protocol["group_index"], protocol.get("adapt_gdn_keys", False))
        if common_protocol is not None and identity != common_protocol:
            raise ValueError("Confirmation checkpoints use different backbones or groups")
        common_protocol = identity
        old_data = json.loads((source / "data.json").read_text())
        digest = data_fingerprint(old_data)
        if digest != training["data_sha256"] or (common_data is not None and common_data != digest):
            raise ValueError("Training data changed or differs across checkpoints")
        common_data = digest
        used_seeds.update(item["case"]["seed"] for item in old_data["train"])
        used_seeds.update(case["seed"] for split in ("validation", "test") for case in old_data[split])
        sources[label] = {"training_run": str(source), "mode": mode, "config": protocol,
                          "training_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()}
    for path in config.get("excluded_case_files", []):
        used_seeds.update(case["seed"] for case in json.loads(Path(path).read_text()))
    cases = (json.loads(Path(config["cases_file"]).read_text()) if "cases_file" in config else
             build_eval_cases("test", config["copy_cases"], config["overwrite_cases"], config["seed_offset"]))
    validation = None
    if config.get("task_protocol") == "copy_generalization_v1":
        from sas_memory.generalization_tasks import validate_generalization_cases
        validation = validate_generalization_cases(cases)
    if config.get("expected_cases_sha256", data_fingerprint(cases)) != data_fingerprint(cases):
        raise ValueError("Cases do not match the locked confirmation digest")
    if used_seeds & {case["seed"] for case in cases}:
        raise ValueError("Confirmation seed leakage")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "cases.json", cases)
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "training_config": sources[config["modes"][0]]["config"], "checkpoint_sources": sources,
                "cases_sha256": data_fingerprint(cases),
                "completed_modes": [], "checkpoint_sha256": {},
                "pre_question_cache_checks": {},
                "completed_oracle_modes": [], "oracle_pre_question_cache_checks": {},
                "task_validation": validation,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT / "sas_memory").glob("*.py"))
                                  + [ROOT / "scripts/run_controller_trial.py", Path(__file__).resolve()]},
                "training": False,
                "evaluation_kind": config.get("evaluation_kind", "fresh_seed_confirmation"),
                "interpretation": "Frozen checkpoints; no optimization or checkpoint selection on this cohort."}
    for relative in manifest["source_sha256"]:
        snapshot = args.output / "sources" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / relative).read_bytes())
    manifest_path = args.output / "manifest.json"
    write_json(manifest_path, manifest)
    import torch
    from safetensors.torch import load_file
    from sas_memory.runtime import TextRuntime
    from sas_memory.controllers import GroupController
    torch.set_num_threads(4)
    protocol = manifest["training_config"]
    model_path = json.loads((ROOT / "configs/models.json").read_text())[protocol["model"]]
    try:
        runtime = TextRuntime(model_path, protocol["model"])
        result = evaluate(runtime, cases, args.output / "native.jsonl")
        write_json(args.output / "native.summary.json", result)
        manifest["completed_modes"].append("native")
        write_json(manifest_path, manifest)
        print(json.dumps({"event": "confirmation_complete", "mode": "native", "rows": result["rows"]}), flush=True)
        oracle_options = config.get("oracle_evaluation")
        if oracle_options:
            result = evaluate_oracle(runtime, cases, args.output / "native.oracle.jsonl", oracle_options)
            write_json(args.output / "native.oracle.summary.json", result)
            manifest["completed_oracle_modes"].append("native")
            write_json(manifest_path, manifest)
            print(json.dumps({"event": "oracle_complete", "mode": "native", "rows": result["rows"]}), flush=True)
        for label in config["modes"]:
            source_spec = sources[label]
            mode, protocol = source_spec["mode"], source_spec["config"]
            checkpoint = Path(source_spec["training_run"]) / mode / "controller.safetensors"
            manifest["checkpoint_sha256"][label] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            controller = GroupController(runtime.model, protocol["group_index"], protocol["modes"][mode], mode,
                                         adapt_gdn_keys=protocol.get("adapt_gdn_keys", False))
            controller.load_state_dict(load_file(str(checkpoint), device=str(runtime.model.device)), strict=True)
            try:
                policy = protocol.get("token_policy", "all")
                if policy == "question":
                    case = cases[0]
                    ledger, _ = prompt_for(runtime, case, "control", case["variants"]["control"]["queries"][0])
                    manifest["pre_question_cache_checks"][label] = {"case_id": case["id"], **verify_native_prefix(runtime, controller, ledger)}
                result = evaluate(runtime, cases, args.output / f"{label}.jsonl", controller, policy)
                write_json(args.output / f"{label}.summary.json", result)
                manifest["completed_modes"].append(label)
                write_json(manifest_path, manifest)
                print(json.dumps({"event": "confirmation_complete", "mode": label, "rows": result["rows"]}), flush=True)
                if oracle_options:
                    result = evaluate_oracle(runtime, cases, args.output / f"{label}.oracle.jsonl", oracle_options, controller, policy)
                    write_json(args.output / f"{label}.oracle.summary.json", result)
                    manifest["completed_oracle_modes"].append(label)
                    if result["pre_question_cache_check"] is not None:
                        manifest["oracle_pre_question_cache_checks"][label] = result["pre_question_cache_check"]
                    write_json(manifest_path, manifest)
                    print(json.dumps({"event": "oracle_complete", "mode": label, "rows": result["rows"]}), flush=True)
            finally:
                controller.close()
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
