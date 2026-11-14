"""Frozen eight-arm COPY transfer, with native-prefix and detached replay checks."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.runtime import TextRuntime
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.check_record_attention_weights import ObservedRecordWeights
from scripts.copy_transfer_data import (DATA_SOURCES, DESTINATION, MODELS, PROTOCOL, read_plan,
    registered_inputs, safety_samples, selected_inputs)
from scripts.run_controller_trial import prompt_for, state_fingerprint, write_json
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_adapter_transfer import verify_question_prefix
from scripts.run_write_locality_load_screen import hook_state, weights_fingerprint

RUNTIME_SOURCES = tuple(sorted(set(DATA_SOURCES) | {
    "scripts/run_copy_dependency_transfer.py", "scripts/analyze_copy_dependency_transfer.py", "scripts/copy_transfer_metrics.py",
}))


def actual_prompt(runtime, cases, query):
    ref = query["representative"]
    case = cases[ref["case_id"]]
    original = next(q for q in case["variants"][ref["variant"]]["queries"] if q["role"] == ref["role"])
    ledger, prompt = prompt_for(runtime, case, ref["variant"], original)
    ids, ledger_ids = runtime.ids(prompt), runtime.ids(ledger)
    if (len(ids) != query["prompt_tokens"] or len(ledger_ids) != query["ledger_tokens"] or
        data_fingerprint(ids) != query["prompt_ids_sha256"] or data_fingerprint(ledger_ids) != query["source_key"] or
        ids[:len(ledger_ids)] != ledger_ids or ids[len(ledger_ids):] != query["question_ids"] or
        original["answer"] != query["answer"] or runtime.continuation_ids(prompt, query["answer"]) != query["answer_ids"]):
        raise ValueError("Actual existing COPY prompt, boundary or own-world answer changed")
    return ledger


def checked_smoke(path, dataset_digest):
    from scripts.analyze_copy_dependency_transfer import audit_run
    report = json.loads(path.read_text())
    if (report["status"] != "audited_smoke" or report["physical_readouts_audited"] != 144 or
        report["dataset_manifest_sha256"] != dataset_digest or audit_run(Path(report["source_run"]), True) != report):
        raise ValueError("Complete independently audited144-query eight-arm smoke required")
    return {"path": str(path), "sha256": streaming_hash(path), "manifest_sha256": report["manifest_sha256"]}


def execute(args, plan, registered, cases, registry, gate):
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    torch.set_num_threads(4)
    physical, bindings = selected_inputs(registry, args.smoke)
    samples = safety_samples(registry)
    case_map = {c["id"]: c for c in cases}
    source_names = set(RUNTIME_SOURCES) | {str(p.relative_to(ROOT)) for p in (ROOT / "sas_memory").glob("*.py")}
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "training": False,
        "new_adapters": False, "reader_policy": "full", "smoke": args.smoke, "dataset": str(args.dataset),
        "dataset_manifest_sha256": streaming_hash(args.dataset / "manifest.json"), "protocol_sha256": streaming_hash(PROTOCOL),
        "registry_sha256": registered["registry_sha256"], "references": registered["references"], "smoke_gate": gate,
        "selected_physical_ids": [q["id"] for q in physical], "bindings_per_model": len(bindings),
        "safety_sample_ids": samples, "source_sha256": {n: streaming_hash(ROOT / n) for n in sorted(source_names)},
        "model_path": plan["model_path"], "models": {}, "completed_models": [], "generation": plan["generation"],
        "generation_config_sha256": {p.name: streaming_hash(p) for p in Path(plan["model_path"]).glob("generation_config.json")}}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    for name in source_names:
        target = args.output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    path = args.output / "manifest.json"
    write_json(path, manifest)
    try:
        runtime = TextRuntime(plan["model_path"], "it")
        manifest.update(python=sys.executable, packages={n: importlib.metadata.version(n) for n in ("torch", "transformers", "fla-core")},
                        gpu=torch.cuda.get_device_name(0), actual_eos_token_id=runtime.model.generation_config.eos_token_id)
        if manifest["packages"] != plan["packages"] or runtime.model.config.layer_types != (["linear_attention"] * 3 + ["full_attention"]) * 6:
            raise ValueError("Native runtime or3:1 hybrid layout changed")
        native_hooks = hook_state(runtime.model)
        manifest["weights_before"] = weights_fingerprint(runtime.model)
        before = manifest["weights_before"]
        if before["sha256"] != plan["native_loaded_weights_sha256"] or not before["all_parameters_frozen"] or not before["no_parameter_gradients"]:
            raise ValueError("Actual loaded native checkpoint/freeze changed")
        write_json(path, manifest)
        native_samples = {}
        for name in MODELS:
            spec = plan["models"][name]
            state = {"completed_readouts": 0, "prefix_checks": [], "attention_checks": [], "native_restore_checks": [],
                     "token_policy": spec["token_policy"] if spec else "none"}
            manifest["models"][name] = state
            controller = None
            try:
                if spec:
                    controller = GroupController(runtime.model, spec["group_index"], spec["rank"], spec["mode"], adapt_gdn_keys=spec["adapt_gdn_keys"])
                    controller.load_state_dict(load_file(spec["checkpoint"], device=str(next(controller.parameters()).device)), strict=True)
                    controller.eval().requires_grad_(False)
                    training = json.loads((Path(spec["training_run"]) / "manifest.json").read_text())
                    state.update(description=controller.description(), adapter_parameters=sum(p.numel() for p in controller.parameters()),
                        checkpoint_sha256=streaming_hash(Path(spec["checkpoint"])), adapter_state_sha256_before=state_fingerprint(controller))
                    if (state["description"] != training["arms"][spec["training_arm"]]["description"] or state["adapter_parameters"] != spec["parameters"] or
                        state["checkpoint_sha256"] != spec["checkpoint_sha256"] or
                        state["adapter_state_sha256_before"] != registered["references"]["training"][name]["checkpoint_state_sha256"]):
                        raise ValueError("Actual loaded fixed adapter, projection scope or parameter count changed")
                with torch.inference_mode():
                    if spec and spec["token_policy"] == "question":
                        for q in physical:
                            if q["id"] in samples:
                                state["prefix_checks"].append(verify_question_prefix(runtime, controller, actual_prompt(runtime, case_map, q), q))
                        print(json.dumps({"event": "copy_native_prefix_identity", "model": name, "prefixes": len(state["prefix_checks"])}), flush=True)
                    with (args.output / (name + ".jsonl")).open("w", buffering=1) as output:
                        for number, q in enumerate(physical, 1):
                            ledger = actual_prompt(runtime, case_map, q)
                            if controller is not None:
                                controller.set_token_policy(spec["token_policy"], q["ledger_tokens"])
                            active_hooks = hook_state(runtime.model)
                            if q["id"] in samples:
                                with ObservedRecordWeights(runtime.model, q["prelude_tokens"], q["ledger_tokens"]) as observer:
                                    result = measure(runtime, ledger, q)
                                if (set(observer.stats) != {3, 7, 11, 15, 19, 23} or hook_state(runtime.model) != active_hooks or any(
                                    s["prefill_calls"] != 1 or s["cached_calls"] < 1 or s["max_record_mass"] <= 0 for s in observer.stats.values())):
                                    raise ValueError("Actual six-layer full/cached historical access or observer cleanup failed")
                                state["attention_checks"].append({"id": q["id"], "stats": observer.stats})
                                if name == "native":
                                    native_samples[q["id"]] = result
                            else:
                                result = measure(runtime, ledger, q)
                            if result["query_tokens"] != q["question_tokens"]:
                                raise ValueError("Actual question token boundary changed")
                            output.write(json.dumps({"id": q["id"], "model": name, "prefill": "full", "reader_policy": "full",
                                "token_policy": state["token_policy"], "registration_sha256": data_fingerprint(q), **result}) + "\n")
                            state["completed_readouts"] = number
                            if number % (6 if args.smoke else 32) == 0:
                                write_json(path, manifest)
                                print(json.dumps({"event": "copy_dependency_transfer", "model": name, "completed": number, "total": len(physical)}), flush=True)
                if controller is not None:
                    state.update(adapter_state_sha256_after=state_fingerprint(controller),
                        all_adapter_parameters_frozen=all(not p.requires_grad and p.grad is None for p in controller.parameters()))
                    if state["adapter_state_sha256_before"] != state["adapter_state_sha256_after"] or not state["all_adapter_parameters_frozen"]:
                        raise ValueError("Fixed adapter tensors/gradients changed")
            finally:
                if controller is not None:
                    controller.close()
            if hook_state(runtime.model) != native_hooks:
                raise ValueError("Controller cleanup failed to restore native hooks")
            with torch.inference_mode():
                for q in physical:
                    if q["id"] in samples:
                        replay = measure(runtime, actual_prompt(runtime, case_map, q), q)
                        if replay != native_samples[q["id"]]:
                            raise ValueError("Detached native full-prefill readout did not replay exactly")
                        state["native_restore_checks"].append({"id": q["id"], "exact": True, "native_readout": replay})
            state.update(hooks_restored=True, results_sha256=streaming_hash(args.output / (name + ".jsonl")))
            manifest["completed_models"].append(name)
            write_json(path, manifest)
            print(json.dumps({"event": "copy_transfer_arm_complete", "model": name, "readouts": len(physical)}), flush=True)
        manifest["weights_after"] = weights_fingerprint(runtime.model)
        manifest["hooks_restored"] = hook_state(runtime.model) == native_hooks
        if manifest["weights_after"] != before or not manifest["hooks_restored"]:
            raise ValueError("Native model weights/gradients or hooks changed")
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DESTINATION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-audit", type=Path)
    args = parser.parse_args()
    if bool(args.smoke_audit) == args.smoke or args.output.exists():
        raise ValueError("Choose a new smoke run or supply its independently completed audit")
    plan = read_plan()
    registered, cases, registry = registered_inputs(args.dataset)
    gate = None if args.smoke else checked_smoke(args.smoke_audit, streaming_hash(args.dataset / "manifest.json"))
    execute(args, plan, registered, cases, registry, gate)


if __name__ == "__main__":
    main()
