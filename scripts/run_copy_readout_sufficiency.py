"""Fixed-prefix oracle/non-slice readout on all existing COPY questions."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.kv_readout import QuestionKVReadout
from sas_memory.runtime import TextRuntime
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.copy_readout_data import DATA_SOURCES, DESTINATION, MODELS, PROTOCOL, REGIMES, read_plan, registered_inputs, sample_ids, selected_inputs, safety_samples
from scripts.copy_readout_runtime import prove_and_measure, readout
from scripts.run_controller_trial import state_fingerprint, write_json
from scripts.run_copy_dependency_transfer import actual_prompt
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_adapter_transfer import verify_question_prefix
from scripts.run_write_locality_load_screen import hook_state, weights_fingerprint

RUNTIME_SOURCES = tuple(sorted(set(DATA_SOURCES) | {
    "scripts/copy_readout_runtime.py", "scripts/copy_readout_metrics.py", "scripts/run_copy_readout_sufficiency.py",
    "scripts/analyze_copy_readout_sufficiency.py",
}))


def checked_smoke(path, digest):
    from scripts.analyze_copy_readout_sufficiency import audit_run
    report = json.loads(path.read_text())
    if (report["status"] != "audited_smoke" or report["new_physical_readouts_audited"] != 288 or
        report["dataset_manifest_sha256"] != digest or audit_run(Path(report["source_run"]), True) != report):
        raise ValueError("Complete independently audited288-query readout smoke required")
    return {"path": str(path), "sha256": streaming_hash(path), "manifest_sha256": report["manifest_sha256"]}


def execute(args, plan, registered, source_plan, cases, registry, masks, full_rows, gate):
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    torch.set_num_threads(4)
    physical, bindings = selected_inputs(registry, args.smoke)
    samples, prefix_samples = sample_ids(cases, registry), safety_samples(registry)
    case_map = {c["id"]: c for c in cases}
    full = {name: {r["id"]: r for r in full_rows[name]} for name in MODELS}
    sources = set(RUNTIME_SOURCES) | {str(p.relative_to(ROOT)) for p in (ROOT / "sas_memory").glob("*.py")}
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "training": False, "new_adapters": False,
        "smoke": args.smoke, "dataset": str(args.dataset), "dataset_manifest_sha256": streaming_hash(args.dataset / "manifest.json"),
        "protocol_sha256": streaming_hash(PROTOCOL), "mask_registry_sha256": registered["mask_registry_sha256"], "references": registered["references"],
        "model_path": plan["model_path"], "selected_physical_ids": [q["id"] for q in physical], "bindings_per_model_regime": len(bindings),
        "sample_query_ids": samples, "native_prefix_sample_ids": prefix_samples, "smoke_gate": gate, "generation": plan["generation"],
        "models": {}, "completed_models": [], "source_sha256": {n: streaming_hash(ROOT / n) for n in sorted(sources)},
        "generation_config_sha256": {p.name: streaming_hash(p) for p in Path(plan["model_path"]).glob("generation_config.json")}}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    for name in sources:
        target = args.output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    path = args.output / "manifest.json"
    write_json(path, manifest)
    try:
        runtime = TextRuntime(plan["model_path"], "it")
        manifest.update(python=sys.executable, gpu=torch.cuda.get_device_name(0), actual_eos_token_id=runtime.model.generation_config.eos_token_id,
            packages={n: importlib.metadata.version(n) for n in ("torch", "transformers", "fla-core")})
        if manifest["packages"] != plan["packages"] or runtime.model.config.layer_types != (["linear_attention"] * 3 + ["full_attention"]) * 6:
            raise ValueError("Actual native runtime or hybrid layout changed")
        native_hooks = hook_state(runtime.model)
        manifest["weights_before"] = weights_fingerprint(runtime.model)
        before = manifest["weights_before"]
        if before["sha256"] != plan["native_loaded_weights_sha256"] or not before["all_parameters_frozen"] or not before["no_parameter_gradients"]:
            raise ValueError("Actual loaded native checkpoint/freeze changed")
        for name in MODELS:
            spec = source_plan["models"][name]
            state = {"completed_queries_per_regime": 0, "prefix_checks": [], "proofs": [], "native_restore_checks": [],
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
                        state["checkpoint_sha256"] != spec["checkpoint_sha256"] or state["adapter_state_sha256_before"] != registered["references"]["training"][name]["checkpoint_state_sha256"]):
                        raise ValueError("Actual loaded frozen adapter state, scope or parameter count changed")
                with torch.inference_mode(), ExitStack() as stack:
                    outputs = {r: stack.enter_context((args.output / f"{name}.{r}.jsonl").open("w", buffering=1)) for r in REGIMES}
                    if spec and spec["token_policy"] == "question":
                        for q in physical:
                            if q["id"] in prefix_samples:
                                state["prefix_checks"].append(verify_question_prefix(runtime, controller, actual_prompt(runtime, case_map, q), q))
                    for number, q in enumerate(physical, 1):
                        ledger = actual_prompt(runtime, case_map, q)
                        if controller is not None:
                            controller.set_token_policy(spec["token_policy"], q["ledger_tokens"])
                        active_hooks = hook_state(runtime.model)
                        keeps = {r: [bit == "1" for bit in masks[q["id"]]["keep_bits"][r]] for r in REGIMES}
                        if q["id"] in samples:
                            proof, actual = prove_and_measure(runtime, ledger, q, keeps, full[name][q["id"]])
                            state["proofs"].append(proof)
                            calls = {r: proof["regimes"][r]["measure_hook_calls"] for r in REGIMES}
                        else:
                            actual, calls = {}, {}
                            for r in REGIMES:
                                with QuestionKVReadout(runtime.model, keeps[r]) as intervention:
                                    actual[r] = measure(runtime, ledger, q)
                                calls[r] = intervention.calls
                        for r in REGIMES:
                            if calls[r]["prefill"] != 1 or calls[r]["cached"] < 1 or actual[r]["query_tokens"] != q["question_tokens"]:
                                raise ValueError("Actual masked prefill/cached execution or query boundary changed")
                            outputs[r].write(json.dumps({"id": q["id"], "model": name, "regime": r, "prefill": "full", "token_policy": state["token_policy"],
                                "registration_sha256": data_fingerprint(q), "mask_registration_sha256": data_fingerprint(masks[q["id"]]),
                                "keep_bits_sha256": data_fingerprint(masks[q["id"]]["keep_bits"][r]), "hook_calls": calls[r], **actual[r]}) + "\n")
                        if hook_state(runtime.model) != active_hooks:
                            raise ValueError("Readout intervention/observer did not restore active controller hooks")
                        state["completed_queries_per_regime"] = number
                        if number % (6 if args.smoke else 32) == 0:
                            write_json(path, manifest)
                            print(json.dumps({"event": "copy_readout", "model": name, "queries_per_regime": number, "total": len(physical)}), flush=True)
                if controller is not None:
                    state.update(adapter_state_sha256_after=state_fingerprint(controller), all_adapter_parameters_frozen=all(not p.requires_grad and p.grad is None for p in controller.parameters()))
                    if state["adapter_state_sha256_before"] != state["adapter_state_sha256_after"] or not state["all_adapter_parameters_frozen"]:
                        raise ValueError("Fixed adapter tensors or gradients changed")
            finally:
                if controller is not None:
                    controller.close()
            if hook_state(runtime.model) != native_hooks:
                raise ValueError("Detach failed to restore all original native hooks")
            with torch.inference_mode():
                for q in physical:
                    if q["id"] in samples:
                        actual = measure(runtime, actual_prompt(runtime, case_map, q), q)
                        if actual != readout(full["native"][q["id"]]):
                            raise ValueError("Detached native full-history readout failed exact replay")
                        state["native_restore_checks"].append({"id": q["id"], "exact": True, "readout": actual})
            state.update(hooks_restored=True, results_sha256={r: streaming_hash(args.output / f"{name}.{r}.jsonl") for r in REGIMES})
            manifest["completed_models"].append(name)
            write_json(path, manifest)
            print(json.dumps({"event": "copy_readout_arm_complete", "model": name, "readouts": 2 * len(physical)}), flush=True)
        manifest["weights_after"] = weights_fingerprint(runtime.model)
        manifest["hooks_restored"] = hook_state(runtime.model) == native_hooks
        if manifest["weights_after"] != before or not manifest["hooks_restored"]:
            raise ValueError("Native tensors, gradients or hooks changed")
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
        raise ValueError("Choose a new operational smoke or provide its independent audit")
    registered, source_plan, cases, registry, masks, rows = registered_inputs(args.dataset)
    gate = None if args.smoke else checked_smoke(args.smoke_audit, streaming_hash(args.dataset / "manifest.json"))
    execute(args, read_plan(), registered, source_plan, cases, registry, masks, rows, gate)


if __name__ == "__main__":
    main()
