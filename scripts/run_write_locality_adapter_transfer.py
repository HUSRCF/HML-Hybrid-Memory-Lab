"""Evaluate the two pinned GDN timing checkpoints on all natural long prefixes."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.runtime import TextRuntime
from sas_memory.write_locality_data import streaming_hash
from scripts.check_record_attention_weights import ObservedRecordWeights
from scripts.locality_transfer_data import (DATA_SOURCES, DESTINATION, MODELS, PROTOCOL,
    read_plan, registered_inputs, safety_samples, selected_inputs)
from scripts.run_controller_trial import prompt_for, state_fingerprint, write_json
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_load_screen import hook_state, weights_fingerprint
from scripts.semantic_order_path_common import assert_complete_cache_equal, cache_fingerprint

RUNTIME_SOURCES = tuple(sorted(set(DATA_SOURCES) | {
    "scripts/locality_transfer_metrics.py", "scripts/run_write_locality_adapter_transfer.py",
    "scripts/analyze_write_locality_adapter_transfer.py", "scripts/check_record_attention_weights.py",
    "scripts/run_oracle_kv_readout.py", "scripts/run_write_locality_load_screen.py",
}))


def checked_smoke(path, dataset_digest):
    from scripts.analyze_write_locality_adapter_transfer import audit_run
    report = json.loads(path.read_text())
    if (report["status"] != "audited_smoke" or report["physical_readouts_audited"] != 48 or
        report["dataset_manifest_sha256"] != dataset_digest or
        audit_run(Path(report["source_run"]), allow_smoke=True) != report):
        raise ValueError("Complete independent native operational smoke audit required")
    return {"path": str(path), "sha256": streaming_hash(path), "manifest_sha256": report["manifest_sha256"]}


def actual_prompt(runtime, cases, query):
    ledger, prompt = prompt_for(runtime, cases[query["cohort"]][query["case_id"]], query["cell"], query)
    ids, ledger_ids = runtime.ids(prompt), runtime.ids(ledger)
    if (len(ids) != query["prompt_tokens"] or data_fingerprint(ids) != query["prompt_ids_sha256"] or
        len(ledger_ids) != query["ledger_tokens"] or data_fingerprint(ledger_ids) != query["ledger_ids_sha256"] or
        ids[:len(ledger_ids)] != ledger_ids or runtime.continuation_ids(prompt, query["answer"]) != query["answer_ids"]):
        raise ValueError("Actual long transfer prompt, boundary or own-world answer changed")
    return ledger


def verify_question_prefix(runtime, controller, ledger, query):
    import torch
    controller.set_token_policy("none")
    native, native_logits, count = runtime.advance(ledger)
    controller.set_token_policy("question", query["ledger_tokens"])
    actual, actual_logits, actual_count = runtime.advance(ledger)
    assert_complete_cache_equal(native, actual)
    if count != query["ledger_tokens"] or actual_count != count or not torch.equal(native_logits, actual_logits):
        raise AssertionError("Question-only adapter changed the complete native ledger prefix or logits")
    return {"source_key": query["source_key"], "ledger_tokens": count,
            "complete_KV_recurrent_conv_bookkeeping_exact": True, "last_prefix_logits_exact": True,
            "prefix_sha256": cache_fingerprint(native)}


def execute(args, plan, registered, contexts, registry, gate):
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    torch.set_num_threads(4)
    pairs, bindings, physical = selected_inputs(registry, args.smoke)
    prefix_keys, query_samples = safety_samples(pairs, bindings, physical)
    cases = {cohort: {r["case"]["id"]: r["case"] for r in context["data"]["validation"]}
             for cohort, context in contexts.items()}
    sources = set(RUNTIME_SOURCES) | {str(p.relative_to(ROOT)) for p in (ROOT / "sas_memory").glob("*.py")}
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "training": False, "new_adapters": False, "smoke": args.smoke, "reader_policy": "full",
                "dataset": str(args.dataset), "dataset_manifest_sha256": streaming_hash(args.dataset / "manifest.json"),
                "protocol_sha256": streaming_hash(PROTOCOL), "registry_sha256": registered["registry_sha256"],
                "smoke_gate": gate, "model_path": registered["model_path"],
                "selected_pair_ids": [p["id"] for p in pairs], "selected_physical_ids": [q["id"] for q in physical],
                "reported_bindings_per_model": len(bindings), "physical_queries_per_model": len(physical),
                "prefix_sample_keys": prefix_keys, "attention_restore_query_ids": query_samples,
                "source_sha256": {name: streaming_hash(ROOT / name) for name in sorted(sources)},
                "models": {}, "completed_models": [], "generation": plan["generation"]}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    for name in sources:
        target = args.output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    path = args.output / "manifest.json"
    write_json(path, manifest)
    try:
        runtime = TextRuntime(registered["model_path"], "it")
        packages = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")}
        if packages != registered["references"]["event"]["packages"] or runtime.model.config.layer_types != (["linear_attention"] * 3 + ["full_attention"]) * 6:
            raise ValueError("Native runtime or3:1 group layout changed")
        manifest.update(python=sys.executable, packages=packages, gpu=torch.cuda.get_device_name(0),
                        actual_eos_token_id=runtime.model.generation_config.eos_token_id)
        hooks = hook_state(runtime.model)
        manifest["weights_before"] = weights_fingerprint(runtime.model)
        if (manifest["weights_before"]["sha256"] != plan["native_loaded_weights_sha256"] or
            not manifest["weights_before"]["all_parameters_frozen"] or not manifest["weights_before"]["no_parameter_gradients"]):
            raise ValueError("Actual loaded native checkpoint changed")
        write_json(path, manifest)
        for name in MODELS:
            spec = plan["models"][name]
            state = {"prefix_checks": [], "full_attention_checks": [], "native_restore_checks": [], "completed_readouts": 0}
            manifest["models"][name] = state
            controller = None
            try:
                controller = GroupController(runtime.model, spec["group_index"], spec["rank"], spec["mode"], adapt_gdn_keys=spec["adapt_gdn_keys"])
                controller.load_state_dict(load_file(spec["checkpoint"], device=str(next(controller.parameters()).device)), strict=True)
                controller.eval().requires_grad_(False)
                training_manifest = json.loads((Path(spec["training_run"]) / "manifest.json").read_text())
                state.update(description=controller.description(), checkpoint_sha256=streaming_hash(Path(spec["checkpoint"])),
                             adapter_state_sha256_before=state_fingerprint(controller), token_policy=spec["token_policy"],
                             adapter_parameters=sum(p.numel() for p in controller.parameters()))
                if (state["description"] != training_manifest["arms"][spec["training_arm"]]["description"] or
                    state["adapter_parameters"] != spec["parameters"] or state["checkpoint_sha256"] != spec["checkpoint_sha256"] or
                    state["adapter_state_sha256_before"] != registered["references"]["training"][name]["checkpoint_state_sha256"]):
                    raise ValueError("Actual loaded adapter tensors, projection scope or parameter count changed")
                with torch.inference_mode():
                    if spec["token_policy"] == "question":
                        for key in prefix_keys:
                            query = next(q for q in physical if q["source_key"] == key)
                            ledger = actual_prompt(runtime, cases, query)
                            state["prefix_checks"].append(verify_question_prefix(runtime, controller, ledger, query))
                        print(json.dumps({"event": "native_long_prefix_identity_passed", "model": name, "prefixes": len(prefix_keys)}), flush=True)
                    with (args.output / (name + ".jsonl")).open("w", buffering=1) as output:
                        for number, query in enumerate(physical, 1):
                            ledger = actual_prompt(runtime, cases, query)
                            controller.set_token_policy(spec["token_policy"], query["ledger_tokens"])
                            if query["id"] in query_samples:
                                with ObservedRecordWeights(runtime.model, query["prelude_tokens"], query["ledger_tokens"]) as observer:
                                    result = measure(runtime, ledger, query)
                                if set(observer.stats) != {3, 7, 11, 15, 19, 23} or any(
                                    stat["prefill_calls"] != 1 or stat["cached_calls"] < 1 or stat["max_record_mass"] <= 0
                                    for stat in observer.stats.values()):
                                    raise ValueError("Actual full-prefill/cached historical Attention access failed")
                                state["full_attention_checks"].append({"id": query["id"], "stats": observer.stats})
                            else:
                                result = measure(runtime, ledger, query)
                            if result["query_tokens"] != query["question_tokens"]:
                                raise ValueError("Actual query boundary changed")
                            output.write(json.dumps({"id": query["id"], "model": name, "reader_policy": "full", "prefill": "full",
                                "token_policy": spec["token_policy"], "registration_sha256": data_fingerprint(query), **result}) + "\n")
                            state["completed_readouts"] = number
                            if number % (12 if args.smoke else 32) == 0:
                                write_json(path, manifest)
                                print(json.dumps({"event": "fixed_checkpoint_transfer", "model": name, "completed": number, "total": len(physical)}), flush=True)
                state["adapter_state_sha256_after"] = state_fingerprint(controller)
                state["all_adapter_parameters_frozen"] = all(not p.requires_grad and p.grad is None for p in controller.parameters())
                if state["adapter_state_sha256_before"] != state["adapter_state_sha256_after"] or not state["all_adapter_parameters_frozen"]:
                    raise ValueError("Fixed evaluated adapter weights or gradients changed")
            finally:
                if controller is not None:
                    controller.close()
            if hook_state(runtime.model) != hooks:
                raise ValueError("Controller detach failed to restore original native hooks")
            with torch.inference_mode():
                for query in physical:
                    if query["id"] not in query_samples:
                        continue
                    ledger = actual_prompt(runtime, cases, query)
                    actual = measure(runtime, ledger, query)
                    expected = query["native_full_readout"]
                    if actual != expected:
                        raise ValueError("Detached native full-prefill readout does not exactly replay the frozen reference")
                    state["native_restore_checks"].append({"id": query["id"], "native_full_readout": actual, "exact": True})
            state.update(hooks_restored=True, results_sha256=streaming_hash(args.output / (name + ".jsonl")))
            manifest["completed_models"].append(name)
            write_json(path, manifest)
            print(json.dumps({"event": "timing_checkpoint_completed", "model": name, "readouts": len(physical)}), flush=True)
        manifest["weights_after"] = weights_fingerprint(runtime.model)
        manifest["hooks_restored"] = hook_state(runtime.model) == hooks
        if manifest["weights_before"] != manifest["weights_after"] or not manifest["hooks_restored"]:
            raise ValueError("Loaded native weights or original hooks changed")
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
        raise ValueError("Choose a new operational smoke or provide its completed independent audit")
    plan = read_plan()
    registered, contexts, registry = registered_inputs(args.dataset)
    gate = None if args.smoke else checked_smoke(args.smoke_audit, streaming_hash(args.dataset / "manifest.json"))
    execute(args, plan, registered, contexts, registry, gate)


if __name__ == "__main__":
    main()
