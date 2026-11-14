"""Fresh full/oracle readouts with exact same-profile full-reference replay."""

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
from scripts.canonical_necessity_common import ADDENDUM_SHA256, check_affine_scope
from scripts.trained_timing_evidence_access_data import (DESTINATION, MODELS, PROTOCOL, MASKED_REGIMES, REGIMES,
    read_plan, registered_inputs, checked_smoke_equivalence, model_order, sources, require_exact_full)
from scripts.copy_readout_data import sample_ids, selected_inputs, safety_samples
from scripts.copy_readout_runtime import prove_and_measure, readout
from scripts.run_controller_trial import state_fingerprint, write_json
from scripts.run_copy_dependency_transfer import actual_prompt
from scripts.run_oracle_kv_readout import measure
from scripts.unseen_shared_timing_runtime import verify_prefix
from scripts.trained_timing_evidence_access_data import unique_prefix_queries
from scripts.run_write_locality_load_screen import hook_state, weights_fingerprint
from scripts.stable_adapter_linears import StableAdapterLinears


def execute(args, plan, registered, source_plan, cases, registry, masks, references, profile, gate):
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    torch.set_num_threads(4)
    physical, bindings = selected_inputs(registry, args.smoke)
    samples, prefix_samples = sample_ids(cases, registry), safety_samples(registry)
    case_map = {c["id"]: c for c in cases}
    reference_rows = {m: {r["id"]: r for r in rows} for m, rows in references.items()}
    order = model_order(args.reverse_model_order)
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "training": False, "new_adapters": False, "smoke": args.smoke,
        "reverse_model_order": args.reverse_model_order, "model_order": order,
        "dataset": str(args.dataset), "dataset_manifest_sha256": streaming_hash(args.dataset / "manifest.json"),
        "protocol_sha256": streaming_hash(PROTOCOL), "precision_addendum_sha256": ADDENDUM_SHA256,
        "arithmetic_profile": profile["profile"],
        "primary_reference": "fresh_full_same_canonical_arithmetic_profile", "cross_profile_rows_in_primary": 0,
        "mask_registry_sha256": registered["mask_registry_sha256"], "references": registered["references"],
        "model_path": plan["model_path"], "selected_physical_ids": [q["id"] for q in physical],
        "bindings_per_model_regime": len(bindings), "sample_query_ids": samples,
        "native_prefix_sample_ids": prefix_samples, "smoke_equivalence_gate": gate, "generation": plan["generation"],
        "models": {}, "completed_models": [], "native_anchors": [],
        "source_sha256": {n: streaming_hash(ROOT / n) for n in sorted(sources())},
        "generation_config_sha256": {p.name: streaming_hash(p) for p in Path(plan["model_path"]).glob("generation_config.json")}}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "precision_addendum.json", profile)
    for name in manifest["source_sha256"]:
        target = args.output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    path = args.output / "manifest.json"
    write_json(path, manifest)
    try:
        runtime = TextRuntime(plan["model_path"], "it")
        manifest.update(python=sys.executable, gpu=torch.cuda.get_device_name(0),
            actual_eos_token_id=runtime.model.generation_config.eos_token_id,
            packages={n: importlib.metadata.version(n) for n in ("torch", "transformers", "fla-core")})
        if (manifest["packages"] != plan["packages"] or
            runtime.model.config.layer_types != (["linear_attention"] * 3 + ["full_attention"]) * 6):
            raise ValueError("Actual native runtime or hybrid layout changed")
        if manifest['actual_eos_token_id']!=registered['references']['actual_eos_token_id'] or manifest['generation_config_sha256']!=registered['references']['generation_config_sha256']:
            raise ValueError('Original generation/stopping configuration changed')
        (args.output/'installed_modeling_source.py').write_bytes(Path(registered['references']['installed_modeling_source']).read_bytes())
        native_hooks = hook_state(runtime.model)
        manifest["weights_before"] = before = weights_fingerprint(runtime.model)
        if (before["sha256"] != plan["native_loaded_weights_sha256"] or
            not before["all_parameters_frozen"] or not before["no_parameter_gradients"]):
            raise ValueError("Actual loaded native checkpoint/freeze changed")
        anchors = {}
        with torch.inference_mode():
            for q in physical:
                if q["id"] in samples:
                    actual = measure(runtime, actual_prompt(runtime, case_map, q), q)
                    require_exact_full(actual, reference_rows["native"][q["id"]], "native_anchor", q["id"])
                    anchors[q["id"]] = actual
                    manifest["native_anchors"].append({"id": q["id"], "readout": actual})
        if list(anchors) != samples or hook_state(runtime.model) != native_hooks:
            raise ValueError("Independent native anchor coverage or hooks changed")
        write_json(path, manifest)
        for name in order:
            spec = source_plan["models"][name]
            state = {"completed_queries_per_regime": 0, "prefix_checks": [], "proofs": [],
                "native_restore_checks": [], "native_before_attach_checks": [], "attached_none_checks": [], "exact_canonical_full_replays": 0, "token_policy": spec["token_policy"] if spec else "none"}
            manifest["models"][name] = state
            controller, arithmetic = None, None
            try:
                with torch.inference_mode():
                    for q in physical:
                        if q['id'] in anchors:
                            actual=measure(runtime,actual_prompt(runtime,case_map,q),q)
                            if actual!=anchors[q['id']]:raise ValueError('Native before-attach replay failed')
                            state['native_before_attach_checks'].append({'id':q['id'],'readout':actual})
                if spec:
                    controller = GroupController(runtime.model, spec["group_index"], spec["rank"], spec["mode"],
                                                 adapt_gdn_keys=spec["adapt_gdn_keys"])
                    controller.load_state_dict(load_file(spec["checkpoint"], device=str(runtime.model.device)), strict=True)
                    controller.eval().requires_grad_(False)
                    training = json.loads((Path(spec["training_run"]) / "manifest.json").read_text())
                    state.update(description=controller.description(), adapter_parameters=sum(p.numel() for p in controller.parameters()),
                        checkpoint_sha256=streaming_hash(Path(spec["checkpoint"])), adapter_state_sha256_before=state_fingerprint(controller))
                    if (state["description"] != training["arms"][spec["training_arm"]]["description"] or
                        state["adapter_parameters"] != spec["parameters"] or state["checkpoint_sha256"] != spec["checkpoint_sha256"] or
                        state["adapter_state_sha256_before"] != registered["references"]["training"][name]["checkpoint_state_sha256"]):
                        raise ValueError("Actual frozen adapter bytes, scope or allocation changed")
                    state["affine_scope"] = check_affine_scope(controller, runtime.model)
                    if not state["affine_scope"]["all_adapter_linear_weights_fp32"]:
                        raise ValueError("Actual adapter parameter storage is not unchangedFP32")
                    arithmetic = StableAdapterLinears(controller)
                    state["arithmetic"] = arithmetic.description()
                with torch.inference_mode(), ExitStack() as stack:
                    outputs = {r: stack.enter_context((args.output / f"{name}.{r}.jsonl").open("w", buffering=1)) for r in REGIMES}
                    if controller:
                        for q in physical:
                            if q['id'] in anchors:
                                controller.set_token_policy('none',q['ledger_tokens'])
                                actual=measure(runtime,actual_prompt(runtime,case_map,q),q)
                                if actual!=anchors[q['id']]:raise ValueError('Attached-none native identity failed')
                                state['attached_none_checks'].append({'id':q['id'],'readout':actual})
                    if spec and spec['token_policy']=='question':
                        for q in unique_prefix_queries(physical):
                            state['prefix_checks'].append(verify_prefix(runtime,controller,actual_prompt(runtime,case_map,q),q))
                    for number, q in enumerate(physical, 1):
                        ledger = actual_prompt(runtime, case_map, q)
                        if controller is not None:
                            controller.set_token_policy(spec["token_policy"], q["ledger_tokens"])
                        active_hooks = hook_state(runtime.model)
                        # This reference belongs to this query/model/profile. Never import old FP32 scores.
                        full = measure(runtime, ledger, q)
                        require_exact_full(full, reference_rows[name][q["id"]], name, q["id"])
                        state["exact_canonical_full_replays"] += 1
                        if name == "native" and q["id"] in anchors and full != anchors[q["id"]]:
                            raise ValueError("Actual native arm differs from its pre-controller anchor")
                        keeps = {r: [bit == "1" for bit in masks[q["id"]]["keep_bits"][r]] for r in MASKED_REGIMES}
                        if q["id"] in samples:
                            proof, actual = prove_and_measure(runtime, ledger, q, keeps, full)
                            state["proofs"].append(proof)
                            calls = {r: proof["regimes"][r]["measure_hook_calls"] for r in MASKED_REGIMES}
                        else:
                            actual, calls = {}, {}
                            for r in MASKED_REGIMES:
                                with QuestionKVReadout(runtime.model, keeps[r]) as intervention:
                                    actual[r] = measure(runtime, ledger, q)
                                calls[r] = intervention.calls
                        actual["full"] = full
                        for r in REGIMES:
                            if actual[r]["query_tokens"] != q["question_tokens"]:
                                raise ValueError("Actual query token boundary changed")
                            if r != "full" and (calls[r]["prefill"] != 1 or calls[r]["cached"] < 1):
                                raise ValueError("Actual masked prefill/cached execution incomplete")
                            outputs[r].write(json.dumps({"id": q["id"], "model": name, "regime": r, "prefill": "full",
                                "token_policy": state["token_policy"], "arithmetic_profile": profile["profile"],
                                "reference_policy": "fresh_full_same_canonical_arithmetic_profile",
                                "registration_sha256": data_fingerprint(q),
                                "canonical_full_reference_sha256": data_fingerprint(readout(reference_rows[name][q["id"]])),
                                "mask_registration_sha256": None if r == "full" else data_fingerprint(masks[q["id"]]),
                                "keep_bits_sha256": None if r == "full" else data_fingerprint(masks[q["id"]]["keep_bits"][r]),
                                "hook_calls": None if r == "full" else calls[r], **actual[r]}) + "\n")
                        if hook_state(runtime.model) != active_hooks:
                            raise ValueError("Readout intervention/observer failed to restore active controller hooks")
                        state["completed_queries_per_regime"] = number
                        if number % (6 if args.smoke else 32) == 0:
                            write_json(path, manifest)
                            print(json.dumps({"event": "trained_timing_evidence_access", "model": name,
                                "queries_per_regime": number, "total": len(physical)}), flush=True)
                if controller is not None:
                    state.update(adapter_state_sha256_after=state_fingerprint(controller),
                        all_adapter_parameters_frozen=all(not p.requires_grad and p.grad is None for p in controller.parameters()),
                        actual_affine_calls=dict(arithmetic.calls))
                    if (state["adapter_state_sha256_before"] != state["adapter_state_sha256_after"] or
                        not state["all_adapter_parameters_frozen"] or any(v < 1 for v in arithmetic.calls.values())):
                        raise ValueError("Frozen adapter tensors, gradients or affine execution changed")
            finally:
                if arithmetic is not None:
                    arithmetic.close()
                    state["arithmetic_hooks_removed"] = all(not m._forward_hooks for m in controller.modules())
                if controller is not None:
                    controller.close()
            if hook_state(runtime.model) != native_hooks:
                raise ValueError("Detach failed to restore original native hooks")
            with torch.inference_mode():
                for q in physical:
                    if q["id"] in samples:
                        actual = measure(runtime, actual_prompt(runtime, case_map, q), q)
                        if actual != anchors[q["id"]]:
                            raise ValueError("Detached native readout failed exact anchor replay")
                        state["native_restore_checks"].append({"id": q["id"], "exact": True, "readout": actual})
            state.update(hooks_restored=True,
                         results_sha256={r: streaming_hash(args.output / f"{name}.{r}.jsonl") for r in REGIMES})
            manifest["completed_models"].append(name)
            write_json(path, manifest)
            print(json.dumps({"event": "evidence_access_arm_complete", "model": name,
                              "new_readouts": 3 * len(physical)}), flush=True)
        manifest["weights_after"] = weights_fingerprint(runtime.model)
        manifest["hooks_restored"] = hook_state(runtime.model) == native_hooks
        if manifest["weights_after"] != before or not manifest["hooks_restored"]:
            raise ValueError("Actual native tensors, gradients or hooks changed")
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
    parser.add_argument("--reverse-model-order", action="store_true")
    parser.add_argument("--smoke-equivalence", type=Path)
    args = parser.parse_args()
    if (bool(args.smoke_equivalence) == args.smoke or args.output.exists() or
        (args.reverse_model_order and not args.smoke)):
        raise ValueError("New forward/reverse operational smoke or gated forward full run required")
    registered, source_plan, cases, registry, masks, references = registered_inputs(args.dataset)
    plan = read_plan()
    profile = json.loads(Path(plan["precision_addendum"]).read_text())
    gate = None if args.smoke else checked_smoke_equivalence(args.smoke_equivalence, streaming_hash(args.dataset / "manifest.json"))
    execute(args, plan, registered, source_plan, cases, registry, masks, references, profile, gate)


if __name__ == "__main__":
    main()
