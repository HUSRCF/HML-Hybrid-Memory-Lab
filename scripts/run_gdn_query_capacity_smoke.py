"""Real-model safety gate for registered, matched query-readout capacity controls."""

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.counterbalance_data import file_digest
from sas_memory.training_data import data_fingerprint
from scripts.analyze_single_fact_probe import audit_run, registered_dataset
from scripts.run_controller_trial import (safety_check, supervised_loss,
                                           verify_blocked_reader, write_json)


def query_safety_check(runtime, encoded, plan):
    import torch
    from sas_memory.gdn_query_control import GDNQueryController
    torch.manual_seed(plan["initialization_seed"])
    with torch.inference_mode():
        native_loss, native_logits = supervised_loss(runtime, encoded, reader_policy="records_blocked")
        native_logits = native_logits.clone()
    controller = GDNQueryController(runtime.model, plan["group_index"], plan["rank"])
    optimizer = torch.optim.AdamW(controller.parameters(), lr=plan["learning_rate"], weight_decay=plan["weight_decay"])
    try:
        description = controller.description()
        if description["trainable_parameters"] != plan["expected_trainable_parameters"] or description["layers"] != plan["layers"]:
            raise ValueError("Actual query-control shape/count differs from registration")
        with torch.inference_mode():
            _, zero_logits = supervised_loss(runtime, encoded, controller, "question", "records_blocked")
            if not torch.equal(native_logits, zero_logits):
                raise ValueError("Zero query control changed native logits")
        loss, _ = supervised_loss(runtime, encoded, controller, "question", "records_blocked")
        loss.backward()
        output_gradients = {name: module.weight.grad.norm().item() if module.weight.grad is not None else None
                            for name, module in controller.outputs.items()}
        if any(value is None or not 0 < value < float("inf") for value in output_gradients.values()):
            raise ValueError(f"Invalid query output gradients: {output_gradients}")
        if any(parameter.requires_grad or parameter.grad is not None for parameter in runtime.model.parameters()):
            raise ValueError("Frozen backbone received gradients")
        torch.nn.utils.clip_grad_norm_(controller.parameters(), plan["max_gradient_norm"], error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        before, after, handles = {}, {}, []
        def observe(destination, name):
            def hook(module, args, output):
                destination[name] = output.detach().cpu().clone()
            return hook
        for projection, name, _ in controller._specs:
            handles.append(projection.register_forward_hook(observe(before, name), prepend=True))
            handles.append(projection.register_forward_hook(observe(after, name)))
        try:
            next_loss, _ = supervised_loss(runtime, encoded, controller, "question", "records_blocked")
        finally:
            for handle in handles:
                handle.remove()
        next_loss.backward()
        encoder_gradients = {name: module.weight.grad.norm().item() if module.weight.grad is not None else None
                             for name, module in controller.encoders.items()}
        if any(value is None or not 0 < value < float("inf") for value in encoder_gradients.values()):
            raise ValueError(f"Invalid query encoder gradients after update: {encoder_gradients}")
        direct = {}
        for _, name, query_dim in controller._specs:
            native, changed = before[name], after[name]
            checks = {"direct_key_value_channels_exact": torch.equal(native[..., query_dim:], changed[..., query_dim:]),
                      "prefix_projection_exact": torch.equal(native[:, :encoded["ledger_tokens"]], changed[:, :encoded["ledger_tokens"]]),
                      "query_residual_nonzero": not torch.equal(native[:, encoded["ledger_tokens"]:, :query_dim], changed[:, encoded["ledger_tokens"]:, :query_dim])}
            if not all(checks.values()):
                raise ValueError(f"Actual packed projection locality failed: {name}: {checks}")
            direct[name] = checks
        with torch.inference_mode():
            controller.set_token_policy("none")
            native_prefix = runtime.model(input_ids=runtime.tensor([encoded["input_ids"][:encoded["ledger_tokens"]]]), use_cache=True, logits_to_keep=1)
            controller.set_token_policy("question", encoded["ledger_tokens"])
            actual_prefix = runtime.model(input_ids=runtime.tensor([encoded["input_ids"][:encoded["ledger_tokens"]]]), use_cache=True, logits_to_keep=1)
            from sas_memory.runtime import assert_cache_equal
            assert_cache_equal(native_prefix.past_key_values, actual_prefix.past_key_values)
            if not torch.equal(native_prefix.logits, actual_prefix.logits):
                raise ValueError("Query control changed pre-question logits after update")
        probabilities = verify_blocked_reader(runtime, encoded)
        return {"status": "passed", **description, "initial_logit_max_error": 0.,
                "native_loss": native_loss.item(), "loss_after_one_step": next_loss.item(),
                "output_gradient_norms": output_gradients, "encoder_gradient_norms_after_one_step": encoder_gradients,
                "direct_projection_checks_after_update": direct, "backbone_gradients_absent": True,
                "token_policy": "question", "reader_policy": "records_blocked",
                "pre_question_cache_and_logits_exact_after_update": True,
                "reader_probability_checks_after_update": probabilities}
    finally:
        controller.close()
        with torch.inference_mode():
            _, restored = supervised_loss(runtime, encoded, reader_policy="records_blocked")
            if not torch.equal(native_logits, restored):
                raise ValueError("Removing query control did not restore native logits")
        del controller, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def native_gate(plan, native_run):
    native = audit_run(native_run)
    registered, data, registry = registered_dataset(Path(plan["data_source"]))
    native_plan = json.loads((native_run / "plan.json").read_text())
    for key in ("data_sha256", "registry_sha256"):
        plan_key = "dataset_registry_sha256" if key == "registry_sha256" else key
        if plan[plan_key] != registered[key] or native_plan[key] != registered[key]:
            raise ValueError("Native gate and capacity plan use different paired inputs")
    for split, conditions in native["splits"].items():
        if conditions["full_history"]["all_four_correct"] != conditions["full_history"]["families"]:
            raise ValueError("Native full-history immediate positive control is incomplete")
    blocked = native["splits"]["validation"]["records_blocked"]
    if blocked["all_four_correct"] == blocked["families"]:
        raise ValueError("Native blocked readout already succeeds: query-capacity training is not triggered")
    if (plan["token_policy"] != "question" or plan["reader_policy"] != "records_blocked" or
        registered["tokenized_input_answer_sha256"] != plan["tokenized_input_answer_sha256"]):
        raise ValueError("Capacity-control protocol changed")
    return registered, data, registry, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/gdn_query_readout_control_plan.json")
    parser.add_argument("--native-run", type=Path, default=Path("/media/990Pro/Code/SAS/runs/memory_update/single-fact-native-probe-20261114-01"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    registered, data, registry, native = native_gate(plan, args.native_run)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    sources = sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__).resolve(),
        ROOT / "scripts/run_controller_trial.py", ROOT / "scripts/run_controller_confirmation.py",
        ROOT / "scripts/run_oracle_kv_readout.py", ROOT / "scripts/check_record_attention_weights.py",
        ROOT / "scripts/analyze_single_fact_probe.py", ROOT / "scripts/prepare_single_fact_control.py",
        ROOT / "scripts/analyze_blocked_reader_adaptation.py", ROOT / "scripts/analyze_controller_trial.py"]
    manifest = {"status": "running", "stage": "safety_smoke_only", "started_utc": datetime.now(timezone.utc).isoformat(),
                "plan_sha256": data_fingerprint(plan), "source_sha256": {str(path.relative_to(ROOT)): file_digest(path) for path in sources},
                "native_gate": native, "dataset_manifest_sha256": file_digest(Path(plan["data_source"]) / "manifest.json"),
                "completed_arms": [], "training_schedule_started": False}
    for relative in manifest["source_sha256"]:
        path = args.output / "sources" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    manifest_path = args.output / "manifest.json"
    write_json(manifest_path, manifest)
    import torch
    from sas_memory.runtime import TextRuntime
    from scripts.run_controller_trial import encode_example
    torch.set_num_threads(4)
    try:
        runtime = TextRuntime(registered["model_path"], "it")
        manifest.update(model_path=registered["model_path"], python=sys.executable, gpu=torch.cuda.get_device_name(0),
                        packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")})
        encoded = encode_example(runtime, data["train"][0], "records_blocked")
        expected = json.loads((Path(plan["data_source"]) / "tokenized_train.json").read_text())[0]
        if encoded != expected:
            raise ValueError("Actual model tokens differ from registered tokens")
        baseline = safety_check(runtime, encoded, "local", 22, plan["group_index"], plan["initialization_seed"],
                                adapt_gdn_keys=True, token_policy="question", reader_policy="records_blocked")
        if baseline["trainable_parameters"] != plan["baseline_trainable_parameters"]:
            raise ValueError("Baseline shape/count differs from registration")
        write_json(args.output / "local.safety.json", baseline)
        manifest["completed_arms"].append("local")
        write_json(manifest_path, manifest)
        query = query_safety_check(runtime, encoded, plan)
        write_json(args.output / "gdn_query.safety.json", query)
        manifest["completed_arms"].append("gdn_query")
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                        backbone_frozen_no_gradients=all(not p.requires_grad and p.grad is None for p in runtime.model.parameters()))
        print(json.dumps({"status": "passed", "parameter_counts": {"local": baseline["trainable_parameters"], "gdn_query": query["trainable_parameters"]}}), flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
