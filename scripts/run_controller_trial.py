"""Train matched-budget causal group controllers on a frozen local 2B model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prompt_for(runtime, case, variant, query):
    from sas_memory.tasks import INSTRUCTIONS, render
    ledger = runtime.prompt_start + INSTRUCTIONS + render(case["prefix"] + case["variants"][variant]["tail"])
    return ledger, ledger + runtime.query_suffix(query["question"])


def encode_example(runtime, example, reader_policy="full"):
    ledger, prompt = prompt_for(runtime, example["case"], example["variant"], example["query"])
    answer_ids = runtime.continuation_ids(prompt, example["query"]["answer"] + "\n")
    prompt_ids = runtime.ids(prompt)
    ledger_ids = runtime.ids(ledger)
    if prompt_ids[:len(ledger_ids)] != ledger_ids:
        raise ValueError("Tokenization crosses the ledger/question mask boundary")
    encoded = {"input_ids": prompt_ids + answer_ids[:-1], "answer_ids": answer_ids,
            "ledger_tokens": len(ledger_ids),
            "case_id": example["case"]["id"], "family": example["case"]["family"],
            "variant": example["variant"], "role": example["query"]["role"]}
    if reader_policy == "records_blocked":
        from sas_memory.record_kv_ablation import record_blocked_mask
        encoded["reader_mask"] = record_blocked_mask(runtime, ledger, prompt)
    elif reader_policy != "full":
        raise ValueError(f"Unknown reader policy: {reader_policy}")
    return encoded


def reader_context(runtime, reader_policy, mask=None):
    if reader_policy == "full":
        return nullcontext(None)
    if reader_policy != "records_blocked" or mask is None:
        raise ValueError("Supported reader policy and registered mask required")
    from sas_memory.kv_readout import QuestionKVReadout
    prelude, boundary = mask["prelude_tokens"], mask["ledger_tokens"]
    expected = "1" * prelude + "0" * (boundary - prelude)
    if not 0 < prelude < boundary or mask["keep_bits"] != expected:
        raise ValueError("Reader mask does not block every historical record")
    if hashlib.sha256(bytes(bit == "1" for bit in expected)).hexdigest() != mask["keep_sha256"]:
        raise ValueError("Reader mask fingerprint changed")
    return QuestionKVReadout(runtime.model, [bit == "1" for bit in expected])


def supervised_loss(runtime, encoded, controller=None, token_policy="all", reader_policy="full", reader_calls=None):
    import torch.nn.functional as F
    inputs = runtime.tensor([encoded["input_ids"]])
    labels = runtime.tensor([encoded["answer_ids"]])
    if controller is not None:
        controller.set_token_policy(token_policy, encoded["ledger_tokens"])
    with reader_context(runtime, reader_policy, encoded.get("reader_mask")) as reader:
        output = runtime.model(input_ids=inputs, use_cache=False, logits_to_keep=labels.shape[1])
    if reader is not None:
        if reader.calls != {"prefill": 1, "cached": 0}:
            raise AssertionError("Supervised forward did not use the registered reader")
        if reader_calls is not None:
            reader_calls.append(dict(reader.calls))
    loss = F.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]), labels.reshape(-1))
    return loss, output.logits


def verify_blocked_reader(runtime, encoded):
    """Observe actual probabilities in teacher forcing and cached answer decoding."""
    import torch
    from sas_memory.runtime import fork_cache
    from scripts.check_record_attention_weights import ObservedRecordWeights
    mask = encoded["reader_mask"]
    args = (runtime.model, mask["prelude_tokens"], mask["ledger_tokens"])
    with torch.inference_mode():
        with ObservedRecordWeights(*args) as control:
            supervised_loss(runtime, encoded)
        with ObservedRecordWeights(*args) as teacher:
            supervised_loss(runtime, encoded, reader_policy="records_blocked")
        prompt_length = len(encoded["input_ids"]) - len(encoded["answer_ids"]) + 1
        with reader_context(runtime, "records_blocked", mask) as reader:
            with ObservedRecordWeights(*args) as cached:
                first = runtime.model(input_ids=runtime.tensor([encoded["input_ids"][:prompt_length]]),
                                      use_cache=True, logits_to_keep=1)
                runtime.model(input_ids=runtime.tensor([[encoded["answer_ids"][0]]]),
                              past_key_values=fork_cache(first.past_key_values), use_cache=True, logits_to_keep=1)
                runtime.greedy_answer(first.past_key_values, first.logits[:, -1].float())
    for index in control.stats:
        if control.stats[index]["max_record_mass"] <= 0:
            raise AssertionError("Unmasked probability control is vacuous")
        for stat in (teacher.stats[index], cached.stats[index]):
            if stat["max_record_probability"] != 0 or stat["max_record_mass"] != 0 or stat["prefill_calls"] != 1:
                raise AssertionError("Blocked record received attention probability")
        if cached.stats[index]["cached_calls"] < 1:
            raise AssertionError("Cached answer forwards were not inspected")
    return {"unmasked_control": control.stats, "teacher_forced": teacher.stats,
            "cached_answer": cached.stats, "reader_calls": reader.calls}


def safety_check(runtime, encoded, mode, rank, group_index, seed, adapt_gdn_keys=False, token_policy="all", reader_policy="full"):
    import torch
    from sas_memory.controllers import GroupController
    torch.manual_seed(seed)
    with torch.inference_mode():
        baseline_loss, baseline_logits = supervised_loss(runtime, encoded, reader_policy=reader_policy)
        baseline_logits = baseline_logits.clone()
        ledger_reference = None
        if token_policy == "question":
            ledger_input = runtime.tensor([encoded["input_ids"][:encoded["ledger_tokens"]]])
            ledger_reference = runtime.model(input_ids=ledger_input, use_cache=True, logits_to_keep=1)
    controller = GroupController(runtime.model, group_index, rank, mode, adapt_gdn_keys=adapt_gdn_keys)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=0.001, weight_decay=0)
    description = controller.description()
    try:
        with torch.inference_mode():
            initial_loss, initial_logits = supervised_loss(runtime, encoded, controller, token_policy, reader_policy)
            identity_error = (initial_logits.float() - baseline_logits.float()).abs().max().item()
        if identity_error != 0:
            raise AssertionError(f"Zero initialization changed native logits: {identity_error}")
        loss, _ = supervised_loss(runtime, encoded, controller, token_policy, reader_policy)
        loss.backward()
        output_gradients = {name: module.weight.grad.float().norm().item() if module.weight.grad is not None else None
                            for name, module in controller.outputs.items()}
        if any(value is None or not (0 < value < float("inf")) for value in output_gradients.values()):
            raise AssertionError(f"Missing, zero or nonfinite output gradients: {output_gradients}")
        if any(p.grad is not None or p.requires_grad for p in runtime.model.parameters()):
            raise AssertionError("Backbone unexpectedly received gradients")
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        next_loss, _ = supervised_loss(runtime, encoded, controller, token_policy, reader_policy)
        next_loss.backward()
        encoder_gradients = {name: module.weight.grad.float().norm().item() if module.weight.grad is not None else None
                             for name, module in controller.encoders.items()}
        if any(value is None or not (0 < value < float("inf")) for value in encoder_gradients.values()):
            raise AssertionError(f"Encoder did not become trainable after first update: {encoder_gradients}")
        ledger_cache_equal = None
        if ledger_reference is not None:
            from sas_memory.runtime import assert_cache_equal
            # After a nonzero optimizer update, question-only adaptation must
            # still leave every pre-question KV/recurrent/convolution tensor exact.
            with torch.inference_mode():
                ledger_actual = runtime.model(input_ids=ledger_input, use_cache=True, logits_to_keep=1)
            assert_cache_equal(ledger_reference.past_key_values, ledger_actual.past_key_values)
            if not torch.equal(ledger_reference.logits, ledger_actual.logits):
                raise AssertionError("Question-only training changed ledger logits")
            ledger_cache_equal = True
        probability_checks = verify_blocked_reader(runtime, encoded) if reader_policy == "records_blocked" else None
        return {"status": "passed", **description, "initial_logit_max_error": identity_error,
                "native_loss": baseline_loss.item(), "initial_loss": initial_loss.item(),
                "loss_after_one_step": next_loss.item(), "output_gradient_norms": output_gradients,
                "encoder_gradient_norms_after_one_step": encoder_gradients,
                "backbone_gradients_absent": True, "token_policy": token_policy,
                "pre_question_cache_and_logits_exact_after_update": ledger_cache_equal,
                "reader_policy": reader_policy, "reader_probability_checks_after_update": probability_checks}
    finally:
        controller.close()
        with torch.inference_mode():
            _, restored_logits = supervised_loss(runtime, encoded, reader_policy=reader_policy)
            if not torch.equal(restored_logits, baseline_logits):
                raise AssertionError("Detaching controller did not restore native logits")
        del controller, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def evaluate(runtime, cases, path, controller=None, token_policy="all", reader_policy="full"):
    rows = []
    with path.open("w", buffering=1) as output:
        for case in cases:
            for variant_name, variant in case["variants"].items():
                for query in variant["queries"]:
                    ledger, prompt = prompt_for(runtime, case, variant_name, query)
                    mask = None
                    if reader_policy == "records_blocked":
                        from sas_memory.record_kv_ablation import record_blocked_mask
                        mask = record_blocked_mask(runtime, ledger, prompt)
                    if controller is not None:
                        controller.set_token_policy(token_policy, len(runtime.ids(ledger)))
                    # Score only the true answer for NLL. No restricted-candidate
                    # correctness is reported: all accuracy is unrestricted greedy.
                    with reader_context(runtime, reader_policy, mask) as reader:
                        result = runtime.probe(ledger, None, query["question"], query["answer"], [query["answer"]],
                                               generate=True, prefill="full", score_batch=False)
                    if reader is not None:
                        if reader.calls["prefill"] != 1 or reader.calls["cached"] < 1:
                            raise AssertionError("Evaluation must mask prefill and cached answer calls")
                        result.update(reader_policy=reader_policy, reader_mask=mask, readout_calls=dict(reader.calls))
                    for field in ("candidate_prediction", "candidate_correct", "candidate_scores"):
                        result.pop(field)
                    row = {"case_id": case["id"], "family": case["family"], "seed": case["seed"],
                           "variant": variant_name, "role": query["role"], "distractor_updates": case["distractor_updates"], **result}
                    rows.append(row)
                    output.write(json.dumps(row) + "\n")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["variant"], row["role"])].append(row)
    summary = [{"family": family, "variant": variant, "role": role, "count": len(members),
                "correct": sum(r["greedy_exact"] for r in members), "accuracy": mean(r["greedy_exact"] for r in members),
                "answer_nll": mean(r["answer_nll"] for r in members)}
               for (family, variant, role), members in sorted(groups.items())]
    write_json(path.with_suffix(".summary.json"), summary)
    return {"rows": len(rows), "grouped": summary}


def validate_adaptation_plan(config, data, plan):
    from sas_memory.training_data import data_fingerprint
    expected = {"model": plan["backbone"], "group_index": plan["group_index"],
                "initialization_seed": plan["initialization_seed"], "steps": plan["steps"],
                "gradient_accumulation": plan["gradient_accumulation"], "reader_policy": plan["reader_policy"],
                "modes": {plan["mode"]: plan["rank"]}}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Configuration differs from the registered blocked-reader plan")
    if config.get("token_policy") not in {arm["token_policy"] for arm in plan["arms"].values()}:
        raise ValueError("Unregistered adaptation arm")
    source = Path(plan["data_source"])
    reference = json.loads((source / "manifest.json").read_text())
    old_data = json.loads((source / "data.json").read_text())
    if reference["status"] != "completed" or reference["data_sha256"] != plan["data_sha256"]:
        raise ValueError("Registered source training run is incomplete or changed")
    if data != old_data or data_fingerprint(data) != plan["data_sha256"]:
        raise ValueError("Training or held-out examples differ from the registered original data")
    for key in ("learning_rate", "weight_decay", "max_gradient_norm", "max_sequence_tokens", "adapt_gdn_keys"):
        if config.get(key) != reference["config"].get(key):
            raise ValueError(f"Optimizer/adapter protocol differs: {key}")
    for relative in ("sas_memory/runtime.py", "sas_memory/controllers.py"):
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != reference["source_sha256"][relative]:
            raise ValueError("Runtime/controller differs from the registered source")


def state_fingerprint(controller):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(controller.state_dict().items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/controller_training_pilot.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--modes", nargs="+", help="Subset of configured modes; all modes use the same data")
    parser.add_argument("--adaptation-plan", type=Path, help="Registered protocol required for blocked-reader adaptation")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    reader_policy = config.get("reader_policy", "full")
    if reader_policy not in ("full", "records_blocked"):
        raise ValueError(f"Unsupported reader policy: {reader_policy}")
    plan = json.loads(args.adaptation_plan.read_text()) if args.adaptation_plan else None
    if (reader_policy == "records_blocked") != (plan is not None):
        raise ValueError("Blocked-reader adaptation requires its registered --adaptation-plan")
    token_policy = config.get("token_policy", "all")
    if token_policy not in ("all", "ledger", "question"):
        raise ValueError(f"Invalid training token policy: {token_policy}")
    modes = args.modes or list(config["modes"])
    if any(mode not in config["modes"] for mode in modes):
        raise ValueError(modes)
    from sas_memory.training_data import build_training_examples, build_eval_cases, data_fingerprint
    data = {
        "train": build_training_examples(config["steps"] * config["gradient_accumulation"]),
        "validation": build_eval_cases("validation", config["validation_copy_cases"], config["validation_overwrite_cases"]),
        "test": build_eval_cases("test", config["test_copy_cases"], config["test_overwrite_cases"]),
    }
    if plan is not None:
        validate_adaptation_plan(config, data, plan)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "data.json", data)
    manifest = {"status": "running", "stage": "smoke" if args.smoke else "training",
                "started_utc": datetime.now(timezone.utc).isoformat(), "config": config, "modes": modes,
                "data_sha256": data_fingerprint(data), "completed_modes": [],
                "reader_policy": reader_policy,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__).resolve(), ROOT / "scripts/check_record_attention_weights.py"]}}
    if plan is not None:
        write_json(args.output / "adaptation_plan.json", plan)
        manifest["adaptation_plan_sha256"] = data_fingerprint(plan)
    manifest_path = args.output / "manifest.json"
    for relative in manifest["source_sha256"]:
        snapshot = args.output / "sources" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / relative).read_bytes())
    write_json(manifest_path, manifest)
    import torch
    from safetensors.torch import save_file
    from sas_memory.runtime import TextRuntime
    from sas_memory.controllers import GroupController
    torch.set_num_threads(4)
    torch.manual_seed(config["initialization_seed"])
    started = time.monotonic()
    try:
        model_path = json.loads((ROOT / "configs/models.json").read_text())[config["model"]]
        runtime = TextRuntime(model_path, config["model"])
        manifest.update(model_path=model_path, gpu=torch.cuda.get_device_name(0), python=sys.executable,
                        packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")})
        encoded = [encode_example(runtime, example, reader_policy) for example in data["train"]]
        if any(len(item["input_ids"]) > config["max_sequence_tokens"] for item in encoded):
            raise ValueError("A training example exceeds the configured length; no truncation is permitted")
        write_json(args.output / "tokenized_train.json", encoded)
        manifest["optimized_input_tokens_per_mode"] = sum(len(item["input_ids"]) for item in encoded)
        manifest["supervised_answer_tokens_per_mode"] = sum(len(item["answer_ids"]) for item in encoded)
        manifest["max_input_tokens"] = max(len(item["input_ids"]) for item in encoded)
        manifest["tokenized_input_answer_sha256"] = data_fingerprint([
            {key: item[key] for key in ("input_ids", "answer_ids")} for item in encoded])
        if plan is not None:
            if (len(encoded) != plan["training_examples"] or
                manifest["supervised_answer_tokens_per_mode"] != plan["supervised_answer_tokens"] or
                manifest["tokenized_input_answer_sha256"] != plan["tokenized_input_answer_sha256"]):
                raise ValueError("Training token budget differs from the registered original inputs/answers")
            from sas_memory.record_kv_ablation import record_blocked_mask
            registry = {"train": [{key: item[key] for key in ("case_id", "variant", "role", "reader_mask")} for item in encoded]}
            for split in ("validation", "test"):
                registry[split] = []
                for case in data[split]:
                    for variant, content in case["variants"].items():
                        for query in content["queries"]:
                            ledger, prompt = prompt_for(runtime, case, variant, query)
                            registry[split].append({"case_id": case["id"], "variant": variant, "role": query["role"],
                                                    "reader_mask": record_blocked_mask(runtime, ledger, prompt)})
            write_json(args.output / "reader_registry.json", registry)
            manifest["reader_registry_sha256"] = data_fingerprint(registry)
            manifest["reader_registry_fixed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        checks = {}
        for mode in modes:
            print(json.dumps({"event": "safety_check", "mode": mode}), flush=True)
            checks[mode] = safety_check(runtime, encoded[0], mode, config["modes"][mode], config["group_index"],
                                        config["initialization_seed"], config.get("adapt_gdn_keys", False), token_policy, reader_policy)
            if plan is not None and checks[mode]["trainable_parameters"] != plan["expected_trainable_parameters"]:
                raise ValueError("Unexpected trainable parameter budget")
            write_json(args.output / "safety_checks.json", checks)
            print(json.dumps({"event": "safety_passed", "mode": mode, "parameters": checks[mode]["trainable_parameters"],
                              "loss_before": checks[mode]["native_loss"], "loss_after_step": checks[mode]["loss_after_one_step"]}), flush=True)
        if not args.smoke:
            print(json.dumps({"event": "native_evaluation"}), flush=True)
            native = {split: evaluate(runtime, data[split], args.output / f"native_{split}.jsonl", reader_policy=reader_policy) for split in ("validation", "test")}
            write_json(args.output / "native_summary.json", native)
            for mode in modes:
                mode_path = args.output / mode
                mode_path.mkdir()
                torch.manual_seed(config["initialization_seed"])
                controller = GroupController(runtime.model, config["group_index"], config["modes"][mode], mode,
                                             adapt_gdn_keys=config.get("adapt_gdn_keys", False))
                write_json(mode_path / "controller.json", {**controller.description(), "initial_state_sha256": state_fingerprint(controller)})
                optimizer = torch.optim.AdamW(controller.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
                torch.cuda.reset_peak_memory_stats()
                mode_started = time.monotonic()
                try:
                    with (mode_path / "training.jsonl").open("w", buffering=1) as log:
                        for step in range(config["steps"]):
                            optimizer.zero_grad(set_to_none=True)
                            losses = []
                            reader_calls = []
                            for micro in range(config["gradient_accumulation"]):
                                item = encoded[step * config["gradient_accumulation"] + micro]
                                loss, _ = supervised_loss(runtime, item, controller, token_policy, reader_policy, reader_calls)
                                if not torch.isfinite(loss):
                                    raise RuntimeError("Nonfinite training loss")
                                (loss / config["gradient_accumulation"]).backward()
                                losses.append(loss.item())
                            grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), config["max_gradient_norm"], error_if_nonfinite=True)
                            optimizer.step()
                            if any(p.grad is not None or p.requires_grad for p in runtime.model.parameters()):
                                raise AssertionError("Backbone changed gradient status")
                            row = {"step": step + 1, "loss": mean(losses), "gradient_norm": grad_norm.item(),
                                   "elapsed_seconds": time.monotonic() - mode_started}
                            if reader_policy == "records_blocked":
                                if len(reader_calls) != config["gradient_accumulation"]:
                                    raise AssertionError("A training microbatch missed the blocked reader")
                                row.update(reader_policy=reader_policy, reader_calls=reader_calls)
                            log.write(json.dumps(row) + "\n")
                            if step == 0 or (step + 1) % 8 == 0:
                                print(json.dumps({"event": "training", "mode": mode, **row}), flush=True)
                    save_file({name: value.detach().cpu().contiguous() for name, value in controller.state_dict().items()},
                              str(mode_path / "controller.safetensors"))
                    training_seconds = time.monotonic() - mode_started
                    result = {split: evaluate(runtime, data[split], mode_path / f"{split}.jsonl", controller, token_policy, reader_policy)
                              for split in ("validation", "test")}
                    result.update(training_seconds=training_seconds, mode_elapsed_seconds=time.monotonic() - mode_started,
                                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
                    write_json(mode_path / "summary.json", result)
                    manifest["completed_modes"].append(mode)
                    write_json(manifest_path, manifest)
                    print(json.dumps({"event": "mode_complete", "mode": mode}), flush=True)
                finally:
                    controller.close()
                    del controller, optimizer
                    gc.collect()
                    torch.cuda.empty_cache()
        manifest.update(status="completed", elapsed_seconds=time.monotonic() - started,
                        completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
