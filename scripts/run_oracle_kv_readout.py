"""Restrict question-time KV access while retaining the complete original prefix."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.generalization_tasks import validate_generalization_cases
from sas_memory.kv_readout import QuestionKVReadout, evidence_token_keep
from sas_memory.oracle import evidence_slice
from sas_memory.readout_controls import readout_event_indices
from sas_memory.tasks import INSTRUCTIONS, Operation, execute
from sas_memory.training_data import data_fingerprint
from scripts.run_controller_trial import prompt_for, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection(runtime, case, name, query, kind="oracle"):
    events = [Operation(**event) for event in case["prefix"] + case["variants"][name]["tail"]]
    chosen = evidence_slice(events, query["entity"], query["version"])
    if str(execute(chosen)[query["version"]][query["entity"]]) != query["answer"]:
        raise AssertionError("Evidence slice changed executable truth")
    indices = readout_event_indices(events, query["entity"], query["version"], kind)
    ledger, _ = prompt_for(runtime, case, name, query)
    prompt = ledger + runtime.query_suffix(query["question"])
    keep = evidence_token_keep(runtime.tokenizer, prompt, ledger, runtime.prompt_start + INSTRUCTIONS,
                               [event.text() for event in events], indices)
    return ledger, prompt, keep, indices


def measure(runtime, ledger, query):
    result = runtime.probe(ledger, None, query["question"], query["answer"], [query["answer"]],
                           generate=True, prefill="full", score_batch=False)
    for field in ("candidate_prediction", "candidate_correct", "candidate_scores"):
        result.pop(field)
    return result


def verify_mask(runtime, ledger, prompt, keep, query, expected):
    import torch
    from sas_memory.runtime import assert_cache_equal
    baseline_cache, baseline_logits, _ = runtime.advance(prompt)
    prefix_cache, prefix_logits, _ = runtime.advance(ledger)
    replay = measure(runtime, ledger, query)
    if replay["generated"] != expected["generated"] or abs(replay["answer_nll"] - expected["answer_nll"]) > 1e-5:
        raise AssertionError("Unrestricted checkpoint replay changed")
    with QuestionKVReadout(runtime.model, [True] * len(keep)) as all_history:
        actual_cache, actual_logits, _ = runtime.advance(prompt)
        assert_cache_equal(baseline_cache, actual_cache)
        if not torch.equal(baseline_logits, actual_logits):
            raise AssertionError("All-history mask changed full-prefill logits")
        actual = measure(runtime, ledger, query)
        if actual != replay:
            raise AssertionError("All-history mask changed scoring or greedy decoding")
    with QuestionKVReadout(runtime.model, keep):
        prefix_actual, logits_actual, _ = runtime.advance(ledger)
        assert_cache_equal(prefix_cache, prefix_actual)
        if not torch.equal(prefix_logits, logits_actual):
            raise AssertionError("Question-only mask changed prefix logits")
        restricted_cache, _, _ = runtime.advance(prompt)
        fields = 0
        for left, right in zip(baseline_cache.layers, restricted_cache.layers):
            for field in ("keys", "values"):
                before, after = getattr(left, field, None), getattr(right, field, None)
                if isinstance(before, torch.Tensor):
                    if not torch.equal(before[..., :len(keep), :], after[..., :len(keep), :]):
                        raise AssertionError("Restricted full prefill changed historical KV")
                    fields += 1
        if fields != 2 * runtime.model.config.layer_types.count("full_attention"):
            raise AssertionError("Missing historical KV fields")
    if not all_history.calls["prefill"] or not all_history.calls["cached"]:
        raise AssertionError("Mask was not exercised in both full and cached forwards")
    return {"all_history_full_cache_and_logits_exact": True, "all_history_scoring_and_greedy_exact": True,
            "pre_question_kv_recurrent_conv_and_logits_exact": True, "historical_kv_exact_in_full_prefill": True,
            "historical_kv_fields_checked": fields, "unrestricted_replay_max_nll_error": abs(replay["answer_nll"] - expected["answer_nll"]),
            "all_history_hook_calls": all_history.calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/oracle_kv_readout.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="One case per checkpoint, separate from the full cohort")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    reference = Path(config["reference_run"])
    original = json.loads((reference / "manifest.json").read_text())
    if original["status"] != "completed" or not set(config["modes"]) <= set(original["completed_oracle_modes"]):
        raise ValueError("Incomplete reference run")
    cases = json.loads((reference / "cases.json").read_text())
    validation = validate_generalization_cases(cases)
    if data_fingerprint(cases) != config["expected_cases_sha256"]:
        raise ValueError("Locked cohort changed")
    if args.smoke:
        cases = cases[:1]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "cases.json", cases)
    sources = sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__).resolve(), ROOT / "scripts/run_controller_trial.py"]
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "smoke": args.smoke, "training": False, "cases_sha256": data_fingerprint(cases), "task_validation": validation,
                "reference_manifest_sha256": digest(reference / "manifest.json"), "reference_results_sha256": {},
                "checkpoint_sha256": {}, "completed_modes": [], "safety_checks": {},
                "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sources}}
    for relative in manifest["source_sha256"]:
        path = args.output / "sources" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    manifest_path = args.output / "manifest.json"
    write_json(manifest_path, manifest)
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    from sas_memory.runtime import TextRuntime
    torch.set_num_threads(4)
    protocol = original["training_config"]
    model_path = json.loads((ROOT / "configs/models.json").read_text())[protocol["model"]]
    try:
        runtime = TextRuntime(model_path, protocol["model"])
        manifest.update(model_path=model_path, python=sys.executable, gpu=torch.cuda.get_device_name(0),
                        packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")})
        for label in config["modes"]:
            controller, policy = None, "all"
            if label != "native":
                spec = original["checkpoint_sources"][label]
                protocol = spec["config"]
                checkpoint = Path(spec["training_run"]) / spec["mode"] / "controller.safetensors"
                if digest(checkpoint) != original["checkpoint_sha256"][label]:
                    raise ValueError("Controller checkpoint changed")
                manifest["checkpoint_sha256"][label] = digest(checkpoint)
                controller = GroupController(runtime.model, protocol["group_index"], protocol["modes"][spec["mode"]],
                                             spec["mode"], adapt_gdn_keys=protocol.get("adapt_gdn_keys", False))
                controller.load_state_dict(load_file(str(checkpoint), device=str(runtime.model.device)), strict=True)
                policy = protocol.get("token_policy", "all")
            raw_path = reference / f"{label}.jsonl"
            manifest["reference_results_sha256"][label] = {"raw": digest(raw_path), "oracle": digest(reference / f"{label}.oracle.jsonl")}
            raw = {(r["case_id"], r["variant"], r["role"]): r for r in (json.loads(line) for line in raw_path.read_text().splitlines())}
            try:
                count = 0
                with (args.output / f"{label}.jsonl").open("w", buffering=1) as output:
                    for case in cases:
                        for name in config["variants"]:
                            query = next(q for q in case["variants"][name]["queries"] if q["role"] == "target_current")
                            ledger, prompt, keep, indices = selection(runtime, case, name, query, config.get("selection_kind", "oracle"))
                            if controller is not None:
                                controller.set_token_policy(policy, len(keep))
                            if count == 0:
                                manifest["safety_checks"][label] = {"case_id": case["id"], "variant": name,
                                    **verify_mask(runtime, ledger, prompt, keep, query, raw[(case["id"], name, query["role"])])}
                                write_json(manifest_path, manifest)
                            with QuestionKVReadout(runtime.model, keep) as intervention:
                                result = measure(runtime, ledger, query)
                            if not intervention.calls["cached"]:
                                raise AssertionError("No cached answer forward was masked")
                            row = {"case_id": case["id"], "variant": name, "role": query["role"],
                                   "selected_event_indices": indices, "ledger_tokens": len(keep), "kept_history_tokens": sum(keep),
                                   "history_keep_sha256": hashlib.sha256(bytes(keep)).hexdigest(),
                                   "prompt_ids_sha256": hashlib.sha256(json.dumps(runtime.ids(prompt)).encode()).hexdigest(),
                                   "hook_calls": intervention.calls, **result}
                            output.write(json.dumps(row) + "\n")
                            count += 1
                manifest["completed_modes"].append(label)
                write_json(manifest_path, manifest)
                print(json.dumps({"event": "oracle_kv_complete", "mode": label, "rows": count}), flush=True)
            finally:
                if controller is not None:
                    controller.close()
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
