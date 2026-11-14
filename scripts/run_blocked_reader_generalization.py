"""Evaluate final matched blocked-reader adapters on the fixed 24-fact cohort."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.counterbalance_data import encode_counterbalance, file_digest
from sas_memory.counterbalance_tasks import ORDERS, validate_counterbalance_cases
from sas_memory.kv_readout import QuestionKVReadout
from sas_memory.record_kv_ablation import record_blocked_mask
from sas_memory.training_data import data_fingerprint
from scripts.analyze_blocked_reader_adaptation import audit_run
from scripts.check_record_attention_weights import ObservedRecordWeights
from scripts.run_controller_confirmation import verify_native_prefix
from scripts.run_controller_trial import write_json
from scripts.run_oracle_kv_readout import measure, verify_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/blocked_reader_generalization.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.config.read_text())
    training_plan = json.loads((ROOT / plan["training_plan"]).read_text())
    dataset = Path(plan["registered_dataset"])
    cases = json.loads((dataset / "cases.json").read_text())
    registry = json.loads((dataset / "registry.json").read_text())
    if (data_fingerprint(cases) != plan["cases_sha256"] or data_fingerprint(registry) != plan["registry_sha256"] or
        validate_counterbalance_cases(cases)["contexts"] != plan["contexts_per_model"]):
        raise ValueError("Registered cohort/masks changed")
    if plan["reader_policy"] != "records_blocked" or plan["training"]:
        raise ValueError("This evaluation is frozen and records-blocked only")
    sources = {label: audit_run(Path(path), training_plan)[0] for label, path in plan["checkpoints"].items()}
    if sources["question_only"]["token_policy"] != "question" or sources["all_tokens"]["token_policy"] != "all":
        raise ValueError("Training arms are mislabeled")
    for key in ("initial_state_sha256", "data_sha256", "tokenized_input_answer_sha256", "reader_registry_sha256", "parameters", "steps"):
        if len({spec[key] for spec in sources.values()}) != 1:
            raise ValueError(f"Unmatched training {key}")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("plan", plan), ("training_plan", training_plan), ("cases", cases), ("registry", registry)):
        write_json(args.output / f"{name}.json", value)
    source_paths = sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__),
        ROOT / "scripts/analyze_blocked_reader_adaptation.py", ROOT / "scripts/run_controller_trial.py",
        ROOT / "scripts/check_record_attention_weights.py", ROOT / "scripts/run_controller_confirmation.py",
        ROOT / "scripts/run_oracle_kv_readout.py", ROOT / "scripts/analyze_controller_trial.py"]
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "training": False,
                "plan_sha256": data_fingerprint(plan), "cases_sha256": plan["cases_sha256"],
                "registry_sha256": plan["registry_sha256"], "checkpoint_sources": sources,
                "completed_models": [], "safety_checks": {}, "native_prefix_checks": {},
                "native_replay_after_detach": {}, "source_sha256": {str(path.relative_to(ROOT)): file_digest(path) for path in source_paths}}
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
    model_path = json.loads((ROOT / "configs/models.json").read_text())[training_plan["backbone"]]
    indexed = {(row["case_id"], row["variant"]): row for row in registry}
    if len(indexed) != plan["contexts_per_model"]:
        raise ValueError("Duplicate/missing mask registry entries")
    try:
        runtime = TextRuntime(model_path, training_plan["backbone"])
        manifest["model_path"] = model_path
        first_ledger, _, _ = encode_counterbalance(runtime, cases[0], ORDERS[0])
        first_query = cases[0]["variants"][ORDERS[0]]["queries"][0]
        first_mask = indexed[(cases[0]["id"], ORDERS[0])]
        with QuestionKVReadout(runtime.model, [bit == "1" for bit in first_mask["keep_bits"]]):
            native = measure(runtime, first_ledger, first_query)
        for label, source in sources.items():
            training = Path(source["path"])
            config = json.loads((training / "manifest.json").read_text())["config"]
            if file_digest(training / "manifest.json") != source["manifest_sha256"]:
                raise ValueError("Training manifest changed after audit")
            checkpoint = training / training_plan["mode"] / "controller.safetensors"
            if file_digest(checkpoint) != source["checkpoint_sha256"]:
                raise ValueError("Final checkpoint changed after audit")
            controller = GroupController(runtime.model, config["group_index"], config["modes"][training_plan["mode"]],
                                         training_plan["mode"], adapt_gdn_keys=config["adapt_gdn_keys"])
            weights = load_file(str(checkpoint), device=str(runtime.model.device))
            controller.load_state_dict(weights, strict=True)
            manifest["safety_checks"][label], manifest["native_prefix_checks"][label] = [], []
            try:
                count = 0
                with (args.output / f"{label}.jsonl").open("w", buffering=1) as output:
                    for index, case in enumerate(cases):
                        for order in ORDERS:
                            ledger, prompt, _ = encode_counterbalance(runtime, case, order)
                            query = case["variants"][order]["queries"][0]
                            registered = indexed[(case["id"], order)]
                            mask = record_blocked_mask(runtime, ledger, prompt)
                            if any(registered[key] != value for key, value in mask.items()):
                                raise ValueError("Evaluation changed original inputs/masks")
                            controller.set_token_policy(config["token_policy"], mask["ledger_tokens"])
                            keep = [bit == "1" for bit in mask["keep_bits"]]
                            sample = index in (0, len(cases) - 1) and order == ORDERS[0]
                            if sample:
                                if config["token_policy"] == "question":
                                    manifest["native_prefix_checks"][label].append({"case_id": case["id"], **verify_native_prefix(runtime, controller, ledger)})
                                args_weights = (runtime.model, mask["prelude_tokens"], mask["ledger_tokens"])
                                with ObservedRecordWeights(*args_weights) as control:
                                    unrestricted = measure(runtime, ledger, query)
                                checks = verify_mask(runtime, ledger, prompt, keep=keep, query=query, expected=unrestricted)
                                with QuestionKVReadout(runtime.model, keep) as reader:
                                    with ObservedRecordWeights(*args_weights) as observed:
                                        actual = measure(runtime, ledger, query)
                                for layer, stat in observed.stats.items():
                                    if stat["max_record_probability"] != 0 or stat["max_record_mass"] != 0 or stat["cached_calls"] < 1:
                                        raise ValueError("Record access was not blocked during cached answers")
                                    if control.stats[layer]["max_record_mass"] <= 0:
                                        raise ValueError("Probability control vacuous")
                                manifest["safety_checks"][label].append({"case_id": case["id"], **checks,
                                    "blocked_probabilities": observed.stats, "unmasked_probabilities": control.stats})
                            else:
                                with QuestionKVReadout(runtime.model, keep) as reader:
                                    actual = measure(runtime, ledger, query)
                            if reader.calls["prefill"] != 1 or reader.calls["cached"] < 1:
                                raise ValueError("A scoring/generation forward missed the blocked reader")
                            row = {"case_id": case["id"], "variant": order, "role": "target_current", "family": "copy",
                                   "base_case_id": case["base_case_id"], "dependency_depth": case["dependency_depth"],
                                   "question_style": case["question_style"], "reader_mask": mask,
                                   "readout_calls": dict(reader.calls), **actual}
                            output.write(json.dumps(row) + "\n")
                            count += 1
                if any(not torch.equal(value, weights[key]) for key, value in controller.state_dict().items()):
                    raise ValueError("Frozen checkpoint changed during evaluation")
                if count != plan["contexts_per_model"]:
                    raise ValueError("Incomplete frozen cohort")
            finally:
                controller.close()
            with QuestionKVReadout(runtime.model, [bit == "1" for bit in first_mask["keep_bits"]]):
                restored = measure(runtime, first_ledger, first_query)
            if restored != native:
                raise ValueError("Detaching adapter did not restore native blocked output")
            manifest["native_replay_after_detach"][label] = True
            manifest["completed_models"].append(label)
            write_json(manifest_path, manifest)
            print(json.dumps({"event": "model_complete", "model": label, "rows": count}), flush=True)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
