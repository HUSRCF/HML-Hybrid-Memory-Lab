"""Frozen native full/blocked-record readout on registered counterfactual families."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.counterbalance_data import file_digest
from sas_memory.kv_readout import QuestionKVReadout
from sas_memory.record_kv_ablation import record_blocked_mask
from sas_memory.training_data import data_fingerprint
from scripts.analyze_single_fact_probe import registered_dataset
from scripts.check_record_attention_weights import ObservedRecordWeights
from scripts.run_controller_trial import prompt_for, write_json
from scripts.run_oracle_kv_readout import measure, verify_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/single_fact_native_probe.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="First complete family in each split")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if (plan["training"] or plan["model"] != "it" or plan["models"] != ["native"] or
        plan["conditions"] != ["full_history", "records_blocked"] or plan["splits"] != ["train", "validation"]):
        raise ValueError("This probe is the registered frozen native two-condition trial")
    dataset = Path(plan["data_source"])
    registered, data, registry = registered_dataset(dataset)
    if registered["data_sha256"] != plan["data_sha256"] or registered["registry_sha256"] != plan["registry_sha256"]:
        raise ValueError("Dataset changed after probe registration")
    evaluated = {}
    for split in plan["splits"]:
        members = data[split]
        if len(members) != plan["examples"][split]:
            raise ValueError("Wrong registered split size")
        if args.smoke:
            family = min(example["case"]["base_case_id"] for example in members)
            members = [example for example in members if example["case"]["base_case_id"] == family]
        evaluated[split] = members
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    source_paths = sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__).resolve(),
        ROOT / "scripts/analyze_single_fact_probe.py", ROOT / "scripts/prepare_single_fact_control.py",
        ROOT / "scripts/analyze_blocked_reader_adaptation.py", ROOT / "scripts/analyze_controller_trial.py",
        ROOT / "scripts/check_record_attention_weights.py", ROOT / "scripts/run_controller_trial.py",
        ROOT / "scripts/run_oracle_kv_readout.py"]
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "training": False,
                "smoke": args.smoke, "plan_sha256": data_fingerprint(plan), "data_manifest_sha256": file_digest(dataset / "manifest.json"),
                "evaluated_case_ids": {split: [example["case"]["id"] for example in members] for split, members in evaluated.items()},
                "safety_checks": {}, "completed_splits": [],
                "source_sha256": {str(path.relative_to(ROOT)): file_digest(path) for path in source_paths}}
    for relative in manifest["source_sha256"]:
        destination = args.output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    manifest_path = args.output / "manifest.json"
    write_json(manifest_path, manifest)
    import torch
    from sas_memory.runtime import TextRuntime
    torch.set_num_threads(4)
    try:
        runtime = TextRuntime(registered["model_path"], "it")
        manifest.update(model_path=registered["model_path"], python=sys.executable, gpu=torch.cuda.get_device_name(0),
                        packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")})
        original_hooks = set(runtime.model._forward_pre_hooks)
        for split, members in evaluated.items():
            indexed = {row["case_id"]: row for row in registry[split]}
            sample_ids = [min(example["case"]["id"] for example in members), max(example["case"]["id"] for example in members)]
            manifest["safety_checks"][split] = []
            with (args.output / f"{split}.full_history.jsonl").open("w", buffering=1) as full_output, (args.output / f"{split}.records_blocked.jsonl").open("w", buffering=1) as blocked_output:
                for example in members:
                    case, query = example["case"], example["query"]
                    row = indexed[case["id"]]
                    ledger, prompt = prompt_for(runtime, case, example["variant"], query)
                    mask = record_blocked_mask(runtime, ledger, prompt)
                    if any(row[key] != value for key, value in mask.items()):
                        raise ValueError("Actual model input/reader mask changed")
                    keep = [bit == "1" for bit in mask["keep_bits"]]
                    sample = case["id"] in sample_ids
                    if sample:
                        with ObservedRecordWeights(runtime.model, mask["prelude_tokens"], mask["ledger_tokens"]) as full_weights:
                            full = measure(runtime, ledger, query)
                        checks = verify_mask(runtime, ledger, prompt, keep=keep, query=query, expected=full)
                        with QuestionKVReadout(runtime.model, keep) as reader:
                            with ObservedRecordWeights(runtime.model, mask["prelude_tokens"], mask["ledger_tokens"]) as blocked_weights:
                                blocked = measure(runtime, ledger, query)
                        for layer, stat in blocked_weights.stats.items():
                            if (stat["max_record_probability"] != 0 or stat["max_record_mass"] != 0 or
                                stat["cached_calls"] < 1 or full_weights.stats[layer]["max_record_mass"] <= 0):
                                raise ValueError("Actual probability check failed")
                        restored = measure(runtime, ledger, query)
                        if restored != full:
                            raise ValueError("Removing the mask did not restore native output")
                        manifest["safety_checks"][split].append({"case_id": case["id"], **checks, "native_replay_after_mask_exact": True,
                                                                 "full_weights": full_weights.stats, "blocked_weights": blocked_weights.stats})
                    else:
                        full = measure(runtime, ledger, query)
                        with QuestionKVReadout(runtime.model, keep) as reader:
                            blocked = measure(runtime, ledger, query)
                    if reader.calls["prefill"] != 1 or reader.calls["cached"] < 1:
                        raise ValueError("A scoring/answer forward missed the blocked mask")
                    for condition, result, output in (("full_history", full, full_output), ("records_blocked", blocked, blocked_output)):
                        record = {"case_id": case["id"], "base_case_id": case["base_case_id"], "value_version": case["value_version"],
                                  "variant": example["variant"], "role": query["role"], "family": case["family"], "seed": case["seed"],
                                  "condition": condition, "registration": row,
                                  "readout_calls": dict(reader.calls) if condition == "records_blocked" else None, **result}
                        output.write(json.dumps(record) + "\n")
            manifest["safety_checks"][split].sort(key=lambda check: check["case_id"])
            manifest["completed_splits"].append(split)
            write_json(manifest_path, manifest)
            print(json.dumps({"event": "split_complete", "split": split, "examples_per_condition": len(members)}), flush=True)
        manifest["frozen_parameters_and_no_gradients"] = all(not parameter.requires_grad and parameter.grad is None for parameter in runtime.model.parameters())
        manifest["hooks_restored"] = set(runtime.model._forward_pre_hooks) == original_hooks
        if not manifest["frozen_parameters_and_no_gradients"] or not manifest["hooks_restored"]:
            raise ValueError("Model freeze or hook restoration failed")
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
