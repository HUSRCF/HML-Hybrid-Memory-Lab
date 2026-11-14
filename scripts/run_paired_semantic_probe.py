"""Frozen full-history native validation on pre-registered semantic updates."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from sas_memory.tasks import INSTRUCTIONS
from sas_memory.training_data import data_fingerprint
from scripts.analyze_paired_semantic_probe import evaluated_members, identity, registered_dataset
from scripts.check_record_attention_weights import ObservedRecordWeights
from scripts.run_controller_trial import prompt_for, write_json
from scripts.run_oracle_kv_readout import measure


def hook_state(model):
    return {name: [list(module._forward_pre_hooks), list(module._forward_hooks)] for name, module in model.named_modules()
            if module._forward_pre_hooks or module._forward_hooks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/paired_semantic_native_probe.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if (plan["training"] or plan["model"] != "it" or plan["models"] != ["native"] or
        plan["reader_policy"] != "full" or plan["splits"] != ["validation"]):
        raise ValueError("Unregistered native semantic protocol")
    dataset = Path(plan["data_source"])
    registered, data, registry = registered_dataset(dataset)
    if (registered["data_sha256"] != plan["data_sha256"] or registered["registry_sha256"] != plan["registry_sha256"] or
        len(data["validation"]) != plan["examples"]["validation"]):
        raise ValueError("Semantic registration differs from plan")
    members = evaluated_members(data, args.smoke)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    sources = sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__).resolve(),
        ROOT / "scripts/analyze_paired_semantic_probe.py", ROOT / "scripts/prepare_paired_semantic_updates.py",
        ROOT / "scripts/check_record_attention_weights.py", ROOT / "scripts/run_controller_trial.py",
        ROOT / "scripts/run_oracle_kv_readout.py", ROOT / "scripts/analyze_blocked_reader_adaptation.py",
        ROOT / "scripts/analyze_controller_trial.py"]
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "training": False,
                "smoke": args.smoke, "reader_policy": "full", "plan_sha256": data_fingerprint(plan),
                "data_manifest_sha256": file_hash(dataset / "manifest.json"), "evaluated_keys": [identity(example) for example in members],
                "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in sources},
                "completed_splits": [], "safety_checks": []}
    for relative in manifest["source_sha256"]:
        path = args.output / "sources" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    manifest_path = args.output / "manifest.json"
    write_json(manifest_path, manifest)
    import torch
    from sas_memory.runtime import TextRuntime
    torch.set_num_threads(4)
    try:
        runtime = TextRuntime(registered["model_path"], "it")
        manifest.update(model_path=registered["model_path"], python=sys.executable, gpu=torch.cuda.get_device_name(0),
                        packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")})
        original_hooks = hook_state(runtime.model)
        indexed = {(row["case_id"], row["variant"], row["role"]): row for row in registry["validation"]}
        keys = sorted((identity(example) for example in members))
        sample_keys = {tuple(keys[0]), tuple(keys[-1])}
        result_path = args.output / "validation.native.jsonl"
        with result_path.open("w", buffering=1) as output:
            for index, example in enumerate(members):
                case, query = example["case"], example["query"]
                key = tuple(identity(example))
                row = indexed[key]
                ledger, prompt = prompt_for(runtime, case, example["variant"], query)
                if data_fingerprint(runtime.ids(prompt)) != row["prompt_ids_sha256"]:
                    raise ValueError("Actual native semantic input differs from registration")
                if key in sample_keys:
                    with ObservedRecordWeights(runtime.model, len(runtime.ids(runtime.prompt_start + INSTRUCTIONS)), row["ledger_tokens"]) as observer:
                        result = measure(runtime, ledger, query)
                    for stats in observer.stats.values():
                        if stats["prefill_calls"] != 1 or stats["cached_calls"] < 1 or stats["max_record_mass"] <= 0:
                            raise ValueError("Native full-history attention observation failed")
                    if measure(runtime, ledger, query) != result:
                        raise ValueError("Native semantic replay differs")
                    manifest["safety_checks"].append({"key": list(key), "native_replay_exact": True, "full_attention_weights": observer.stats})
                else:
                    result = measure(runtime, ledger, query)
                output.write(json.dumps({"case_id": case["id"], "variant": example["variant"], "role": query["role"],
                    **{field: case[field] for field in ("base_case_id", "value_version", "family", "seed", "distractor_updates")},
                    "reader_policy": "full", "registration": row, **result}) + "\n")
                if (index + 1) % 36 == 0:
                    print(json.dumps({"event": "evaluation", "completed_examples": index + 1, "total_examples": len(members)}), flush=True)
        manifest["safety_checks"].sort(key=lambda check: check["key"])
        manifest["frozen_parameters_and_no_gradients"] = all(not parameter.requires_grad and parameter.grad is None for parameter in runtime.model.parameters())
        manifest["hooks_restored"] = hook_state(runtime.model) == original_hooks
        if not manifest["frozen_parameters_and_no_gradients"] or not manifest["hooks_restored"]:
            raise ValueError("Native semantic freeze/hook restoration failed")
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                        completed_splits=["validation"], result_sha256=file_hash(result_path))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
