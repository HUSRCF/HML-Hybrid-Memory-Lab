"""Disable learned branches without retraining; never change checkpoint weights."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_controller_trial import evaluate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mode", default="shared")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporal", action="store_true", help="Mask full-prefill token positions instead of splitting caches")
    parser.add_argument("--gdn-components", action="store_true", help="Separate learned GDN Key and decay/write-gate residuals")
    args = parser.parse_args()
    trained = json.loads((args.run / "manifest.json").read_text())
    config = trained["config"]
    if args.temporal and args.gdn_components:
        raise ValueError("Choose temporal or GDN-component ablation")
    if args.gdn_components and not config.get("adapt_gdn_keys", False):
        raise ValueError("GDN-component ablation requires trained GDN Key residuals")
    if args.temporal and config.get("token_policy", "all") != "all":
        raise ValueError("Temporal disablement expects a checkpoint trained on all token positions")
    if args.mode not in trained["completed_modes"]:
        raise ValueError("Requested controller did not finish training and evaluation")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    from sas_memory.runtime import TextRuntime
    torch.set_num_threads(4)
    model_path = json.loads((ROOT / "configs/models.json").read_text())[config["model"]]
    checkpoint = args.run / args.mode / "controller.safetensors"
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "training_run": str(args.run), "mode": args.mode,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "data_sha256": trained["data_sha256"], "completed_ablations": [],
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT / "sas_memory").glob("*.py"))
                                  + [ROOT / "scripts/run_controller_trial.py", Path(__file__).resolve()]},
                "interpretation": "Post-training branch interventions; disablement is not matched-budget retraining."}
    manifest["temporal"] = args.temporal
    manifest["gdn_components"] = args.gdn_components
    for relative in manifest["source_sha256"]:
        snapshot = args.output / "sources" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / relative).read_bytes())
    write_json(args.output / "manifest.json", manifest)
    try:
        runtime = TextRuntime(model_path, config["model"])
        controller = GroupController(runtime.model, config["group_index"], config["modes"][args.mode], args.mode,
                                     adapt_gdn_keys=config.get("adapt_gdn_keys", False))
        controller.load_state_dict(load_file(str(checkpoint), device=str(runtime.model.device)), strict=True)
        cases = json.loads((args.run / "data.json").read_text())["test"]
        names = list(controller.outputs)
        enabled_sets = {
            "all_enabled": names,
            "gdn_only_enabled": [name for name in names if name.startswith("gdn_")],
            "attention_only_enabled": [name for name in names if name.startswith("attention_")],
            "attention_query_only": [name for name in names if name == "attention_q_proj"],
            "attention_key_only": [name for name in names if name == "attention_k_proj"],
            "all_disabled": [],
        }
        if args.gdn_components:
            keys = [name for name in names if name.startswith("gdn_") and name.endswith("_key")]
            gates = [name for name in names if name.startswith("gdn_") and not name.endswith("_key")]
            attention = [name for name in names if name.startswith("attention_")]
            enabled_sets = {"all_enabled": names, "gdn_keys_only": keys, "gdn_gates_only": gates,
                            "attention_and_gdn_keys": attention + keys,
                            "attention_and_gdn_gates": attention + gates, "all_disabled": []}
        policies = {label: config.get("token_policy", "all") for label in enabled_sets}
        if args.temporal:
            enabled_sets = {"all_tokens": names, "ledger_only": names, "question_only": names, "all_disabled": []}
            policies = {"all_tokens": "all", "ledger_only": "ledger", "question_only": "question", "all_disabled": "none"}
        manifest["planned_ablations"] = enabled_sets
        write_json(args.output / "manifest.json", manifest)
        try:
            for label, selected in enabled_sets.items():
                controller.set_enabled_outputs(selected)
                result = evaluate(runtime, cases, args.output / f"{label}.jsonl", controller, policies.get(label, "all"))
                write_json(args.output / f"{label}.summary.json", result)
                if label in ("all_enabled", "all_tokens", "all_disabled"):
                    expected_path = (args.run / args.mode / "test.jsonl" if label in ("all_enabled", "all_tokens") else args.run / "native_test.jsonl")
                    expected = [json.loads(line) for line in expected_path.read_text().splitlines()]
                    actual = [json.loads(line) for line in (args.output / f"{label}.jsonl").read_text().splitlines()]
                    if len(expected) != len(actual):
                        raise AssertionError("Checkpoint replay row count differs")
                    max_error = 0.0
                    for before, after in zip(expected, actual):
                        if (before["case_id"], before["variant"], before["role"]) != (after["case_id"], after["variant"], after["role"]):
                            raise AssertionError("Checkpoint replay order differs")
                        if before["generated"] != after["generated"]:
                            raise AssertionError("Checkpoint replay greedy answer differs")
                        max_error = max(max_error, abs(before["answer_nll"] - after["answer_nll"]))
                    if max_error > 1e-5:
                        raise AssertionError(f"Checkpoint replay NLL differs: {max_error}")
                    manifest[f"{label}_max_nll_replay_error"] = max_error
                manifest["completed_ablations"].append(label)
                write_json(args.output / "manifest.json", manifest)
                print(json.dumps({"event": "ablation_complete", "label": label, "outputs": selected}), flush=True)
        finally:
            controller.close()
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
