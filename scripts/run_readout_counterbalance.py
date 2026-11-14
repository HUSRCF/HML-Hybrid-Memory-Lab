"""Frozen-checkpoint inference on registered position-balanced KV controls."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.counterbalance_data import encode_counterbalance, file_digest, locked_training_sources
from sas_memory.counterbalance_tasks import MASKS, ORDERS, validate_counterbalance_cases
from sas_memory.kv_readout import QuestionKVReadout
from sas_memory.training_data import data_fingerprint
from scripts.run_controller_confirmation import verify_native_prefix
from scripts.run_controller_trial import write_json
from scripts.run_oracle_kv_readout import measure, verify_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="First and last registered cases, all four checkpoints")
    args = parser.parse_args()
    data_manifest = json.loads((args.data / "manifest.json").read_text())
    plan = json.loads((args.data / "plan.json").read_text())
    all_cases = json.loads((args.data / "cases.json").read_text())
    registry = json.loads((args.data / "registry.json").read_text())
    validation = validate_counterbalance_cases(all_cases)
    if data_manifest["status"] != "validated_before_inference" or validation != data_manifest["task_validation"]:
        raise ValueError("Cohort has not passed preparation")
    for key, value in (("cases", all_cases), ("registry", registry), ("plan", plan)):
        if data_fingerprint(value) != data_manifest[key + "_sha256"]:
            raise ValueError(f"Registered {key} changed")
    sources = locked_training_sources(plan, ROOT, {case["seed"] for case in all_cases})
    if sources != data_manifest["checkpoint_sources"]:
        raise ValueError("Registered training sources changed")
    for name, fingerprint in data_manifest["tokenizer_sha256"].items():
        if file_digest(Path(data_manifest["model_path"]) / name) != fingerprint:
            raise ValueError("Registered tokenizer changed")
    cases = [all_cases[0], all_cases[-1]] if args.smoke else all_cases
    indexed = {(row["case_id"], row["variant"]): row for row in registry}
    if len(indexed) != validation["contexts"]:
        raise ValueError("Duplicate or incomplete registered contexts")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("plan.json", "cases.json", "registry.json"):
        (args.output / name).write_bytes((args.data / name).read_bytes())
    (args.output / "data_manifest.json").write_bytes((args.data / "manifest.json").read_bytes())
    source_paths = sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__).resolve(),
        ROOT / "scripts/prepare_readout_counterbalance.py", ROOT / "scripts/run_oracle_kv_readout.py",
        ROOT / "scripts/run_controller_confirmation.py", ROOT / "scripts/run_controller_trial.py"]
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "training": False,
                "smoke": args.smoke, "data_path": str(args.data), "data_manifest_sha256": file_digest(args.data / "manifest.json"),
                "cases_sha256": data_manifest["cases_sha256"], "registry_sha256": data_manifest["registry_sha256"],
                "evaluated_case_ids": [case["id"] for case in cases], "checkpoint_sources": sources,
                "modes": plan["checkpoints"], "regimes": plan["evaluation_regimes"], "completed_modes": [],
                "safety_checks": {}, "native_prefix_checks": {}, "native_replay_after_detach": {},
                "source_sha256": {str(path.relative_to(ROOT)): file_digest(path) for path in source_paths}}
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
    try:
        runtime = TextRuntime(data_manifest["model_path"], "it")
        manifest.update(model_path=data_manifest["model_path"], python=sys.executable, gpu=torch.cuda.get_device_name(0),
                        packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")})
        native_reference = None
        for label in manifest["modes"]:
            controller = None
            weights = None
            if label != "native":
                spec = sources[label]
                protocol = spec["config"]
                controller = GroupController(runtime.model, protocol["group_index"], protocol["modes"][spec["mode"]],
                                             spec["mode"], adapt_gdn_keys=protocol.get("adapt_gdn_keys", False))
                weights = load_file(str(Path(spec["training_run"]) / spec["mode"] / "controller.safetensors"), device=str(runtime.model.device))
                controller.load_state_dict(weights, strict=True)
            manifest["safety_checks"][label], manifest["native_prefix_checks"][label] = [], []
            try:
                with ExitStack() as stack:
                    outputs = {kind: stack.enter_context((args.output / f"{label}.{kind}.jsonl").open("w", buffering=1))
                               for kind in manifest["regimes"]}
                    contexts = 0
                    for index, case in enumerate(cases):
                        for order in ORDERS:
                            ledger, prompt, record = encode_counterbalance(runtime, case, order)
                            if record != indexed[(case["id"], order)]:
                                raise ValueError("Runtime input/masks differ from pre-inference registration")
                            query = case["variants"][order]["queries"][0]
                            if controller is not None:
                                controller.set_token_policy("question", record["ledger_tokens"])
                            raw = measure(runtime, ledger, query)
                            if native_reference is None and label == "native":
                                native_reference = raw
                            check = index in (0, len(cases) - 1) and order == ORDERS[0]
                            if check and controller is not None:
                                manifest["native_prefix_checks"][label].append({"case_id": case["id"], "variant": order,
                                    **verify_native_prefix(runtime, controller, ledger)})
                            for kind in manifest["regimes"]:
                                row = {"case_id": case["id"], "variant": order, "role": "target_current", "regime": kind,
                                       **{key: record[key] for key in ("ledger_tokens", "prompt_tokens", "prompt_ids_sha256")}}
                                if kind == "raw_full":
                                    result = raw
                                else:
                                    mask = record["masks"][kind]
                                    keep = [bit == "1" for bit in mask["keep_bits"]]
                                    if check:
                                        manifest["safety_checks"][label].append({"case_id": case["id"], "variant": order, "regime": kind,
                                            **verify_mask(runtime, ledger, prompt, keep, query, raw)})
                                    with QuestionKVReadout(runtime.model, keep) as intervention:
                                        result = measure(runtime, ledger, query)
                                    if intervention.calls["prefill"] != 1 or not intervention.calls["cached"]:
                                        raise ValueError("Missing prefill/cached mask intervention")
                                    row.update({key: mask[key] for key in ("selected_event_indices", "kept_history_tokens", "keep_sha256")},
                                               hook_calls=intervention.calls)
                                outputs[kind].write(json.dumps({**row, **result}) + "\n")
                            contexts += 1
                            if contexts % 48 == 0:
                                print(json.dumps({"event": "counterbalance_progress", "mode": label, "contexts": contexts,
                                                  "rows": contexts * len(manifest["regimes"])}), flush=True)
                    if controller is not None and any(not torch.equal(value, weights[key]) for key, value in controller.state_dict().items()):
                        raise ValueError("Frozen adapter weights changed during evaluation")
                manifest["completed_modes"].append(label)
                write_json(manifest_path, manifest)
                print(json.dumps({"event": "counterbalance_complete", "mode": label, "contexts": contexts}), flush=True)
            finally:
                if controller is not None:
                    controller.close()
            if label != "native":
                first = cases[0]
                ledger, _, _ = encode_counterbalance(runtime, first, ORDERS[0])
                replay = measure(runtime, ledger, first["variants"][ORDERS[0]]["queries"][0])
                error = abs(replay["answer_nll"] - native_reference["answer_nll"])
                if error > 1e-5 or replay["generated"] != native_reference["generated"]:
                    raise ValueError("Removing controller failed to restore native behavior")
                manifest["native_replay_after_detach"][label] = {"max_nll_error": error, "greedy_exact_match": True,
                                                                 "adapter_weights_unchanged": True}
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
