"""Block historical record KV during questions, keeping ordinary hybrid prefix processing."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.counterbalance_data import encode_counterbalance, file_digest
from sas_memory.counterbalance_tasks import ORDERS
from sas_memory.kv_readout import QuestionKVReadout
from sas_memory.record_kv_ablation import record_blocked_mask, reference_protocol
from sas_memory.training_data import data_fingerprint
from scripts.analyze_controller_trial import load_rows
from scripts.run_controller_confirmation import verify_native_prefix
from scripts.run_controller_trial import write_json
from scripts.run_oracle_kv_readout import measure, verify_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="First/last registered cases; all frozen checkpoints")
    args = parser.parse_args()
    data = json.loads((args.data / "manifest.json").read_text())
    plan = json.loads((args.data / "plan.json").read_text())
    cases = json.loads((args.data / "cases.json").read_text())
    registry = json.loads((args.data / "registry.json").read_text())
    reference, original, _, ref_cases, sources, hashes = reference_protocol(plan, ROOT)
    if data["status"] != "validated_before_inference" or cases != ref_cases or sources != data["checkpoint_sources"]:
        raise ValueError("Registered data or checkpoint provenance changed")
    for key, value in (("plan", plan), ("cases", cases), ("registry", registry)):
        if data_fingerprint(value) != data[key + "_sha256"]:
            raise ValueError(f"Registered {key} changed")
    if hashes != data["reference_result_sha256"] or file_digest(reference / "manifest.json") != data["reference_manifest_sha256"]:
        raise ValueError("Reference scores/manifest changed")
    for name, fingerprint in data["tokenizer_sha256"].items():
        if file_digest(Path(data["model_path"]) / name) != fingerprint:
            raise ValueError("Registered tokenizer changed")
    all_expected = {(case["id"], order, "target_current"): case["variants"][order]["queries"][0]["answer"] for case in cases for order in ORDERS}
    reference_rows = {model: {regime: {(row["case_id"], row["variant"]): row for row in load_rows(reference / f"{model}.{regime}.jsonl", all_expected)}
                             for regime in ("raw_full", "oracle")} for model in plan["models"]}
    old_registry = json.loads((reference / "registry.json").read_text())
    if data_fingerprint(old_registry) != original["registry_sha256"]:
        raise ValueError("Reference oracle masks changed")
    oracle_masks = {(row["case_id"], row["variant"]): row["masks"]["oracle"] for row in old_registry}
    indexed = {(row["case_id"], row["variant"]): row for row in registry}
    if len(indexed) != len(all_expected):
        raise ValueError("Incomplete registered record masks")
    evaluated = [cases[0], cases[-1]] if args.smoke else cases
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("plan.json", "reference_plan.json", "reference_manifest.json", "reference.audit.json", "cases.json", "registry.json"):
        (args.output / name).write_bytes((args.data / name).read_bytes())
    (args.output / "data_manifest.json").write_bytes((args.data / "manifest.json").read_bytes())
    source_paths = sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__).resolve(),
        ROOT / "scripts/prepare_record_kv_ablation.py", ROOT / "scripts/run_oracle_kv_readout.py",
        ROOT / "scripts/run_controller_confirmation.py", ROOT / "scripts/run_controller_trial.py"]
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "training": False,
                "smoke": args.smoke, "data_path": str(args.data), "data_manifest_sha256": file_digest(args.data / "manifest.json"),
                "cases_sha256": data["cases_sha256"], "registry_sha256": data["registry_sha256"],
                "evaluated_case_ids": [case["id"] for case in evaluated], "models": plan["models"], "completed_models": [],
                "checkpoint_sources": sources, "safety_checks": {}, "native_prefix_checks": {}, "oracle_replay": {},
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
    try:
        runtime = TextRuntime(data["model_path"], "it")
        manifest.update(model_path=data["model_path"], python=sys.executable, gpu=torch.cuda.get_device_name(0),
                        packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")})
        if manifest["packages"] != original["packages"]:
            raise ValueError("Runtime packages differ from the reference evaluation")
        for label in plan["models"]:
            controller, weights = None, None
            if label != "native":
                spec, protocol = sources[label], sources[label]["config"]
                controller = GroupController(runtime.model, protocol["group_index"], protocol["modes"][spec["mode"]],
                                             spec["mode"], adapt_gdn_keys=protocol.get("adapt_gdn_keys", False))
                weights = load_file(str(Path(spec["training_run"]) / spec["mode"] / "controller.safetensors"), device=str(runtime.model.device))
                controller.load_state_dict(weights, strict=True)
            for field in ("safety_checks", "native_prefix_checks", "oracle_replay"):
                manifest[field][label] = []
            try:
                count = 0
                with (args.output / f"{label}.records_blocked.jsonl").open("w", buffering=1) as output:
                    for index, case in enumerate(evaluated):
                        for order in ORDERS:
                            ledger, prompt, _ = encode_counterbalance(runtime, case, order)
                            record = {"case_id": case["id"], "variant": order, **record_blocked_mask(runtime, ledger, prompt)}
                            if record != indexed[(case["id"], order)]:
                                raise ValueError("Runtime mask/input changed from pre-inference registration")
                            query = case["variants"][order]["queries"][0]
                            if controller is not None:
                                controller.set_token_policy("question", record["ledger_tokens"])
                            keep = [bit == "1" for bit in record["keep_bits"]]
                            if index in (0, len(evaluated) - 1) and order == ORDERS[0]:
                                if controller is not None:
                                    manifest["native_prefix_checks"][label].append({"case_id": case["id"], "variant": order,
                                        **verify_native_prefix(runtime, controller, ledger)})
                                manifest["safety_checks"][label].append({"case_id": case["id"], "variant": order,
                                    **verify_mask(runtime, ledger, prompt, keep, query, reference_rows[label]["raw_full"][(case["id"], order)])})
                                with QuestionKVReadout(runtime.model, [bit == "1" for bit in oracle_masks[(case["id"], order)]["keep_bits"]]):
                                    actual = measure(runtime, ledger, query)
                                expected = reference_rows[label]["oracle"][(case["id"], order)]
                                error = abs(actual["answer_nll"] - expected["answer_nll"])
                                if error > 1e-5 or actual["generated"] != expected["generated"]:
                                    raise ValueError("Oracle reference replay changed")
                                manifest["oracle_replay"][label].append({"case_id": case["id"], "variant": order,
                                                                        "max_nll_error": error, "greedy_exact_match": True})
                                write_json(manifest_path, manifest)
                            with QuestionKVReadout(runtime.model, keep) as intervention:
                                result = measure(runtime, ledger, query)
                            if intervention.calls["prefill"] != 1 or intervention.calls["cached"] < 1:
                                raise ValueError("Missing prefill/cached record-KV restriction")
                            output.write(json.dumps({**{key: value for key, value in record.items() if key != "keep_bits"},
                                                     "role": "target_current", "regime": "records_blocked",
                                                     "hook_calls": intervention.calls, **result}) + "\n")
                            count += 1
                            if count % 48 == 0:
                                print(json.dumps({"event": "record_kv_progress", "model": label, "contexts": count}), flush=True)
                if controller is not None and any(not torch.equal(value, weights[key]) for key, value in controller.state_dict().items()):
                    raise ValueError("Frozen adapter weights changed")
                manifest["completed_models"].append(label)
                write_json(manifest_path, manifest)
                print(json.dumps({"event": "record_kv_complete", "model": label, "contexts": count}), flush=True)
            finally:
                if controller is not None:
                    controller.close()
            if label != "native":
                case = evaluated[0]
                ledger, _, _ = encode_counterbalance(runtime, case, ORDERS[0])
                actual = measure(runtime, ledger, case["variants"][ORDERS[0]]["queries"][0])
                expected = reference_rows["native"]["raw_full"][(case["id"], ORDERS[0])]
                error = abs(actual["answer_nll"] - expected["answer_nll"])
                if error > 1e-5 or actual["generated"] != expected["generated"]:
                    raise ValueError("Controller removal did not restore native behavior")
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
