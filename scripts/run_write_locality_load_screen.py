"""Run the complete frozen, full-history native write-locality load screen."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.tasks import INSTRUCTIONS
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import (DATA_SOURCES, PROTOCOL, evaluated_members, identity,
                                           read_protocol, registered_dataset, streaming_hash)
from scripts.check_record_attention_weights import ObservedRecordWeights
from scripts.run_controller_trial import prompt_for, write_json
from scripts.run_oracle_kv_readout import measure


def hook_state(model):
    return {name: [list(module._forward_pre_hooks), list(module._forward_hooks)]
            for name, module in model.named_modules() if module._forward_pre_hooks or module._forward_hooks}


def weights_fingerprint(model):
    """Hash every actual loaded state tensor before/after, without saving copies."""
    import torch
    digest, total_bytes = hashlib.sha256(), 0
    state = model.state_dict()
    with torch.inference_mode():
        for name, tensor in sorted(state.items()):
            digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
            flat = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
            for offset in range(0, flat.numel(), 16 * 1024 * 1024):
                block = flat[offset:offset + 16 * 1024 * 1024].cpu().numpy().tobytes()
                digest.update(block)
                total_bytes += len(block)
    return {"sha256": digest.hexdigest(), "state_tensors": len(state), "state_bytes": total_bytes,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "all_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
            "no_parameter_gradients": all(parameter.grad is None for parameter in model.parameters())}


def checked_smoke(path, dataset_manifest):
    from scripts.analyze_write_locality_load_screen import audit_run
    report = json.loads(path.read_text())
    if report["status"] != "audited_smoke" or report["records_audited"] != 12:
        raise ValueError("Complete independent operational smoke audit is required")
    actual = audit_run(Path(report["source_run"]), allow_smoke=True)
    actual["analysis_source_sha256"] = streaming_hash(ROOT / "scripts/analyze_write_locality_load_screen.py")
    if report != actual or report["dataset_manifest_sha256"] != dataset_manifest:
        raise ValueError("Operational smoke audit/source/data changed")
    return {"report_path": str(path), "report_sha256": streaming_hash(path),
            "manifest_sha256": report["manifest_sha256"], "records_audited": report["records_audited"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-audit", type=Path)
    args = parser.parse_args()
    if bool(args.smoke_audit) == args.smoke:
        raise ValueError("Choose operational --smoke or provide --smoke-audit for the complete run")
    plan = read_protocol()
    registered, data, registry = registered_dataset(args.dataset)
    dataset_digest = streaming_hash(args.dataset / "manifest.json")
    gate = None if args.smoke else checked_smoke(args.smoke_audit, dataset_digest)
    members = evaluated_members(data, args.smoke)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    sources = set(DATA_SOURCES) | {"sas_memory/write_locality_metrics.py",
        "scripts/run_write_locality_load_screen.py", "scripts/analyze_write_locality_load_screen.py",
        "scripts/check_record_attention_weights.py", "scripts/run_oracle_kv_readout.py"}
    sources.update(str(path.relative_to(ROOT)) for path in (ROOT / "sas_memory").glob("*.py"))
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "training": False, "adapters": False, "smoke": args.smoke, "reader_policy": "full",
                "dataset": str(args.dataset), "dataset_manifest_sha256": dataset_digest,
                "protocol_sha256": streaming_hash(PROTOCOL), "reference_gate": registered["reference_gate"],
                "smoke_gate": gate, "model_path": registered["model_path"], "safety_checks": [],
                "generation_config_sha256": {path.name: streaming_hash(path) for path in
                                              Path(registered["model_path"]).glob("generation_config.json")},
                "evaluated_keys": [identity(example) for example in members],
                "source_sha256": {relative: streaming_hash(ROOT / relative) for relative in sorted(sources)},
                "decoding": {"maximum_new_tokens": 12, "candidate_restriction": False, "gold_answer_nll_tokens": 3},
                "screen_signal_implementation": "Same axis/fixed-other-bit/role/load contrast regresses in at least two distinct base streams; value siblings count once."}
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
        manifest.update(python=sys.executable, gpu=torch.cuda.get_device_name(0),
                        actual_eos_token_id=runtime.model.generation_config.eos_token_id,
                        packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")})
        before_hooks = hook_state(runtime.model)
        manifest["weights_before"] = weights_fingerprint(runtime.model)
        write_json(manifest_path, manifest)
        indexed = {(row["case_id"], row["variant"], row["role"]): row for row in registry["validation"]}
        ordered_keys = sorted(tuple(identity(example)) for example in members)
        sample_keys = {ordered_keys[0], ordered_keys[-1]}
        result_path = args.output / "validation.native.jsonl"
        with result_path.open("w", buffering=1) as output:
            for number, example in enumerate(members, 1):
                case, query = example["case"], example["query"]
                key = tuple(identity(example))
                registration = indexed[key]
                ledger, prompt = prompt_for(runtime, case, example["variant"], query)
                if data_fingerprint(runtime.ids(prompt)) != registration["prompt_ids_sha256"]:
                    raise ValueError("Actual locality prompt differs from frozen token registration")
                if key in sample_keys:
                    with ObservedRecordWeights(runtime.model, len(runtime.ids(runtime.prompt_start + INSTRUCTIONS)), registration["ledger_tokens"]) as observer:
                        result = measure(runtime, ledger, query)
                    if set(observer.stats) != {3, 7, 11, 15, 19, 23} or any(
                            stat["prefill_calls"] != 1 or stat["cached_calls"] < 1 or stat["max_record_mass"] <= 0
                            for stat in observer.stats.values()):
                        raise ValueError("Actual full-history prefill/cached Attention safety failed")
                    if measure(runtime, ledger, query) != result:
                        raise ValueError("Unobserved native replay differs from observed result")
                    manifest["safety_checks"].append({"key": list(key), "native_replay_exact": True,
                                                       "full_attention_weights": observer.stats})
                else:
                    result = measure(runtime, ledger, query)
                if result["query_tokens"] != registration["prompt_tokens"] - registration["ledger_tokens"]:
                    raise ValueError("Actual query-token length differs from registration")
                output.write(json.dumps({"case_id": case["id"], "variant": example["variant"], "role": query["role"],
                    **{field: case[field] for field in ("base_stream_id", "base_case_id", "seed", "active_accounts", "value_version")},
                    "reader_policy": "full", "registration": registration, **result}) + "\n")
                if number % (12 if args.smoke else 48) == 0:
                    print(json.dumps({"event": "native_locality_evaluation", "completed": number, "total": len(members)}), flush=True)
        manifest["weights_after"] = weights_fingerprint(runtime.model)
        manifest["hooks_restored"] = hook_state(runtime.model) == before_hooks
        if (manifest["weights_before"] != manifest["weights_after"] or not manifest["hooks_restored"] or
            not manifest["weights_after"]["all_parameters_frozen"] or not manifest["weights_after"]["no_parameter_gradients"]):
            raise ValueError("Native weight identity, freeze or observer-hook restoration failed")
        manifest["safety_checks"].sort(key=lambda check: check["key"])
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(), result_sha256=streaming_hash(result_path))
        print(json.dumps({"event": "native_locality_completed", "examples": len(members), "weights_identical": True}), flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
