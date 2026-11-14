"""Run the registered native-only no-recent-initial-repeat history diagnostic."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.runtime import TextRuntime
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.check_record_attention_weights import ObservedRecordWeights
from scripts.history_no_repeat_data import (DATA_SOURCES, DESTINATION, PROTOCOL, evaluated_members,
    identity, read_plan, registered_inputs, safety_samples)
from scripts.run_controller_trial import prompt_for, write_json
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_load_screen import hook_state, weights_fingerprint

RUNTIME_SOURCES = tuple(sorted(set(DATA_SOURCES) | {
    "scripts/run_semantic_history_no_repeat.py", "scripts/analyze_semantic_history_no_repeat.py",
    "scripts/history_no_repeat_metrics.py", "scripts/check_record_attention_weights.py",
    "scripts/run_oracle_kv_readout.py", "scripts/run_write_locality_load_screen.py",
}))


def checked_smoke(path, dataset_digest):
    from scripts.analyze_semantic_history_no_repeat import audit_run
    report = json.loads(path.read_text())
    if (report["status"] != "audited_smoke" or report["records_audited"] != 24 or
        report["dataset_manifest_sha256"] != dataset_digest or
        audit_run(Path(report["source_run"]), allow_smoke=True) != report):
        raise ValueError("Complete independently audited24-query operational smoke required")
    return {"path": str(path), "sha256": streaming_hash(path), "manifest_sha256": report["manifest_sha256"]}


def execute(args, plan, registered, data, registry, gate):
    import torch
    torch.set_num_threads(4)
    members = evaluated_members(data, args.smoke)
    sample_keys = safety_samples(members)
    sources = set(RUNTIME_SOURCES) | {str(p.relative_to(ROOT)) for p in (ROOT / "sas_memory").glob("*.py")}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "training": False, "adapters": False, "reader_policy": "full", "smoke": args.smoke,
        "dataset": str(args.dataset), "dataset_manifest_sha256": streaming_hash(args.dataset / "manifest.json"),
        "protocol_sha256": streaming_hash(PROTOCOL), "registry_sha256": registered["registry_sha256"],
        "reference_gate": registered["reference_gate"], "smoke_gate": gate, "model_path": registered["model_path"],
        "evaluated_keys": [identity(row) for row in members], "safety_sample_keys": sample_keys,
        "safety_checks": [], "completed_readouts": 0,
        "source_sha256": {name: streaming_hash(ROOT / name) for name in sorted(sources)},
        "generation_config_sha256": {p.name: streaming_hash(p) for p in Path(registered["model_path"]).glob("generation_config.json")},
        "decoding": {"maximum_new_tokens": 12, "candidate_restriction": False, "gold_answer_nll_tokens": 3}}
    for name in sources:
        destination = args.output / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    manifest_path = args.output / "manifest.json"
    write_json(manifest_path, manifest)
    try:
        runtime = TextRuntime(registered["model_path"], "it")
        manifest.update(python=sys.executable, gpu=torch.cuda.get_device_name(0),
            actual_eos_token_id=runtime.model.generation_config.eos_token_id,
            packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")})
        if (manifest["packages"] != registered["reference_gate"]["packages"] or
            runtime.model.config.layer_types != (["linear_attention"] * 3 + ["full_attention"]) * 6):
            raise ValueError("Native runtime or3:1 hybrid layer layout changed")
        before_hooks = hook_state(runtime.model)
        manifest["weights_before"] = weights_fingerprint(runtime.model)
        if (manifest["weights_before"]["sha256"] != plan["native_loaded_weights_sha256"] or
            not manifest["weights_before"]["all_parameters_frozen"] or not manifest["weights_before"]["no_parameter_gradients"]):
            raise ValueError("Actual loaded native checkpoint/freeze changed")
        write_json(manifest_path, manifest)
        indexed = {(row["case_id"], row["variant"], row["role"]): row for row in registry["validation"]}
        result_path = args.output / "validation.native.jsonl"
        with torch.inference_mode(), result_path.open("w", buffering=1) as output:
            for number, example in enumerate(members, 1):
                case, query = example["case"], example["query"]
                key = identity(example)
                registration = indexed[tuple(key)]
                prefix = registry["prefixes"][registration["prefix_key"]]
                ledger, prompt = prompt_for(runtime, case, example["variant"], query)
                ids, ledger_ids = runtime.ids(prompt), runtime.ids(ledger)
                if (data_fingerprint(ids) != registration["prompt_ids_sha256"] or
                    data_fingerprint(ledger_ids) != prefix["ledger_ids_sha256"] or ids[:len(ledger_ids)] != ledger_ids or
                    ids[len(ledger_ids):] != registration["question_ids"] or
                    runtime.continuation_ids(prompt, query["answer"]) != registration["answer_ids"]):
                    raise ValueError("Actual source world, prompt boundary or own-world answer tokens changed")
                if key in sample_keys:
                    with ObservedRecordWeights(runtime.model, prefix["prelude_tokens"], prefix["ledger_tokens"]) as observer:
                        result = measure(runtime, ledger, query)
                    if (set(observer.stats) != {3, 7, 11, 15, 19, 23} or any(
                        stat["prefill_calls"] != 1 or stat["cached_calls"] < 1 or stat["max_record_mass"] <= 0
                        for stat in observer.stats.values()) or hook_state(runtime.model) != before_hooks):
                        raise ValueError("Actual six-layer full-history access/hook restoration failed")
                    replay = measure(runtime, ledger, query)
                    if replay != result:
                        raise ValueError("Actual unobserved native full-prefill replay differs")
                    manifest["safety_checks"].append({"key": key, "native_replay_exact": True,
                        "native_replay": replay, "full_attention_weights": observer.stats, "hooks_restored": True})
                else:
                    result = measure(runtime, ledger, query)
                if result["query_tokens"] != len(registration["question_ids"]):
                    raise ValueError("Actual question length differs from frozen registry")
                output.write(json.dumps({"case_id": case["id"], "variant": example["variant"], "role": query["role"],
                    **{field: case[field] for field in ("cohort", "base_stream_id", "base_case_id", "seed", "active_accounts", "value_version")},
                    "reader_policy": "full", "prefill": "full", "registration_sha256": data_fingerprint(registration), **result}) + "\n")
                manifest["completed_readouts"] = number
                if number % (12 if args.smoke else 32) == 0:
                    write_json(manifest_path, manifest)
                    print(json.dumps({"event": "native_no_repeat_history", "completed": number, "total": len(members)}), flush=True)
        manifest["weights_after"] = weights_fingerprint(runtime.model)
        manifest["hooks_restored"] = hook_state(runtime.model) == before_hooks
        if manifest["weights_before"] != manifest["weights_after"] or not manifest["hooks_restored"]:
            raise ValueError("Native loaded weights/gradients or original hooks changed")
        manifest["safety_checks"].sort(key=lambda row: row["key"])
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(), result_sha256=streaming_hash(result_path))
        print(json.dumps({"event": "native_no_repeat_history_completed", "readouts": len(members), "weights_identical": True}), flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DESTINATION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-audit", type=Path)
    args = parser.parse_args()
    if bool(args.smoke_audit) == args.smoke or args.output.exists():
        raise ValueError("Choose a new operational smoke or supply its independent audit")
    plan = read_plan()
    registered, data, registry = registered_inputs(args.dataset)
    gate = None if args.smoke else checked_smoke(args.smoke_audit, streaming_hash(args.dataset / "manifest.json"))
    execute(args, plan, registered, data, registry, gate)


if __name__ == "__main__":
    main()
