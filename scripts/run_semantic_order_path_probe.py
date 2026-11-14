"""Cross natural recurrent/conv and KV origins at matched question positions."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from sas_memory.training_data import data_fingerprint
from sas_memory.tasks import INSTRUCTIONS
from scripts.analyze_paired_semantic_probe import identity
from scripts.check_record_attention_weights import ObservedRecordWeights
from scripts.run_blocked_reader_budget_calibration import atomic_json
from scripts.run_controller_trial import prompt_for, write_json
from scripts.run_paired_semantic_probe import hook_state
from scripts.semantic_order_path_common import (CONDITIONS, assert_complete_cache_equal, cache_fingerprint,
    check_origins, functional_report, reference_gate, registered_probe_inputs, selected_examples)


def cached_measure(runtime, ledger, cache, query):
    result = runtime.probe(ledger, cache, query["question"], query["answer"], [query["answer"]],
                           generate=True, prefill="cached", score_batch=False)
    for field in ("candidate_prediction", "candidate_correct", "candidate_scores"):
        result.pop(field)
    return result


def same_readout(left, right):
    return (left["generated"] == right["generated"] and left["greedy_exact"] == right["greedy_exact"] and
            left["query_tokens"] == right["query_tokens"] and abs(left["answer_nll"] - right["answer_nll"]) <= 1e-5)


def execute(args, plan, base, data, cases, registry, reference):
    import torch
    from sas_memory.interventions import transplant
    from sas_memory.runtime import TextRuntime, fork_cache
    torch.set_num_threads(4)
    sources = set(json.loads((Path(plan["augmentation_reference_run"]) / "manifest.json").read_text())["source_sha256"])
    sources.update(json.loads((Path(plan["full_reference_order_run"]) / "manifest.json").read_text())["source_sha256"])
    sources.update("scripts/" + name for name in ("run_semantic_order_path_probe.py", "analyze_semantic_order_path_probe.py", "semantic_order_path_common.py", "run_paired_semantic_probe.py"))
    fingerprints = {relative: file_hash(ROOT / relative) for relative in sorted(sources)}
    selected = selected_examples(data, args.smoke)
    keys = [tuple(identity(example)) for example in selected]
    sample_keys = {min(keys), max(keys)}
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "training": False,
        "smoke": args.smoke, "pid": os.getpid(), "reader_policy": "full", "plan_sha256": data_fingerprint(plan),
        "source_sha256": fingerprints, "reference_audit": reference, "evaluated_keys": [list(key) for key in keys],
        "prefix_checks": [], "safety_checks": [], "completed_conditions": []}
    write_json(args.output / "plan.json", plan)
    for relative in fingerprints:
        destination = args.output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    path = args.output / "manifest.json"
    atomic_json(path, manifest)
    try:
        runtime = TextRuntime(base["model_path"], "it")
        packages = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")}
        native = json.loads((Path(plan["full_reference_native_run"]) / "manifest.json").read_text())
        if packages != native["packages"] or runtime.model.config.layer_types != (["linear_attention"]*3+["full_attention"])*6:
            raise ValueError("Native path runtime or hybrid layer layout changed")
        manifest.update(model_path=base["model_path"], packages=packages, python=sys.executable, gpu=torch.cuda.get_device_name(0))
        original_hooks = hook_state(runtime.model)
        rows = {name: [] for name in CONDITIONS}
        with torch.inference_mode(), ExitStack() as stack:
            outputs = {name: stack.enter_context((args.output / f"{name}.jsonl").open("w", buffering=1)) for name in CONDITIONS}
            for number, example in enumerate(selected, 1):
                key = tuple(identity(example))
                query = example["query"]
                natural_cases = {"canonical": example["case"], "permuted": cases[example["case"]["id"]]}
                ledgers, originals = {}, {}
                for origin, case in natural_cases.items():
                    ledger, prompt = prompt_for(runtime, case, example["variant"], query)
                    registration = registry[origin][key]
                    if (data_fingerprint(runtime.ids(prompt)) != registration["prompt_ids_sha256"] or
                        len(runtime.ids(ledger)) != registration["ledger_tokens"]):
                        raise ValueError("Actual native prefix/query differs from registration")
                    ledgers[origin] = ledger
                    originals[origin], _, _ = runtime.advance(ledger)
                saved = {origin: fork_cache(cache) for origin, cache in originals.items()}
                caches = {"canonical_natural": originals["canonical"], "permuted_natural": originals["permuted"],
                    "canonical_state_permuted_kv": transplant(originals["canonical"], originals["permuted"], runtime.model.config.layer_types, plan["groups"], "kv"),
                    "permuted_state_canonical_kv": transplant(originals["canonical"], originals["permuted"], runtime.model.config.layer_types, plan["groups"], "recurrent")}
                hashes = {origin: cache_fingerprint(cache) for origin, cache in originals.items()}
                query_results = {}
                for name, origins in CONDITIONS.items():
                    check_origins(caches[name], originals, origins)
                    ledger = ledgers["permuted" if name == "permuted_natural" else "canonical"]
                    observer = None
                    if key in sample_keys:
                        with ObservedRecordWeights(runtime.model, len(runtime.ids(runtime.prompt_start + INSTRUCTIONS)), registry["canonical"][key]["ledger_tokens"]) as observer:
                            result = cached_measure(runtime, ledger, caches[name], query)
                        if set(observer.stats) != {3,7,11,15,19,23} or any(s["prefill_calls"] != 0 or s["cached_calls"] < 1 or s["max_record_mass"] <= 0 for s in observer.stats.values()):
                            raise ValueError("Cached query did not inspect historical attention at every layer")
                        manifest["safety_checks"].append({"key": list(key), "condition": name, "full_attention_weights": observer.stats})
                    else:
                        result = cached_measure(runtime, ledger, caches[name], query)
                    if result["query_tokens"] != registry["canonical"][key]["prompt_tokens"] - registry["canonical"][key]["ledger_tokens"]:
                        raise ValueError("Cached question used an unregistered token boundary")
                    for origin in originals:
                        assert_complete_cache_equal(originals[origin], saved[origin])
                    check_origins(caches[name], originals, origins)
                    query_results[name] = result
                    case = example["case"]
                    row = {"case_id": case["id"], "variant": example["variant"], "role": query["role"],
                        **{field: case[field] for field in ("base_case_id", "family", "seed", "value_version", "distractor_updates")},
                        "condition": name, "cache_origins": origins, "reader_policy": "full", "prefill": "cached",
                        "registrations": {origin: registry[origin][key] for origin in registry}, "prefix_sha256": hashes,
                        "source_cache_unchanged": True, **result}
                    rows[name].append(row)
                    outputs[name].write(json.dumps(row) + "\n")
                replay_checks = []
                if key in sample_keys:
                    for origin in originals:
                        other = "permuted" if origin == "canonical" else "canonical"
                        baseline = query_results[f"{origin}_natural"]
                        for label, recipient in (("self", originals[origin]), ("all_path_donor", originals[other])):
                            control = transplant(recipient, originals[origin], runtime.model.config.layer_types, plan["groups"], "both")
                            assert_complete_cache_equal(control, originals[origin])
                            replay = cached_measure(runtime, ledgers[origin], control, query)
                            if not same_readout(baseline, replay):
                                raise ValueError("Self/all-path replay differs from the matching cached natural readout")
                            replay_checks.append({"origin": origin, "kind": label, "readout_matches": True,
                                                  "nll_absolute_error": abs(replay["answer_nll"]-baseline["answer_nll"])})
                            for source in originals:
                                assert_complete_cache_equal(originals[source], saved[source])
                manifest["prefix_checks"].append({"key": list(key), "prefix_sha256": hashes,
                    "ledger_tokens": registry["canonical"][key]["ledger_tokens"], "all_condition_origins_verified": True,
                    "sources_unchanged_after_each_query": True, "replay_checks": replay_checks})
                if number % 36 == 0:
                    atomic_json(path, manifest)
                    print(json.dumps({"event": "native_path_queries", "examples_per_condition": number, "conditions": 4}), flush=True)
        manifest["hooks_restored"] = hook_state(runtime.model) == original_hooks
        manifest["frozen_parameters_and_no_gradients"] = all(not p.requires_grad and p.grad is None for p in runtime.model.parameters())
        if not manifest["hooks_restored"] or not manifest["frozen_parameters_and_no_gradients"]:
            raise ValueError("Native path probe changed hooks or frozen parameter scope")
        manifest.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
            completed_conditions=list(CONDITIONS), results_sha256={name: file_hash(args.output / f"{name}.jsonl") for name in CONDITIONS},
            functional_summary=functional_report(rows), natural_prefix_prefills=2*len(selected),
            unique_prefixes_per_order=len({(e["case"]["id"],e["variant"]) for e in selected}))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/semantic_order_path_probe_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    base, data, cases, registry = registered_probe_inputs(plan)
    reference = reference_gate(plan)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, plan, base, data, cases, registry, reference)


if __name__ == "__main__":
    main()
