"""Cross aligned native recurrent/conv and KV sources with A/C commit guards."""

import argparse
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.interventions import transplant
from sas_memory.kv_readout import QuestionKVReadout
from sas_memory.runtime import TextRuntime, fork_cache
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.locality_event_data import DATA_SOURCES, DESTINATION, registered_inputs
from scripts.locality_event_runtime import (HistoryWeights, assert_cache_layout, checked_ledger,
    replay_pair_ids, same_readout, score_worlds, selected_pairs, supporting_kv_equal)
from scripts.run_controller_trial import write_json
from scripts.run_write_locality_load_screen import hook_state, weights_fingerprint
from scripts.semantic_order_path_common import assert_complete_cache_equal, cache_fingerprint, check_origins


RUNTIME_SOURCES = tuple(sorted(set(DATA_SOURCES) | {
    "scripts/locality_event_runtime.py", "scripts/run_write_locality_event_paths.py",
    "scripts/analyze_write_locality_event_paths.py", "scripts/semantic_order_path_common.py",
}))


def smoke_gate(path, dataset_digest):
    from scripts.analyze_write_locality_event_paths import audit_run
    report = json.loads(path.read_text())
    if (report["status"] != "audited_smoke" or report["dataset_manifest_sha256"] != dataset_digest or
        audit_run(Path(report["source_run"]), allow_smoke=True) != report):
        raise ValueError("Complete independent operational smoke audit required")
    return {"path": str(path), "sha256": streaming_hash(path), "manifest_sha256": report["manifest_sha256"]}


def execute(args, registered, contexts, registry, gate):
    import torch
    torch.set_num_threads(4)
    pairs = selected_pairs(registry, args.smoke)
    replay_ids = replay_pair_ids(registry, pairs, args.smoke)
    cases = {cohort: {row["case"]["id"]: row["case"] for row in context["data"]["validation"]}
             for cohort, context in contexts.items()}
    sources = set(RUNTIME_SOURCES) | {str(p.relative_to(ROOT)) for p in (ROOT / "sas_memory").glob("*.py")}
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "training": False, "adapters": False, "smoke": args.smoke, "dataset": str(args.dataset),
                "dataset_manifest_sha256": streaming_hash(args.dataset / "manifest.json"), "smoke_gate": gate,
                "model_path": registered["model_path"], "selected_pair_ids": [p["id"] for p in pairs],
                "replay_pair_ids": sorted(replay_ids), "source_sha256": {name: streaming_hash(ROOT / name) for name in sorted(sources)},
                "decoding": {"maximum_new_tokens": 12, "candidate_restriction": False, "gold_tokens": 3,
                             "one_generation_per_query_and_cache": True, "separate_world_scoring": True},
                "source_checks": [], "attention_checks": [], "physical_ledger_prefills": 0,
                "full_prefill_C_baselines": 0, "readout_counts": {}}
    args.output.mkdir(parents=True, exist_ok=False)
    for name in sources:
        destination = args.output / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    path = args.output / "manifest.json"
    write_json(path, manifest)
    try:
        runtime = TextRuntime(registered["model_path"], "it")
        manifest.update(python=sys.executable, gpu=torch.cuda.get_device_name(0),
                        packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "fla-core")},
                        actual_eos_token_id=runtime.model.generation_config.eos_token_id)
        reference = json.loads(Path(registered["references"]["replica"]["audit_path"]).read_text())
        if manifest["packages"] != reference["packages"] or runtime.model.config.layer_types != (["linear_attention"] * 3 + ["full_attention"]) * 6:
            raise ValueError("Native runtime or3:1 layer layout changed")
        hooks = hook_state(runtime.model)
        manifest["weights_before"] = weights_fingerprint(runtime.model)
        if manifest["weights_before"]["sha256"] != reference["weights_sha256"]:
            raise ValueError("Actual loaded model differs from the replicated native checkpoint")
        write_json(path, manifest)
        fingerprints = {}
        with torch.inference_mode(), ExitStack() as stack:
            output = stack.enter_context((args.output / "readouts.jsonl").open("w", buffering=1))
            calibrations = stack.enter_context((args.output / "calibrations.jsonl").open("w", buffering=1))

            def write(pair, kind, condition, label, result, **extra):
                key = f"{kind}/{label}"
                manifest["readout_counts"][key] = manifest["readout_counts"].get(key, 0) + 1
                output.write(json.dumps({"pair_id": pair["id"], "kind": kind, "condition": condition,
                                         "label": label, **result, **extra}) + "\n")

            for number, pair in enumerate(pairs, 1):
                case = cases[pair["cohort"]][pair["case_id"]]
                originals, ledgers, prefix_hashes, layouts = {}, {}, {}, {}
                for side, key in pair["source_keys"].items():
                    prefix = registry["prefixes"][key]
                    ledger = checked_ledger(runtime, case, prefix, pair["queries"]["preservation"])
                    ledgers[side] = ledger
                    originals[side], _, count = runtime.advance(ledger)
                    if count != prefix["ledger_tokens"]:
                        raise ValueError("Source computation included a question or omitted ledger tokens")
                    manifest["physical_ledger_prefills"] += 1
                    layouts[side] = assert_cache_layout(originals[side], count)
                    prefix_hashes[side] = cache_fingerprint(originals[side])
                    if fingerprints.setdefault(key, prefix_hashes[side]) != prefix_hashes[side]:
                        raise ValueError("Recomputed natural source fingerprint changed")
                saved = {side: fork_cache(cache) for side, cache in originals.items()}
                evidence = supporting_kv_equal(originals, pair["alignment"]["support_span"])
                source_check = {"pair_id": pair["id"], "source_keys": pair["source_keys"],
                                "prefix_sha256": prefix_hashes, "layouts": layouts, "supporting_KV": evidence,
                                "all_condition_origins_verified": False, "sources_unchanged_after_each_query": False}
                sample = pair["id"] in replay_ids
                prefix = registry["prefixes"][pair["source_keys"]["left"]]
                full_c = {}
                if "current_C" in pair["queries"]:
                    for side in originals:
                        full_c[side] = score_worlds(runtime, ledgers[side], None, pair["queries"]["current_C"], prefill="full")
                        write(pair, "full_C_baseline", side, "current_C", full_c[side])
                        manifest["full_prefill_C_baselines"] += 1
                results = {}
                for index, spec in enumerate(registry["conditions"]):
                    origins = {"recurrent_and_convolution": spec["recurrent_conv"], "kv": spec["kv"]}
                    cache = transplant(originals[spec["recurrent_conv"]], originals[spec["kv"]], runtime.model.config.layer_types, list(range(6)), "kv")
                    for label, query in pair["queries"].items():
                        check_origins(cache, originals, origins)
                        side = spec["recurrent_conv"]
                        observer = HistoryWeights(runtime.model, prefix["prelude_tokens"], prefix["ledger_tokens"]) if sample and label == "preservation" else nullcontext()
                        with observer:
                            result = score_worlds(runtime, ledgers[side], cache, query)
                        if sample and label == "preservation":
                            observer.validate()
                            manifest["attention_checks"].append({"pair_id": pair["id"], "kind": "full", "condition": index, "stats": observer.stats})
                        if result["query_tokens"] != len(query["endpoints"][side]["question_ids"]):
                            raise ValueError("Cached question length differs from registration")
                        for origin in originals:
                            assert_complete_cache_equal(originals[origin], saved[origin])
                        check_origins(cache, originals, origins)
                        results[(index, label)] = result
                        write(pair, "cache_condition", index, label, result, cache_origins=origins, source_cache_unchanged=True)
                        if index < 2:
                            side = "left" if index == 0 else "right"
                            registered_reference = query["endpoints"][side]["full_prefill_reference"]
                            if label == "current_C":
                                full = full_c[side]["world_scores"][side]
                                full_generated = full_c[side]["generated"]
                                reference_hash = data_fingerprint(full_c[side])
                            else:
                                full = contexts[pair["cohort"]]["raw"][tuple(registered_reference["key"])]
                                full_generated, reference_hash = full["generated"], registered_reference["row_sha256"]
                            score = result["world_scores"][side]
                            calibrations.write(json.dumps({"pair_id": pair["id"], "label": label, "side": side,
                                "full_reference_sha256": reference_hash, "full_generated": full_generated,
                                "full_own_world_correct": full["greedy_exact"], "full_own_world_nll": full["answer_nll"],
                                "cached_generated": result["generated"], "cached_own_world_correct": score["greedy_exact"],
                                "cached_own_world_nll": score["answer_nll"],
                                "cached_minus_full_nll": score["answer_nll"] - full["answer_nll"]}) + "\n")
                    del cache
                for side in originals:
                    keep_set = set(pair["oracle"]["kept_positions"])
                    keep = [i in keep_set for i in range(pair["oracle"]["ledger_tokens"])]
                    if data_fingerprint(keep) != pair["oracle"]["history_keep_sha256"]:
                        raise ValueError("Oracle mask differs from frozen instruction/evidence selection")
                    observer = HistoryWeights(runtime.model, prefix["prelude_tokens"], prefix["ledger_tokens"], keep) if sample else nullcontext()
                    with QuestionKVReadout(runtime.model, keep) as mask, observer:
                        result = score_worlds(runtime, ledgers[side], originals[side], pair["queries"]["preservation"])
                    if mask.calls["prefill"] != 0 or mask.calls["cached"] < 1:
                        raise ValueError("Oracle incorrectly recomputed the ledger or missed cached forwards")
                    if sample:
                        observer.validate()
                        manifest["attention_checks"].append({"pair_id": pair["id"], "kind": "oracle", "condition": side, "stats": observer.stats})
                    for origin in originals:
                        assert_complete_cache_equal(originals[origin], saved[origin])
                    write(pair, "oracle", side, "preservation", result, mask_calls=mask.calls, source_cache_unchanged=True)
                if sample:
                    for index, side in enumerate(("left", "right")):
                        other = "right" if side == "left" else "left"
                        for kind, recipient in (("self", originals[side]), ("all_path_donor", originals[other])):
                            clone = transplant(recipient, originals[side], runtime.model.config.layer_types, list(range(6)), "both")
                            assert_complete_cache_equal(clone, originals[side])
                            replay = score_worlds(runtime, ledgers[side], clone, pair["queries"]["preservation"])
                            if not same_readout(results[(index, "preservation")], replay):
                                raise ValueError("Self/all-path clone failed native readout replay")
                            for origin in originals:
                                assert_complete_cache_equal(originals[origin], saved[origin])
                            write(pair, "replay", side, "preservation", replay, clone_kind=kind, exact_cache_and_metadata=True)
                            del clone
                source_check.update(all_condition_origins_verified=True, sources_unchanged_after_each_query=True)
                manifest["source_checks"].append(source_check)
                manifest["unique_natural_prefixes"] = len(fingerprints)
                write_json(path, manifest)
                print(json.dumps({"event": "event_path_pair_completed", "pairs": number, "total": len(pairs),
                                  "readout_counts": manifest["readout_counts"]}), flush=True)
                del originals, saved
        manifest["weights_after"] = weights_fingerprint(runtime.model)
        manifest["hooks_restored"] = hook_state(runtime.model) == hooks
        if (manifest["weights_before"] != manifest["weights_after"] or not manifest["hooks_restored"] or
            not manifest["weights_after"]["all_parameters_frozen"] or not manifest["weights_after"]["no_parameter_gradients"]):
            raise ValueError("Native weights, gradients or observer hooks changed")
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                        results_sha256={name: streaming_hash(args.output / name) for name in ("readouts.jsonl", "calibrations.jsonl")})
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DESTINATION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-audit", type=Path)
    args = parser.parse_args()
    if bool(args.smoke_audit) == args.smoke or args.output.exists():
        raise ValueError("Choose a new operational smoke or supply its complete independent audit")
    registered, contexts, registry = registered_inputs(args.dataset)
    gate = None if args.smoke else smoke_gate(args.smoke_audit, streaming_hash(args.dataset / "manifest.json"))
    execute(args, registered, contexts, registry, gate)


if __name__ == "__main__":
    main()
