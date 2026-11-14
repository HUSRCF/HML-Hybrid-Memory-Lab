"""Fixed-checkpoint transfer inputs; keep all sealed native sources unchanged."""

from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.locality_event_data import DATA_SOURCES as EVENT_SOURCES, registered_inputs as event_inputs
from scripts.run_controller_trial import prompt_for

PROTOCOL = ROOT / "configs/write_locality_adapter_transfer_plan.json"
PROTOCOL_SHA256 = "847adbb277ae6440293cce349ed536bcbc2b91df0574e8f1d8068cfc6986457f"
DESTINATION = Path("/media/990Pro/Code/SAS/datasets/memory_update/write-locality-adapter-transfer-v1")
MODELS = ("gdn_all_tokens", "gdn_question_only")
DATA_SOURCES = tuple(sorted(set(EVENT_SOURCES) | {
    "configs/write_locality_adapter_transfer_plan.json", "scripts/locality_transfer_data.py",
    "scripts/prepare_write_locality_adapter_transfer.py", "scripts/analyze_write_locality_event_paths.py",
    "scripts/run_write_locality_event_paths.py", "scripts/locality_event_runtime.py",
    "scripts/semantic_order_path_common.py",
}))


def read_plan():
    plan = json.loads(PROTOCOL.read_text())
    if (streaming_hash(PROTOCOL) != PROTOCOL_SHA256 or list(plan["models"]) != list(MODELS) or
        plan["profile"] != "fixed_checkpoint_transfer_at_native_write_locality_endpoints" or
        plan["training"] or plan["new_adapters"] or plan["model_key"] != "it" or not plan["local_files_only"] or
        plan["reader_policy"] != "full" or plan["counts"] != {
            "models": 2, "reported_pair_endpoint_readouts_per_model": 448,
            "unique_physical_queries_per_model": 384, "total_reported_bindings": 896,
            "total_physical_queries": 768, "pairs_per_cohort_stratum": 16,
            "independent_base_streams_per_cohort": 4}):
        raise ValueError("Registered timing-transfer scope changed")
    for name, policy in zip(MODELS, ("all", "question")):
        spec = plan["models"][name]
        expected = {"final_step": 432, "token_policy": policy, "mode": "gdn_only", "rank": 84,
                    "group_index": 3, "adapt_gdn_keys": True, "parameters": 696192}
        if any(spec[key] != value for key, value in expected.items()):
            raise ValueError("Registered frozen timing checkpoint changed")
    return plan


def validate_training_reference(spec):
    """Check pinned completed audits and actual artifacts, not rerun old training."""
    run = Path(spec["training_run"])
    audit_path = Path(spec["audit"])
    audit = json.loads(audit_path.read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    checkpoint = Path(spec["checkpoint"])
    metadata = json.loads((checkpoint.parent / "checkpoint.json").read_text())
    if (audit["status"] != "completed_and_audited" or audit["source_run"] != str(run) or
        streaming_hash(audit_path) != spec["audit_sha256"] or
        streaming_hash(audit_path.with_suffix(".source.py")) != audit["analysis_source_sha256"] or
        streaming_hash(run / "manifest.json") != spec["manifest_sha256"] or
        audit["manifest_sha256"] != spec["manifest_sha256"] or manifest["status"] != "completed" or manifest["smoke"] or
        streaming_hash(run / "plan.json") != spec["training_plan_sha256"] or
        manifest["plan_sha256"] != data_fingerprint(plan) or
        streaming_hash(checkpoint) != spec["checkpoint_sha256"] or
        streaming_hash(checkpoint.parent / "checkpoint.json") != spec["checkpoint_metadata_sha256"] or
        metadata["controller_sha256"] != spec["checkpoint_sha256"] or metadata["step"] != 432 or
        metadata["plan_sha256"] != data_fingerprint(plan) or metadata["arm"] != spec["training_arm"] or
        metadata["policy"] != spec["token_policy"] or metadata["optimized_input_tokens"] != 527808 or
        metadata["supervised_answer_tokens"] != 3456 or
        streaming_hash(checkpoint.parent / "optimizer.pt") != metadata["optimizer_sha256"] or
        manifest["arms"][spec["training_arm"]]["initial_state_sha256"] != spec["initial_state_sha256"] or
        manifest["arms"][spec["training_arm"]]["description"]["trainable_parameters"] != spec["parameters"]):
        raise ValueError("Pinned completed training/checkpoint reference changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / "sources" / name) != digest:
            raise ValueError("Actual/archived frozen training implementation changed")
    return {"audit_path": str(audit_path), "audit_sha256": spec["audit_sha256"],
            "manifest_sha256": spec["manifest_sha256"], "checkpoint_sha256": spec["checkpoint_sha256"],
            "checkpoint_state_sha256": metadata["state_sha256"], "packages": manifest["packages"],
            "model_path": manifest["model_path"], "reference_audit_reexecuted": False}


def checked_references(plan, *, reexecute_event=False):
    path = Path(plan["event_audit"])
    audit = json.loads(path.read_text())
    run = Path(audit["source_run"])
    manifest = json.loads((run / "manifest.json").read_text())
    if (streaming_hash(path) != plan["event_audit_sha256"] or audit["status"] != "completed_and_audited" or
        audit["readouts_audited"] != 1200 or audit["calibrations_audited"] != 448 or audit["pairs_audited"] != 96 or
        streaming_hash(path.with_suffix(".source.py")) != audit["analysis_source_sha256"] or
        streaming_hash(run / "manifest.json") != audit["manifest_sha256"] or manifest["status"] != "completed" or
        manifest["smoke"] or audit["weights_sha256"] != plan["native_loaded_weights_sha256"] or
        streaming_hash(Path(plan["event_dataset"]) / "manifest.json") != plan["event_dataset_manifest_sha256"] or
        audit["dataset_manifest_sha256"] != plan["event_dataset_manifest_sha256"] or
        audit["registry_sha256"] != plan["event_registry_sha256"]):
        raise ValueError("Complete native event diagnosis gate changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / "sources" / name) != digest:
            raise ValueError("Actual/archived event diagnosis implementation changed")
    for name, digest in manifest["results_sha256"].items():
        if streaming_hash(run / name) != digest:
            raise ValueError("Actual native event observations changed")
    if reexecute_event:
        from scripts.analyze_write_locality_event_paths import audit_run
        if audit_run(run) != audit:
            raise ValueError("Reexecuted full event audit differs from pinned report")
    training = {name: validate_training_reference(spec) for name, spec in plan["models"].items()}
    if (len({spec["initial_state_sha256"] for spec in plan["models"].values()}) != 1 or
        any(ref["packages"] != manifest["packages"] or ref["model_path"] != manifest["model_path"] for ref in training.values())):
        raise ValueError("Timing models are not paired on the same native runtime")
    return {"event": {"audit_path": str(path), "audit_sha256": streaming_hash(path),
                      "manifest_sha256": audit["manifest_sha256"], "run": str(run),
                      "weights_sha256": audit["weights_sha256"], "packages": manifest["packages"]},
            "training": training}


def deduplicate_queries(candidates):
    """Merge only identical natural sources/questions, and reject inconsistent truth."""
    physical, bindings = {}, []
    for candidate, binding in candidates:
        key = data_fingerprint([candidate["source_key"], candidate["question_ids_sha256"]])
        if key in physical and physical[key] != {"id": key, **candidate}:
            raise ValueError("Repeated source/question identity has conflicting tokens, truth or native reference")
        physical[key] = {"id": key, **candidate}
        bindings.append({**binding, "physical_id": key})
    if len({(b["pair_id"], b["side"], b["label"]) for b in bindings}) != len(bindings):
        raise ValueError("Duplicate registered readout binding")
    return list(physical.values()), bindings


def build_registry(plan, contexts, event_registry, references, runtime):
    raw = [json.loads(line) for line in (Path(references["event"]["run"]) / "readouts.jsonl").read_text().splitlines()]
    c_refs = {(r["pair_id"], r["condition"]): r for r in raw if r["kind"] == "full_C_baseline"}
    cases = {cohort: {row["case"]["id"]: row["case"] for row in context["data"]["validation"]}
             for cohort, context in contexts.items()}
    candidates, pairs = [], []
    for pair in event_registry["pairs"]:
        pairs.append({key: pair[key] for key in ("id", "cohort", "stratum", "case_id", "seed", "base_stream_id", "value_version", "active_accounts", "source_keys")})
        case = cases[pair["cohort"]][pair["case_id"]]
        for side, source in pair["source_keys"].items():
            prefix = event_registry["prefixes"][source]
            for label, query in pair["queries"].items():
                endpoint = query["endpoints"][side]
                answer = query["world_answers"][side]
                ledger, prompt = prompt_for(runtime, case, prefix["cell"], {"question": query["question"]})
                ids, ledger_ids = runtime.ids(prompt), runtime.ids(ledger)
                answer_ids = runtime.continuation_ids(prompt, answer)
                if (data_fingerprint(ids) != endpoint["prompt_ids_sha256"] or ids[:len(ledger_ids)] != ledger_ids or
                    data_fingerprint(ledger_ids) != prefix["ledger_ids_sha256"] or len(ids) != 2865 or
                    ids[len(ledger_ids):] != endpoint["question_ids"] or answer_ids != endpoint["world_answer_ids"][side]):
                    raise ValueError("Actual natural transfer prompt or own-world target changed")
                if label == "current_C":
                    original = c_refs[(pair["id"], side)]
                    score = original["world_scores"][side]
                    native = {"answer": score["answer"], "answer_nll": score["answer_nll"],
                              "greedy_exact": score["greedy_exact"], "generated": original["generated"],
                              "query_tokens": original["query_tokens"]}
                    descriptor = {"kind": "event_full_C", "pair_id": pair["id"], "side": side,
                                  "row_sha256": data_fingerprint(original)}
                else:
                    descriptor = endpoint["full_prefill_reference"]
                    original = contexts[pair["cohort"]]["raw"][tuple(descriptor["key"])]
                    native = {key: original[key] for key in ("answer", "answer_nll", "greedy_exact", "generated", "query_tokens")}
                    descriptor = {"kind": "native_full_screen", "cohort": pair["cohort"], **descriptor}
                if native["answer"] != answer or native["query_tokens"] != len(endpoint["question_ids"]):
                    raise ValueError("Native full-prefill baseline uses another natural-world answer")
                candidate = {"cohort": pair["cohort"], "source_key": source, "case_id": pair["case_id"],
                             "cell": prefix["cell"], "question": query["question"], "answer": answer,
                             "ledger_tokens": prefix["ledger_tokens"], "ledger_ids_sha256": prefix["ledger_ids_sha256"],
                             "prelude_tokens": prefix["prelude_tokens"], "prompt_tokens": endpoint["prompt_tokens"],
                             "prompt_ids_sha256": endpoint["prompt_ids_sha256"],
                             "question_ids_sha256": endpoint["question_ids_sha256"],
                             "question_tokens": len(endpoint["question_ids"]), "answer_ids": answer_ids,
                             "native_reference": descriptor, "native_full_readout": native}
                candidates.append((candidate, {"pair_id": pair["id"], "side": side, "label": label}))
    physical, bindings = deduplicate_queries(candidates)
    if len(physical) != 384 or len(bindings) != 448 or len(pairs) != 96:
        raise ValueError("Complete registered transfer cohort required")
    counts = Counter((pair["cohort"], pair["stratum"]) for pair in pairs)
    if len(counts) != 6 or set(counts.values()) != {16}:
        raise ValueError("A cohort or stratum was filtered")
    first = {}
    for pair in sorted(pairs, key=lambda p: p["id"]):
        first.setdefault((pair["cohort"], pair["stratum"]), pair["id"])
    return {"physical_queries": physical, "bindings": bindings, "pairs": pairs,
            "smoke_pair_ids": list(first.values()), "counts": plan["counts"]}


def selected_inputs(registry, smoke):
    pairs = [p for p in registry["pairs"] if not smoke or p["id"] in registry["smoke_pair_ids"]]
    keys = {p["id"] for p in pairs}
    bindings = [b for b in registry["bindings"] if b["pair_id"] in keys]
    physical_ids = {b["physical_id"] for b in bindings}
    physical = [q for q in registry["physical_queries"] if q["id"] in physical_ids]
    return pairs, bindings, physical


def safety_samples(pairs, bindings, physical):
    groups = defaultdict(list)
    for pair in pairs:
        groups[(pair["cohort"], pair["stratum"])].append(pair["id"])
    selected = {key for values in groups.values() for key in (min(values), max(values))}
    prefix_keys = {source for pair in pairs if pair["id"] in selected for source in pair["source_keys"].values()}
    query_ids = {b["physical_id"] for b in bindings if b["pair_id"] in selected and b["label"] == "preservation"}
    if {q["source_key"] for q in physical if q["id"] in query_ids} != prefix_keys:
        raise ValueError("Safety samples missed a prescribed source prefix")
    return sorted(prefix_keys), sorted(query_ids)


def registered_inputs(path=DESTINATION):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    plan = read_plan()
    references = checked_references(plan)
    registered, contexts, event_registry = event_inputs(Path(plan["event_dataset"]))
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    registry = build_registry(plan, contexts, event_registry, references, tokenizer_runtime(registered["model_path"]))
    if (manifest["status"] != "registered_before_inference" or manifest["model_inference"] or manifest["training"] or
        manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or manifest["references"] != references or
        manifest["registry_sha256"] != data_fingerprint(registry) or manifest["model_path"] != registered["model_path"] or
        manifest["event_dataset_manifest_sha256"] != plan["event_dataset_manifest_sha256"] or
        not manifest["complete_event_audit_reexecuted_at_registration"] or
        manifest["counts"] != plan["counts"] or set(manifest["source_sha256"]) != set(DATA_SOURCES) or
        json.loads((path / "registry.json").read_text()) != registry or json.loads((path / "plan.json").read_text()) != plan or
        datetime.fromisoformat(plan["registered_utc"]) >= datetime.fromisoformat(manifest["registered_utc"])):
        raise ValueError("Transfer registration, actual sources, deduplication or targets changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / "sources" / name) != digest:
            raise ValueError("Actual/archived transfer data provider changed")
    return manifest, contexts, registry
