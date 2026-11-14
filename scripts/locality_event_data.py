"""Frozen inputs for the registered semantic-tail cache-source diagnosis.

This provider adds no model operations and leaves both completed native engines
unchanged. All queries originate from ledger-only sources; companion questions
have separate worlds rather than a best-of-two target.
"""

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.locality_replication import (
    RUNTIME_SOURCES, discovery_reference, primary_recurrence,
    read_plan as replica_plan, registered_dataset as replica_dataset,
)
from sas_memory.tasks import INSTRUCTIONS, Operation, execute
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import registered_dataset as discovery_dataset, streaming_hash
from scripts.run_controller_trial import prompt_for

STORAGE = Path("/media/990Pro/Code/SAS")
DESTINATION = STORAGE / "datasets/memory_update/fixed-length-write-locality-event-paths-v1"
REPLICA_AUDIT = STORAGE / "artifacts/memory_update/write-locality-replication-20261114.json"
PROTOCOLS = (
    "configs/semantic_write_locality_event_paths_plan.json",
    "configs/semantic_write_locality_event_paths_addendum.json",
    "configs/semantic_write_locality_event_paths_commit_guard.json",
)
DATA_SOURCES = tuple(sorted(set(RUNTIME_SOURCES) | set(PROTOCOLS) | {
    "scripts/locality_event_data.py", "scripts/prepare_write_locality_event_paths.py",
    "sas_memory/kv_readout.py",
}))
CONDITIONS = (
    {"recurrent_conv": "left", "kv": "left"},
    {"recurrent_conv": "right", "kv": "right"},
    {"recurrent_conv": "left", "kv": "right"},
    {"recurrent_conv": "right", "kv": "left"},
)
STRATA = (
    {"id": "unrelated_B_at32", "active_accounts": 32, "axis": "A", "fixed_other_bit": 1,
     "role": "protected_current", "left_cell": "A0_C1", "right_cell": "A1_C1"},
    {"id": "initial_A_at128", "active_accounts": 128, "axis": "A", "fixed_other_bit": 1,
     "role": "target_initial", "left_cell": "A0_C1", "right_cell": "A1_C1"},
    {"id": "unrelated_C_on_initial_A_at128", "active_accounts": 128, "axis": "C", "fixed_other_bit": 1,
     "role": "target_initial", "left_cell": "A1_C0", "right_cell": "A1_C1"},
)
COUNTS = {
    "datasets": 2, "strata": 3, "pairs": 96, "pairs_per_stratum_per_dataset": 16,
    "unique_natural_prefixes": 160, "unique_natural_prefixes_per_dataset": 80,
    "primary_preservation_readouts": 384, "A_current_companion_readouts": 384,
    "C_current_companion_readouts": 128, "oracle_history_readouts": 192,
    "new_full_prefill_C_current_baselines": 64,
    "cached_natural_endpoint_calibrations_including_companions": 448,
    "prescribed_self_and_all_path_replays": 48,
}


def read_protocols():
    plan, addendum, guard = [json.loads((ROOT / name).read_text()) for name in PROTOCOLS]
    if (streaming_hash(ROOT / plan["replication_protocol"]) != plan["replication_protocol_sha256"] or
        streaming_hash(ROOT / addendum["base_protocol"]) != addendum["base_protocol_sha256"] or
        streaming_hash(ROOT / guard["base_protocol"]) != guard["base_protocol_sha256"] or
        streaming_hash(ROOT / guard["A_current_addendum"]) != guard["A_current_addendum_sha256"] or
        plan["strata"] != list(STRATA) or plan["conditions"] != list(CONDITIONS) or
        plan["training"] or plan["adapters"] or plan["model_key"] != "it" or not plan["local_files_only"] or
        guard["query_template"] != "What is the current balance of account {other_entity}?" or
        any(guard["combined_counts"][key] != COUNTS[key] for key in guard["combined_counts"])):
        raise ValueError("Registered event paths, companion worlds or commit guards changed")
    return {name: value for name, value in zip(PROTOCOLS, (plan, addendum, guard))}


def checked_references(replica_audit=REPLICA_AUDIT, *, reexecute=False):
    """Verify actual audits and their observations; optionally rerun both auditors."""
    plan = replica_plan()
    discovery = discovery_reference(plan, reexecute=reexecute)
    path = Path(replica_audit)
    replica = json.loads(path.read_text())
    trial = Path(replica["source_run"])
    manifest = json.loads((trial / "manifest.json").read_text())
    recurrence = primary_recurrence(replica["metrics"], plan)
    if (replica["status"] != "completed_and_audited" or replica["records_audited"] != 576 or
        replica["primary_recurrence"] != recurrence or not recurrence["event_path_gate_passed"] or
        replica["weights_sha256"] != discovery["weights_sha256"] or
        replica["protocol_sha256"] != streaming_hash(ROOT / "configs/semantic_write_locality_replication_plan.json") or
        replica["manifest_sha256"] != streaming_hash(trial / "manifest.json") or manifest["status"] != "completed" or
        manifest["smoke"] or streaming_hash(trial / "validation.native.jsonl") != manifest["result_sha256"] or
        streaming_hash(path.with_suffix(".source.py")) != replica["analysis_source_sha256"]):
        raise ValueError("Complete fresh-seed replication gate or actual observations invalid")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(trial / "sources" / name) != digest:
            raise ValueError("Replica implementation or source archive changed")
    if reexecute:
        from scripts.analyze_write_locality_replication import audit_run
        if audit_run(trial) != replica:
            raise ValueError("Reexecuted full replica audit disagrees with saved report")
    result = {}
    for cohort, source, report in (("discovery", Path(plan["discovery_audit"]), discovery),
                                   ("replica", path, replica)):
        run = Path(report["source_run"])
        result[cohort] = {"audit_path": str(source), "audit_sha256": streaming_hash(source),
                          "run": str(run), "manifest_sha256": streaming_hash(run / "manifest.json"),
                          "raw_sha256": streaming_hash(run / "validation.native.jsonl"),
                          "weights_sha256": report["weights_sha256"], "records_audited": report["records_audited"]}
    return result, recurrence


def load_contexts(references, protocols):
    plan = protocols[PROTOCOLS[0]]
    contexts = {}
    for name, loader in (("discovery", discovery_dataset), ("replica", replica_dataset)):
        manifest, data, registry = loader(Path(plan["datasets"][name]))
        report = json.loads(Path(references[name]["audit_path"]).read_text())
        if (report["dataset_manifest_sha256"] != streaming_hash(Path(plan["datasets"][name]) / "manifest.json") or
            report["data_sha256"] != data_fingerprint(data) or report["registry_sha256"] != data_fingerprint(registry)):
            raise ValueError("Natural data do not belong to the audited native observations")
        raw = [json.loads(line) for line in (Path(references[name]["run"]) / "validation.native.jsonl").read_text().splitlines()]
        indexed = {(row["case_id"], row["variant"], row["role"]): row for row in raw}
        registrations = {(row["case_id"], row["variant"], row["role"]): row for row in registry["validation"]}
        if len(indexed) != 576 or indexed.keys() != registrations.keys():
            raise ValueError("Natural source query/reference coverage differs")
        contexts[name] = {"manifest": manifest, "data": data, "registry": registry,
                          "raw": indexed, "registrations": registrations}
    if contexts["discovery"]["manifest"]["model_path"] != contexts["replica"]["manifest"]["model_path"]:
        raise ValueError("Event sources use different local models")
    return contexts


def cases_for_stratum(data, stratum):
    cases = {row["case"]["id"]: row["case"] for row in data["validation"]
             if row["case"]["active_accounts"] == stratum["active_accounts"]}
    if len(cases) != 16 or len({case["base_stream_id"] for case in cases.values()}) != 4:
        raise ValueError("Every stratum needs all four streams and all four values")
    return [cases[key] for key in sorted(cases)]


def world_truth(case, cell):
    return execute([Operation(**event) for event in case["prefix"] + case["variants"][cell]["tail"]])


def paired_queries(case, stratum):
    cells = {side: stratum[side + "_cell"] for side in ("left", "right")}
    truth = {side: world_truth(case, cell) for side, cell in cells.items()}
    role = stratum["role"]
    original = {side: next(q for q in case["variants"][cell]["queries"] if q["role"] == role)
                for side, cell in cells.items()}
    queries = {"preservation": original}
    for label, entity in (("current_A", case["target_entity"]), ("current_C", case["other_entity"])):
        if label == "current_C" and stratum["axis"] != "C":
            continue
        queries[label] = {side: {"role": "target_current" if label == "current_A" else "other_current",
                                "entity": entity, "version": "current",
                                "question": f"What is the current balance of account {entity}?",
                                "answer": str(truth[side]["current"][entity])} for side in cells}
    if (original["left"] != original["right"] or
        (queries["current_A"]["left"]["answer"] != queries["current_A"]["right"]["answer"]) != (stratum["axis"] == "A") or
        (stratum["axis"] == "C" and queries["current_C"]["left"]["answer"] == queries["current_C"]["right"]["answer"])):
        raise ValueError("Invariant query or legitimate update worlds invalid")
    return queries


def check_source_alignment(left_ids, right_ids, prefix, stratum):
    if len(left_ids) != len(right_ids) or len(left_ids) != prefix["ledger_tokens"]:
        raise ValueError("Source ledger lengths differ")
    tail_record = 160 if stratum["axis"] == "A" else 161
    allowed = set(prefix["records"][tail_record]["value_tokens"])
    changed = {i for i, (left, right) in enumerate(zip(left_ids, right_ids)) if left != right}
    if not changed or not changed <= allowed:
        raise ValueError("Source pair changed outside its registered tail value digits")
    support = prefix["records"][1 if stratum["role"] == "protected_current" else 0]
    start, end = support["token_start"], support["token_end"]
    if left_ids[start:end] != right_ids[start:end] or end > min(allowed):
        raise ValueError("Supporting evidence or its causal age changed")
    return {"changed_token_positions": sorted(changed), "support_record": 1 if stratum["role"] == "protected_current" else 0,
            "support_ids_sha256": data_fingerprint(left_ids[start:end]), "support_span": [start, end]}


def make_registry(contexts, runtime):
    from sas_memory.kv_readout import evidence_token_keep
    pairs, prefixes = [], {}
    for cohort, context in contexts.items():
        for stratum in STRATA:
            for case in cases_for_stratum(context["data"], stratum):
                cells = {side: stratum[side + "_cell"] for side in ("left", "right")}
                queries = paired_queries(case, stratum)
                ledger_ids, records, prompts = {}, {}, {}
                sources = {}
                registered_queries = {}
                for side, cell in cells.items():
                    ledger, prompt = prompt_for(runtime, case, cell, queries["preservation"][side])
                    prompts[side] = (ledger, prompt)
                    ledger_ids[side] = runtime.ids(ledger)
                    local_key = case["id"] + "/" + cell
                    key = cohort + "/" + local_key
                    records[side] = context["registry"]["prefixes"][local_key]
                    if data_fingerprint(ledger_ids[side]) != records[side]["ledger_ids_sha256"]:
                        raise ValueError("Natural source registry no longer matches ledger-only tokens")
                    sources[side] = key
                    source = {"cohort": cohort, "case_id": case["id"], "cell": cell, **records[side]}
                    if key in prefixes and prefixes[key] != source:
                        raise ValueError("Repeated source identity changed")
                    prefixes[key] = source
                if records["left"]["records"] != records["right"]["records"]:
                    raise ValueError("Source record boundaries differ")
                alignment = check_source_alignment(ledger_ids["left"], ledger_ids["right"], records["left"], stratum)
                for label, worlds in queries.items():
                    endpoints = {}
                    for side, query in worlds.items():
                        cell = cells[side]
                        ledger, prompt = prompt_for(runtime, case, cell, query)
                        ids = runtime.ids(prompt)
                        boundary = len(ledger_ids[side])
                        answer_ids = {world: runtime.continuation_ids(prompt, other["answer"]) for world, other in worlds.items()}
                        if (ids[:boundary] != ledger_ids[side] or len(ids) != 2865 or
                            any(len(value) != 3 for value in answer_ids.values())):
                            raise ValueError("Ledger/question boundary or frozen three-digit targets changed")
                        registration = {"prompt_tokens": len(ids), "prompt_ids_sha256": data_fingerprint(ids),
                                        "question_ids": ids[boundary:], "question_ids_sha256": data_fingerprint(ids[boundary:]),
                                        "world_answer_ids": answer_ids}
                        if label != "current_C":
                            key = (case["id"], cell, query["role"])
                            original = context["registrations"][key]
                            row = context["raw"][key]
                            if (registration["prompt_ids_sha256"] != original["prompt_ids_sha256"] or
                                answer_ids[side] != original["answer_ids"] or row["answer"] != query["answer"] or
                                row["registration"] != original):
                                raise ValueError("Natural endpoint no longer reproduces the complete frozen reference")
                            registration["full_prefill_reference"] = {"key": list(key), "row_sha256": data_fingerprint(row)}
                        else:
                            registration["full_prefill_reference"] = None
                        endpoints[side] = registration
                    if endpoints["left"]["question_ids"] != endpoints["right"]["question_ids"]:
                        raise ValueError("Query suffix depends on source world")
                    registered_queries[label] = {"question": worlds["left"]["question"],
                                                 "world_answers": {side: query["answer"] for side, query in worlds.items()},
                                                 "endpoints": endpoints}
                keep = {}
                for side, cell in cells.items():
                    ledger, prompt = prompts[side]
                    events = [Operation(**event).text() for event in case["prefix"] + case["variants"][cell]["tail"]]
                    mask = evidence_token_keep(runtime.tokenizer, prompt, ledger, runtime.prompt_start + INSTRUCTIONS,
                                               events, [alignment["support_record"]])
                    expected = [i < records[side]["prelude_tokens"] or alignment["support_span"][0] <= i < alignment["support_span"][1]
                                for i in range(len(ledger_ids[side]))]
                    if mask != expected:
                        raise ValueError("Oracle mask is not exactly prelude plus one original supporting record")
                    keep[side] = mask
                if keep["left"] != keep["right"]:
                    raise ValueError("Oracle evidence access differs between source worlds")
                pairs.append({"id": f"{cohort}/{stratum['id']}/{case['id']}", "cohort": cohort,
                              "stratum": stratum["id"], "case_id": case["id"],
                              **{key: case[key] for key in ("seed", "value_version", "base_stream_id", "active_accounts")},
                              "source_keys": sources, "alignment": alignment, "queries": registered_queries,
                              "oracle": {"kept_positions": [i for i, flag in enumerate(keep["left"]) if flag],
                                         "history_keep_sha256": data_fingerprint(keep["left"]),
                                         "ledger_tokens": len(keep["left"])}})
    if len(pairs) != 96 or len(prefixes) != 160 or len({p["id"] for p in pairs}) != 96:
        raise ValueError("Incomplete event-path cohort")
    for cohort in contexts:
        if sum(p["cohort"] == cohort for p in prefixes.values()) != 80:
            raise ValueError("Incomplete natural prefix source set")
    groups = Counter((p["cohort"], p["stratum"]) for p in pairs)
    if len(groups) != 6 or set(groups.values()) != {16}:
        raise ValueError("All-case strata missing or filtered")
    replay_pairs = []
    for group in groups:
        members = sorted(p["id"] for p in pairs if (p["cohort"], p["stratum"]) == group)
        replay_pairs.extend((members[0], members[-1]))
    counts = {**COUNTS}
    actual = {"primary_preservation_readouts": len(pairs) * 4, "A_current_companion_readouts": len(pairs) * 4,
              "C_current_companion_readouts": sum("current_C" in p["queries"] for p in pairs) * 4,
              "oracle_history_readouts": len(pairs) * 2,
              "new_full_prefill_C_current_baselines": sum("current_C" in p["queries"] for p in pairs) * 2,
              "cached_natural_endpoint_calibrations_including_companions": sum(len(p["queries"]) for p in pairs) * 2,
              "prescribed_self_and_all_path_replays": len(replay_pairs) * 4}
    if any(counts[key] != value for key, value in actual.items()):
        raise ValueError("Companion, calibration, oracle or replay count differs from registration")
    return {"prefixes": prefixes, "pairs": pairs, "replay_pair_ids": replay_pairs,
            "conditions": list(CONDITIONS), "counts": counts}


def registered_inputs(path=DESTINATION):
    """Reconstruct all semantic/token bindings without running a model."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    protocols = read_protocols()
    references, recurrence = checked_references(Path(manifest["replica_audit_path"]))
    contexts = load_contexts(references, protocols)
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    registry = make_registry(contexts, tokenizer_runtime(manifest["model_path"]))
    if (manifest["status"] != "registered_before_inference" or manifest["model_inference"] or manifest["training"] or
        manifest["references"] != references or manifest["primary_recurrence"] != recurrence or
        not manifest["complete_reference_audits_reexecuted_at_registration"] or
        manifest["model_path"] != contexts["replica"]["manifest"]["model_path"] or
        manifest["counts"] != registry["counts"] or manifest["registry_sha256"] != data_fingerprint(registry) or
        manifest["source_sha256"].keys() != set(DATA_SOURCES) or
        manifest["protocol_sha256"] != {name: streaming_hash(ROOT / name) for name in PROTOCOLS} or
        json.loads((path / "protocols.json").read_text()) != protocols or
        json.loads((path / "registry.json").read_text()) != registry):
        raise ValueError("Event data, source references, gate or complete registry changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / "sources" / name) != digest:
            raise ValueError("Event registration implementation changed")
    for cohort, descriptor in references.items():
        if streaming_hash(path / (cohort + "_audit.json")) != descriptor["audit_sha256"]:
            raise ValueError("Archived complete native reference audit changed")
    return manifest, contexts, registry
