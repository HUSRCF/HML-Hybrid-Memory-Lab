"""Counterbalanced overwrite histories with identical final executable states."""

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.tasks import INSTRUCTIONS, Operation, execute, render
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.history_no_repeat_data import identity, model_artifacts
from scripts.run_semantic_history_no_repeat import RUNTIME_SOURCES as NATIVE_SOURCES

PROTOCOL = ROOT / "configs/semantic_overwrite_history_plan.json"
PROTOCOL_SHA256 = "a6f66f16f8e6f58ff140739727dbf2c2ee164749409b5ea4286d0a5eb66c247d"
DESTINATION = Path("/media/990Pro/Code/SAS/datasets/memory_update/counterbalanced-overwrite-history-v1")
CELLS = ("XX", "XY", "YX", "YY")
ROLES = ("target_initial", "target_current", "other_current")
CONTRASTS = {"final_X": ("YX", "XX"), "final_Y": ("XY", "YY")}
DATA_SOURCES = tuple(sorted(set(NATIVE_SOURCES) | {
    "configs/semantic_overwrite_history_plan.json", "scripts/overwrite_history_data.py",
    "scripts/prepare_semantic_overwrite_history.py",
}))


def read_plan():
    plan = json.loads(PROTOCOL.read_text())
    if (streaming_hash(PROTOCOL) != PROTOCOL_SHA256 or
        plan["profile"] != "same_final_state_counterbalanced_overwrite_histories" or
        plan["training"] or plan["adapters"] or not plan["local_files_only"] or plan["model_key"] != "it" or
        plan["reader_policy"] != "full" or plan["cohorts"] != {
            "discovery": [97600, 97601, 97602, 97603], "replica": [97700, 97701, 97702, 97703]} or
        plan["grid"] != {"active_accounts": 128, "prefix_records": 160, "tail_records": 2,
            "value_versions": 4, "cells": list(CELLS), "query_roles": list(ROLES), "case_values": 32,
            "natural_prefixes": 128, "full_prefill_queries": 384, "queries_per_cohort": 192,
            "same_final_value_pairs": 64, "pairs_per_cohort": 32, "independent_base_streams_per_cohort": 4,
            "maximum_prompt_tokens": 4096, "no_truncation": True} or
        plan["construction"]["paired_contrasts"] != {
            name: {"reference_novel": pair[0], "candidate_repeat": pair[1]} for name, pair in CONTRASTS.items()}):
        raise ValueError("Registered counterbalanced overwrite-history scope changed")
    return plan


def reference_gate(plan, reexecute=False):
    path = Path(plan["reference"]["audit"])
    audit = json.loads(path.read_text())
    run = Path(audit["source_run"])
    manifest = json.loads((run / "manifest.json").read_text())
    if (streaming_hash(path) != plan["reference"]["audit_sha256"] or audit["status"] != "completed_and_audited" or
        audit["records_audited"] != 384 or audit["metrics"]["primary_signal"]["present_in_both_cohorts"] or
        audit["weights_sha256"] != plan["native_loaded_weights_sha256"] or
        streaming_hash(path.with_suffix(".source.py")) != audit["analysis_source_sha256"] or
        streaming_hash(run / "manifest.json") != audit["manifest_sha256"] or manifest["status"] != "completed" or
        manifest["smoke"] or manifest["weights_before"] != manifest["weights_after"] or
        streaming_hash(run / "validation.native.jsonl") != manifest["result_sha256"]):
        raise ValueError("Prior native evidence/failed historical primary changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / "sources" / name) != digest:
            raise ValueError("Actual/archived prior native implementation changed")
    if reexecute:
        from scripts.analyze_semantic_history_no_repeat import audit_run
        if audit_run(run) != audit:
            raise ValueError("Complete previous native audit does not reproduce")
    return {"audit_path": str(path), "audit_sha256": streaming_hash(path),
        "manifest_sha256": audit["manifest_sha256"], "weights_sha256": audit["weights_sha256"],
        "packages": manifest["packages"], "model_path": manifest["model_path"], "previous_history_primary_passed": False}


def events_for(case, cell):
    prefix = list(case["prefix_template"])
    prefix[3] = {**prefix[3], "value": case["variants"][cell]["early_value"]}
    return prefix + case["variants"][cell]["tail"]


def prompt_for(runtime, case, cell, query):
    ledger = runtime.prompt_start + INSTRUCTIONS + render(events_for(case, cell))
    return ledger, ledger + runtime.query_suffix(query["question"])


def build_data(plan):
    cases, selectors = [], []
    for cohort, seeds in plan["cohorts"].items():
        for seed in seeds:
            entities = [f"E{number}" for number in range(700, 900)]
            random.Random(seed).shuffle(entities)
            a, b, c = entities[:3]
            all_values = random.Random(seed + 410009).sample(range(100, 1000), 4 * 162)
            rng, distractors = random.Random(seed + 310003 + 128), []
            while len(distractors) < 156:
                cycle = entities[3:128].copy()
                rng.shuffle(cycle)
                distractors.extend(cycle)
            targets = [a, b, c, a] + distractors[:156]
            for version in range(4):
                values = all_values[version * 162:(version + 1) * 162]
                template = [asdict(Operation("SET", target, value)) for target, value in zip(targets, values)]
                case = {"id": f"overwrite-history-s{seed}-a128-v{version}",
                    "base_stream_id": f"overwrite-history-s{seed}", "base_case_id": f"overwrite-history-s{seed}-a128",
                    "family": "counterbalanced_overwrite_history", "cohort": cohort, "seed": seed,
                    "active_accounts": 128, "value_version": version, "target_entity": a,
                    "protected_entity": b, "other_entity": c, "X": values[3], "Y": values[160],
                    "prefix_template": template, "variants": {}}
                for cell in CELLS:
                    case["variants"][cell] = {"early_value": case[cell[0]],
                        "tail": [asdict(Operation("SET", a, case[cell[1]])), asdict(Operation("SET", c, values[161]))]}
                    truth = execute([Operation(**event) for event in events_for(case, cell)])
                    queries = []
                    for role, entity, kind in ((ROLES[0], a, "initial"), (ROLES[1], a, "current"), (ROLES[2], c, "current")):
                        question = (f"What was the initial balance of account {entity}?" if kind == "initial"
                                    else f"What is the current balance of account {entity}?")
                        queries.append({"role": role, "entity": entity, "version": kind,
                            "question": question, "answer": str(truth[kind][entity])})
                    case["variants"][cell]["queries"] = queries
                    selectors.extend({"case_id": case["id"], "variant": cell, "role": role} for role in ROLES)
                cases.append(case)
    return {"cases": cases, "validation": selectors}


def evaluated_members(data, smoke=False):
    cases = {c["id"]: c for c in data["cases"]}
    first = {cohort: min(c["id"] for c in data["cases"] if c["cohort"] == cohort) for cohort in ("discovery", "replica")}
    members = []
    for selector in data["validation"]:
        case = cases[selector["case_id"]]
        if smoke and case["id"] != first[case["cohort"]]:
            continue
        cell = selector["variant"]
        query = next(q for q in case["variants"][cell]["queries"] if q["role"] == selector["role"])
        members.append({"case": case, "variant": cell, "query": query})
    return members


def safety_samples(members):
    grouped = defaultdict(set)
    for row in members:
        grouped[row["case"]["cohort"]].add(row["case"]["id"])
    selected = {key for values in grouped.values() for key in (min(values), max(values))}
    return sorted(identity(row) for row in members if row["case"]["id"] in selected and row["query"]["role"] == ROLES[1])


def validate_data(data, plan):
    if data != build_data(plan):
        raise ValueError("Frozen factorial schedule, selectors, identities or labels changed")
    members = evaluated_members(data)
    if (len(data["cases"]) != 32 or len(members) != 384 or len({tuple(identity(r)) for r in members}) != 384 or
        Counter(c["cohort"] for c in data["cases"]) != {"discovery": 16, "replica": 16}):
        raise ValueError("Incomplete compact case/query registry")
    siblings = defaultdict(list)
    for case in data["cases"]:
        truths = {}
        a, b, c = (case[key] for key in ("target_entity", "protected_entity", "other_entity"))
        for cell in CELLS:
            events = events_for(case, cell)
            if (len(events) != 162 or len({e["target"] for e in events}) != 128 or
                [e["target"] for e in events[:4]] != [a, b, c, a] or
                any(e["target"] in {a, b, c} for e in events[4:160]) or
                [e["target"] for e in events[160:]] != [a, c] or
                sum(e["value"] == events[0]["value"] for e in events) != 1 or
                len({events[0]["value"], case["X"], case["Y"], events[161]["value"]}) != 4 or
                events[3]["value"] != case[cell[0]] or events[160]["value"] != case[cell[1]]):
                raise ValueError("Counterbalance, evidence count, write scope or live-account count changed")
            truths[cell] = execute([Operation(**event) for event in events])
            for query in case["variants"][cell]["queries"]:
                if query["answer"] != str(truths[cell][query["version"]][query["entity"]]):
                    raise ValueError("Own-world answer differs from executable truth")
                siblings[case["base_case_id"], cell, query["role"]].append(query["answer"])
        for novel, repeat in CONTRASTS.values():
            if truths[novel] != truths[repeat]:
                raise ValueError("Same-final comparison changes the complete executable state")
        if truths["XX"]["current"][a] == truths["XY"]["current"][a] or truths["YX"]["current"][a] == truths["YY"]["current"][a]:
            raise ValueError("Legitimate final-value sensitivity was lost")
    if any(len(answers) != 4 or len(set(answers)) != 4 for answers in siblings.values()):
        raise ValueError("Four-value query siblings are incomplete")
    return {"examples": 384, "value_cases": 32, "same_final_pairs": 64, "natural_prefixes": 128,
            "base_streams": 8, "cohorts": 2, "active_accounts": 128, "four_value_query_groups": len(siblings)}


def token_registry(data, plan, runtime):
    prefixes, registrations, groups = {}, [], defaultdict(dict)
    sibling_signatures, case_signatures = {}, {}
    for example in evaluated_members(data):
        case, cell, query = example["case"], example["variant"], example["query"]
        ledger, prompt = prompt_for(runtime, case, cell, query)
        prefix_key = case["id"] + "/" + cell
        if prefix_key not in prefixes:
            encoded = runtime.tokenizer(ledger, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
            chars = {char: i for i, (left, right) in enumerate(offsets) for char in range(left, right)}
            matches = list(re.finditer(r"\b\d{3}\b", ledger))
            events = events_for(case, cell)
            if len(matches) != 162 or sum(int(m.group()) == events[0]["value"] for m in matches) != 1:
                raise ValueError("Initial numeric evidence repeats or numeric records are missing")
            cursor, records = len(runtime.prompt_start + INSTRUCTIONS), []
            for event, match in zip(events, matches):
                end = cursor + len(Operation(**event).text())
                left, right = chars[cursor], chars[end - 1] + 1
                positions = [chars[match.start() + digit] for digit in range(3)]
                if (int(match.group()) != event["value"] or offsets[left][0] != cursor or offsets[right - 1][1] != end or
                    [tuple(offsets[i]) for i in positions] != [(match.start() + j, match.start() + j + 1) for j in range(3)]):
                    raise ValueError("Actual record or individual value-digit boundaries are not aligned")
                records.append({"char_start": cursor, "char_end": end, "token_start": left,
                    "token_end": right, "value_tokens": positions})
                cursor = end
            if cursor != len(ledger):
                raise ValueError("Record boundaries do not cover the complete ledger")
            prefixes[prefix_key] = {"ledger_tokens": len(ids), "ledger_ids_sha256": data_fingerprint(ids),
                "prelude_tokens": len(runtime.ids(runtime.prompt_start + INSTRUCTIONS)), "records": records,
                "initial_numeric_occurrences": 1, "initial_value_record": 0}
        prefix = prefixes[prefix_key]
        ids, ledger_ids = runtime.ids(prompt), runtime.ids(ledger)
        if (ids[:len(ledger_ids)] != ledger_ids or len(ids) > plan["grid"]["maximum_prompt_tokens"] or
            data_fingerprint(ledger_ids) != prefix["ledger_ids_sha256"]):
            raise ValueError("Actual source prefix, question boundary or budget changed")
        positions = {i for record in prefix["records"] for i in record["value_tokens"]}
        registration = {"case_id": case["id"], "variant": cell, "role": query["role"],
            **{key: case[key] for key in ("cohort", "base_stream_id", "base_case_id", "seed", "active_accounts", "value_version")},
            "prefix_key": prefix_key, "ledger_tokens": len(ledger_ids), "prompt_tokens": len(ids),
            "prompt_ids_sha256": data_fingerprint(ids), "question_ids": ids[len(ledger_ids):],
            "question_ids_sha256": data_fingerprint(ids[len(ledger_ids):]), "answer": query["answer"],
            "answer_ids": runtime.continuation_ids(prompt, query["answer"]),
            "normalized_prompt_sha256": data_fingerprint([-1 if i in positions else t for i, t in enumerate(ids)])}
        if len(registration["answer_ids"]) != 3:
            raise ValueError("Answer is not the fixed three-digit tokenization")
        signature = registration["normalized_prompt_sha256"], tuple(sorted(positions))
        if sibling_signatures.setdefault((case["base_case_id"], cell, query["role"]), signature) != signature:
            raise ValueError("Value siblings differ outside numeric values")
        signature = registration["ledger_tokens"], registration["prompt_tokens"], registration["question_ids_sha256"], prefix["records"]
        if case_signatures.setdefault((case["id"], query["role"]), signature) != signature:
            raise ValueError("Factorial cells changed record/query positions or prompt lengths")
        groups[case["id"], query["role"]][cell] = ids, prefix, registration
        registrations.append(registration)
    for group in groups.values():
        baseline, prefix, _ = group["XX"]
        for cell, (ids, other, _) in group.items():
            allowed = set(prefix["records"][3]["value_tokens"] if cell[0] != "X" else ())
            allowed.update(prefix["records"][160]["value_tokens"] if cell[1] != "X" else ())
            changed = {i for i, (a, b) in enumerate(zip(baseline, ids)) if a != b}
            if len(baseline) != len(ids) or not changed <= allowed or bool(changed) != bool(allowed):
                raise ValueError("Factorial manipulation changed an unregistered token")
        for novel, repeat in CONTRASTS.values():
            left, lp, lr = group[novel]
            right, rp, rr = group[repeat]
            changed = {i for i, (a, b) in enumerate(zip(left, right)) if a != b}
            if (not changed or not changed <= set(lp["records"][3]["value_tokens"]) or lp["records"] != rp["records"] or
                lr["question_ids"] != rr["question_ids"] or lr["answer_ids"] != rr["answer_ids"] or lr["answer"] != rr["answer"]):
                raise ValueError("Same-final pair changed query/answer tokens or final evidence")
        for early in "XY":
            left, lp, lr = group[early + "X"]
            right, rp, rr = group[early + "Y"]
            changed = {i for i, (a, b) in enumerate(zip(left, right)) if a != b}
            if not changed or not changed <= set(lp["records"][160]["value_tokens"]):
                raise ValueError("Fixed-early final toggle changed an unregistered token")
            if lr["role"] == ROLES[1] and (lr["answer"] == rr["answer"] or lr["answer_ids"] == rr["answer_ids"]):
                raise ValueError("Final-value change lost its current-answer sensitivity")
    if len(prefixes) != 128 or len(registrations) != 384:
        raise ValueError("Complete actual-token factorial registry required")
    return {"prefixes": prefixes, "validation": registrations}


def registered_inputs(path=DESTINATION):
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    plan = read_plan()
    data = json.loads((path / "data.json").read_text())
    registry = json.loads((path / "registry.json").read_text())
    model = json.loads((ROOT / plan["model_config"]).read_text())[plan["model_key"]]
    if (manifest["status"] != "registered_before_inference" or manifest["training"] or manifest["model_inference"] or
        not manifest["complete_reference_audit_reexecuted_at_registration"] or manifest["model_path"] != model or
        manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or manifest["reference_gate"] != reference_gate(plan) or
        manifest["counts"] != validate_data(data, plan) or manifest["data_sha256"] != data_fingerprint(data) or
        manifest["registry_sha256"] != data_fingerprint(registry) or json.loads((path / "plan.json").read_text()) != plan or
        set(manifest["source_sha256"]) != set(DATA_SOURCES) or
        manifest["evaluation_keys_sha256"] != data_fingerprint([identity(row) for row in evaluated_members(data)]) or
        datetime.fromisoformat(plan["registered_utc"]) >= datetime.fromisoformat(manifest["registered_utc"])):
        raise ValueError("Counterbalanced data/protocol/reference registration changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / "sources" / name) != digest:
            raise ValueError("Actual/archived counterbalanced data implementation changed")
    tokenizer, checkpoint = model_artifacts(model)
    if manifest["tokenizer_sha256"] != tokenizer or manifest["checkpoint_sha256"] != checkpoint:
        raise ValueError("Actual tokenizer/config/checkpoint changed")
    if token_registry(data, plan, tokenizer_runtime(model)) != registry:
        raise ValueError("Frozen factorial registry differs from actual reconstructed tokens")
    return manifest, data, registry
