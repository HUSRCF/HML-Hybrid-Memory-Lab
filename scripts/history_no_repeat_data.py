"""Native historical-version diagnostic with no recent initial-value repeat."""

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
from sas_memory.tasks import INSTRUCTIONS, Operation, execute
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import prompt_for
from scripts.run_write_locality_control_panel import RUNTIME_SOURCES as PANEL_SOURCES

PROTOCOL = ROOT / "configs/semantic_history_no_recent_repeat_plan.json"
PROTOCOL_SHA256 = "849d7df7adca1f6ad78bae230f2faaf94eaff273f01d71898c9f1fea65af23c0"
DESTINATION = Path("/media/990Pro/Code/SAS/datasets/memory_update/history-no-recent-repeat-v1")
CELLS = ("hold_middle", "update_new")
ROLES = ("target_initial", "target_current", "other_current")
DATA_SOURCES = tuple(sorted(set(PANEL_SOURCES) | {
    "configs/semantic_history_no_recent_repeat_plan.json", "configs/models.json",
    "scripts/history_no_repeat_data.py", "scripts/prepare_semantic_history_no_repeat.py",
    "scripts/analyze_blocked_reader_adaptation.py", "scripts/run_controller_trial.py",
    "sas_memory/tasks.py", "sas_memory/runtime.py", "sas_memory/training_data.py",
}))


def read_plan():
    plan = json.loads(PROTOCOL.read_text())
    if (streaming_hash(PROTOCOL) != PROTOCOL_SHA256 or
        plan["profile"] != "historical_version_preservation_without_recent_initial_value_repeat" or
        plan["training"] or plan["adapters"] or not plan["local_files_only"] or
        plan["model_key"] != "it" or plan["reader_policy"] != "full" or
        plan["cohorts"] != {"discovery": [97400, 97401, 97402, 97403], "replica": [97500, 97501, 97502, 97503]} or
        plan["grid"] != {"active_accounts": [32, 128], "prefix_records": 160, "tail_records": 2,
            "value_versions": 4, "tail_cells": list(CELLS), "query_roles": list(ROLES),
            "primary_active_accounts": 128, "case_pairs_per_cohort_load": 16, "all_case_pairs": 64,
            "natural_prefixes": 128, "full_prefill_queries": 384, "full_prefill_queries_per_cohort": 192,
            "independent_base_streams_per_cohort": 4, "maximum_prompt_tokens": 4096, "no_truncation": True}):
        raise ValueError("Registered no-repeat history diagnostic scope changed")
    return plan


def reference_gate(plan, reexecute=False):
    path = Path(plan["gate"]["required_control_panel_audit"])
    audit = json.loads(path.read_text())
    run = Path(audit["source_run"])
    manifest = json.loads((run / "manifest.json").read_text())
    if (streaming_hash(path) != plan["gate"]["required_control_panel_audit_sha256"] or
        audit["status"] != "completed_and_audited" or audit["physical_readouts_audited"] != 1920 or
        audit["reported_bindings_audited"] != 2240 or
        audit["weights_sha256"] != plan["native_loaded_weights_sha256"] or
        streaming_hash(path.with_suffix(".source.py")) != audit["analysis_source_sha256"] or
        streaming_hash(run / "manifest.json") != audit["manifest_sha256"] or
        manifest["status"] != "completed" or manifest["smoke"] or
        manifest["weights_before"] != manifest["weights_after"]):
        raise ValueError("Completed five-checkpoint reference gate changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / "sources" / name) != digest:
            raise ValueError("Actual/archived reference implementation changed")
    for name, state in manifest["models"].items():
        if streaming_hash(run / (name + ".jsonl")) != state["results_sha256"]:
            raise ValueError("Actual reference model observations changed")
    if reexecute:
        from scripts.analyze_write_locality_control_panel import audit_run
        if audit_run(run) != audit:
            raise ValueError("Complete actual reference audit does not reproduce")
    return {"audit_path": str(path), "audit_sha256": streaming_hash(path),
            "manifest_sha256": audit["manifest_sha256"], "weights_sha256": audit["weights_sha256"],
            "packages": manifest["packages"], "model_path": manifest["model_path"]}


def identity(example):
    return [example["case"]["id"], example["variant"], example["query"]["role"]]


def build_data(plan):
    rows = []
    for cohort, seeds in plan["cohorts"].items():
        for seed in seeds:
            entities = [f"E{number}" for number in range(700, 900)]
            random.Random(seed).shuffle(entities)
            a, b, c = entities[:3]
            all_values = random.Random(seed + 410009).sample(range(100, 1000), 4 * 162)
            for load in plan["grid"]["active_accounts"]:
                rng, distractors = random.Random(seed + 310003 + load), []
                while len(distractors) < 156:
                    cycle = entities[3:load].copy()
                    rng.shuffle(cycle)
                    distractors.extend(cycle)
                targets = [a, b, c, a] + distractors[:156]
                for version in range(4):
                    values = all_values[version * 162:(version + 1) * 162]
                    prefix = [Operation("SET", target, value) for target, value in zip(targets, values)]
                    variants = {}
                    for cell in CELLS:
                        tail = [Operation("SET", a, values[3] if cell == CELLS[0] else values[160]),
                                Operation("SET", c, values[161])]
                        truth = execute(prefix + tail)
                        queries = []
                        for role, entity, kind in ((ROLES[0], a, "initial"), (ROLES[1], a, "current"), (ROLES[2], c, "current")):
                            question = (f"What was the initial balance of account {entity}?" if kind == "initial"
                                        else f"What is the current balance of account {entity}?")
                            queries.append({"role": role, "entity": entity, "version": kind,
                                            "question": question, "answer": str(truth[kind][entity])})
                        variants[cell] = {"tail": [asdict(event) for event in tail], "queries": queries}
                    case = {"id": f"history-no-repeat-s{seed}-a{load:03d}-v{version}",
                            "base_stream_id": f"history-no-repeat-s{seed}",
                            "base_case_id": f"history-no-repeat-s{seed}-a{load:03d}",
                            "family": "history_no_recent_repeat", "cohort": cohort, "seed": seed,
                            "active_accounts": load, "value_version": version,
                            "target_entity": a, "protected_entity": b, "other_entity": c,
                            "prefix": [asdict(event) for event in prefix], "variants": variants}
                    for cell in CELLS:
                        for query in variants[cell]["queries"]:
                            rows.append({"case": case, "variant": cell, "query": query})
    return {"validation": rows}


def validate_data(data, plan):
    if data != build_data(plan):
        raise ValueError("Frozen schedule, source worlds, labels or evaluation order changed")
    cases, seen, siblings = {}, set(), defaultdict(list)
    for example in data["validation"]:
        case, cell, query = example["case"], example["variant"], example["query"]
        key = tuple(identity(example))
        if key in seen:
            raise ValueError("Duplicate history query")
        seen.add(key)
        cases[case["id"]] = case
        prefix, tail = case["prefix"], case["variants"][cell]["tail"]
        a, b, c = (case[k] for k in ("target_entity", "protected_entity", "other_entity"))
        events = prefix + tail
        if (len(prefix) != 160 or len(tail) != 2 or
            len({e["target"] for e in prefix}) != case["active_accounts"] or
            [e["target"] for e in prefix[:4]] != [a, b, c, a] or
            any(e["target"] in {a, b, c} for e in prefix[4:]) or
            [e["target"] for e in tail] != [a, c] or
            sum(e["value"] == prefix[0]["value"] for e in events) != 1 or
            prefix[0]["value"] in {prefix[3]["value"], tail[0]["value"]} or
            (cell == CELLS[0] and tail[0]["value"] != prefix[3]["value"])):
            raise ValueError("Initial evidence repetition, current-value shortcut or record/load drift")
        truth = execute([Operation(**event) for event in events])
        if query["answer"] != str(truth[query["version"]][query["entity"]]):
            raise ValueError("History own-world truth differs from executable SET semantics")
        siblings[(case["base_case_id"], cell, query["role"])].append(example)
    if (len(seen) != 384 or len(cases) != 64 or
        Counter((case["cohort"], case["active_accounts"]) for case in cases.values()) != {
            (cohort, load): 16 for cohort in plan["cohorts"] for load in (32, 128)} or
        any(len(group) != 4 or len({e["query"]["answer"] for e in group}) != 4 for group in siblings.values())):
        raise ValueError("Missing complete cohort/load/four-value siblings")
    return {"examples": 384, "case_pairs": 64, "natural_prefixes": 128, "base_streams": 8,
            "cohorts": 2, "loads": [32, 128], "records_per_prefix": 162,
            "four_value_query_groups": len(siblings)}


def token_registry(data, plan, runtime):
    prefixes, registrations, comparisons = {}, [], defaultdict(dict)
    load_signatures, value_signatures, evidence_signatures = {}, {}, {}
    for example in data["validation"]:
        case, cell, query = example["case"], example["variant"], example["query"]
        ledger, prompt = prompt_for(runtime, case, cell, query)
        prefix_key = f"{case['id']}/{cell}"
        if prefix_key not in prefixes:
            encoded = runtime.tokenizer(ledger, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
            chars = {char: i for i, (left, right) in enumerate(offsets) for char in range(left, right)}
            matches = list(re.finditer(r"\b\d{3}\b", ledger))
            events = case["prefix"] + case["variants"][cell]["tail"]
            if len(matches) != 162 or sum(int(m.group()) == events[0]["value"] for m in matches) != 1:
                raise ValueError("Initial numeric evidence repeats or numeric record spans are missing")
            cursor, records = len(runtime.prompt_start + INSTRUCTIONS), []
            for event, match in zip(events, matches):
                end = cursor + len(Operation(**event).text())
                left, right = chars[cursor], chars[end - 1] + 1
                positions = [chars[match.start() + digit] for digit in range(3)]
                if (int(match.group()) != event["value"] or offsets[left][0] != cursor or offsets[right - 1][1] != end or
                    [tuple(offsets[i]) for i in positions] != [(match.start() + j, match.start() + j + 1) for j in range(3)]):
                    raise ValueError("Record/value identity or individual digit alignment failed")
                records.append({"char_start": cursor, "char_end": end, "token_start": left,
                                "token_end": right, "value_tokens": positions})
                cursor = end
            if cursor != len(ledger):
                raise ValueError("Record boundaries do not cover the whole ledger")
            prefixes[prefix_key] = {"ledger_tokens": len(ids), "ledger_ids_sha256": data_fingerprint(ids),
                "prelude_tokens": len(runtime.ids(runtime.prompt_start + INSTRUCTIONS)), "records": records,
                "common_prefix_ids_sha256": data_fingerprint(ids[:records[160]["token_start"]]),
                "initial_value_numeric_occurrences": 1, "initial_value_record": 0}
        prefix = prefixes[prefix_key]
        ids, ledger_ids = runtime.ids(prompt), runtime.ids(ledger)
        if (ids[:len(ledger_ids)] != ledger_ids or len(ids) > plan["grid"]["maximum_prompt_tokens"] or
            data_fingerprint(ledger_ids) != prefix["ledger_ids_sha256"]):
            raise ValueError("Question boundary, source prefix or prompt budget changed")
        positions = {i for record in prefix["records"] for i in record["value_tokens"]}
        row = {"case_id": case["id"], "variant": cell, "role": query["role"],
            **{key: case[key] for key in ("cohort", "base_stream_id", "base_case_id", "seed", "active_accounts", "value_version")},
            "prefix_key": prefix_key, "ledger_tokens": len(ledger_ids), "prompt_tokens": len(ids),
            "prompt_ids_sha256": data_fingerprint(ids), "question_ids": ids[len(ledger_ids):],
            "question_ids_sha256": data_fingerprint(ids[len(ledger_ids):]), "answer": query["answer"],
            "answer_ids": runtime.continuation_ids(prompt, query["answer"]),
            "normalized_prompt_sha256": data_fingerprint([-1 if i in positions else token for i, token in enumerate(ids)])}
        if len(row["answer_ids"]) != 3:
            raise ValueError("Answer is not the fixed three-digit tokenization")
        signature = (row["ledger_tokens"], row["prompt_tokens"], row["question_ids_sha256"], prefix["records"])
        if load_signatures.setdefault((case["seed"], case["value_version"], query["role"]), signature) != signature:
            raise ValueError("Loads/worlds changed record positions, prompt lengths or query suffix")
        signature = row["normalized_prompt_sha256"], tuple(sorted(positions))
        if value_signatures.setdefault((case["base_case_id"], cell, query["role"]), signature) != signature:
            raise ValueError("Value siblings changed non-value content or positions")
        signature = [ledger_ids[r["token_start"]:r["token_end"]] for i, r in enumerate(prefix["records"]) if i in (0, 1, 2, 3, 160, 161)]
        if evidence_signatures.setdefault((case["seed"], case["value_version"], cell), signature) != signature:
            raise ValueError("Load manipulation changed A/B/C evidence content")
        comparisons[(case["id"], query["role"])][cell] = ids, prefix
        registrations.append(row)
    for group in comparisons.values():
        left, lp = group[CELLS[0]]
        right, rp = group[CELLS[1]]
        changed = {i for i, (a, b) in enumerate(zip(left, right)) if a != b}
        if (len(left) != len(right) or not changed or not changed <= set(lp["records"][160]["value_tokens"]) or
            lp["records"] != rp["records"] or lp["common_prefix_ids_sha256"] != rp["common_prefix_ids_sha256"]):
            raise ValueError("Tail worlds differ outside the registered final A value")
    if len(registrations) != 384 or len(prefixes) != 128:
        raise ValueError("Token registry is missing prescribed natural worlds")
    return {"prefixes": prefixes, "validation": registrations}


def evaluated_members(data, smoke):
    if not smoke:
        return data["validation"]
    first = {}
    for row in data["validation"]:
        case = row["case"]
        group = case["cohort"], case["active_accounts"]
        first[group] = min(first.get(group, case["id"]), case["id"])
    return [row for row in data["validation"] if row["case"]["id"] in first.values()]


def safety_samples(members):
    grouped = defaultdict(set)
    for example in members:
        case = example["case"]
        grouped[(case["cohort"], case["active_accounts"])].add(case["id"])
    selected = {key for values in grouped.values() for key in (min(values), max(values))}
    return sorted(identity(row) for row in members if row["case"]["id"] in selected and row["query"]["role"] == ROLES[0])


def model_artifacts(model):
    model = Path(model)
    tokenizer = {p.name: streaming_hash(p) for p in sorted(model.iterdir()) if p.is_file() and
                 ("token" in p.name or p.name == "config.json" or p.suffix == ".jinja")}
    checkpoint = {p.name: streaming_hash(p) for p in sorted(model.glob("*.safetensors"))}
    if not tokenizer or not checkpoint:
        raise ValueError("Existing local tokenizer/checkpoint is incomplete")
    return tokenizer, checkpoint


def registered_inputs(path=DESTINATION):
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    plan = read_plan()
    data = json.loads((path / "data.json").read_text())
    registry = json.loads((path / "registry.json").read_text())
    model = json.loads((ROOT / plan["model_config"]).read_text())[plan["model_key"]]
    if (manifest["status"] != "registered_before_inference" or manifest["training"] or manifest["model_inference"] or
        not manifest["complete_control_panel_audit_reexecuted_at_registration"] or
        manifest["model_path"] != model or manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or
        manifest["reference_gate"] != reference_gate(plan) or manifest["counts"] != validate_data(data, plan) or
        manifest["data_sha256"] != data_fingerprint(data) or manifest["registry_sha256"] != data_fingerprint(registry) or
        json.loads((path / "plan.json").read_text()) != plan or set(manifest["source_sha256"]) != set(DATA_SOURCES) or
        manifest["evaluation_keys_sha256"] != data_fingerprint([identity(row) for row in data["validation"]]) or
        datetime.fromisoformat(plan["registered_utc"]) >= datetime.fromisoformat(manifest["registered_utc"])):
        raise ValueError("No-repeat data/protocol/reference registration changed")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / "sources" / name) != digest:
            raise ValueError("Actual/archived no-repeat data implementation changed")
    tokenizer, checkpoint = model_artifacts(model)
    if manifest["tokenizer_sha256"] != tokenizer or manifest["checkpoint_sha256"] != checkpoint:
        raise ValueError("Actual local tokenizer/config/checkpoint changed")
    if token_registry(data, plan, tokenizer_runtime(model)) != registry:
        raise ValueError("Frozen registry differs from actual reconstructed source/query tokens")
    return manifest, data, registry
