"""Executable, fixed-length value-update controls for the native hybrid model."""

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re

from sas_memory.tasks import INSTRUCTIONS, Operation, execute
from sas_memory.training_data import data_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/semantic_write_locality_load_plan.json"
CELLS = ("A0_C0", "A1_C0", "A0_C1", "A1_C1")
ROLES = ("target_current", "protected_current", "target_initial")
DATA_SOURCES = (
    "configs/models.json",
    "sas_memory/write_locality_data.py", "sas_memory/tasks.py", "sas_memory/runtime.py",
    "sas_memory/training_data.py", "sas_memory/calibration.py",
    "scripts/prepare_write_locality_load_screen.py", "scripts/analyze_blocked_reader_adaptation.py",
    "scripts/analyze_controller_trial.py", "scripts/run_controller_trial.py",
)


def streaming_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_protocol():
    plan = json.loads(PROTOCOL.read_text())
    grid = plan["development_grid"]
    if (plan["profile"] != "fixed_length_semantic_write_locality_load_screen" or
        plan["training"] or plan["adapters"] or plan["reader_policy"] != "full" or
        plan["model_key"] != "it" or plan["model_class"] != "Qwen3_5ForCausalLM" or not plan["local_files_only"] or
        grid != {"base_seeds": [97000, 97001, 97002, 97003], "active_accounts": [8, 32, 128],
                 "prefix_records": 160, "value_versions": 4, "tail_cells": list(CELLS),
                 "query_roles": list(ROLES), "expected_free_answers": 576,
                 "expected_distinct_prefixes_with_tail": 192, "independent_base_streams": 4,
                 "maximum_prompt_tokens": 4096, "no_truncation": True}):
        raise ValueError("Unsupported or changed registered locality protocol")
    return plan


def reference_gate(plan):
    """Pin the completed independent audit; do not claim to reexecute it."""
    path = Path(plan["gate"]["required_audit"])
    audit = json.loads(path.read_text())
    source = path.with_suffix(".source.py")
    trial = Path(audit["source_run"])
    manifest = json.loads((trial / "manifest.json").read_text())
    if (audit["status"] != plan["gate"]["required_status"] or
        audit["validation_records_audited"] != plan["gate"]["required_validation_records"] or
        audit["teacher_train_records_audited"] != plan["gate"]["required_teacher_train_records"] or
        audit["paired_initialization_verified"] is not True or manifest["status"] != "completed" or
        streaming_hash(trial / "manifest.json") != audit["manifest_sha256"] or
        streaming_hash(source) != audit["analysis_source_sha256"]):
        raise ValueError("Completed independent GDN timing gate changed or is incomplete")
    return {"report_path": str(path), "report_sha256": streaming_hash(path),
            "analysis_source_sha256": streaming_hash(source), "manifest_sha256": audit["manifest_sha256"],
            "status": audit["status"], "reference_audits_reexecuted": False}


def identity(example):
    return [example["case"]["id"], example["variant"], example["query"]["role"]]


def build_data(plan):
    rows = []
    for seed in plan["development_grid"]["base_seeds"]:
        entities = [f"E{number}" for number in range(700, 900)]
        random.Random(seed).shuffle(entities)
        a, b, c = entities[:3]
        all_values = random.Random(seed + 410009).sample(range(100, 1000), 4 * 162)
        for load in plan["development_grid"]["active_accounts"]:
            rng = random.Random(seed + 310003 + load)
            distractors = []
            while len(distractors) < 157:
                cycle = entities[3:load].copy()
                rng.shuffle(cycle)
                distractors.extend(cycle)
            targets = entities[:3] + distractors[:157]
            for version in range(4):
                values = all_values[version * 162:(version + 1) * 162]
                prefix = [Operation("SET", target, value) for target, value in zip(targets, values)]
                variants = {}
                for cell in CELLS:
                    tail = [Operation("SET", a, values[160] if cell[1] == "1" else values[0]),
                            Operation("SET", c, values[161] if cell[-1] == "1" else values[2])]
                    truth = execute(prefix + tail)
                    queries = []
                    for role, entity, kind in ((ROLES[0], a, "current"), (ROLES[1], b, "current"),
                                               (ROLES[2], a, "initial")):
                        question = (f"What is the current balance of account {entity}?" if kind == "current"
                                    else f"What was the initial balance of account {entity}?")
                        queries.append({"role": role, "entity": entity, "version": kind,
                                        "question": question, "answer": str(truth[kind][entity])})
                    variants[cell] = {"tail": [asdict(event) for event in tail], "queries": queries}
                case = {"id": f"write-locality-s{seed}-a{load:03d}-v{version}",
                        "base_stream_id": f"write-locality-s{seed}",
                        "base_case_id": f"write-locality-s{seed}-a{load:03d}",
                        "family": "write_locality_load", "seed": seed, "active_accounts": load,
                        "value_version": version, "target_entity": a, "protected_entity": b,
                        "other_entity": c, "prefix": [asdict(event) for event in prefix], "variants": variants}
                for cell in CELLS:
                    for query in variants[cell]["queries"]:
                        rows.append({"case": case, "variant": cell, "query": query})
    return {"validation": rows}


def validate_data(data, plan):
    rows = data["validation"]
    expected = build_data(plan)
    if data != expected:
        raise ValueError("Locality events, identities, labels or registered evaluation order changed")
    cases, groups, seen = {}, defaultdict(list), set()
    for example in rows:
        case, cell, query = example["case"], example["variant"], example["query"]
        key = tuple(identity(example))
        if key in seen:
            raise ValueError("Duplicate locality observation")
        seen.add(key)
        cases[case["id"]] = case
        if len(case["prefix"]) != 160 or len({event["target"] for event in case["prefix"]}) != case["active_accounts"]:
            raise ValueError("Record count or live-account load changed")
        a, b, c = (case[field] for field in ("target_entity", "protected_entity", "other_entity"))
        if [event["target"] for event in case["prefix"][:3]] != [a, b, c] or any(
                event["target"] in {a, b, c} for event in case["prefix"][3:]):
            raise ValueError("Query fact age or unrelated-write scope changed")
        events = [Operation(**event) for event in case["prefix"] + case["variants"][cell]["tail"]]
        truth = execute(events)
        if query["answer"] != str(truth[query["version"]][query["entity"]]):
            raise ValueError("Locality answer disagrees with executable truth")
        if truth["current"][b] != case["prefix"][1]["value"] or truth["initial"][a] != case["prefix"][0]["value"]:
            raise ValueError("Protected or initial truth changed")
        groups[(case["base_case_id"], cell, query["role"])].append(example)
    for siblings in groups.values():
        if len(siblings) != 4 or len({row["query"]["answer"] for row in siblings}) != 4:
            raise ValueError("Missing four-answer value siblings")
    if len(rows) != 576 or len(cases) != 48 or len({case["base_stream_id"] for case in cases.values()}) != 4:
        raise ValueError("Incomplete locality development cohort")
    return {"examples": len(rows), "value_cases": len(cases), "base_streams": 4,
            "loads": [8, 32, 128], "prefixes_with_tail": len(cases) * 4,
            "records_per_prefix": 162, "four_value_query_groups": len(groups)}


def token_registry(data, plan, runtime):
    from scripts.run_controller_trial import prompt_for
    prefixes, registrations = {}, []
    load_signatures, value_signatures, cells = {}, {}, defaultdict(dict)
    for example in data["validation"]:
        case, cell, query = example["case"], example["variant"], example["query"]
        ledger, prompt = prompt_for(runtime, case, cell, query)
        prefix_key = f"{case['id']}/{cell}"
        if prefix_key not in prefixes:
            tokens = runtime.tokenizer(ledger, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = tokens["input_ids"], tokens["offset_mapping"]
            characters = {char: index for index, (left, right) in enumerate(offsets) for char in range(left, right)}
            matches = list(re.finditer(r"\b\d{3}\b", ledger))
            if len(matches) != 162:
                raise ValueError("SET numeric spans do not match the 162 records")
            cursor = len(runtime.prompt_start + INSTRUCTIONS)
            records = []
            for event, match in zip(case["prefix"] + case["variants"][cell]["tail"], matches):
                end = cursor + len(Operation(**event).text())
                left, right = characters[cursor], characters[end - 1] + 1
                positions = [characters[match.start() + digit] for digit in range(3)]
                if (offsets[left][0] != cursor or offsets[right - 1][1] != end or
                    [tuple(offsets[index]) for index in positions] != [(match.start() + i, match.start() + i + 1) for i in range(3)]):
                    raise ValueError("Record boundaries or individual value digits are not token-aligned")
                records.append({"char_start": cursor, "char_end": end, "token_start": left,
                                "token_end": right, "value_tokens": positions})
                cursor = end
            if cursor != len(ledger):
                raise ValueError("Record registry does not cover the complete ledger")
            prefixes[prefix_key] = {"ledger_tokens": len(ids), "ledger_ids_sha256": data_fingerprint(ids),
                                    "prelude_tokens": len(runtime.ids(runtime.prompt_start + INSTRUCTIONS)),
                                    "records": records}
        prefix = prefixes[prefix_key]
        ids, ledger_ids = runtime.ids(prompt), runtime.ids(ledger)
        if ids[:len(ledger_ids)] != ledger_ids or len(ids) > plan["development_grid"]["maximum_prompt_tokens"]:
            raise ValueError("Question boundary crossed or prompt budget exceeded")
        positions = {position for record in prefix["records"] for position in record["value_tokens"]}
        row = {"case_id": case["id"], "variant": cell, "role": query["role"],
               "base_stream_id": case["base_stream_id"], "seed": case["seed"], "active_accounts": case["active_accounts"],
               "value_version": case["value_version"], "prefix_key": prefix_key,
               "ledger_tokens": len(ledger_ids), "prompt_tokens": len(ids),
               "prompt_ids_sha256": data_fingerprint(ids),
               "question_ids_sha256": data_fingerprint(ids[len(ledger_ids):]),
               "normalized_prompt_sha256": data_fingerprint([-1 if index in positions else token for index, token in enumerate(ids)]),
               "answer_ids": runtime.continuation_ids(prompt, query["answer"])}
        if len(row["answer_ids"]) != 3 or data_fingerprint(ledger_ids) != prefix["ledger_ids_sha256"]:
            raise ValueError("Answer-token or reused-prefix identity changed")
        key = (case["seed"], case["value_version"], query["role"])
        signature = (row["ledger_tokens"], row["prompt_tokens"], row["question_ids_sha256"])
        if load_signatures.setdefault(key, signature) != signature:
            raise ValueError("Loads or tail cells differ in prompt length, query positions or suffix")
        key = (case["base_case_id"], cell, query["role"])
        signature = (row["normalized_prompt_sha256"], tuple(sorted(positions)))
        if value_signatures.setdefault(key, signature) != signature:
            raise ValueError("Value siblings differ outside registered numeric spans")
        cells[(case["id"], query["role"])][cell] = (ids, prefix)
        registrations.append(row)
    for group in cells.values():
        baseline, reference = group["A0_C0"]
        for cell, (ids, prefix) in group.items():
            if prefix["records"] != reference["records"]:
                raise ValueError("Tail-cell record or value positions changed")
            allowed = set(reference["records"][160]["value_tokens"] if cell[1] == "1" else ())
            allowed.update(reference["records"][161]["value_tokens"] if cell[-1] == "1" else ())
            changed = {index for index, (left, right) in enumerate(zip(baseline, ids)) if left != right}
            if not changed <= allowed or bool(allowed) != bool(changed):
                raise ValueError("Tail toggle changed an unregistered token or failed to change a value")
    return {"prefixes": prefixes, "validation": registrations}


def evaluated_members(data, smoke):
    if not smoke:
        return data["validation"]
    return [row for row in data["validation"] if row["case"]["seed"] == 97000 and
            row["case"]["value_version"] == 0 and row["case"]["active_accounts"] in (8, 128) and
            row["variant"] in (CELLS[0], CELLS[-1])]


def registered_dataset(path):
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    path = Path(path)
    plan = read_protocol()
    manifest = json.loads((path / "manifest.json").read_text())
    data = json.loads((path / "data.json").read_text())
    registry = json.loads((path / "registry.json").read_text())
    if (manifest["status"] != "registered_before_inference" or manifest["training"] or manifest["model_inference"] or
        json.loads((path / "plan.json").read_text()) != plan or manifest["protocol_sha256"] != streaming_hash(PROTOCOL) or
        manifest["counts"] != validate_data(data, plan) or manifest["data_sha256"] != data_fingerprint(data) or
        manifest["registry_sha256"] != data_fingerprint(registry) or
        manifest["reference_gate"] != reference_gate(plan) or
        manifest["model_path"] != json.loads((ROOT / plan["model_config"]).read_text())[plan["model_key"]] or
        set(manifest["source_sha256"]) != set(DATA_SOURCES) or
        manifest["evaluation_keys_sha256"] != data_fingerprint([[row["case_id"], row["variant"], row["role"]] for row in registry["validation"]])):
        raise ValueError("Locality data/protocol registration changed")
    model = Path(manifest["model_path"])
    expected_tokenizer = {file.name for file in model.iterdir() if file.is_file() and
                          ("token" in file.name or file.name == "config.json" or file.suffix == ".jinja")}
    if (set(manifest["tokenizer_sha256"]) != expected_tokenizer or not manifest["checkpoint_sha256"] or
        set(manifest["checkpoint_sha256"]) != {file.name for file in model.glob("*.safetensors")}):
        raise ValueError("Tokenizer/config/checkpoint coverage is incomplete")
    for relative, digest in manifest["source_sha256"].items():
        if streaming_hash(path / "sources" / relative) != digest or streaming_hash(ROOT / relative) != digest:
            raise ValueError("Locality data/runtime source changed")
    for group in ("tokenizer_sha256", "checkpoint_sha256"):
        for name, digest in manifest[group].items():
            if streaming_hash(Path(manifest["model_path"]) / name) != digest:
                raise ValueError("Locality tokenizer/config/checkpoint changed")
    runtime = tokenizer_runtime(manifest["model_path"])
    if token_registry(data, plan, runtime) != registry:
        raise ValueError("Locality token registry does not match actual reconstructed prompts")
    return manifest, data, registry
