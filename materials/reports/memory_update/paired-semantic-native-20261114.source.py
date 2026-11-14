"""Audit registered multi-entity semantic prompts and unrestricted native readouts."""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from sas_memory.semantic_metrics import summarize_semantics
from sas_memory.tasks import Operation, independent
from sas_memory.training_data import data_fingerprint
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.analyze_controller_trial import load_rows
from scripts.prepare_paired_semantic_updates import build_data, validate_data
from scripts.run_controller_trial import encode_example, prompt_for, write_json


def registered_dataset(path):
    manifest = json.loads((path / "manifest.json").read_text())
    data = json.loads((path / "data.json").read_text())
    registry = json.loads((path / "registry.json").read_text())
    if (manifest["status"] != "registered_before_inference" or manifest["training"] or manifest["model_inference"] or
        manifest["reader_policy"] != "full" or validate_data(data, 4, 2) != manifest["counts"] or
        data_fingerprint(data) != manifest["data_sha256"] or data_fingerprint(registry) != manifest["registry_sha256"] or
        data != build_data()):
        raise ValueError("Registered semantic events/truth/coverage changed")
    for relative, digest in manifest["source_sha256"].items():
        if file_hash(path / "sources" / relative) != digest or file_hash(ROOT / relative) != digest:
            raise ValueError("Semantic data/runtime source differs from registration")
    for name, digest in manifest["tokenizer_sha256"].items():
        if file_hash(Path(manifest["model_path"]) / name) != digest:
            raise ValueError("Semantic tokenizer/model config changed")
    runtime = tokenizer_runtime(manifest["model_path"])
    for split in ("train", "validation"):
        if len(registry[split]) != len(data[split]):
            raise ValueError("Missing semantic token registrations")
        groups = defaultdict(list)
        for example, row in zip(data[split], registry[split]):
            case, query = example["case"], example["query"]
            ledger, prompt = prompt_for(runtime, case, example["variant"], query)
            tokens = runtime.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
            positions = []
            for match in re.finditer(r"\b\d{3}\b", ledger):
                indices = [index for index, (left, right) in enumerate(tokens["offset_mapping"]) if left < match.end() and right > match.start()]
                if len(indices) != 3 or [tuple(tokens["offset_mapping"][index]) for index in indices] != [(match.start() + i, match.start() + i + 1) for i in range(3)]:
                    raise ValueError("Non-isolated semantic SET value tokens")
                positions.extend(indices)
            positions, boundary = sorted(set(positions)), len(runtime.ids(ledger))
            position_set, ids = set(positions), tokens["input_ids"]
            expected = {"case_id": case["id"], "base_case_id": case["base_case_id"], "variant": example["variant"],
                        "role": query["role"], "value_version": case["value_version"], "ledger_tokens": boundary,
                        "prompt_tokens": len(ids), "value_token_positions": positions, "prompt_ids_sha256": data_fingerprint(ids),
                        "normalized_prompt_sha256": data_fingerprint([-1 if index in position_set else token for index, token in enumerate(ids)]),
                        "question_ids_sha256": data_fingerprint(ids[boundary:])}
            if row != expected or ids[:boundary] != runtime.ids(ledger):
                raise ValueError("Registered semantic input identity changed")
            if case["family"] in ("independent_swap", "dependent_swap", "copy") and example["variant"] != "before":
                pair = [Operation(**event) for event in case["variants"][example["variant"]]["tail"]]
                if len(pair) != 2 or independent(*pair) != (case["family"] == "independent_swap"):
                    raise ValueError("Executable read/write dependency label invalid")
            groups[case["base_case_id"], example["variant"], query["role"]].append(row)
        for siblings in groups.values():
            signatures = {(row["ledger_tokens"], row["prompt_tokens"], tuple(row["value_token_positions"]),
                           row["normalized_prompt_sha256"], row["question_ids_sha256"]) for row in siblings}
            if len(siblings) != 4 or len(signatures) != 1:
                raise ValueError("Semantic counterfactual token/position match failed")
    encoded = [encode_example(runtime, example, "full") for example in data["train"]]
    if (encoded != json.loads((path / "tokenized_train.json").read_text()) or
        data_fingerprint([{key: item[key] for key in ("input_ids", "answer_ids")} for item in encoded]) != manifest["tokenized_input_answer_sha256"]):
        raise ValueError("Registered semantic training tokens changed")
    return manifest, data, registry


def evaluated_members(data, smoke):
    if not smoke:
        return data["validation"]
    # Complete long COPY family, not a performance-selected subset.
    family = max(example["case"]["base_case_id"] for example in data["validation"] if example["case"]["family"] == "copy")
    return [example for example in data["validation"] if example["case"]["base_case_id"] == family]


def identity(example):
    return [example["case"]["id"], example["variant"], example["query"]["role"]]


def audit_run(path, allow_smoke=False):
    manifest = json.loads((path / "manifest.json").read_text())
    plan = json.loads((path / "plan.json").read_text())
    dataset = Path(plan["data_source"])
    registered, data, registry = registered_dataset(dataset)
    if (manifest["status"] != "completed" or manifest["training"] or
        manifest["plan_sha256"] != data_fingerprint(plan) or manifest["data_manifest_sha256"] != file_hash(dataset / "manifest.json") or
        manifest["model_path"] != registered["model_path"] or manifest["completed_splits"] != ["validation"] or
        manifest["reader_policy"] != "full" or not manifest["frozen_parameters_and_no_gradients"] or not manifest["hooks_restored"]):
        raise ValueError("Incomplete or non-native semantic probe")
    if manifest["smoke"] and not allow_smoke:
        raise ValueError("Smoke is not full-cohort semantic evidence")
    for stamp in (registered["registered_utc"], plan["registered_utc"]):
        if datetime.fromisoformat(stamp) >= datetime.fromisoformat(manifest["started_utc"]):
            raise ValueError("Semantic registration was not fixed before inference")
    if registered["data_sha256"] != plan["data_sha256"] or registered["registry_sha256"] != plan["registry_sha256"]:
        raise ValueError("Plan/data identity changed")
    for relative, digest in manifest["source_sha256"].items():
        if file_hash(path / "sources" / relative) != digest:
            raise ValueError("Archived semantic probe source changed")
    for relative in ("sas_memory/runtime.py", "sas_memory/tasks.py", "scripts/run_controller_trial.py"):
        if file_hash(ROOT / relative) != manifest["source_sha256"][relative]:
            raise ValueError("Live semantic runtime differs from archived run")
    members = evaluated_members(data, manifest["smoke"])
    if manifest["evaluated_keys"] != [identity(example) for example in members]:
        raise ValueError("Evaluated semantic cases/order changed")
    expected = {tuple(identity(example)): example["query"]["answer"] for example in members}
    rows = load_rows(path / "validation.native.jsonl", expected)
    if file_hash(path / "validation.native.jsonl") != manifest["result_sha256"]:
        raise ValueError("Native result file changed")
    indexed = {(row["case_id"], row["variant"], row["role"]): row for row in registry["validation"]}
    for row, example in zip(rows, members):
        key = tuple(identity(example))
        if ((row["case_id"], row["variant"], row["role"]) != key or row["registration"] != indexed[key] or
            row["reader_policy"] != "full" or
            any(row[field] != example["case"][field] for field in ("family", "seed", "base_case_id", "value_version", "distractor_updates"))):
            raise ValueError("Native semantic result metadata/input changed")
    sample_keys = sorted((identity(example) for example in members))
    if [check["key"] for check in manifest["safety_checks"]] != [sample_keys[0], sample_keys[-1]]:
        raise ValueError("Missing prescribed native semantic replay checks")
    for check in manifest["safety_checks"]:
        if not check["native_replay_exact"] or set(check["full_attention_weights"]) != {str(index) for index in (3, 7, 11, 15, 19, 23)}:
            raise ValueError("Native semantic replay/attention layers failed")
        for stats in check["full_attention_weights"].values():
            if stats["prefill_calls"] != 1 or stats["cached_calls"] < 1 or stats["max_record_mass"] <= 0:
                raise ValueError("Native full-history access control failed")
    return {"status": "audited_smoke" if manifest["smoke"] else "completed_and_audited", "source_run": str(path),
            "manifest_sha256": file_hash(path / "manifest.json"), "records_audited": len(rows),
            "metrics": summarize_semantics(rows), "interpretation": plan["interpretation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    report = audit_run(args.run, args.allow_smoke)
    report["analysis_source_sha256"] = file_hash(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: report["metrics"][key] for key in ("examples", "correct", "four_value_query_groups", "joint_roles", "all_twelve_value_variant_groups")}))


if __name__ == "__main__":
    main()
