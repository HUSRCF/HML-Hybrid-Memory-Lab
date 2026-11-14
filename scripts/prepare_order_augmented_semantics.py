"""Register same-budget initial-record order augmentation without model inference."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from sas_memory.tasks import Operation, execute
from sas_memory.training_data import data_fingerprint
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.analyze_paired_semantic_probe import registered_dataset
from scripts.prepare_paired_semantic_updates import validate_data
from scripts.run_controller_trial import encode_example, prompt_for, write_json


SEED_OFFSET = 991003


def build_augmented(data):
    """Replace half the train initial orders; no examples, facts or labels are added."""
    augmented, orders = deepcopy(data), {}
    original_cases = {example["case"]["id"]: example["case"] for example in data["train"]}
    for example in augmented["train"]:
        case = example["case"]
        base = case["base_case_id"]
        index = int(base.rsplit("-", 1)[1])
        if index not in range(4):
            raise ValueError("Augmentation requires the four registered train streams per task")
        initial = original_cases[case["id"]]["prefix"][:8]
        if len(initial) != 8 or any(event["kind"] != "SET" for event in initial) or len({event["target"] for event in initial}) != 8:
            raise ValueError("Initial order augmentation requires eight independent SETs")
        if base not in orders:
            permutation = list(range(8))
            if index in (1, 2):
                rng = random.Random(case["seed"] + SEED_OFFSET)
                for _ in range(1000):
                    rng.shuffle(permutation)
                    if all(position != source for position, source in enumerate(permutation)):
                        break
                else:
                    raise ValueError("No deterministic derangement")
            orders[base] = permutation
        # Deepcopy preserves the shared case object inside a sibling; read only
        # from the original on every repeated query to avoid applying a shuffle twice.
        case["prefix"] = [deepcopy(initial[source]) for source in orders[base]] + deepcopy(original_cases[case["id"]]["prefix"][8:])
        original = original_cases[case["id"]]
        if case["variants"] != original["variants"]:
            raise ValueError("Augmentation changed a tail or question")
        for name, variant in case["variants"].items():
            before = execute([Operation(**event) for event in original["prefix"] + variant["tail"]])
            after = execute([Operation(**event) for event in case["prefix"] + variant["tail"]])
            if before != after:
                raise ValueError("Augmentation changed current or initial truth")
    if augmented["validation"] != data["validation"]:
        raise ValueError("Canonical validation changed during train augmentation")
    validate_data(augmented, 4, 2)
    groups = {}
    for example in augmented["train"]:
        case = example["case"]
        groups[case["base_case_id"]] = case
    balance = Counter((case["family"], case["distractor_updates"], orders[base] != list(range(8))) for base, case in groups.items())
    if len(groups) != 24 or len(balance) != 24 or set(balance.values()) != {1}:
        raise ValueError("Every task/length needs one canonical and one permuted train stream")
    return augmented, orders


def build_registry(runtime, examples, reference):
    rows, groups = [], defaultdict(list)
    if len(examples) != len(reference):
        raise ValueError("Augmented sample count changed")
    for example, original in zip(examples, reference):
        case, query = example["case"], example["query"]
        ledger, prompt = prompt_for(runtime, case, example["variant"], query)
        encoded = runtime.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids, positions = encoded["input_ids"], []
        matches = list(re.finditer(r"\b\d{3}\b", ledger))
        if len(matches) != sum(event["kind"] == "SET" for event in case["prefix"] + case["variants"][example["variant"]]["tail"]):
            raise ValueError("Unexpected numeric span")
        for match in matches:
            indices = [index for index, (left, right) in enumerate(encoded["offset_mapping"]) if left < match.end() and right > match.start()]
            if len(indices) != 3 or [tuple(encoded["offset_mapping"][index]) for index in indices] != [(match.start()+i, match.start()+i+1) for i in range(3)]:
                raise ValueError("Value digit token boundaries changed")
            positions.extend(indices)
        positions, boundary = sorted(set(positions)), len(runtime.ids(ledger))
        row = {"case_id": case["id"], "base_case_id": case["base_case_id"], "variant": example["variant"], "role": query["role"],
               "value_version": case["value_version"], "ledger_tokens": boundary, "prompt_tokens": len(ids), "value_token_positions": positions,
               "prompt_ids_sha256": data_fingerprint(ids),
               "normalized_prompt_sha256": data_fingerprint([-1 if index in set(positions) else token for index, token in enumerate(ids)]),
               "question_ids_sha256": data_fingerprint(ids[boundary:])}
        unchanged = ("case_id", "base_case_id", "variant", "role", "value_version", "ledger_tokens", "prompt_tokens", "value_token_positions", "question_ids_sha256")
        if ids[:boundary] != runtime.ids(ledger) or any(row[key] != original[key] for key in unchanged):
            raise ValueError("Augmentation changed sample identity, query IDs/position or token count")
        rows.append(row)
        groups[case["base_case_id"], example["variant"], query["role"]].append(row)
    for siblings in groups.values():
        signatures = {(row["ledger_tokens"], row["prompt_tokens"], tuple(row["value_token_positions"]), row["normalized_prompt_sha256"], row["question_ids_sha256"]) for row in siblings}
        if len(siblings) != 4 or len(signatures) != 1:
            raise ValueError("Augmented four-value matching failed")
    return rows


def registered_augmented_dataset(path):
    manifest = json.loads((path / "manifest.json").read_text())
    base_path = Path(manifest["base_data"])
    base, original, original_registry = registered_dataset(base_path)
    if (manifest["status"] != "registered_before_training" or manifest["training"] or manifest["model_inference"] or
        file_hash(base_path / "manifest.json") != manifest["base_data_manifest_sha256"] or manifest["base_data_sha256"] != base["data_sha256"] or
        manifest["model_path"] != base["model_path"] or manifest["reader_policy"] != "full"):
        raise ValueError("Augmented data/base registration changed")
    for relative, digest in manifest["source_sha256"].items():
        if file_hash(ROOT / relative) != digest or file_hash(path / "sources" / relative) != digest:
            raise ValueError("Registered augmentation source changed")
    data, orders = build_augmented(original)
    runtime = tokenizer_runtime(base["model_path"])
    registry = {"train": build_registry(runtime, data["train"], original_registry["train"]), "validation": original_registry["validation"]}
    encoded = [encode_example(runtime, example, "full") for example in data["train"]]
    original_encoded = json.loads((base_path / "tokenized_train.json").read_text())
    if any(a["answer_ids"] != b["answer_ids"] or len(a["input_ids"]) != len(b["input_ids"]) or a["ledger_tokens"] != b["ledger_tokens"] for a, b in zip(encoded, original_encoded)):
        raise ValueError("Augmentation changed an actual optimized token budget or label")
    values = (("data", data), ("orders", orders), ("registry", registry), ("tokenized_train", encoded))
    for name, value in values:
        if json.loads((path / f"{name}.json").read_text()) != value or data_fingerprint(value) != manifest[f"{name}_sha256"]:
            raise ValueError("Registered augmented tokens, labels or permutation changed")
    token_hash = data_fingerprint([{key: item[key] for key in ("input_ids", "answer_ids")} for item in encoded])
    if token_hash != manifest["tokenized_input_answer_sha256"] or validate_data(data, 4, 2) != manifest["counts"]:
        raise ValueError("Registered augmented counts or supervised token identity changed")
    return manifest, data, registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, default=Path("/media/990Pro/Code/SAS/datasets/memory_update/paired-semantic-updates-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, original, original_registry = registered_dataset(args.base_data)
    data, orders = build_augmented(original)
    runtime = tokenizer_runtime(base["model_path"])
    registry = {"train": build_registry(runtime, data["train"], original_registry["train"]), "validation": original_registry["validation"]}
    encoded = [encode_example(runtime, example, "full") for example in data["train"]]
    old_encoded = json.loads((args.base_data / "tokenized_train.json").read_text())
    if any(a["answer_ids"] != b["answer_ids"] or len(a["input_ids"]) != len(b["input_ids"]) or a["ledger_tokens"] != b["ledger_tokens"] for a, b in zip(encoded, old_encoded)):
        raise ValueError("Optimized training input/answer token budgets changed")
    args.output.mkdir(parents=True, exist_ok=False)
    values = (("data", data), ("orders", orders), ("registry", registry), ("tokenized_train", encoded))
    for name, value in values:
        write_json(args.output / f"{name}.json", value)
    sources = [Path(__file__).resolve()] + [ROOT / relative for relative in (
        "scripts/analyze_paired_semantic_probe.py", "scripts/prepare_paired_semantic_updates.py", "scripts/analyze_blocked_reader_adaptation.py",
        "scripts/run_controller_trial.py", "sas_memory/tasks.py", "sas_memory/training_data.py", "sas_memory/runtime.py")]
    manifest = {"status": "registered_before_training", "registered_utc": datetime.now(timezone.utc).isoformat(), "training": False, "model_inference": False,
                "reader_policy": "full", "model_path": base["model_path"], "base_data": str(args.base_data),
                "base_data_manifest_sha256": file_hash(args.base_data / "manifest.json"), "base_data_sha256": base["data_sha256"],
                "counts": validate_data(data, 4, 2), **{f"{name}_sha256": data_fingerprint(value) for name, value in values},
                "tokenized_input_answer_sha256": data_fingerprint([{key: item[key] for key in ("input_ids", "answer_ids")} for item in encoded]),
                "seed_offset": SEED_OFFSET, "canonical_train_stream_indices": [0, 3], "deranged_train_stream_indices": [1, 2],
                "source_sha256": {str(source.relative_to(ROOT)): file_hash(source) for source in sources},
                "interpretation": "Ordinary deterministic 50/50 initial-order augmentation, not a semantic architecture. Same train examples, labels, questions, per-example input/answer lengths and ordering; half the base streams change only eight independent initial SET records. Each task and length has one canonical and one deranged stream. Four siblings share the same permutation. Canonical validation stays byte-for-byte unchanged; both observed validation orders remain development, not untouched confirmation."}
    for relative in manifest["source_sha256"]:
        destination = args.output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({key:manifest[key] for key in ("status", "counts", "data_sha256", "registry_sha256", "tokenized_input_answer_sha256")}))


if __name__ == "__main__":
    main()
