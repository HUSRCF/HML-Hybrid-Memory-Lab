"""Register same-question, different-value single-fact carrier positive controls."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.tasks import Operation, execute
from sas_memory.training_data import data_fingerprint


def build_data(train_groups=32, validation_groups=16, values_per_group=4):
    data = {}
    for split, count, entity_range, seed in (("train", train_groups, range(100, 500), 91000),
                                             ("validation", validation_groups, range(500, 700), 92000)):
        rng = random.Random(seed)
        entities = rng.sample(list(entity_range), count)
        examples = []
        for index, number in enumerate(entities):
            entity = f"E{number}"
            values = rng.sample([value for value in range(200, 800) if value != number], values_per_group)
            for version, value in enumerate(values):
                event = Operation("SET", entity, value)
                answer = str(execute([event])["current"][entity])
                query = {"role": "target_current", "entity": entity, "version": "current",
                         "question": f"What is the current balance of account {entity}?", "answer": answer}
                case = {"id": f"single-fact-{split}-g{index:03d}-v{version}", "base_case_id": f"single-fact-{split}-g{index:03d}",
                        "family": "single_fact", "seed": seed + index, "value_version": version, "distractor_updates": 0,
                        "prefix": [asdict(event)], "variants": {"control": {"tail": [], "queries": [query]}}}
                examples.append({"case": case, "variant": "control", "query": query})
        random.Random(seed + 771).shuffle(examples)
        data[split] = examples
    validate_data(data, train_groups, validation_groups, values_per_group)
    return data


def validate_data(data, train_groups, validation_groups, values_per_group=4):
    entities = {}
    counts = {}
    for split, expected_groups, low, high in (("train", train_groups, 100, 500), ("validation", validation_groups, 500, 700)):
        groups, seen, used = defaultdict(list), set(), set()
        for example in data[split]:
            case, query = example["case"], example["query"]
            if (case["id"] in seen or example["variant"] != "control" or set(case["variants"]) != {"control"} or
                case["family"] != "single_fact" or case["distractor_updates"] != 0):
                raise ValueError("Duplicate identity or unsupported variant")
            seen.add(case["id"])
            if case["variants"]["control"] != {"tail": [], "queries": [query]} or len(case["prefix"]) != 1:
                raise ValueError("Expected one fact, no tail, and one current-value question")
            event = Operation(**case["prefix"][0])
            number = int(event.target[1:])
            if (event.kind != "SET" or not low <= number < high or event.value == number or
                event.source is not None or event.value is None or not 200 <= event.value < 800 or
                query != {"role": "target_current", "entity": event.target, "version": "current",
                          "question": f"What is the current balance of account {event.target}?",
                          "answer": str(execute([event])["current"][event.target])}):
                raise ValueError("Fact, query, label or namespace invalid")
            used.add(event.target)
            groups[case["base_case_id"]].append(example)
        if len(groups) != expected_groups or len(data[split]) != expected_groups * values_per_group or len(used) != expected_groups:
            raise ValueError("Wrong family/entity coverage")
        for siblings in groups.values():
            if (len(siblings) != values_per_group or {item["case"]["value_version"] for item in siblings} != set(range(values_per_group)) or
                len({item["query"]["question"] for item in siblings}) != 1 or
                len({item["query"]["answer"] for item in siblings}) != values_per_group):
                raise ValueError("Counterfactual siblings must share a question and have distinct labels")
        entities[split] = used
        counts[split] = {"examples": len(data[split]), "groups": len(groups), "siblings_per_group": values_per_group,
                         "question_text_lookup_correct_ceiling": sum(max(Counter(item["query"]["answer"] for item in siblings).values()) for siblings in groups.values())}
    if entities["train"] & entities["validation"]:
        raise ValueError("Train/validation entities overlap")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from sas_memory.record_kv_ablation import record_blocked_mask
    from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
    from scripts.run_controller_trial import encode_example, prompt_for, write_json
    data = build_data()
    counts = validate_data(data, 32, 16)
    model_path = Path(json.loads((ROOT / "configs/models.json").read_text())["it"])
    runtime = tokenizer_runtime(str(model_path))
    registry, encoded_train = {}, []
    for split in ("train", "validation"):
        registry[split] = []
        groups = defaultdict(list)
        for example in data[split]:
            case, query = example["case"], example["query"]
            ledger, prompt = prompt_for(runtime, case, "control", query)
            mask = record_blocked_mask(runtime, ledger, prompt)
            encoded = runtime.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
            value = query["answer"]
            if ledger.count(value) != 1:
                raise ValueError("Value occurrence not uniquely isolated in the fact")
            start, end = ledger.index(value), ledger.index(value) + len(value)
            indices = [index for index, (left, right) in enumerate(encoded["offset_mapping"]) if left < end and right > start]
            if len(indices) != 3 or any(not start <= encoded["offset_mapping"][index][0] < encoded["offset_mapping"][index][1] <= end for index in indices):
                raise ValueError("Fact value is not exactly three isolated digit tokens")
            ids = encoded["input_ids"]
            normalized = [-1 if index in indices else token for index, token in enumerate(ids)]
            row = {"case_id": case["id"], "base_case_id": case["base_case_id"], "value_version": case["value_version"],
                   "value_token_positions": indices, "normalized_prompt_sha256": data_fingerprint(normalized),
                   "question_ids_sha256": data_fingerprint(ids[mask["ledger_tokens"]:]), **mask}
            registry[split].append(row)
            groups[case["base_case_id"]].append(row)
            if split == "train":
                encoded_train.append(encode_example(runtime, example, "records_blocked"))
        for siblings in groups.values():
            signatures = {(row["ledger_tokens"], row["prompt_tokens"], tuple(row["value_token_positions"]),
                           row["normalized_prompt_sha256"], row["question_ids_sha256"], row["keep_sha256"]) for row in siblings}
            if len(signatures) != 1:
                raise ValueError("Counterfactual siblings differ beyond the three fact-value tokens")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("data", data), ("registry", registry), ("tokenized_train", encoded_train)):
        write_json(args.output / f"{name}.json", value)
    import hashlib
    source_paths = [Path(__file__).resolve()] + [ROOT / relative for relative in (
        "sas_memory/tasks.py", "sas_memory/training_data.py", "sas_memory/record_kv_ablation.py",
        "sas_memory/runtime.py", "sas_memory/kv_readout.py", "sas_memory/counterbalance_data.py",
        "scripts/analyze_blocked_reader_adaptation.py", "scripts/run_controller_trial.py")]
    manifest = {"status": "registered_before_inference", "registered_utc": datetime.now(timezone.utc).isoformat(),
                "training": False, "model_inference": False, "model_path": str(model_path), "python": sys.executable, "counts": counts,
                "data_sha256": data_fingerprint(data), "registry_sha256": data_fingerprint(registry),
                "tokenized_input_answer_sha256": data_fingerprint([{key: item[key] for key in ("input_ids", "answer_ids")} for item in encoded_train]),
                "supervised_training_answer_tokens": sum(len(item["answer_ids"]) for item in encoded_train),
                "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
                "tokenizer_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(model_path.iterdir()) if path.is_file() and ("token" in path.name or path.suffix == ".jinja" or path.name == "config.json")},
                "primary_metric": "All four different-value siblings answered correctly by unrestricted greedy decoding; also report individual-answer accuracy.",
                "interpretation": "Immediate single-fact carrier/readout positive control, not a long-memory or semantic-update result. Within each family only three value-token IDs differ; question text, token positions, all other input IDs and allowed history columns match. A deterministic prefix-independent answer can get at most one of four siblings correct. Retained carriers include recurrent/convolution state built with normal hybrid prefix attention."}
    for path in source_paths:
        source = args.output / "sources" / path.relative_to(ROOT)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(path.read_bytes())
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "counts": counts, "supervised_training_answer_tokens": manifest["supervised_training_answer_tokens"]}))


if __name__ == "__main__":
    main()
