"""Register commuting initial-record permutations with identical query positions."""

import argparse
from collections import defaultdict
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
from scripts.run_controller_trial import prompt_for, write_json


def build_cases(data):
    originals = {example["case"]["id"]: example["case"] for example in data["validation"]}
    cases, permutations = {}, {}
    for case_id, original in originals.items():
        prefix = original["prefix"]
        if len(prefix) < 8 or any(event["kind"] != "SET" for event in prefix[:8]) or len({event["target"] for event in prefix[:8]}) != 8:
            raise ValueError("Expected eight independent initial SET records")
        group = original["base_case_id"]
        if group not in permutations:
            rng = random.Random(original["seed"] + 880001)
            for _ in range(1000):
                permutation = list(range(8))
                rng.shuffle(permutation)
                if all(index != source for index, source in enumerate(permutation)):
                    break
            else:
                raise ValueError("Failed to construct a deterministic derangement")
            permutations[group] = permutation
        case = deepcopy(original)
        case["prefix"] = [deepcopy(prefix[index]) for index in permutations[group]] + deepcopy(prefix[8:])
        for name, variant in original["variants"].items():
            before = execute([Operation(**event) for event in prefix + variant["tail"]])
            after = execute([Operation(**event) for event in case["prefix"] + case["variants"][name]["tail"]])
            if before != after:
                raise ValueError("Independent initial permutation changed executable truth")
        cases[case_id] = case
    return cases, permutations


def token_registry(runtime, data, cases, original_registry):
    result, groups = [], defaultdict(list)
    for example, original in zip(data["validation"], original_registry["validation"]):
        case, query = cases[example["case"]["id"]], example["query"]
        ledger, prompt = prompt_for(runtime, case, example["variant"], query)
        tokens = runtime.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        boundary, positions = len(runtime.ids(ledger)), []
        for match in re.finditer(r"\b\d{3}\b", ledger):
            indices = [index for index, (left, right) in enumerate(tokens["offset_mapping"]) if left < match.end() and right > match.start()]
            if len(indices) != 3 or [tuple(tokens["offset_mapping"][index]) for index in indices] != [(match.start() + i, match.start() + i + 1) for i in range(3)]:
                raise ValueError("Permuted SET value digit token identity failed")
            positions.extend(indices)
        positions, ids = sorted(set(positions)), tokens["input_ids"]
        position_set = set(positions)
        row = {"case_id": case["id"], "base_case_id": case["base_case_id"], "variant": example["variant"], "role": query["role"],
               "value_version": case["value_version"], "ledger_tokens": boundary, "prompt_tokens": len(ids),
               "value_token_positions": positions, "prompt_ids_sha256": data_fingerprint(ids),
               "normalized_prompt_sha256": data_fingerprint([-1 if index in position_set else token for index, token in enumerate(ids)]),
               "question_ids_sha256": data_fingerprint(ids[boundary:])}
        if (ids[:boundary] != runtime.ids(ledger) or
            any(row[key] != original[key] for key in ("ledger_tokens", "prompt_tokens", "question_ids_sha256"))):
            raise ValueError("Permutation changed query token IDs or absolute query positions")
        if row["prompt_ids_sha256"] == original["prompt_ids_sha256"]:
            raise ValueError("Initial record permutation did not change the input")
        result.append(row)
        groups[case["base_case_id"], example["variant"], query["role"]].append(row)
    for siblings in groups.values():
        signatures = {(row["ledger_tokens"], row["prompt_tokens"], tuple(row["value_token_positions"]), row["normalized_prompt_sha256"], row["question_ids_sha256"]) for row in siblings}
        if len(siblings) != 4 or len(signatures) != 1:
            raise ValueError("Permuted value siblings lost non-value input identity")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, default=Path("/media/990Pro/Code/SAS/datasets/memory_update/paired-semantic-updates-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, data, original_registry = registered_dataset(args.base_data)
    cases, permutations = build_cases(data)
    runtime = tokenizer_runtime(base["model_path"])
    registry = token_registry(runtime, data, cases, original_registry)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("cases", cases), ("permutations", permutations), ("registry", registry)):
        write_json(args.output / f"{name}.json", value)
    sources = [Path(__file__).resolve(), ROOT / "scripts/analyze_paired_semantic_probe.py", ROOT / "scripts/prepare_paired_semantic_updates.py",
               ROOT / "scripts/run_controller_trial.py", ROOT / "scripts/analyze_blocked_reader_adaptation.py",
               ROOT / "sas_memory/tasks.py", ROOT / "sas_memory/training_data.py", ROOT / "sas_memory/runtime.py"]
    manifest = {"status": "registered_before_inference", "registered_utc": datetime.now(timezone.utc).isoformat(), "training": False, "model_inference": False,
                "base_data": str(args.base_data), "base_data_manifest_sha256": file_hash(args.base_data / "manifest.json"),
                "base_data_sha256": base["data_sha256"], "model_path": base["model_path"], "condition": "initial_record_derangement",
                "cases": len(cases), "base_families": len(permutations), "examples": len(registry),
                "cases_sha256": data_fingerprint(cases), "permutations_sha256": data_fingerprint(permutations), "registry_sha256": data_fingerprint(registry),
                "source_sha256": {str(source.relative_to(ROOT)): file_hash(source) for source in sources},
                "interpretation": "Reused-validation functional order diagnostic, not untouched confirmation. All eight independent initial SET records move; tails, questions, executable current/initial truth, query IDs and absolute query positions stay fixed. Recomputed recurrent states and contextual KV differ, so failures do not isolate GDN write damage. Four-value matching still holds within each question/variant. No training or method selection on this condition is registered yet."}
    for relative in manifest["source_sha256"]:
        destination = args.output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "cases", "base_families", "examples", "cases_sha256", "registry_sha256")}))


if __name__ == "__main__":
    main()
