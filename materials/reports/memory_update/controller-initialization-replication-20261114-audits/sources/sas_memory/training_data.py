"""Versioned, executable adapter data with disjoint entity namespaces."""

from __future__ import annotations

import hashlib
import json
import random
import re

from sas_memory.tasks import make_case


ENTITY_RANGES = {"train": (100, 500), "validation": (500, 700), "test": (700, 1000)}
SEED_BASES = {"train": 10000, "validation": 20000, "test": 30000}


def namespaced_case(family: str, seed: int, updates: int, split: str) -> dict:
    case = make_case(family, seed, updates)
    start, stop = ENTITY_RANGES[split]
    names = sorted({event["target"] for event in case["prefix"]})
    replacement = random.Random(seed + 99991).sample(range(start, stop), len(names))
    mapping = dict(zip(names, (f"E{value}" for value in replacement)))
    for event in case["prefix"] + [e for v in case["variants"].values() for e in v["tail"]]:
        event["target"] = mapping[event["target"]]
        if event["source"] is not None:
            event["source"] = mapping[event["source"]]
    for variant in case["variants"].values():
        for query in variant["queries"]:
            query["entity"] = mapping[query["entity"]]
            query["question"] = re.sub(r"E\d{3}", lambda match: mapping[match.group()], query["question"])
    case["id"] = f"{split}-{case['id']}"
    case["entity_namespace"] = [start, stop]
    return case


def build_training_examples(count: int) -> list[dict]:
    examples = []
    # Exactly balance roles in every four microbatches; COPY dominates the task
    # mix, with ordinary assignment/swap examples guarding against specialization.
    families = ["copy"] * 6 + ["overwrite"] * 2 + ["dependent_swap", "independent_swap"]
    rng = random.Random(7027)
    for index in range(count):
        family = rng.choice(families)
        case = namespaced_case(family, SEED_BASES["train"] + index, rng.choice((8, 16, 32)), "train")
        variant_name = "control" if rng.random() < 0.5 else "treatment"
        role = ("target_current", "target_current", "target_initial", "protected_current")[index % 4]
        query = next(q for q in case["variants"][variant_name]["queries"] if q["role"] == role)
        examples.append({"case": case, "variant": variant_name, "query": query})
    return examples


def build_eval_cases(split: str, copy_cases: int, overwrite_cases: int, seed_offset: int = 0) -> list[dict]:
    if split not in ("validation", "test"):
        raise ValueError(split)
    families = ["copy"] * copy_cases + ["overwrite"] * overwrite_cases
    return [namespaced_case(family, SEED_BASES[split] + seed_offset + index, (8, 64)[index % 2], split)
            for index, family in enumerate(families)]


def data_fingerprint(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
