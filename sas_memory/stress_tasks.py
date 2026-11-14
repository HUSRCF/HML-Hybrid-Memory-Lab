"""Overwrite stress cases with independently controlled duration and entity load."""

import random

from sas_memory.tasks import Operation, execute
from dataclasses import asdict


def make_stress_case(seed: int, distractor_updates: int, distractor_entities: int) -> dict:
    if not 4 <= distractor_entities <= 896 or distractor_updates < 0:
        raise ValueError("Need 4..896 distractor entities and nonnegative update count")
    entity_rng = random.Random(seed)
    entities = [f"E{i}" for i in entity_rng.sample(range(100, 1000), 900)]
    a, b, c, d = entities[:4]
    pool = entities[4:4 + distractor_entities]
    # Draw targets before any distractors; all target labels and initial facts
    # remain identical across duration/load conditions for the same seed.
    value_rng = random.Random(seed + 104729)
    values = value_rng.sample(range(200, 800), 12)
    prefix = [Operation("SET", name, value) for name, value in zip(entities[:8], values[:8])]
    allowed_values = [value for value in range(200, 800) if value not in values]
    write_rng = random.Random(seed + 130363)
    address_rng = random.Random(seed + 155921)
    prefix += [Operation("SET", pool[int(address_rng.random() * len(pool))], write_rng.choice(allowed_values))
               for _ in range(distractor_updates)]
    tails = {"before": [], "control": [Operation("SET", c, values[8])],
             "treatment": [Operation("SET", a, values[9])]}
    variants = {}
    for name, tail in tails.items():
        state = execute(prefix + tail)
        queries = []
        for role, entity, version in (("target_current", a, "current"), ("protected_current", b, "current"),
                                      ("target_initial", a, "initial")):
            question = (f"What is the current balance of account {entity}?" if version == "current"
                        else f"What was the initial balance of account {entity}?")
            queries.append({"role": role, "entity": entity, "version": version,
                            "question": question, "answer": str(state[version][entity])})
        variants[name] = {"tail": [asdict(event) for event in tail], "queries": queries}
    return {"id": f"overwrite-s{seed}-n{distractor_updates}-e{distractor_entities}", "protocol_version": 2,
            "family": "overwrite", "seed": seed, "distractor_updates": distractor_updates,
            "distractor_entities": distractor_entities, "observed_distractor_entities": len({e.target for e in prefix[4:]}),
            "prefix": [asdict(event) for event in prefix], "variants": variants,
            "candidates": sorted({q["answer"] for v in variants.values() for q in v["queries"]} | {str(values[8])})}
