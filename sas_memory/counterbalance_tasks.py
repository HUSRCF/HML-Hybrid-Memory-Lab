"""Fresh COPY chains with equal-valued, position-balanced late SET controls."""

from copy import deepcopy
from collections import defaultdict
from dataclasses import asdict

from sas_memory.generalization_tasks import copy_variant
from sas_memory.oracle import evidence_slice
from sas_memory.tasks import Operation, execute, independent
from sas_memory.training_data import namespaced_case

ORDERS = ("source_first", "unrelated_first")
MASKS = ("oracle", "extra_source", "extra_unrelated")


def build_counterbalance_cases(seeds=range(70000, 70024), depths=(1, 2, 3),
                               styles=("original", "entity_first")):
    cases = []
    for index, seed in enumerate(seeds):
        base = namespaced_case("copy", seed, (8, 64)[index % 2], "test")
        for depth in depths:
            for style in styles:
                original = copy_variant(base, depth, style)
                copies = [Operation(**e) for e in original["variants"]["control"]["tail"][:-1]]
                source_update = Operation(**original["variants"]["control"]["tail"][-1])
                queries = original["variants"]["control"]["queries"]
                query = deepcopy(next(q for q in queries if q["role"] == "target_current"))
                unrelated = next(q["entity"] for q in queries if q["role"] == "protected_current")
                unrelated_update = Operation("SET", unrelated, source_update.value)
                variants = {}
                for order, updates in (("source_first", [source_update, unrelated_update]),
                                       ("unrelated_first", [unrelated_update, source_update])):
                    variants[order] = {"tail": [asdict(e) for e in copies + updates], "queries": [deepcopy(query)]}
                cases.append({"id": "counterbalance-" + original["id"], "base_case_id": original["base_case_id"],
                              "family": "copy_counterbalance", "seed": seed, "distractor_updates": original["distractor_updates"],
                              "dependency_depth": depth, "question_style": style, "prefix": original["prefix"],
                              "source_entity": source_update.target, "unrelated_entity": unrelated,
                              "target_entity": query["entity"], "new_value": source_update.value,
                              "variants": variants})
    return cases


def counterbalance_indices(case, order, kind):
    if order not in ORDERS or kind not in MASKS:
        raise ValueError("Unknown counterbalance condition")
    events = [Operation(**e) for e in case["prefix"] + case["variants"][order]["tail"]]
    chosen = evidence_slice(events, case["target_entity"], "current")
    ids = {id(event) for event in chosen}
    selected = {i for i, event in enumerate(events) if id(event) in ids}
    if kind != "oracle":
        entity = case["source_entity"] if kind == "extra_source" else case["unrelated_entity"]
        extra = next(i for i in range(len(events) - 2, len(events)) if events[i].target == entity)
        if extra in selected:
            raise ValueError("Late control record is necessary evidence")
        selected.add(extra)
    return sorted(selected)


def validate_counterbalance_cases(cases):
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Empty or duplicate counterbalance cases")
    conditions, clusters, prefixes = set(), defaultdict(set), {}
    for case in cases:
        condition = (case["dependency_depth"], case["question_style"])
        base = case["base_case_id"]
        if case["family"] != "copy_counterbalance" or condition in clusters[base]:
            raise ValueError("Wrong family or duplicate condition")
        conditions.add(condition)
        clusters[base].add(condition)
        identity = (case["seed"], case["distractor_updates"], case["prefix"], case["new_value"],
                    case["source_entity"], case["unrelated_entity"], case["target_entity"])
        if base in prefixes and prefixes[base] != identity:
            raise ValueError("Matched conditions changed base facts")
        prefixes[base] = identity
        if set(case["variants"]) != set(ORDERS):
            raise ValueError("Missing counterbalanced order")
        source, unrelated, target = (case[name + "_entity"] for name in ("source", "unrelated", "target"))
        if len({source, unrelated, target}) != 3:
            raise ValueError("Control accounts must differ")
        prefix = [Operation(**e) for e in case["prefix"]]
        old_source = execute(prefix)["current"][source]
        if old_source == case["new_value"]:
            raise ValueError("Late value does not differ from snapshot")
        tails = []
        for order in ORDERS:
            variant = case["variants"][order]
            tail = [Operation(**e) for e in variant["tail"]]
            copies, updates = tail[:-2], tail[-2:]
            if len(copies) != case["dependency_depth"] or any(e.kind != "COPY" for e in copies):
                raise ValueError("Incorrect COPY depth")
            path = [copies[0].source] + [e.target for e in copies]
            if path[0] != source or path[-1] != target or len(set(path)) != len(path) or unrelated in path:
                raise ValueError("Invalid distinct-account COPY path")
            if any(e.source != previous for previous, e in zip(path, copies)):
                raise ValueError("Broken COPY chain")
            expected_updates = [Operation("SET", source, case["new_value"]), Operation("SET", unrelated, case["new_value"])]
            if order == "unrelated_first":
                expected_updates.reverse()
            if updates != expected_updates or not independent(*updates):
                raise ValueError("Late SETs are not equal-valued independent updates")
            queries = variant["queries"]
            if len(queries) != 1 or queries[0]["role"] != "target_current" or queries[0]["entity"] != target or queries[0]["version"] != "current":
                raise ValueError("Wrong counterbalance readout")
            question = (f"What is the current balance of account {target}?" if case["question_style"] == "original" else
                        f"For account {target}, what is its current balance?" if case["question_style"] == "entity_first" else None)
            if queries[0]["question"] != question or queries[0]["answer"] != str(old_source):
                raise ValueError("Gold/question disagrees with snapshot semantics")
            events = prefix + tail
            if execute(events)["current"][target] != old_source:
                raise ValueError("Late updates changed the copied target")
            for kind in MASKS:
                indices = counterbalance_indices(case, order, kind)
                if execute([events[i] for i in indices])["current"][target] != old_source:
                    raise ValueError("Selected evidence changed the answer")
            tails.append(tail)
        if tails[0][:-2] != tails[1][:-2] or tails[0][-2:] != tails[1][-2:][::-1]:
            raise ValueError("Variants differ beyond the two late positions")
    if any(members != conditions for members in clusters.values()):
        raise ValueError("Unbalanced conditions across base facts")
    return {"cases": len(cases), "base_cases": len(clusters), "contexts": len(cases) * len(ORDERS),
            "conditions": [{"dependency_depth": depth, "question_style": style} for depth, style in sorted(conditions)]}
