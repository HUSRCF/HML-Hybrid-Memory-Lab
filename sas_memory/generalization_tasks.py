"""Matched, held-out COPY chains and question wording, with executable truth.

Conditions share a base case. Their rows are dependent and must be clustered by
base_case_id rather than treated as independent examples in pooled intervals.
"""

from copy import deepcopy
from collections import defaultdict
from dataclasses import asdict

from sas_memory.tasks import Operation, execute
from sas_memory.training_data import namespaced_case


def copy_variant(base: dict, depth: int = 1, question_style: str = "original") -> dict:
    if base["family"] != "copy" or depth < 1 or question_style not in ("original", "entity_first"):
        raise ValueError((base["family"], depth, question_style))
    case = deepcopy(base)
    first_copy, source_update = [Operation(**event) for event in base["variants"]["control"]["tail"]]
    if first_copy.kind != "COPY" or source_update.kind != "SET" or source_update.target != first_copy.source:
        raise ValueError("Expected a snapshot-copy/source-update base case")
    protected = next(q["entity"] for q in base["variants"]["control"]["queries"] if q["role"] == "protected_current")
    entities = list(dict.fromkeys(event["target"] for event in base["prefix"]))
    available = [entity for entity in entities if entity not in (first_copy.target, first_copy.source, protected)]
    if depth - 1 > len(available):
        raise ValueError("Not enough distinct intermediate accounts")
    path = [first_copy.source, *available[:depth - 1], first_copy.target]
    copies = [Operation("COPY", target=target, source=source) for source, target in zip(path, path[1:])]
    tails = {"before": [], "control": copies + [source_update], "treatment": [source_update] + copies}
    case.update(id=f"{base['id']}-depth{depth}-{question_style}", base_case_id=base["id"],
                dependency_depth=depth, question_style=question_style)
    for name, tail in tails.items():
        variant = case["variants"][name]
        variant["tail"] = [asdict(event) for event in tail]
        state = execute([Operation(**event) for event in case["prefix"]] + tail)
        for query in variant["queries"]:
            query["answer"] = str(state[query["version"]][query["entity"]])
            if question_style == "entity_first":
                predicate = "what is its current balance?" if query["version"] == "current" else "what was its initial balance?"
                query["question"] = f"For account {query['entity']}, {predicate}"
    case["candidates"] = sorted(set(base["candidates"]) | {q["answer"] for v in case["variants"].values() for q in v["queries"]})
    return case


def build_generalization_cases(seeds=tuple(range(60000, 60008)), depths=(1, 2, 3),
                               question_styles=("original", "entity_first")) -> list[dict]:
    result = []
    for index, seed in enumerate(seeds):
        base = namespaced_case("copy", seed, (8, 64)[index % 2], "test")
        for depth in depths:
            for style in question_styles:
                result.append(copy_variant(base, depth, style))
    return result


def validate_generalization_cases(cases: list[dict]) -> dict:
    """Validate gold semantics and matched clusters before interpreting model data."""
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Empty cohort or duplicate case IDs")
    clusters, conditions = defaultdict(set), set()
    prefixes = {}
    roles = {"target_current", "target_initial", "protected_current"}
    for case in cases:
        if case["family"] != "copy":
            raise ValueError("Expected COPY generalization cases")
        base = case["base_case_id"]
        condition = (case["dependency_depth"], case["question_style"])
        if condition in clusters[base]:
            raise ValueError("Duplicate matched condition")
        clusters[base].add(condition)
        conditions.add(condition)
        identity = (case["seed"], case["distractor_updates"], case["prefix"])
        if base in prefixes and prefixes[base] != identity:
            raise ValueError("Matched conditions do not share the same base facts")
        prefixes[base] = identity
        if set(case["variants"]) != {"before", "control", "treatment"}:
            raise ValueError("Incomplete order variants")
        for variant in case["variants"].values():
            events = [Operation(**event) for event in case["prefix"] + variant["tail"]]
            state = execute(events)
            queries = variant["queries"]
            if len(queries) != 3 or {query["role"] for query in queries} != roles:
                raise ValueError("Incomplete functional readouts")
            for query in queries:
                if str(state[query["version"]][query["entity"]]) != query["answer"]:
                    raise ValueError("Gold answer disagrees with executable semantics")
    if any(members != conditions for members in clusters.values()):
        raise ValueError("Unbalanced matched conditions across base cases")
    return {"cases": len(cases), "base_cases": len(clusters),
            "conditions": [{"dependency_depth": depth, "question_style": style} for depth, style in sorted(conditions)]}


def copy_update_challenges(case: dict) -> dict[str, list[Operation]]:
    """Add a matched source/unrelated SET after a snapshot COPY chain.

    Both source and unrelated accounts are initialized in the shared short prefix.
    The only difference between the two update conditions is the updated account.
    """
    from sas_memory.oracle import evidence_slice
    variant = case["variants"]["control"]
    events = [Operation(**event) for event in case["prefix"] + variant["tail"]]
    query = next(query for query in variant["queries"] if query["role"] == "target_current")
    protected = next(query["entity"] for query in variant["queries"] if query["role"] == "protected_current")
    minimal = evidence_slice(events, query["entity"], "current")
    update = events[-1]
    if update.kind != "SET" or any(event is update for event in minimal):
        raise ValueError("Expected a final source update irrelevant to the copied target")
    involved = set().union(*(event.reads | event.writes for event in minimal))
    if protected in involved or update.target not in involved:
        raise ValueError("Update accounts do not match the dependency control")
    initialization = next(event for event in events[:len(case["prefix"])] if event.target == protected)
    if initialization.kind != "SET":
        raise ValueError("Expected an initialized unrelated account")
    chosen = {id(event) for event in minimal + [initialization]}
    known = [event for event in events if id(event) in chosen]
    conditions = {"minimal": minimal, "known_accounts": known, "source_update": known + [update],
                  "unrelated_update": known + [Operation("SET", protected, update.value)]}
    for selected in conditions.values():
        if str(execute(selected)["current"][query["entity"]]) != query["answer"]:
            raise ValueError("Update challenge changed the copied target's truth")
    return conditions
