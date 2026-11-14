"""Functional update metrics; equal or different wrong answers are not success."""

from collections import Counter, defaultdict
import re
from statistics import mean

ROLES = {"target_current", "protected_current", "target_initial"}
VARIANTS = {"before", "control", "treatment"}


def numeric_prediction(row):
    match = re.fullmatch(r"\s*(-?\d+)\s*", row["generated"])
    return match.group(1) if match else None


def summarize_semantics(rows):
    if not rows:
        raise ValueError("Empty semantic evaluation")
    query_groups, contexts, cases, cells = (defaultdict(list) for _ in range(4))
    seen = set()
    for row in rows:
        identity = row["case_id"], row["variant"], row["role"]
        if identity in seen or row["variant"] not in VARIANTS or row["role"] not in ROLES:
            raise ValueError("Duplicate or invalid semantic readout")
        if row["greedy_exact"] != (numeric_prediction(row) == row["answer"]):
            raise ValueError("Correctness disagrees with unrestricted generation")
        seen.add(identity)
        query_groups[(row["base_case_id"], row["variant"], row["role"])].append(row)
        contexts[(row["case_id"], row["variant"])].append(row)
        cases[row["case_id"]].append(row)
        cells[(row["family"], row["variant"], row["role"])].append(row)
    histogram, cell_families = Counter(), defaultdict(list)
    for siblings in query_groups.values():
        if (len(siblings) != 4 or {row["value_version"] for row in siblings} != {0, 1, 2, 3} or
            len({row["answer"] for row in siblings}) != 4):
            raise ValueError("Incomplete four-value semantic family")
        correct = sum(row["greedy_exact"] for row in siblings)
        histogram[correct] += 1
        first = siblings[0]
        cell_families[(first["family"], first["variant"], first["role"])].append(correct == 4)
    joint, value_variant = {}, defaultdict(list)
    for key, members in contexts.items():
        if len(members) != 3 or {row["role"] for row in members} != ROLES:
            raise ValueError("Incomplete current/protected/initial context")
        joint[key] = all(row["greedy_exact"] for row in members)
        value_variant[(members[0]["base_case_id"], members[0]["variant"])].append((members[0]["value_version"], joint[key]))
    if any(len(members) != 4 or {version for version, _ in members} != {0, 1, 2, 3} for members in value_variant.values()):
        raise ValueError("Incomplete twelve-readout semantic group")
    preservation, order_pairs, regressions = defaultdict(list), defaultdict(list), []
    for case_id, members in cases.items():
        if len(members) != 9 or {(row["variant"], row["role"]) for row in members} != {(variant, role) for variant in VARIANTS for role in ROLES}:
            raise ValueError("Incomplete before/control/treatment case")
        indexed = {(row["variant"], row["role"]): row for row in members}
        family = members[0]["family"]
        for role in sorted(ROLES):
            before, control, treatment = [indexed[variant, role] for variant in ("before", "control", "treatment")]
            if role != "target_current" and len({row["answer"] for row in (before, control, treatment)}) != 1:
                raise ValueError("Protected/initial executable truth changed")
            for after in (control, treatment):
                if before["answer"] != after["answer"]:
                    continue
                item = {"before_correct": before["greedy_exact"], "after_correct": after["greedy_exact"]}
                preservation[family, role].append(item)
                if item["before_correct"] and not item["after_correct"]:
                    regressions.append({"case_id": case_id, "base_case_id": after["base_case_id"], "family": family,
                                        "value_version": after["value_version"], "variant": after["variant"], "role": role,
                                        "answer": after["answer"], "before_generated": before["generated"], "after_generated": after["generated"]})
            first, second = numeric_prediction(control), numeric_prediction(treatment)
            order_pairs[family, role].append({"gold_changes": control["answer"] != treatment["answer"],
                "both_correct": control["greedy_exact"] and treatment["greedy_exact"],
                "control_correct": control["greedy_exact"], "treatment_correct": treatment["greedy_exact"],
                "valid_pair": first is not None and second is not None,
                "predictions_change": first != second if first is not None and second is not None else None})
    return {"examples": len(rows), "correct": sum(row["greedy_exact"] for row in rows),
            "accuracy": mean(row["greedy_exact"] for row in rows), "answer_nll": mean(row["answer_nll"] for row in rows),
            "four_value_query_groups": {"groups": len(query_groups), "all_four_correct": histogram[4],
                "more_than_one_correct": sum(count for correct, count in histogram.items() if correct > 1),
                "correct_per_group_histogram": {str(index): histogram[index] for index in range(5)}},
            "joint_roles": {"contexts": len(joint), "all_three_correct": sum(joint.values())},
            "all_twelve_value_variant_groups": {"groups": len(value_variant),
                "all_twelve_correct": sum(all(correct for _, correct in members) for members in value_variant.values())},
            "cells": [{"family": family, "variant": variant, "role": role, "examples": len(members),
                "correct": sum(row["greedy_exact"] for row in members), "answer_nll": mean(row["answer_nll"] for row in members),
                "four_value_groups": len(cell_families[family, variant, role]),
                "all_four_correct": sum(cell_families[family, variant, role])}
                for (family, variant, role), members in sorted(cells.items())],
            "before_after_preservation": [{"family": family, "role": role, "pairs": len(members),
                "before_correct": sum(item["before_correct"] for item in members),
                "after_correct": sum(item["after_correct"] for item in members),
                "regressions": sum(item["before_correct"] and not item["after_correct"] for item in members),
                "rescues": sum(not item["before_correct"] and item["after_correct"] for item in members)}
                for (family, role), members in sorted(preservation.items())],
            "preservation_regressions": sorted(regressions, key=lambda row: (row["case_id"], row["variant"], row["role"])),
            "order_pairs": [{"family": family, "role": role, "pairs": len(members),
                "gold_change_pairs": sum(item["gold_changes"] for item in members),
                "both_correct": sum(item["both_correct"] for item in members),
                "control_correct": sum(item["control_correct"] for item in members),
                "treatment_correct": sum(item["treatment_correct"] for item in members),
                "valid_prediction_pairs": sum(item["valid_pair"] for item in members),
                "predicted_change_pairs": sum(item["predictions_change"] is True for item in members),
                "predicted_unchanged_pairs": sum(item["predictions_change"] is False for item in members)}
                for (family, role), members in sorted(order_pairs.items())],
            "interpretation": "Functional readout diagnostics. Equal/different wrong predictions are not semantic success. Before/after lengths, positions and downstream computation differ, so regressions do not isolate recurrent write damage."}
