"""Paired functional metrics; candidate ranking is not free-generation accuracy."""

from collections import defaultdict
from statistics import mean


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["distractor_updates"], row.get("distractor_entities", 4), row["variant"], row["role"])].append(row)
    grouped = []
    for (family, length, entities, variant, role), members in sorted(groups.items()):
        generated = [m["greedy_exact"] for m in members if m["greedy_exact"] is not None]
        grouped.append({"family": family, "distractor_updates": length, "distractor_entities": entities, "variant": variant, "role": role,
                        "count": len(members), "candidate_accuracy": mean(m["candidate_correct"] for m in members),
                        "greedy_exact_accuracy": mean(generated) if generated else None,
                        "answer_nll": mean(m["answer_nll"] for m in members)})
    cases = defaultdict(dict)
    for row in rows:
        cases[row["case_id"]][(row["variant"], row["role"])] = row
    paired = []
    for case_id, members in sorted(cases.items()):
        for role in ("protected_current", "target_initial"):
            required = [(variant, role) for variant in ("before", "control", "treatment")]
            if not all(key in members for key in required):
                continue
            before, control, treatment = (members[key] for key in required)
            if len({m["answer"] for m in (before, control, treatment)}) != 1:
                raise AssertionError("Preservation metric requires the same true answer")
            paired.append({"case_id": case_id, "family": before["family"], "role": role,
                           "distractor_updates": before["distractor_updates"],
                           "distractor_entities": before.get("distractor_entities", 4),
                           "nll_treatment_minus_control": treatment["answer_nll"] - control["answer_nll"],
                           "before_correct": before["greedy_exact"],
                           "control_correct": control["greedy_exact"], "treatment_correct": treatment["greedy_exact"],
                           "same_ledger_token_length": control["ledger_tokens"] == treatment["ledger_tokens"]})
    return {
        "rows": len(rows), "cases": len(cases), "grouped": grouped, "paired_preservation": paired,
        "interpretation": "Small controlled pilot; no causal attribution or architecture benefit established.",
    }
