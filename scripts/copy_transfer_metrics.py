"""Full COPY order/preservation outcomes and paired, base-stream comparisons."""

from collections import defaultdict
from statistics import mean

from scripts.copy_transfer_data import MODELS, ROLES, VARIANTS, selected_inputs


def paired_outcomes(candidate, reference):
    if [c["case_id"] for c in candidate] != [c["case_id"] for c in reference]:
        raise ValueError("Paired COPY comparisons need identical complete cases")
    pairs = [{"case_id": c["case_id"], "base_case_id": c["base_case_id"], "dependency_depth": c["dependency_depth"],
        "question_style": c["question_style"], "gain": c["joint_after"] and not r["joint_after"],
        "loss": r["joint_after"] and not c["joint_after"]} for c, r in zip(candidate, reference)]
    def counts(group):
        gains, losses = sum(p["gain"] for p in group), sum(p["loss"] for p in group)
        return {"cases": len(group), "gains": gains, "losses": losses, "signed_net_gain": gains - losses}
    streams = [{"base_case_id": stream, **counts([p for p in pairs if p["base_case_id"] == stream])}
               for stream in sorted({p["base_case_id"] for p in pairs})]
    depths = [{"depth": depth, **counts([p for p in pairs if p["dependency_depth"] == depth])} for depth in (1, 2, 3)]
    checks = {"positive_overall": counts(pairs)["signed_net_gain"] > 0,
              "at_least_three_positive_streams": sum(s["signed_net_gain"] > 0 for s in streams) >= 3,
              "positive_at_every_depth": all(d["signed_net_gain"] > 0 for d in depths)}
    return {**counts(pairs), "base_stream_results": streams, "depth_results": depths, "individual_pairs": pairs,
            "candidate_gate_checks": checks, "candidate_passed": all(checks.values())}


def summarize(all_rows, cases, registry, plan):
    physical, bindings = selected_inputs(registry)
    keys = [q["id"] for q in physical]
    if list(all_rows) != list(MODELS) or any([r["id"] for r in all_rows[name]] != keys for name in MODELS):
        raise ValueError("All eight complete ordered physical COPY evaluations required")
    results, outcomes = {}, {}
    for name in MODELS:
        indexed = {r["id"]: r for r in all_rows[name]}
        expanded = [{**b, **indexed[b["physical_id"]]} for b in bindings]
        by_case = {(r["case_id"], r["variant"], r["role"]): r for r in expanded}
        case_results = []
        for case in cases:
            def correct(variant, role):
                return by_case[case["id"], variant, role]["greedy_exact"]
            item = {key: case[key] for key in ("base_case_id", "dependency_depth", "question_style", "distractor_updates")}
            item.update(case_id=case["id"],
                both_order_targets_correct=all(correct(v, ROLES[0]) for v in VARIANTS[1:]),
                before_all_correct=all(correct(VARIANTS[0], role) for role in ROLES),
                joint_after=all(correct(v, role) for v in VARIANTS[1:] for role in ROLES),
                joint_all_nine=all(correct(v, role) for v in VARIANTS for role in ROLES),
                preservation={role: {v: {"baseline_correct": correct("before", role), "after_correct": correct(v, role),
                    "regression": correct("before", role) and not correct(v, role),
                    "rescue": not correct("before", role) and correct(v, role)} for v in VARIANTS[1:]} for role in ROLES[1:]})
            case_results.append(item)
        outcomes[name] = case_results
        cells = []
        for depth in (1, 2, 3):
            for style in ("original", "entity_first"):
                for variant in VARIANTS:
                    for role in ROLES:
                        group = [r for r in expanded if (r["dependency_depth"], r["question_style"], r["variant"], r["role"]) == (depth, style, variant, role)]
                        cells.append({"depth": depth, "question_style": style, "variant": variant, "role": role,
                            "correct": sum(r["greedy_exact"] for r in group), "examples": len(group),
                            "mean_answer_nll": mean(r["answer_nll"] for r in group)})
        by_role = {role: {v: {"correct": sum(r["greedy_exact"] for r in expanded if r["variant"] == v and r["role"] == role),
            "examples": 48} for v in VARIANTS} for role in ROLES}
        results[name] = {"physical_correct": sum(r["greedy_exact"] for r in all_rows[name]), "physical_queries": len(physical),
            "binding_correct": sum(r["greedy_exact"] for r in expanded), "bindings": len(bindings), "role_variant_counts": by_role,
            "joint_six_after": sum(c["joint_after"] for c in case_results),
            "joint_all_nine": sum(c["joint_all_nine"] for c in case_results),
            "both_order_targets_correct": sum(c["both_order_targets_correct"] for c in case_results),
            "before_all_correct": sum(c["before_all_correct"] for c in case_results), "cases": len(cases),
            "cells": cells, "case_outcomes": case_results}
    comparisons = [{"candidate": c, "reference": r, **paired_outcomes(outcomes[c], outcomes[r])}
                   for c, r in plan["analysis"]["comparisons"]]
    required = [c for c in comparisons if c["candidate"] == "shared" and c["reference"] in ("independent", "local_question", "gdn_question_only")]
    if len(required) != 3:
        raise ValueError("All three registered shared-control comparisons required")
    return {"models": results, "comparisons": comparisons,
        "descriptive_shared_candidate_gate": {"passed": all(c["candidate_passed"] for c in required),
            "comparators": {c["reference"]: c["candidate_gate_checks"] for c in required},
            "scope": "Reused development cohort; descriptive screen only, not architecture success, significance or fresh replication."},
        "interpretation": plan["scope"]}
