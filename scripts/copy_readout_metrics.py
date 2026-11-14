"""Separate frozen-prefix repair from program-address-specific improvement."""

from statistics import mean
from scripts.copy_transfer_data import MODELS, ROLES, selected_inputs
from scripts.copy_transfer_metrics import summarize as transfer_summary


def paired_current(candidate, reference, bindings):
    pairs = []
    for b in bindings:
        if b["variant"] == "before" or b["role"] != ROLES[0]:
            continue
        c, r = candidate[b["physical_id"]], reference[b["physical_id"]]
        pairs.append({**b, "candidate_correct": c["greedy_exact"], "reference_correct": r["greedy_exact"],
            "gain": c["greedy_exact"] and not r["greedy_exact"], "loss": r["greedy_exact"] and not c["greedy_exact"],
            "nll_candidate_minus_reference": c["answer_nll"] - r["answer_nll"]})
    if len(pairs) != 96 or len({(p["case_id"], p["variant"]) for p in pairs}) != 96:
        raise ValueError("All96current-target-after comparisons required")
    def counts(group):
        gains, losses = sum(p["gain"] for p in group), sum(p["loss"] for p in group)
        return {"pairs": len(group), "gains": gains, "losses": losses, "signed_net_gain": gains - losses}
    streams = [{"base_case_id": s, **counts([p for p in pairs if p["base_case_id"] == s])}
               for s in sorted({p["base_case_id"] for p in pairs})]
    return {**counts(pairs), "positive_base_streams": [s["base_case_id"] for s in streams if s["signed_net_gain"] > 0],
        "base_stream_results": streams, "depth_results": [{"depth": d, **counts([p for p in pairs if p["dependency_depth"] == d])} for d in (1, 2, 3)],
        "mean_nll_candidate_minus_reference": mean(p["nll_candidate_minus_reference"] for p in pairs), "individual_pairs": pairs}


def sufficiency_checks(full, oracle, comparison):
    def after(result, role):
        return sum(result["role_variant_counts"][role][v]["correct"] for v in ("control", "treatment"))
    return {"oracle_current_at_least90_of96": after(oracle, ROLES[0]) >= 90,
        "historical_after_no_lower_than_full": after(oracle, ROLES[2]) >= after(full, ROLES[2]),
        "unrelated_after_no_lower_than_full": after(oracle, ROLES[1]) >= after(full, ROLES[1]),
        "positive_net_current_repair": comparison["signed_net_gain"] > 0,
        "at_least_four_positive_base_streams": len(comparison["positive_base_streams"]) >= 4}


def summarize(all_rows, full_rows, cases, registry, source_plan, plan):
    if list(all_rows) != list(MODELS) or any(list(all_rows[m]) != ["oracle", "nonslice"] for m in MODELS):
        raise ValueError("Every fixed model and both registered regimes required")
    models = {}
    for regime in ("full", "oracle", "nonslice"):
        rows = full_rows if regime == "full" else {m: all_rows[m][regime] for m in MODELS}
        # Reuse only complete role/order/case statistics, never reinterpret the
        # old shared-candidate gate under oracle intervention.
        stats = transfer_summary(rows, cases, registry, source_plan)["models"]
        models[regime] = stats
    _, bindings = selected_inputs(registry)
    comparisons = {}
    for model in MODELS:
        indexed = {r: {v["id"]: v for v in (full_rows[model] if r == "full" else all_rows[model][r])}
                   for r in ("full", "oracle", "nonslice")}
        comparisons[model] = {r: paired_current(indexed["oracle"], indexed[r], bindings) for r in ("full", "nonslice")}
    gates, address = {}, {}
    for model in ("local_question", "gdn_question_only"):
        gates[model] = sufficiency_checks(models["full"][model], models["oracle"][model], comparisons[model]["full"])
        c = comparisons[model]["nonslice"]
        address[model] = {"positive_net_oracle_over_nonslice": c["signed_net_gain"] > 0,
                          "at_least_four_positive_base_streams": len(c["positive_base_streams"]) >= 4}
    return {"models_by_regime": models, "current_target_paired_comparisons": comparisons,
        "frozen_prefix_sufficiency": {"passed": all(all(g.values()) for g in gates.values()), "model_checks": gates},
        "address_specific_gain": {"passed": all(all(g.values()) for g in address.values()), "model_checks": address},
        "prior_shared_history_repeat_primaries_remain_failed": True, "interpretation": plan["scope"]}
