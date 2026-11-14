"""Separate direct SET-leaf necessity from the sealed oracle-sufficiency result."""

from scripts.copy_necessity_data import MODELS, REGIMES, ROLES, selected_inputs
from scripts.copy_readout_metrics import paired_current
from scripts.copy_transfer_metrics import summarize as transfer_summary


def after_count(result, role):
    return sum(result["role_variant_counts"][role][v]["correct"] for v in ("control", "treatment"))


def necessity_checks(full, other, contrast):
    return {"other_current_after_no_lower_than_full": after_count(other, ROLES[0]) >= after_count(full, ROLES[0]),
        "other_history_after_no_lower_than_full": after_count(other, ROLES[2]) >= after_count(full, ROLES[2]),
        "other_unrelated_after_no_lower_than_full": after_count(other, ROLES[1]) >= after_count(full, ROLES[1]),
        "other_before_joint_no_lower_than_full": other["before_all_correct"] >= full["before_all_correct"],
        "positive_net_support_removal_loss": contrast["signed_net_gain"] > 0,
        "at_least_four_positive_loss_streams": len(contrast["positive_base_streams"]) >= 4}


def summarize(all_rows, full_rows, cases, registry, source_plan, plan):
    if list(all_rows) != list(MODELS) or any(list(all_rows[m]) != list(REGIMES) for m in MODELS):
        raise ValueError("Every frozen model and both whole-record interventions required")
    models = {}
    for regime in ("full", *REGIMES):
        rows = full_rows if regime == "full" else {m: all_rows[m][regime] for m in MODELS}
        models[regime] = transfer_summary(rows, cases, registry, source_plan)["models"]
    _, bindings = selected_inputs(registry)
    effects, versus_full = {}, {}
    for m in MODELS:
        indexed = {r: {v["id"]: v for v in (full_rows[m] if r == "full" else all_rows[m][r])}
                   for r in ("full", *REGIMES)}
        effects[m] = paired_current(indexed["remove_matched_other"], indexed["remove_support"], bindings)
        versus_full[m] = {r: paired_current(indexed[r], indexed["full"], bindings) for r in REGIMES}
    checks = {m: necessity_checks(models["full"][m], models["remove_matched_other"][m], effects[m])
              for m in ("local_question", "gdn_question_only")}
    return {"models_by_regime": models, "support_removal_effects": effects, "current_comparisons_vs_full": versus_full,
        "contrast_direction": "Candidate=remove_matched_other,reference=remove_support: gains mean support-removal harm; losses mean support-removal rescue; signed_net_gain is net support-removal harm.",
        "whole_record_necessity": {"passed": all(all(c.values()) for c in checks.values()), "model_checks": checks},
        "prior_sufficiency_and_address_gates_unchanged_passed": True,
        "prior_shared_history_repeat_primaries_remain_failed": True, "interpretation": plan["scope"]}
