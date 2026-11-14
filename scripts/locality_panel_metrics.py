"""Own-world and joint-update comparisons for the complete five-model panel."""

from collections import defaultdict
from statistics import mean

from scripts.analyze_write_locality_event_paths import paired_outcomes
from scripts.locality_panel_data import MODELS
from scripts.locality_transfer_data import selected_inputs

COMPARISONS = {"shared_versus_independent": ("independent", "shared"),
               "shared_versus_local": ("local", "shared"),
               "local_question_versus_local": ("local", "local_question")}


def summarize(registry, rows_by_model, smoke=False):
    pairs, bindings, physical = selected_inputs(registry, smoke)
    queries = {q["id"]: q for q in physical}
    indexed = {"native": {key: q["native_full_readout"] for key, q in queries.items()}}
    if list(rows_by_model) != list(MODELS):
        raise ValueError("All five registered fixed models required")
    for model, rows in rows_by_model.items():
        indexed[model] = {row["id"]: row for row in rows}
        if len(rows) != len(queries) or indexed[model].keys() != queries.keys():
            raise ValueError("Panel metrics require exactly every selected physical query")
    binding_index = {(b["pair_id"], b["side"], b["label"]): b["physical_id"] for b in bindings}
    labels = {p["id"]: [b["label"] for b in bindings if b["pair_id"] == p["id"] and b["side"] == "left"] for p in pairs}

    def readout(model, pair, side, label):
        return indexed[model][binding_index[(pair["id"], side, label)]]

    def correct(model, pair, side, label):
        return readout(model, pair, side, label)["greedy_exact"]

    def joint(model, pair, side):
        return all(correct(model, pair, side, label) for label in labels[pair["id"]])

    def counts(group):
        result = {}
        for model in indexed:
            result[model] = {}
            for side in ("left", "right"):
                result[model][side] = {label: {"correct": sum(correct(model, p, side, label) for p in group),
                    "examples": len(group), "answer_nll": mean(readout(model, p, side, label)["answer_nll"] for p in group)}
                    for label in labels[group[0]["id"]]}
                result[model][side]["all_roles_joint"] = {"correct": sum(joint(model, p, side) for p in group), "examples": len(group)}
        return result

    def compare(group, before, after):
        result = {}
        for side in ("left", "right"):
            result[side] = {label: paired_outcomes([correct(before, p, side, label) for p in group],
                                                   [correct(after, p, side, label) for p in group])
                            for label in labels[group[0]["id"]]}
            result[side]["all_roles_joint"] = paired_outcomes([joint(before, p, side) for p in group],
                                                              [joint(after, p, side) for p in group])
        return result

    groups, streams = defaultdict(list), defaultdict(list)
    for pair in pairs:
        groups[(pair["cohort"], pair["stratum"])].append(pair)
        streams[(pair["cohort"], pair["stratum"], pair["base_stream_id"])].append(pair)
    grouped = []
    for (cohort, stratum), members in groups.items():
        induced = [p for p in members if correct("native", p, "left", "preservation") and not correct("native", p, "right", "preservation")]
        grouped.append({"cohort": cohort, "stratum": stratum, "pairs": len(members),
            "base_streams": sorted({p["base_stream_id"] for p in members}), "counts": counts(members),
            "versus_native": {model: compare(members, "native", model) for model in MODELS},
            "architecture_comparisons": {name: {"reference": before, "candidate": after,
                "outcomes": compare(members, before, after)} for name, (before, after) in COMPARISONS.items()},
            "preservation_left_to_right": {model: paired_outcomes(
                [correct(model, p, "left", "preservation") for p in members],
                [correct(model, p, "right", "preservation") for p in members]) for model in indexed},
            "native_full_update_regression_cases": [p["case_id"] for p in induced],
            "native_full_update_regression_repairs": {model: {
                "preservation_repaired": [p["case_id"] for p in induced if correct(model, p, "right", "preservation")],
                "all_roles_joint_repaired": [p["case_id"] for p in induced if joint(model, p, "right")]} for model in MODELS}})
    by_stream = [{"cohort": key[0], "stratum": key[1], "base_stream_id": key[2], "value_cases": len(members),
        "counts": counts(members), "all_values_roles_and_sides_correct": {
            model: all(joint(model, p, side) for p in members for side in ("left", "right")) for model in indexed}}
        for key, members in streams.items()]
    unique = {model: {"correct": sum(r["greedy_exact"] for r in rows.values()), "physical_queries": len(queries),
        "reported_binding_correct": sum(rows[b["physical_id"]]["greedy_exact"] for b in bindings),
        "reported_bindings": len(bindings)} for model, rows in indexed.items()}
    return {"unique_query_totals": unique, "cohort_stratum_results": grouped, "base_stream_results": by_stream,
        "interpretation": "Identical natural full-prefill queries and own-world truth for all five checkpoints. Related value/readout bindings are not independent samples. One training initialization, approximately matched parameter counts with different ranks, short-to-long distribution shift and weaker Attention-only teacher fit limit causal/architecture claims. Question-stage computation includes both GDN and Attention; sampled native ledger identity is not a pure Attention-Q intervention."}
