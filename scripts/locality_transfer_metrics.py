"""Paired own-world transfer outcomes, without pooling related readouts."""

from collections import defaultdict
from statistics import mean

from scripts.analyze_write_locality_event_paths import paired_outcomes
from scripts.locality_transfer_data import MODELS, selected_inputs


def summarize(registry, rows_by_model, smoke=False):
    pairs, bindings, physical = selected_inputs(registry, smoke)
    queries = {q["id"]: q for q in physical}
    indexed = {"native": {key: q["native_full_readout"] for key, q in queries.items()}}
    if list(rows_by_model) != list(MODELS):
        raise ValueError("Both pre-registered timing models required")
    for model, rows in rows_by_model.items():
        indexed[model] = {row["id"]: row for row in rows}
        if len(rows) != len(queries) or indexed[model].keys() != queries.keys():
            raise ValueError("Transfer metrics require exactly the selected unique queries")
    binding_index = {(b["pair_id"], b["side"], b["label"]): b["physical_id"] for b in bindings}
    labels = {pair["id"]: [b["label"] for b in bindings if b["pair_id"] == pair["id"] and b["side"] == "left"] for pair in pairs}
    models = list(indexed)

    def readout(model, pair, side, label):
        return indexed[model][binding_index[(pair["id"], side, label)]]

    def correct(model, pair, side, label):
        return readout(model, pair, side, label)["greedy_exact"]

    def joint(model, pair, side):
        return all(correct(model, pair, side, label) for label in labels[pair["id"]])

    def counts(group):
        result = {}
        for model in models:
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
        item = {"cohort": cohort, "stratum": stratum, "pairs": len(members),
                "base_streams": sorted({p["base_stream_id"] for p in members}), "counts": counts(members),
                "versus_native": {model: compare(members, "native", model) for model in MODELS},
                "question_versus_all_token": compare(members, MODELS[0], MODELS[1]),
                "preservation_left_to_right": {model: paired_outcomes(
                    [correct(model, p, "left", "preservation") for p in members],
                    [correct(model, p, "right", "preservation") for p in members]) for model in models},
                "native_full_update_regression_cases": [p["case_id"] for p in induced],
                "native_full_update_regression_repairs": {model: {
                    "preservation_repaired": [p["case_id"] for p in induced if correct(model, p, "right", "preservation")],
                    "all_roles_joint_repaired": [p["case_id"] for p in induced if joint(model, p, "right")]} for model in MODELS}}
        grouped.append(item)
    by_stream = [{"cohort": key[0], "stratum": key[1], "base_stream_id": key[2], "value_cases": len(members),
                  "counts": counts(members),
                  "all_values_roles_and_sides_correct": {model: all(joint(model, p, side) for p in members for side in ("left", "right")) for model in models}}
                 for key, members in streams.items()]
    unique = {}
    for model in models:
        unique[model] = {"correct": sum(r["greedy_exact"] for r in indexed[model].values()), "physical_queries": len(queries),
                         "reported_binding_correct": sum(indexed[model][b["physical_id"]]["greedy_exact"] for b in bindings),
                         "reported_bindings": len(bindings)}
    return {"unique_query_totals": unique, "cohort_stratum_results": grouped, "base_stream_results": by_stream,
            "interpretation": "All natural endpoints use their own full-prefill source-world truth. Reused bindings are not extra model executions or independent samples. Transfer is from the earlier short eight-account training distribution; no checkpoint selection or new training. Question-phase adaptation includes recurrent and KV updates during question/answer processing, not a pure Attention-Q intervention."}
