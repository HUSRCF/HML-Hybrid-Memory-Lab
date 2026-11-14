"""Signed repeat costs for same-final-value pairs with preservation guards."""

from statistics import mean

from scripts.overwrite_history_data import CELLS, CONTRASTS, ROLES, evaluated_members, identity


def paired_statistics(pairs):
    def stats(eligible):
        return {"pairs": len(eligible),
            "novel_correct": sum(p["novel_current_correct"] for p in eligible),
            "repeat_correct": sum(p["repeat_current_correct"] for p in eligible),
            "losses": sum(p["raw_repeat_loss"] for p in eligible),
            "rescues": sum(p["raw_repeat_rescue"] for p in eligible),
            "signed_net_loss": sum(int(p["raw_repeat_loss"]) - int(p["raw_repeat_rescue"]) for p in eligible),
            "both_correct": sum(p["novel_current_correct"] and p["repeat_current_correct"] for p in eligible),
            "both_wrong": sum(not p["novel_current_correct"] and not p["repeat_current_correct"] for p in eligible)}
    return {"raw": stats(pairs), "guard_qualified": stats([p for p in pairs if p["preservation_guards_correct"]]),
        "mean_nll_repeat_minus_novel": mean(p["current_nll_repeat_minus_novel"] for p in pairs),
        "loss_pair_ids": [p["id"] for p in pairs if p["raw_repeat_loss"]],
        "rescue_pair_ids": [p["id"] for p in pairs if p["raw_repeat_rescue"]],
        "qualified_loss_pair_ids": [p["id"] for p in pairs if p["qualified_repeat_loss"]],
        "qualified_rescue_pair_ids": [p["id"] for p in pairs if p["qualified_repeat_rescue"]],
        "guard_failed_loss_pair_ids": [p["id"] for p in pairs if p["raw_repeat_loss"] and not p["preservation_guards_correct"]]}


def summarize(rows, data):
    members = evaluated_members(data)
    keys = [[row["case_id"], row["variant"], row["role"]] for row in rows]
    if keys != [identity(r) for r in members] or len(rows) != 384 or len({tuple(k) for k in keys}) != 384:
        raise ValueError("Complete ordered384-query factorial observations required")
    indexed = {tuple(key): row for key, row in zip(keys, rows)}

    def read(case_id, cell, role):
        return indexed[case_id, cell, role]

    def correct(case_id, cell, role):
        return read(case_id, cell, role)["greedy_exact"]

    def joint(case_id, cell):
        return all(correct(case_id, cell, role) for role in ROLES)

    pairs, case_outcomes = [], []
    for case in data["cases"]:
        case_id = case["id"]
        worlds = {cell: {"all_roles_joint": joint(case_id, cell), "roles": {
            role: {key: read(case_id, cell, role)[key] for key in ("answer", "generated", "greedy_exact", "answer_nll")}
            for role in ROLES}} for cell in CELLS}
        case_outcomes.append({"case_id": case_id,
            **{key: case[key] for key in ("cohort", "base_stream_id", "seed", "value_version")},
            "worlds": worlds, "all_current_A_cells_correct": all(correct(case_id, c, ROLES[1]) for c in CELLS),
            "both_final_values_correct_given_early": {
                early: all(correct(case_id, early + final, ROLES[1]) for final in "XY") for early in "XY"},
            "all_roles_and_cells_correct": all(joint(case_id, cell) for cell in CELLS)})
        for name, (novel, repeat) in CONTRASTS.items():
            if any(read(case_id, novel, role)["answer"] != read(case_id, repeat, role)["answer"] for role in ROLES):
                raise ValueError("Same-final contrast changes a task answer")
            before, after = correct(case_id, novel, ROLES[1]), correct(case_id, repeat, ROLES[1])
            guard = all(correct(case_id, cell, role) for cell in (novel, repeat) for role in (ROLES[0], ROLES[2]))
            loss, rescue = before and not after, not before and after
            pairs.append({"id": case_id + "/" + name, "case_id": case_id, "final_identity": name,
                **{key: case[key] for key in ("cohort", "base_stream_id", "seed", "value_version")},
                "novel_cell": novel, "repeat_cell": repeat, "answer": read(case_id, novel, ROLES[1])["answer"],
                "novel_current_correct": before, "repeat_current_correct": after,
                "raw_repeat_loss": loss, "raw_repeat_rescue": rescue, "preservation_guards_correct": guard,
                "qualified_repeat_loss": loss and guard, "qualified_repeat_rescue": rescue and guard,
                "current_nll_repeat_minus_novel": read(case_id, repeat, ROLES[1])["answer_nll"] - read(case_id, novel, ROLES[1])["answer_nll"],
                "worlds": {cell: worlds[cell] for cell in (novel, repeat)}})
    cohorts, streams, final_results = [], [], []
    for cohort in ("discovery", "replica"):
        members = [r for r in rows if r["cohort"] == cohort]
        group = [p for p in pairs if p["cohort"] == cohort]
        outcomes = [p for p in case_outcomes if p["cohort"] == cohort]
        if len(members) != 192 or len(group) != 32 or len(outcomes) != 16:
            raise ValueError("Incomplete cohort or final-value pairs")
        counts = {cell: {role: {"correct": sum(r["greedy_exact"] for r in members if r["variant"] == cell and r["role"] == role),
            "examples": 16, "answer_nll": mean(r["answer_nll"] for r in members if r["variant"] == cell and r["role"] == role)}
            for role in ROLES} for cell in CELLS}
        for cell in CELLS:
            counts[cell]["all_roles_joint"] = {"correct": sum(p["worlds"][cell]["all_roles_joint"] for p in outcomes), "examples": 16}
        per_stream = []
        for stream in sorted({p["base_stream_id"] for p in group}):
            stream_pairs = [p for p in group if p["base_stream_id"] == stream]
            stream_cases = [p for p in outcomes if p["base_stream_id"] == stream]
            if len(stream_pairs) != 8 or len(stream_cases) != 4:
                raise ValueError("Incomplete four-value/two-final stream")
            item = {"cohort": cohort, "base_stream_id": stream, **paired_statistics(stream_pairs),
                "all_four_value_cases_roles_cells_correct": all(p["all_roles_and_cells_correct"] for p in stream_cases)}
            streams.append(item)
            per_stream.append(item)
        stats = paired_statistics(group)
        positive = [p["base_stream_id"] for p in per_stream if p["guard_qualified"]["signed_net_loss"] > 0]
        gate = {"novel_baseline_at_least24_of32": stats["raw"]["novel_correct"] >= 24,
            "raw_net_repeat_loss_positive": stats["raw"]["signed_net_loss"] > 0,
            "qualified_net_repeat_loss_positive": stats["guard_qualified"]["signed_net_loss"] > 0,
            "at_least_two_positive_qualified_streams": len(positive) >= 2}
        cohorts.append({"cohort": cohort, "counts": counts, **stats,
            "positive_qualified_base_streams": positive, "primary_gate_checks": gate, "primary_passed": all(gate.values()),
            "all_current_A_cells_correct": sum(p["all_current_A_cells_correct"] for p in outcomes),
            "all_roles_cells_correct": sum(p["all_roles_and_cells_correct"] for p in outcomes),
            "both_final_values_correct_given_early": {early: sum(p["both_final_values_correct_given_early"][early] for p in outcomes) for early in "XY"}})
        for name in CONTRASTS:
            final_results.append({"cohort": cohort, "final_identity": name,
                **paired_statistics([p for p in group if p["final_identity"] == name])})
    return {"examples": len(rows), "correct": sum(r["greedy_exact"] for r in rows),
        "cohort_results": cohorts, "base_stream_results": streams, "final_identity_results": final_results,
        "individual_same_final_pairs": pairs, "individual_value_cases": case_outcomes,
        "primary_signal": {"present_in_both_cohorts": all(c["primary_passed"] for c in cohorts),
            "cohort_passed": {c["cohort"]: c["primary_passed"] for c in cohorts},
            "cohort_gate_checks": {c["cohort"]: c["primary_gate_checks"] for c in cohorts},
            "definition": "Each cohort: novel-current baseline >=24/32, positive raw and qualified signed repeat losses, positive qualified direction in >=2 distinct base streams. Descriptive replicated direction check, not significance."},
        "interpretation": "Final answers and complete executable states are matched and value identities counterbalanced. All raw, guarded and unqualified outcomes remain visible. Related value/final/cell observations are not independent trials; no recurrent-damage or shared-interface necessity conclusion follows automatically. The earlier historical-preservation primary remains failed."}
