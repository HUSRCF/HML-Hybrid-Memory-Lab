"""Functional tail contrasts; related value siblings never count as new streams."""

from collections import defaultdict
from statistics import mean

from sas_memory.write_locality_data import CELLS, ROLES


def paired(members, left_cell, right_cell, invariant=True):
    indexed = {(row["case_id"], row["variant"]): row for row in members}
    case_ids = sorted({row["case_id"] for row in members})
    pairs = [(indexed[case_id, left_cell], indexed[case_id, right_cell]) for case_id in case_ids]
    if invariant and any(left["answer"] != right["answer"] for left, right in pairs):
        raise ValueError("Answer-changing comparison was labeled as preservation")
    regressions = [right for left, right in pairs if left["greedy_exact"] and not right["greedy_exact"]]
    rescues = [right for left, right in pairs if not left["greedy_exact"] and right["greedy_exact"]]
    baseline_correct = sum(left["greedy_exact"] for left, right in pairs)
    return {"units": len(pairs), "reference_correct": baseline_correct,
            "candidate_correct": sum(right["greedy_exact"] for left, right in pairs),
            "rescued": len(rescues), "regressed": len(regressions),
            "both_correct": sum(left["greedy_exact"] and right["greedy_exact"] for left, right in pairs),
            "both_wrong": sum(not left["greedy_exact"] and not right["greedy_exact"] for left, right in pairs),
            "mean_nll_candidate_minus_reference": mean(right["answer_nll"] - left["answer_nll"] for left, right in pairs),
            "regressed_case_ids": [row["case_id"] for row in regressions],
            "regressed_base_streams": sorted({row["base_stream_id"] for row in regressions}),
            "rescued_case_ids": [row["case_id"] for row in rescues],
            "secondary_baseline_correct_regression_rate": {
                "regressed": len(regressions), "baseline_correct": baseline_correct,
                "rate": len(regressions) / baseline_correct if baseline_correct else None}}


def all_correct_units(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return {"correct": sum(all(row["greedy_exact"] for row in members) for members in grouped.values()),
            "units": len(grouped)}


def summarize(rows):
    if len(rows) != 576 or len({(r['case_id'], r['variant'], r['role']) for r in rows}) != 576:
        raise ValueError("Locality metrics require the complete registered grid")
    cells, contrasts, target_updates, load_units, streams = [], [], [], [], []
    for load in (8, 32, 128):
        members = [row for row in rows if row["active_accounts"] == load]
        if len(members) != 192:
            raise ValueError("Locality load cohort is incomplete")
        for cell in CELLS:
            for role in ROLES:
                group = [row for row in members if row["variant"] == cell and row["role"] == role]
                if len(group) != 16:
                    raise ValueError("Locality cell is incomplete")
                cells.append({"active_accounts": load, "cell": cell, "role": role,
                              "correct": sum(row["greedy_exact"] for row in group), "examples": len(group),
                              "answer_nll": mean(row["answer_nll"] for row in group)})
        for bit in (0, 1):
            for role in ROLES[1:]:
                group = [row for row in members if row["role"] == role]
                contrasts.append({"active_accounts": load, "axis": "A", "fixed_other_bit": bit, "role": role,
                                  **paired(group, f"A0_C{bit}", f"A1_C{bit}")})
            for role in ROLES:
                group = [row for row in members if row["role"] == role]
                contrasts.append({"active_accounts": load, "axis": "C", "fixed_other_bit": bit, "role": role,
                                  **paired(group, f"A{bit}_C0", f"A{bit}_C1")})
            target_updates.append({"active_accounts": load, "fixed_C_bit": bit,
                                   **paired([row for row in members if row["role"] == ROLES[0]],
                                            f"A0_C{bit}", f"A1_C{bit}", invariant=False)})
        load_units.append({"active_accounts": load,
                           "individual": {"correct": sum(row["greedy_exact"] for row in members), "units": len(members)},
                           "all_three_roles": all_correct_units(members, ["case_id", "variant"]),
                           "all_four_values": all_correct_units(members, ["base_case_id", "variant", "role"]),
                           "all_twelve_cell_role_readouts": all_correct_units(members, ["case_id"]),
                           "all_48_value_cell_role_readouts": all_correct_units(members, ["base_case_id"])})
        for seed in (97000, 97001, 97002, 97003):
            group = [row for row in members if row["seed"] == seed]
            if len(group) != 48:
                raise ValueError("Base-stream value/cell/role cluster is incomplete")
            streams.append({"seed": seed, "active_accounts": load, "correct": sum(row["greedy_exact"] for row in group),
                            "examples": len(group), "answer_nll": mean(row["answer_nll"] for row in group)})
    signals = [row for row in contrasts if len(row["regressed_base_streams"]) >= 2]
    return {"examples": len(rows), "correct": sum(row["greedy_exact"] for row in rows),
            "cell_role_counts": cells, "preservation_contrasts": contrasts, "target_A_update_pairs": target_updates,
            "functional_units_by_load": load_units, "base_stream_results": streams,
            "screen_signal": {"present": bool(signals), "contrasts": signals,
                              "definition": "The same axis, fixed-other-bit, role and load regresses in at least two distinct base streams. Multiple value siblings from one stream count once.",
                              "interpretation": "Descriptive development screen only; no significance, recurrent-damage or architecture-necessity claim."}}
