"""Complete paired history outcomes with legitimate current-value commit guards."""

from collections import Counter, defaultdict
import re
from statistics import mean

from sas_memory.write_locality_metrics import paired
from scripts.history_no_repeat_data import CELLS, ROLES, identity


def summarize(rows, data):
    expected = [identity(example) for example in data["validation"]]
    keys = [[row["case_id"], row["variant"], row["role"]] for row in rows]
    if keys != expected or len(rows) != 384 or len({tuple(key) for key in keys}) != 384:
        raise ValueError("Complete ordered384-query natural grid required")
    cases = {row["case"]["id"]: row["case"] for row in data["validation"]}
    indexed = {(row["case_id"], row["variant"], row["role"]): row for row in rows}

    def read(case_id, cell, role):
        return indexed[case_id, cell, role]

    def correct(case_id, cell, role):
        return read(case_id, cell, role)["greedy_exact"]

    def joint(case_id, cell):
        return all(correct(case_id, cell, role) for role in ROLES)

    individual = []
    for case_id, case in cases.items():
        lhist, rhist = (correct(case_id, cell, ROLES[0]) for cell in CELLS)
        regressed, rescued = lhist and not rhist, not lhist and rhist
        committed = all(correct(case_id, cell, role) for cell in CELLS for role in ROLES[1:])
        item = {key: case[key] for key in ("cohort", "base_stream_id", "base_case_id", "seed", "active_accounts", "value_version")}
        item.update(case_id=case_id, history_regressed=regressed, history_rescued=rescued,
            all_current_correct_both_worlds=committed, commit_qualified_history_regression=regressed and committed,
            worlds={cell: {"all_roles_joint": joint(case_id, cell), "roles": {
                role: {key: read(case_id, cell, role)[key] for key in ("answer", "generated", "greedy_exact", "answer_nll")}
                for role in ROLES}} for cell in CELLS})
        for cell in CELLS:
            hist = read(case_id, cell, ROLES[0])
            match = re.fullmatch(r"\s*(-?\d+)\s*", hist["generated"])
            generated = match.group(1) if match else None
            named_values = {"current_A": read(case_id, cell, ROLES[1])["answer"],
                            "middle_A": str(case["prefix"][3]["value"]),
                            "current_C": read(case_id, cell, ROLES[2])["answer"]}
            item["worlds"][cell]["historical_error_numeric_matches"] = (
                [] if hist["greedy_exact"] else [name for name, value in named_values.items() if generated == value])
            item["worlds"][cell]["historical_output_is_integer"] = bool(match)
        individual.append(item)
    grouped = []
    for cohort in ("discovery", "replica"):
        for load in (32, 128):
            members = [r for r in rows if r["cohort"] == cohort and r["active_accounts"] == load]
            pairs = [p for p in individual if p["cohort"] == cohort and p["active_accounts"] == load]
            if len(members) != 96 or len(pairs) != 16 or len({p["base_stream_id"] for p in pairs}) != 4:
                raise ValueError("A cohort/load/base-stream group is incomplete")
            counts = {cell: {role: {"correct": sum(r["greedy_exact"] for r in members if r["variant"] == cell and r["role"] == role),
                "examples": 16, "answer_nll": mean(r["answer_nll"] for r in members if r["variant"] == cell and r["role"] == role)}
                for role in ROLES} for cell in CELLS}
            for cell in CELLS:
                counts[cell]["all_roles_joint"] = {"correct": sum(p["worlds"][cell]["all_roles_joint"] for p in pairs), "examples": 16}
            qualified = [p for p in pairs if p["commit_qualified_history_regression"]]
            regressed = [p for p in pairs if p["history_regressed"]]
            streams = sorted({p["base_stream_id"] for p in qualified})
            grouped.append({"cohort": cohort, "active_accounts": load, "pairs": 16, "counts": counts,
                "paired_roles": {role: paired([r for r in members if r["role"] == role], CELLS[0], CELLS[1], invariant=role != ROLES[1]) for role in ROLES},
                "commit_qualified_regression_cases": [p["case_id"] for p in qualified],
                "commit_qualified_regression_base_streams": streams,
                "unqualified_regression_cases": [p["case_id"] for p in regressed if not p["all_current_correct_both_worlds"]],
                "all_current_correct_both_worlds": sum(p["all_current_correct_both_worlds"] for p in pairs),
                "baseline_historical_floor": counts[CELLS[0]][ROLES[0]]["correct"] <= 4,
                "descriptive_two_stream_signal": len(streams) >= 2,
                "primary": load == 128})
    by_stream = []
    for cohort in ("discovery", "replica"):
        for load in (32, 128):
            streams = sorted({r["base_stream_id"] for r in rows if r["cohort"] == cohort})
            for stream in streams:
                members = [r for r in rows if r["cohort"] == cohort and r["active_accounts"] == load and r["base_stream_id"] == stream]
                pairs = [p for p in individual if p["cohort"] == cohort and p["active_accounts"] == load and p["base_stream_id"] == stream]
                by_stream.append({"cohort": cohort, "active_accounts": load, "base_stream_id": stream,
                    "examples": len(members), "correct": sum(r["greedy_exact"] for r in members),
                    "all_four_values_roles_worlds_correct": all(joint(p["case_id"], cell) for p in pairs for cell in CELLS),
                    "historical_by_world": {cell: sum(correct(p["case_id"], cell, ROLES[0]) for p in pairs) for cell in CELLS},
                    "qualified_regression_cases": [p["case_id"] for p in pairs if p["commit_qualified_history_regression"]]})
    primary = [g for g in grouped if g["primary"]]
    return {"examples": len(rows), "correct": sum(r["greedy_exact"] for r in rows),
        "cohort_load_results": grouped, "base_stream_results": by_stream, "individual_pairs": individual,
        "primary_signal": {"present_in_both_cohorts": all(g["descriptive_two_stream_signal"] for g in primary),
            "cohort_qualified_base_streams": {g["cohort"]: g["commit_qualified_regression_base_streams"] for g in primary},
            "cohort_baseline_floor": {g["cohort"]: g["baseline_historical_floor"] for g in primary},
            "definition": "At128accounts in each cohort, history regresses in at least two distinct streams while current A/C remain correct in both worlds. Descriptive replication gate, not significance."},
        "interpretation": "Initial evidence occurs once in the same distant position and common prefix; both final A values differ from the initial answer. A remaining functional effect does not isolate recurrent-state damage or justify shared-interface necessity. Low hold-world history competence limits absence claims. Related value siblings/roles/worlds are not independent observations."}
