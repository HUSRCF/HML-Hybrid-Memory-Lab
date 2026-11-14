"""Audit and aggregate saved diagnostic runs without loading a model."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import mean


def audit_run(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    cases_text = (path / "cases.jsonl").read_text()
    if hashlib.sha256(cases_text.encode()).hexdigest() != manifest["cases_sha256"]:
        raise ValueError(f"Case file changed: {path}")
    cases = [json.loads(line) for line in cases_text.splitlines()]
    expected = {(case["id"], name, query["role"]): query["answer"] for case in cases
                for name, variant in case["variants"].items() for query in variant["queries"]}
    rows = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
    seen = set()
    for row in rows:
        key = (row["case_id"], row["variant"], row["role"])
        if key in seen or expected.get(key) != row["answer"]:
            raise ValueError(f"Duplicate or incorrect result label: {key}")
        seen.add(key)
    if manifest["status"] == "completed" and seen != expected.keys():
        raise ValueError(f"Completed run has missing results: {path}")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["distractor_updates"], row.get("distractor_entities", 4), row["variant"], row["role"])].append(row)
    aggregates = []
    for (family, length, load, variant, role), members in sorted(groups.items()):
        generated = [r for r in members if r["generated"] is not None]
        valid = [r for r in generated if re.fullmatch(r"\s*-?\d+\s*", r["generated"])]
        aggregates.append({"family": family, "distractor_updates": length, "distractor_entities": load,
                           "variant": variant, "role": role, "count": len(members),
                           "greedy_correct": sum(r["greedy_exact"] for r in generated),
                           "generation_count": len(generated), "integer_format_valid": len(valid),
                           "candidate_correct": sum(r["candidate_correct"] for r in members),
                           "answer_nll": mean(r["answer_nll"] for r in members)})
    case_rows = defaultdict(dict)
    for row in rows:
        case_rows[row["case_id"]][(row["variant"], row["role"])] = row
    preservation = []
    for case_id, members in case_rows.items():
        keys = [(v, "protected_current") for v in ("before", "control", "treatment")]
        if not all(key in members for key in keys):
            continue
        before, control, treatment = (members[key] for key in keys)
        if len({r["answer"] for r in (before, control, treatment)}) != 1:
            raise ValueError("Protected answer changed")
        preservation.append({"case_id": case_id,
                             "before_correct": before["greedy_exact"],
                             "control_correct": control["greedy_exact"], "treatment_correct": treatment["greedy_exact"],
                             "candidate_control_correct": control["candidate_correct"],
                             "candidate_treatment_correct": treatment["candidate_correct"],
                             "same_token_length": control["ledger_tokens"] == treatment["ledger_tokens"],
                             "nll_treatment_minus_control": treatment["answer_nll"] - control["answer_nll"]})
    return {"path": str(path), "model": manifest["model"], "status": manifest["status"],
            "cases": len(cases), "seeds": sorted({c["seed"] for c in cases}), "rows": len(rows),
            "unique_common_prefixes": len({json.dumps(c["prefix"], sort_keys=True) for c in cases}),
            "complete_label_coverage": seen == expected.keys(), "aggregates": aggregates,
            "protected_comparisons": preservation,
            "greedy_protected_regressions_vs_matched_control": sum(r["control_correct"] is True and r["treatment_correct"] is False for r in preservation),
            "greedy_protected_regressions_vs_before": sum(r["before_correct"] is True and r["treatment_correct"] is False for r in preservation),
            "candidate_protected_regressions_vs_matched_control": sum(r["candidate_control_correct"] and not r["candidate_treatment_correct"] for r in preservation),
            "numerical_verification": json.loads((path / "verification.json").read_text()),
            "warning": "Repeated families share prefixes. Counts are not independent samples; no significance claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [audit_run(path) for path in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": reports}, indent=2) + "\n")
    for report, path in zip(reports, args.runs):
        rows = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
        print(json.dumps({key: report[key] for key in ("model", "status", "cases", "rows", "unique_common_prefixes", "complete_label_coverage")}), flush=True)
        for role in ("target_current", "protected_current", "target_initial"):
            members = [r for r in rows if r["variant"] == "treatment" and r["role"] == role]
            print(json.dumps({"role": role, "treatment_greedy_correct": sum(r["greedy_exact"] is True for r in members),
                              "integer_format_valid": sum(bool(re.fullmatch(r"\s*-?\d+\s*", r["generated"] or "")) for r in members),
                              "candidate_correct": sum(r["candidate_correct"] for r in members), "count": len(members)}), flush=True)


if __name__ == "__main__":
    main()
