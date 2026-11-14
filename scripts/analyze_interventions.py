"""Summarize counterfactual cache effects against each recipient's own baseline."""

from collections import defaultdict
import argparse
import json
from pathlib import Path
from statistics import mean


def analyze(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    verification = json.loads((path / "verification.json").read_text())
    rows = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
    native = {(r["case_id"], r["recipient"], r["role"], r["condition"]): r
              for r in rows if r["condition"].startswith("natural_")}
    grouped = defaultdict(list)
    individual = []
    numerical_errors = []
    for key, row in native.items():
        if key[-1] == "natural_cached":
            full = native[(*key[:-1], "natural_full")]
            numerical_errors.extend(abs(row["candidate_scores"][c]["nll_mean"] - full["candidate_scores"][c]["nll_mean"])
                                    for c in row["candidate_scores"])
    for row in rows:
        if row["condition"] != "transplant":
            continue
        baseline = native[(row["case_id"], row["recipient"], row["role"], "natural_cached")]
        recipient_score = baseline["candidate_scores"][row["recipient_answer"]]["nll_sum"]
        donor_score = baseline["candidate_scores"][row["donor_answer"]]["nll_sum"]
        effect = {"case_id": row["case_id"], "recipient": row["recipient"], "group": row["group"],
                  "path": row["path"], "role": row["role"],
                  "recipient_answer_nll_delta": row["recipient_answer_nll"] - baseline["recipient_answer_nll"],
                  "donor_log_odds_delta": row["donor_vs_recipient_log_odds"] - (recipient_score - donor_score),
                  "prediction_changed": row["candidate_prediction"] != baseline["candidate_prediction"]}
        individual.append(effect)
        grouped[(row["group"], row["path"], row["role"])].append(effect)
    aggregates = []
    for (group, pathway, role), members in sorted(grouped.items()):
        aggregates.append({"group": group, "path": pathway, "role": role, "count": len(members),
                           "mean_recipient_nll_delta": mean(r["recipient_answer_nll_delta"] for r in members),
                           "mean_abs_recipient_nll_delta": mean(abs(r["recipient_answer_nll_delta"]) for r in members),
                           "mean_donor_log_odds_delta": mean(r["donor_log_odds_delta"] for r in members),
                           "prediction_changes": sum(r["prediction_changed"] for r in members)})
    return {"status": manifest["status"], "rows": len(rows), "case_ids": manifest["case_ids"],
            "verification": verification, "aggregates": aggregates, "individual": individual,
            "natural_full_cached_max_candidate_nll_error": max(numerical_errors),
            "natural_full_cached_mean_candidate_nll_error": mean(numerical_errors),
            "interpretation": "Effects of unnatural mixed caches. Path attribution is provisional; small effects must be compared with numerical discrepancies. Not evidence that joint training is necessary."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "rows", "verification", "natural_full_cached_max_candidate_nll_error")}))
    for row in report["aggregates"]:
        if row["group"] == "all":
            print(json.dumps(row))


if __name__ == "__main__":
    main()
