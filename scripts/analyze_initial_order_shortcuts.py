"""Describe changed wrong answers and legacy-slot matches after independent reordering."""

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from sas_memory.training_data import data_fingerprint
from scripts.analyze_controller_trial import load_rows
from scripts.analyze_initial_order_control import MODELS, canonical_rows
from scripts.compare_paired_semantic_controllers import compare
from scripts.run_controller_trial import write_json


def number(row):
    match = re.fullmatch(r"\s*(-?\d+)\s*", row["generated"])
    return match.group(1) if match else None


def analyze(audit):
    run = Path(audit["source_run"])
    if audit["status"] != "completed_and_audited" or file_hash(run / "manifest.json") != audit["manifest_sha256"]:
        raise ValueError("A current full independent audit is required")
    manifest = json.loads((run / "manifest.json").read_text())
    compare(manifest["training_reference_audit"])  # Recheck original raw-file hashes before reuse.
    plan = json.loads((run / "plan.json").read_text())
    registration = json.loads((Path(plan["data_source"]) / "manifest.json").read_text())
    data = json.loads((Path(registration["base_data"]) / "data.json").read_text())
    cases = json.loads((Path(plan["data_source"]) / "cases.json").read_text())
    if data_fingerprint(data) != registration["base_data_sha256"] or data_fingerprint(cases) != plan["cases_sha256"]:
        raise ValueError("Canonical/permuted facts changed since registration")
    examples = data["validation"]
    expected = {(example["case"]["id"], example["variant"], example["query"]["role"]): example["query"]["answer"] for example in examples}
    canonical = canonical_rows(plan, expected)
    models, details = {}, []
    for name in MODELS:
        output = run / f"{name}.jsonl"
        if file_hash(output) != manifest["models"][name]["results_sha256"]:
            raise ValueError("Permuted outputs changed since independent audit")
        rows = load_rows(output, expected)
        originals = {(row["case_id"], row["variant"], row["role"]): row for row in canonical[name]}
        statistics = {"examples": len(rows), "both_wrong_same_number": 0, "both_wrong_different_number_or_invalid": 0,
                      "unchanged_fact_readouts": 0, "unchanged_fact_errors": 0, "unchanged_fact_errors_matching_legacy_initial_slot": 0,
                      "unchanged_fact_regressions": 0, "unchanged_fact_regressions_matching_legacy_initial_slot": 0}
        for row, example in zip(rows, examples):
            key = (row["case_id"], row["variant"], row["role"])
            original = originals[key]
            if not original["greedy_exact"] and not row["greedy_exact"]:
                field = "both_wrong_same_number" if number(original) is not None and number(original) == number(row) else "both_wrong_different_number_or_invalid"
                statistics[field] += 1
            if row["role"] not in ("protected_current", "target_initial"):
                continue
            statistics["unchanged_fact_readouts"] += 1
            initial = example["case"]["prefix"][:8]
            slot = next(index for index, event in enumerate(initial) if event["target"] == example["query"]["entity"])
            occupying = cases[row["case_id"]]["prefix"][slot]
            legacy_value = str(occupying["value"])
            if occupying["target"] == example["query"]["entity"] or legacy_value == row["answer"]:
                raise ValueError("Legacy-slot alternative is not distinct from preserved truth")
            if not row["greedy_exact"]:
                matched = number(row) == legacy_value
                regression = original["greedy_exact"]
                statistics["unchanged_fact_errors"] += 1
                statistics["unchanged_fact_errors_matching_legacy_initial_slot"] += matched
                statistics["unchanged_fact_regressions"] += regression
                statistics["unchanged_fact_regressions_matching_legacy_initial_slot"] += regression and matched
                details.append({"model": name, "key": list(key), "gold": row["answer"], "canonical_generated": original["generated"],
                                "permuted_generated": row["generated"], "canonical_correct": regression, "legacy_initial_slot": slot,
                                "entity_now_in_legacy_slot": occupying["target"], "legacy_slot_initial_value": legacy_value, "matches_legacy_slot_value": matched})
        models[name] = statistics
    return {"status": "descriptive_error_analysis", "source_run": str(run), "manifest_sha256": audit["manifest_sha256"], "models": models,
            "unchanged_fact_errors": details,
            "interpretation": "Reused-validation description only. Legacy-slot matches count wrong numeric answers equal to the initial value now occupying the queried entity's old initial-record slot; no chance correction, significance or unique causal shortcut attribution. Other wrong answers and both-wrong changes are retained. Current target queries are excluded from slot matching because updates can change their truth."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(json.loads(args.audit.read_text()))
    report.update(audit_report_sha256=file_hash(args.audit), analysis_source_sha256=file_hash(Path(__file__)))
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report["models"]))


if __name__ == "__main__":
    main()
