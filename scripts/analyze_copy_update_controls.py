"""Audit matched short-program update controls and cluster their readout effects."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.generalization_tasks import copy_update_challenges, validate_generalization_cases
from sas_memory.training_data import data_fingerprint
from scripts.analyze_controller_trial import load_rows
from scripts.analyze_controller_confirmation import scalar_metrics, paired_interval, unary_cluster_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    config = manifest["config"]
    if manifest["status"] != "completed" or manifest["completed_modes"] != config["modes"]:
        raise ValueError("Update controls are incomplete")
    cases = json.loads((args.run / "cases.json").read_text())
    validation = validate_generalization_cases(cases)
    if data_fingerprint(cases) != manifest["cases_sha256"] or manifest["cases_sha256"] != config["expected_cases_sha256"]:
        raise ValueError("Control cases changed")
    reference = Path(config["reference_run"])
    if hashlib.sha256((reference / "manifest.json").read_bytes()).hexdigest() != manifest["reference_manifest_sha256"]:
        raise ValueError("Reference manifest changed")
    original = json.loads((reference / "manifest.json").read_text())
    for relative, digest in manifest["source_sha256"].items():
        if hashlib.sha256((args.run / "sources" / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source snapshot changed: {relative}")
    for relative in ("sas_memory/runtime.py", "sas_memory/controllers.py"):
        if manifest["source_sha256"][relative] != original["source_sha256"][relative]:
            raise ValueError("Reused minimal scores have a different runtime/controller implementation")
    lookup = {case["id"]: case for case in cases}
    clusters = {case["id"]: case["base_case_id"] for case in cases}
    expected = {(case["id"], regime, "target_current"): next(q["answer"] for q in case["variants"]["control"]["queries"] if q["role"] == "target_current")
                for case in cases for regime in config["regimes"]}
    expected_reference = {(case["id"], variant, "target_current"): next(q["answer"] for q in case["variants"][variant]["queries"] if q["role"] == "target_current")
                          for case in cases for variant in ("control", "treatment")}
    report = {"status": "completed", "cases": len(cases), "base_cases": validation["base_cases"], "models": {},
              "interpretation": "Exploratory matched programs; all target answers remain fixed. Account-specific versus general extra-update effects are functional readout effects, not localization to recurrent state, KV or query construction."}
    for mode in config["modes"]:
        reference_path = reference / f"{mode}.oracle.jsonl"
        if hashlib.sha256(reference_path.read_bytes()).hexdigest() != manifest["reference_oracle_sha256"][mode]:
            raise ValueError("Minimal evidence results changed")
        if manifest["matched_prompt_lengths"][mode] != len(cases):
            raise ValueError("Prompt-length controls incomplete")
        replay = manifest["minimal_replay"][mode]
        if replay["max_nll_error"] > 1e-5 or not replay["greedy_exact_match"]:
            raise ValueError("Minimal evidence replay failed")
        if mode != "native":
            spec = original["checkpoint_sources"][mode]
            checkpoint = Path(spec["training_run"]) / spec["mode"] / "controller.safetensors"
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["checkpoint_sha256"][mode]:
                raise ValueError("Controller checkpoint changed")
            if spec["config"].get("token_policy", "all") == "question":
                checks = manifest["pre_question_cache_checks"].get(mode, {})
                if any(not checks.get(regime, {}).get("kv_recurrent_conv_and_logits_exact") for regime in config["regimes"]):
                    raise ValueError("Missing native-prefix identity check")
        rows = load_rows(args.run / f"{mode}.jsonl", expected)
        indexed = {(row["case_id"], row["variant"]): row for row in rows}
        for row in rows:
            case = lookup[row["case_id"]]
            programs = copy_update_challenges(case)
            if row["source_variant"] != "control" or row["events"] != [asdict(event) for event in programs[row["variant"]]]:
                raise ValueError("Rendered control program changed")
        for case in cases:
            if indexed[(case["id"], "source_update")]["prompt_tokens"] != indexed[(case["id"], "unrelated_update")]["prompt_tokens"]:
                raise ValueError("Source/unrelated prompt lengths differ")
        minimal = [row for row in load_rows(reference_path, expected_reference) if row["variant"] == "control"]
        for row in minimal:
            if row["oracle_events"] != [asdict(event) for event in copy_update_challenges(lookup[row["case_id"]])["minimal"]]:
                raise ValueError("Reused minimal program differs")
        regimes = {"minimal": minimal, **{regime: [row for row in rows if row["variant"] == regime] for regime in config["regimes"]}}
        scores = {regime: unary_cluster_scores(members, clusters) for regime, members in regimes.items()}
        comparisons = {f"{left}_minus_{right}": paired_interval(scores[left], scores[right]) for left, right in
                       (("source_update", "unrelated_update"), ("source_update", "known_accounts"), ("unrelated_update", "known_accounts"))}
        for value in comparisons.values():
            value["unit"] = "base_case_id, averaging all six matched depth/wording conditions"
        conditions = sorted({(case["dependency_depth"], case["question_style"]) for case in cases})
        errors = {}
        for regime in ("source_update", "unrelated_update"):
            categories = []
            for row in regimes[regime]:
                case = lookup[row["case_id"]]
                entity = next(query["entity"] for query in case["variants"]["control"]["queries"] if query["role"] == "protected_current")
                initial = next(event["value"] for event in case["prefix"] if event["target"] == entity)
                prediction = row["generated"].strip()
                categories.append("correct" if row["greedy_exact"] else "newly_written_value" if prediction == str(row["events"][-1]["value"])
                                  else "unrelated_initial_value" if prediction == str(initial) else "other")
            errors[regime] = dict(Counter(categories))
        baseline = {row["case_id"]: row for row in regimes["known_accounts"]}
        preservation = {regime: {"initially_correct": sum(row["greedy_exact"] for row in baseline.values()),
                                "regressions": sum(baseline[row["case_id"]]["greedy_exact"] and not row["greedy_exact"] for row in regimes[regime]),
                                "recoveries": sum(not baseline[row["case_id"]]["greedy_exact"] and row["greedy_exact"] for row in regimes[regime])}
                        for regime in ("source_update", "unrelated_update")}
        report["models"][mode] = {"regimes": {name: scalar_metrics(members) for name, members in regimes.items()},
                                  "paired_differences": comparisons, "error_types": errors, "paired_preservation": preservation,
                                  "by_condition": [{"dependency_depth": depth, "question_style": style,
                                                    "regimes": {name: scalar_metrics([row for row in members if
                                                        lookup[row["case_id"]]["dependency_depth"] == depth and lookup[row["case_id"]]["question_style"] == style])
                                                                for name, members in regimes.items()}}
                                                   for depth, style in conditions]}
    report["source_run"] = str(args.run)
    report["manifest_sha256"] = hashlib.sha256((args.run / "manifest.json").read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for mode, result in report["models"].items():
        print(mode, {name: f"{value['correct']}/{value['count']}" for name, value in result["regimes"].items()})
        print(json.dumps(result["paired_differences"]["source_update_minus_unrelated_update"]))


if __name__ == "__main__":
    main()
