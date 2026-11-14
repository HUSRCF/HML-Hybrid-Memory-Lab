"""Audit frozen confirmation and bootstrap paired joint-readout differences by case."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_controller_trial import aggregate, load_rows
from sas_memory.training_data import data_fingerprint


def case_scores(rows):
    contexts = defaultdict(list)
    for row in rows:
        if row["family"] == "copy" and row["variant"] != "before":
            contexts[(row["case_id"], row["variant"])].append(row)
    by_case = defaultdict(list)
    for (case, _), members in contexts.items():
        if len(members) != 3:
            raise ValueError("Incomplete COPY readout group")
        by_case[case].append(all(member["greedy_exact"] for member in members))
    if any(len(values) != 2 for values in by_case.values()):
        raise ValueError("Missing COPY order variant")
    return {case: mean(values) for case, values in by_case.items()}


def paired_interval(left, right, samples=5000):
    if left.keys() != right.keys() or not left:
        raise ValueError("Case keys do not match")
    differences = [left[key] - right[key] for key in sorted(left)]
    rng = random.Random(9173)
    draws = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(samples))
    return {"difference": mean(differences), "ci95": [draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]],
            "cases": len(differences), "bootstrap_samples": samples,
            "unit": "case, averaging its two order variants", "multiple_comparison_adjustment": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    expected_modes = ["native", *manifest["config"]["modes"]]
    if manifest["status"] != "completed" or manifest["completed_modes"] != expected_modes:
        raise ValueError("Confirmation is incomplete")
    cases = json.loads((args.run / "cases.json").read_text())
    if data_fingerprint(cases) != manifest["cases_sha256"]:
        raise ValueError("Confirmation cases changed")
    if manifest["config"].get("expected_cases_sha256", manifest["cases_sha256"]) != manifest["cases_sha256"]:
        raise ValueError("Confirmation differs from its locked case digest")
    for relative, digest in manifest["source_sha256"].items():
        snapshot = args.run / "sources" / relative
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source snapshot mismatch: {relative}")
    for mode in manifest["config"]["modes"]:
        spec = manifest.get("checkpoint_sources", {}).get(mode, {"training_run": manifest["config"].get("training_run"), "mode": mode})
        training_run = Path(spec["training_run"])
        checkpoint = training_run / spec["mode"] / "controller.safetensors"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["checkpoint_sha256"][mode]:
            raise ValueError(f"Checkpoint changed since confirmation: {mode}")
        if "training_manifest_sha256" in spec and hashlib.sha256((training_run / "manifest.json").read_bytes()).hexdigest() != spec["training_manifest_sha256"]:
            raise ValueError(f"Training manifest changed: {mode}")
        if spec.get("config", {}).get("token_policy") == "question":
            if not manifest.get("pre_question_cache_checks", {}).get(mode, {}).get("kv_recurrent_conv_and_logits_exact"):
                raise ValueError("Missing native prefix-memory identity check")
    expected = {(case["id"], variant, query["role"]): query["answer"] for case in cases
                for variant, content in case["variants"].items() for query in content["queries"]}
    models, scores, stratified = {}, {}, {}
    for mode in expected_modes:
        rows = load_rows(args.run / f"{mode}.jsonl", expected)
        models[mode], scores[mode] = aggregate(rows), case_scores(rows)
        strata = defaultdict(list)
        for row in rows:
            if row["family"] == "copy" and row["variant"] != "before":
                strata[(row["distractor_updates"], row["variant"])].append(row)
        stratified[mode] = [{"distractor_updates": length, "variant": variant, **aggregate(members)}
                            for (length, variant), members in sorted(strata.items())]
    reference = manifest["config"].get("comparison_reference", "shared")
    pairs = manifest["config"].get("paired_comparisons", [(reference, mode) for mode in expected_modes if mode != reference])
    comparisons = {f"{left}_minus_{right}": paired_interval(scores[left], scores[right]) for left, right in pairs}
    protocols = [spec["config"] for spec in manifest.get("checkpoint_sources", {}).values()]
    seeds = sorted({protocol["initialization_seed"] for protocol in protocols or [manifest["training_config"]]})
    report = {"status": "completed", "cases": len(cases), "cases_sha256": manifest["cases_sha256"],
              "evaluation_kind": manifest.get("evaluation_kind", "fresh_seed_confirmation"),
              "training_initialization_seeds": seeds, "models": models, "paired_copy_joint": comparisons,
              "copy_by_length_and_order": stratified,
              "pre_question_cache_checks": manifest.get("pre_question_cache_checks", {}),
              "interpretation": "Frozen evaluation on the recorded cohort and initialization seeds. Paired intervals resample cases, conditional on the specified checkpoints; no equivalence or architecture novelty claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for mode, result in models.items():
        print(mode, next(row for row in result["joint_all_three_correct"] if row["family"] == "copy"))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
