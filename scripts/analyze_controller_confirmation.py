"""Audit frozen confirmation and bootstrap paired joint-readout differences by case."""

import argparse
from collections import defaultdict
from dataclasses import asdict
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


def cluster_scores(values, lookup):
    grouped = defaultdict(list)
    for case, value in values.items():
        grouped[lookup[case]].append(value)
    return {cluster: mean(members) for cluster, members in grouped.items()}


def scalar_metrics(rows):
    return {"correct": sum(row["greedy_exact"] for row in rows), "count": len(rows),
            "accuracy": mean(row["greedy_exact"] for row in rows),
            "answer_nll": mean(row["answer_nll"] for row in rows)}


def unary_cluster_scores(rows, lookup):
    grouped = defaultdict(list)
    for row in rows:
        grouped[lookup[row["case_id"]]].append(row["greedy_exact"])
    return {cluster: mean(values) for cluster, values in grouped.items()}


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
    config = manifest["config"]
    if config.get("task_protocol") == "copy_generalization_v1":
        from sas_memory.generalization_tasks import validate_generalization_cases
        if validate_generalization_cases(cases) != manifest["task_validation"]:
            raise ValueError("Generalization validation changed")
    case_lookup = {case["id"]: case for case in cases}
    cluster_key = config.get("cluster_key")
    clusters = {case["id"]: case[cluster_key] if cluster_key else case["id"] for case in cases}
    strata_fields = config.get("strata", [])
    oracle_options = config.get("oracle_evaluation")
    if oracle_options and manifest.get("completed_oracle_modes") != expected_modes:
        raise ValueError("Oracle evaluation is incomplete")
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
            if oracle_options and not manifest.get("oracle_pre_question_cache_checks", {}).get(mode, {}).get("kv_recurrent_conv_and_logits_exact"):
                raise ValueError("Missing oracle-prefix memory identity check")
    expected = {(case["id"], variant, query["role"]): query["answer"] for case in cases
                for variant, content in case["variants"].items() for query in content["queries"]}
    models, scores, stratified, by_condition, oracle_metrics = {}, {}, {}, {}, {}
    for mode in expected_modes:
        rows = load_rows(args.run / f"{mode}.jsonl", expected)
        models[mode], scores[mode] = aggregate(rows), cluster_scores(case_scores(rows), clusters)
        strata = defaultdict(list)
        for row in rows:
            if row["family"] == "copy" and row["variant"] != "before":
                strata[(row["distractor_updates"], row["variant"])].append(row)
        stratified[mode] = [{"distractor_updates": length, "variant": variant, **aggregate(members)}
                            for (length, variant), members in sorted(strata.items())]
        if strata_fields:
            conditions = defaultdict(list)
            for row in rows:
                condition = tuple(case_lookup[row["case_id"]][field] for field in strata_fields)
                conditions[condition].append(row)
            by_condition[mode] = [{**dict(zip(strata_fields, condition)),
                                   "base_cases": len({clusters[row["case_id"]] for row in members}),
                                   **aggregate(members)} for condition, members in sorted(conditions.items())]
        if oracle_options:
            if set(oracle_options["roles"]) != {"target_current"} or set(oracle_options["variants"]) != {"control", "treatment"}:
                raise ValueError("Current oracle analysis expects both order variants and target_current only")
            from sas_memory.oracle import evidence_slice
            from sas_memory.tasks import Operation, execute
            expected_oracle = {key: answer for key, answer in expected.items()
                               if key[1] in oracle_options["variants"] and key[2] in oracle_options["roles"]}
            oracle_rows = load_rows(args.run / f"{mode}.oracle.jsonl", expected_oracle)
            for row in oracle_rows:
                case = case_lookup[row["case_id"]]
                variant = case["variants"][row["variant"]]
                query = next(query for query in variant["queries"] if query["role"] == row["role"])
                events = [Operation(**event) for event in case["prefix"] + variant["tail"]]
                selected = evidence_slice(events, query["entity"], query["version"])
                if row["oracle_events"] != [asdict(event) for event in selected] or row["selected_event_count"] != len(selected) or row["original_event_count"] != len(events):
                    raise ValueError("Oracle evidence or recorded counts changed")
                if str(execute(selected)[query["version"]][query["entity"]]) != query["answer"]:
                    raise ValueError("Oracle evidence does not preserve truth")
            current_rows = [row for row in rows if (row["case_id"], row["variant"], row["role"]) in expected_oracle]
            oracle_conditions, raw_conditions = defaultdict(list), defaultdict(list)
            for destination, members in ((oracle_conditions, oracle_rows), (raw_conditions, current_rows)):
                for row in members:
                    condition = tuple(case_lookup[row["case_id"]][field] for field in strata_fields)
                    destination[condition].append(row)
            gain = paired_interval(unary_cluster_scores(oracle_rows, clusters), unary_cluster_scores(current_rows, clusters))
            gain["unit"] = f"{cluster_key or 'case'}, averaging matched conditions and order variants"
            oracle_metrics[mode] = {"raw_current": scalar_metrics(current_rows), "oracle_current": scalar_metrics(oracle_rows),
                                   "oracle_minus_raw_current": gain,
                                   "by_condition": [{**dict(zip(strata_fields, condition)),
                                                     "raw_current": scalar_metrics(raw_conditions[condition]),
                                                     "oracle_current": scalar_metrics(members)}
                                                    for condition, members in sorted(oracle_conditions.items())]}
    reference = manifest["config"].get("comparison_reference", "shared")
    pairs = manifest["config"].get("paired_comparisons", [(reference, mode) for mode in expected_modes if mode != reference])
    comparisons = {f"{left}_minus_{right}": paired_interval(scores[left], scores[right]) for left, right in pairs}
    if cluster_key:
        for result in comparisons.values():
            result["unit"] = f"{cluster_key}, averaging matched conditions and two order variants"
    protocols = [spec["config"] for spec in manifest.get("checkpoint_sources", {}).values()]
    seeds = sorted({protocol["initialization_seed"] for protocol in protocols or [manifest["training_config"]]})
    report = {"status": "completed", "cases": len(cases), "cases_sha256": manifest["cases_sha256"],
              "evaluation_kind": manifest.get("evaluation_kind", "fresh_seed_confirmation"),
              "training_initialization_seeds": seeds, "models": models, "paired_copy_joint": comparisons,
              "independent_copy_clusters": len(next(iter(scores.values()))),
              "evaluation_clusters": len(set(clusters.values())), "cluster_key": cluster_key or "case_id",
              "by_condition": by_condition, "oracle_current": oracle_metrics,
              "copy_by_length_and_order": stratified,
              "pre_question_cache_checks": manifest.get("pre_question_cache_checks", {}),
              "oracle_pre_question_cache_checks": manifest.get("oracle_pre_question_cache_checks", {}),
              "limitations": (["Oracle filtering changes positions, token count and recurrent computation; gains do not isolate retrieval, GDN memory or reasoning causes."] if oracle_options else []),
              "interpretation": "Frozen evaluation on the recorded cohort and initialization seeds. Paired intervals resample cases, conditional on the specified checkpoints; no equivalence or architecture novelty claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for mode, result in models.items():
        print(mode, next(row for row in result["joint_all_three_correct"] if row["family"] == "copy"))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
