"""Decompose the earlier two-record interference without selecting failures."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.counterbalance_data import file_digest
from scripts.analyze_controller_trial import load_rows
from scripts.analyze_controller_confirmation import scalar_metrics, paired_interval, unary_cluster_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target_manifest = json.loads((args.target / "manifest.json").read_text())
    paths = {"oracle": Path(target_manifest["config"]["oracle_kv_reference"]),
             "target_initial_only": args.target, "source_other_only": args.source,
             "competing_versions": Path(target_manifest["config"]["both_reference"])}
    audit_dir = args.output.with_suffix("").with_name(args.output.stem + "-audits")
    audit_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for kind, path in paths.items():
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["config"].get("selection_kind", "oracle") != kind:
            raise ValueError("Incorrect split protocol")
        subprocess.run([sys.executable, str(ROOT / "scripts/analyze_oracle_kv_readout.py"), str(path), "--output", str(audit_dir / f"{kind}.json")], check=True)
        manifests[kind] = manifest
    cases = json.loads((args.target / "cases.json").read_text())
    expected = {(case["id"], order, "target_current"): next(q["answer"] for q in case["variants"][order]["queries"] if q["role"] == "target_current") for case in cases for order in ("control", "treatment")}
    lookup = {case["id"]: case for case in cases}
    clusters = {case["id"]: case["base_case_id"] for case in cases}
    validation = Path(target_manifest["config"]["control_validation"])
    if file_digest(validation) != target_manifest["config"]["control_validation_sha256"]:
        raise ValueError("Singleton token-budget registration changed")
    registered = {(row["case_id"], row["variant"]): row for row in json.loads(validation.read_text())["rows"]}
    for kind, manifest in manifests.items():
        if any(manifest[key] != target_manifest[key] for key in ("cases_sha256", "model_path", "reference_manifest_sha256")):
            raise ValueError("Split cohort/backbone/reference differs")
        if manifest["checkpoint_sha256"]["local_question"] != target_manifest["checkpoint_sha256"]["local_question"]:
            raise ValueError("Split checkpoints differ")
    rows = {kind: load_rows(path / "local_question.jsonl", expected) for kind, path in paths.items()}
    indexed = {kind: {(row["case_id"], row["variant"]): row for row in members} for kind, members in rows.items()}
    for key in expected:
        case_id, order, _ = key
        pair = (case_id, order)
        members = {kind: values[pair] for kind, values in indexed.items()}
        if len({row["prompt_ids_sha256"] for row in members.values()}) != 1:
            raise ValueError("Split inputs changed")
        if members["target_initial_only"]["kept_history_tokens"] != members["source_other_only"]["kept_history_tokens"]:
            raise ValueError("Singleton retained-token counts differ")
        sets = {kind: set(row["selected_event_indices"]) for kind, row in members.items()}
        if sets["target_initial_only"] & sets["source_other_only"] != sets["oracle"] or sets["target_initial_only"] | sets["source_other_only"] != sets["competing_versions"]:
            raise ValueError("Singleton selections do not decompose the original two-record mask")
        for kind, row in members.items():
            if row["selected_event_indices"] != registered[pair]["event_indices"][kind]:
                raise ValueError("Singleton registration changed")
    conditions = sorted({(case["dependency_depth"], case["question_style"]) for case in cases})
    # An exploratory second difference in NLL, not a new primary outcome.
    interaction = defaultdict(list)
    for key in indexed["oracle"]:
        values = {kind: members[key]["answer_nll"] for kind, members in indexed.items()}
        interaction[clusters[key[0]]].append(values["competing_versions"] - values["target_initial_only"] - values["source_other_only"] + values["oracle"])
    interaction_mean = {base: mean(values) for base, values in interaction.items()}
    second_difference = paired_interval(interaction_mean, {base: 0. for base in interaction_mean})
    second_difference["unit"] = "base_case_id; exploratory answer-NLL second difference, nats per gold answer token"
    errors = {}
    for kind, members in rows.items():
        labels = []
        for row in members:
            case, pair = lookup[row["case_id"]], (row["case_id"], row["variant"])
            query = next(q for q in case["variants"][row["variant"]]["queries"] if q["role"] == "target_current")
            initial = next(event["value"] for event in case["prefix"] if event["target"] == query["entity"])
            extra, = set(indexed["source_other_only"][pair]["selected_event_indices"]) - set(indexed["oracle"][pair]["selected_event_indices"])
            source_value = (case["prefix"] + case["variants"][row["variant"]]["tail"])[extra]["value"]
            prediction = row["generated"].strip()
            labels.append("correct" if row["greedy_exact"] else "target_initial_value" if prediction == str(initial)
                          else "unnecessary_source_value" if prediction == str(source_value) else "other")
        errors[kind] = dict(Counter(labels))
    report = {"status": "completed", "base_cases": len(set(clusters.values())), "queries_per_regime": len(expected),
              "training_initializations": 1, "checkpoint": "question_only_seed1701", "runs": {kind: str(path) for kind, path in paths.items()},
              "manifest_sha256": {kind: file_digest(path / "manifest.json") for kind, path in paths.items()},
              "regimes": {kind: scalar_metrics(members) for kind, members in rows.items()},
              "error_types": errors,
              "singleton_accuracy_differences_vs_oracle": {kind: paired_interval(unary_cluster_scores(rows[kind], clusters), unary_cluster_scores(rows["oracle"], clusters)) for kind in ("target_initial_only", "source_other_only")},
              "exploratory_nll_second_difference": second_difference,
              "by_order": {order: {kind: scalar_metrics([row for row in members if row["variant"] == order]) for kind, members in rows.items()} for order in ("control", "treatment")},
              "by_condition": [{"dependency_depth": depth, "question_style": style,
                                "regimes": {kind: scalar_metrics([row for row in members if lookup[row["case_id"]]["dependency_depth"] == depth and lookup[row["case_id"]]["question_style"] == style]) for kind, members in rows.items()}} for depth, style in conditions],
              "interpretation": "Exploratory decomposition on all old 60000-series cases, one frozen initialization. Singleton retained-token budgets match and their union is the previously evaluated two-extra mask. Necessary evidence and original input remain intact. Target-initial and unused-source records differ in positions and values; this identifies components of the bundled effect, not a fully isolated semantic-version cause. No recurrent-damage or architecture claim."}
    for value in report["singleton_accuracy_differences_vs_oracle"].values():
        value["unit"] = "base_case_id, averaging six depth/wording conditions and both COPY orders"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["regimes"], indent=2))
    print(json.dumps(report["by_order"], indent=2))


if __name__ == "__main__":
    main()
