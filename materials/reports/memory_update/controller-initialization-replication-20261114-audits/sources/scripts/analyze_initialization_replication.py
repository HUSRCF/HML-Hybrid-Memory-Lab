"""Audit fixed-cohort paired training-seed replication without pooling dependent rows."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from scripts.analyze_controller_ablations import replay_error
from scripts.analyze_controller_confirmation import case_scores, paired_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/controller_replication_analysis.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    audits = args.output.parent / f"{args.output.stem}-audits"
    audits.mkdir(parents=True, exist_ok=True)
    rows_by_seed, paired_scores, inputs = {}, {}, []
    cohort, native, optimization = None, None, None
    max_native_error = 0.0
    for requested in config["runs"]:
        run = Path(requested["path"])
        audit_path = audits / f"{run.name}.json"
        audit = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_controller_confirmation.py"),
                                str(run), "--output", str(audit_path)], capture_output=True, text=True)
        if audit.returncode:
            raise RuntimeError(f"Source-run audit failed: {run}\n{audit.stdout}\n{audit.stderr}")
        report = json.loads(audit_path.read_text())
        manifest = json.loads((run / "manifest.json").read_text())
        if cohort is not None and report["cases_sha256"] != cohort:
            raise ValueError("Replication cohorts differ")
        cohort = report["cases_sha256"]
        baseline = [json.loads(line) for line in (run / "native.jsonl").read_text().splitlines()]
        if native is not None:
            max_native_error = max(max_native_error, replay_error(baseline, native))
        native = baseline
        inputs.append({"run": str(run), "audit": str(audit_path),
                       "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest()})
        for pair in requested["pairs"]:
            seed = pair["seed"]
            if seed in rows_by_seed:
                raise ValueError("Repeated training initialization")
            values, scores, checks = {}, {}, {}
            for policy in ("all", "question"):
                label = pair[policy]
                spec = manifest["checkpoint_sources"][label]
                protocol = spec["config"]
                source = Path(spec["training_run"])
                if (spec["mode"] != "local" or protocol["initialization_seed"] != seed
                        or protocol.get("token_policy", "all") != policy):
                    raise ValueError("Controller identity differs from the locked pair")
                training = json.loads((source / "manifest.json").read_text())
                data = json.loads((source / "data.json").read_text())
                if data_fingerprint(data) != training["data_sha256"] or training["data_sha256"] != config["training_data_sha256"]:
                    raise ValueError("Training data changed or differs across seeds")
                encoded = json.loads((source / "tokenized_train.json").read_text())
                digest = data_fingerprint([{key: item[key] for key in ("input_ids", "answer_ids")} for item in encoded])
                if digest != config["tokenized_input_answer_sha256"]:
                    raise ValueError("Actual training inputs/answer tokens differ")
                description = json.loads((source / "local/controller.json").read_text())
                if description["trainable_parameters"] != config["parameters_per_controller"]:
                    raise ValueError("Controller parameter budgets differ")
                identity = {key: protocol[key] for key in ("model", "group_index", "steps", "gradient_accumulation",
                            "learning_rate", "weight_decay", "max_gradient_norm", "adapt_gdn_keys")}
                identity["rank"] = protocol["modes"]["local"]
                if optimization is not None and identity != optimization:
                    raise ValueError("Architecture or optimization differs across seeds/policies")
                optimization = identity
                rows = [json.loads(line) for line in (run / f"{label}.jsonl").read_text().splitlines()]
                scores[policy] = case_scores(rows)
                values[policy] = report["models"][label]
                checks[policy] = {"run": str(source), "parameters": description["trainable_parameters"],
                                  "training_token_sha256": digest}
            rows_by_seed[seed] = {"models": values, "question_minus_all": paired_interval(scores["question"], scores["all"]),
                                  "checks": checks}
            paired_scores[seed] = scores
    if sorted(rows_by_seed) != sorted(config["initialization_seeds"]):
        raise ValueError("A prespecified initialization is missing")
    keys = next(iter(paired_scores.values()))["all"].keys()
    if any(scores[policy].keys() != keys for scores in paired_scores.values() for policy in ("all", "question")):
        raise ValueError("Paired case keys differ across seeds")
    fixed_seed_scores = {policy: {case: mean(scores[policy][case] for scores in paired_scores.values()) for case in keys}
                         for policy in ("all", "question")}
    average_difference = paired_interval(fixed_seed_scores["question"], fixed_seed_scores["all"])
    average_difference["unit"] = "case, averaging two orders and the fixed prespecified training seeds"
    report = {"status": "completed", "evaluation_kind": "fixed_cohort_initialization_replication", "cases_sha256": cohort,
              "initialization_seeds": sorted(rows_by_seed), "per_seed": rows_by_seed,
              "mean_joint_accuracy": {policy: mean(values.values()) for policy, values in fixed_seed_scores.items()},
              "fixed_seed_average_difference": average_difference, "native_replay_max_nll_error": max_native_error,
              "inputs": inputs, "analysis_config": config,
              "interpretation": "Every seed reported. Case bootstrap is conditional on these few training seeds; do not treat repeated seeds/orders/questions as independent cases or infer equivalence/general architecture superiority."}
    report["analysis_source_sha256"] = {}
    for path in (Path(__file__).resolve(), ROOT / "scripts/analyze_controller_confirmation.py",
                 ROOT / "scripts/analyze_controller_ablations.py", ROOT / "scripts/analyze_controller_trial.py",
                 ROOT / "sas_memory/training_data.py", args.config.resolve()):
        relative = str(path.relative_to(ROOT))
        report["analysis_source_sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot = audits / "sources" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for seed, result in sorted(rows_by_seed.items()):
        print(seed, {policy: next(v for v in model["joint_all_three_correct"] if v["family"] == "copy")
                     for policy, model in result["models"].items()})
    print(json.dumps(average_difference, indent=2))


if __name__ == "__main__":
    main()
