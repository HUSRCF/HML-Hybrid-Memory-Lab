"""Audit fixed-checkpoint branch/time ablations and their exact replay controls."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from scripts.analyze_controller_trial import aggregate, load_rows


def replay_error(actual, expected):
    left = {(row["case_id"], row["variant"], row["role"]): row for row in actual}
    right = {(row["case_id"], row["variant"], row["role"]): row for row in expected}
    if left.keys() != right.keys():
        raise ValueError("Replay row keys differ")
    error = 0.0
    for key, row in left.items():
        if row["generated"] != right[key]["generated"]:
            raise ValueError("Replay greedy answers differ")
        error = max(error, abs(row["answer_nll"] - right[key]["answer_nll"]))
    if error > 1e-5:
        raise ValueError(f"Replay NLL differs: {error}")
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "completed", "runs": [], "interpretation": [
        "Post-training disablement can introduce distribution shifts; this is not matched-budget retraining.",
        "Ledger-only changes both recurrent state formation and evidence KV construction. It does not isolate GDN writes.",
        "Question-only includes the fixed question suffix and answer generation, not just one attention query.",
        "Source provenance varies across older runs; inspect each run's source audit separately from endpoint replay."
    ]}
    for run in args.runs:
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError(f"Ablation incomplete: {run}")
        missing_snapshots = []
        for relative, digest in manifest["source_sha256"].items():
            snapshot = run / "sources" / relative
            if not snapshot.exists():
                missing_snapshots.append(relative)
            elif hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source snapshot changed: {relative}")
        training = Path(manifest["training_run"])
        data = json.loads((training / "data.json").read_text())
        if data_fingerprint(data) != manifest["data_sha256"]:
            raise ValueError("Training/evaluation data changed")
        checkpoint = training / manifest["mode"] / "controller.safetensors"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed")
        expected = {(case["id"], variant, query["role"]): query["answer"] for case in data["test"]
                    for variant, content in case["variants"].items() for query in content["queries"]}
        enabled = "all_tokens" if manifest.get("temporal") else "all_enabled"
        labels = ([enabled, "ledger_only", "question_only", "all_disabled"] if manifest.get("temporal") else
                  [enabled, "gdn_only_enabled", "attention_only_enabled", "attention_query_only", "attention_key_only", "all_disabled"])
        if "planned_ablations" in manifest:
            labels = list(manifest["planned_ablations"])
        if manifest["completed_ablations"] != labels:
            raise ValueError("Ablation labels incomplete")
        rows = {label: load_rows(run / f"{label}.jsonl", expected) for label in labels}
        errors = {enabled: replay_error(rows[enabled], load_rows(training / manifest["mode"] / "test.jsonl", expected)),
                  "all_disabled": replay_error(rows["all_disabled"], load_rows(training / "native_test.jsonl", expected))}
        result = {"path": str(run), "mode": manifest["mode"], "temporal": manifest.get("temporal", False),
                  "gdn_components": manifest.get("gdn_components", False),
                  "source_audit": {"missing_snapshots": missing_snapshots,
                                   "evaluation_helper_recorded": "scripts/run_controller_trial.py" in manifest["source_sha256"]},
                  "endpoint_max_nll_error": errors,
                  "conditions": {label: aggregate(members) for label, members in rows.items()}}
        report["runs"].append(result)
        for label, values in result["conditions"].items():
            print(manifest["mode"], label, next(row for row in values["joint_all_three_correct"] if row["family"] == "copy"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
