"""Audit matched-budget trials and report version-preserving joint competence."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint


def load_rows(path, expected):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    seen = set()
    for row in rows:
        key = (row["case_id"], row["variant"], row["role"])
        if key in seen or expected.get(key) != row["answer"]:
            raise ValueError(f"Duplicate result or wrong gold label: {key}")
        seen.add(key)
        valid = re.fullmatch(r"\s*(-?\d+)\s*", row["generated"])
        if row["greedy_exact"] != bool(valid and valid.group(1) == row["answer"]):
            raise ValueError("Stored correctness disagrees with raw generation")
        if not math.isfinite(row["answer_nll"]) or row["answer_nll"] < 0:
            raise ValueError("Invalid answer NLL")
    if seen != expected.keys():
        raise ValueError(f"Missing evaluation rows: {path}")
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["variant"] != "before":
            grouped[(row["family"], row["role"])].append(row)
    metrics = [{"family": family, "role": role, "correct": sum(r["greedy_exact"] for r in members),
                "count": len(members), "accuracy": mean(r["greedy_exact"] for r in members),
                "answer_nll": mean(r["answer_nll"] for r in members)}
               for (family, role), members in sorted(grouped.items())]
    contexts = defaultdict(list)
    for row in rows:
        if row["variant"] != "before":
            contexts[(row["family"], row["case_id"], row["variant"])].append(row)
    joint = defaultdict(list)
    for (family, _, _), members in contexts.items():
        if {r["role"] for r in members} != {"target_current", "protected_current", "target_initial"}:
            raise ValueError("Incomplete functional context")
        joint[family].append(all(r["greedy_exact"] for r in members))
    return {"metrics": metrics, "joint_all_three_correct": [
        {"family": family, "correct": sum(values), "count": len(values), "accuracy": mean(values)}
        for family, values in sorted(joint.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    report = {"models": {}, "runs": [], "interpretation": "Pilot; query pairs share cases. No significance or semantic-interface claim."}
    shared_hash, shared_token_hash, native_rows = None, None, None
    for path in args.runs:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["status"] != "completed" or set(manifest["completed_modes"]) != set(manifest["modes"]):
            raise ValueError(f"Run is incomplete: {path}")
        data = json.loads((path / "data.json").read_text())
        if data_fingerprint(data) != manifest["data_sha256"]:
            raise ValueError("Data fingerprint mismatch")
        if shared_hash is None:
            shared_hash = manifest["data_sha256"]
        if shared_hash != manifest["data_sha256"]:
            raise ValueError("Cannot compare trials with different data")
        encoded = json.loads((path / "tokenized_train.json").read_text())
        token_hash = data_fingerprint([{key: item[key] for key in ("input_ids", "answer_ids")} for item in encoded])
        if shared_token_hash is not None and shared_token_hash != token_hash:
            raise ValueError("Tokenized training inputs or supervised answers differ")
        shared_token_hash = token_hash
        for relative, digest in manifest["source_sha256"].items():
            if hashlib.sha256((path / "sources" / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source snapshot mismatch: {relative}")
        expected = {(case["id"], variant, query["role"]): query["answer"] for case in data["test"]
                    for variant, content in case["variants"].items() for query in content["queries"]}
        native = load_rows(path / "native_test.jsonl", expected)
        if native_rows is not None and native != native_rows:
            raise ValueError("Native baselines differ across runs")
        native_rows = native
        report["models"]["native"] = aggregate(native)
        for mode in manifest["modes"]:
            policy = manifest["config"].get("token_policy", "all")
            label = mode if policy == "all" else f"{mode}_{policy}"
            if label in report["models"]:
                raise ValueError(f"Repeated mode and token policy: {label}")
            checks = json.loads((path / "safety_checks.json").read_text())[mode]
            if checks["status"] != "passed" or checks["initial_logit_max_error"] != 0:
                raise ValueError("Model safety checks did not pass")
            if checks.get("token_policy", "all") != policy:
                raise ValueError("Safety check and training token policies differ")
            log = [json.loads(line) for line in (path / mode / "training.jsonl").read_text().splitlines()]
            if [r["step"] for r in log] != list(range(1, manifest["config"]["steps"] + 1)):
                raise ValueError("Training steps are incomplete")
            if any(not math.isfinite(r["loss"]) or not math.isfinite(r["gradient_norm"]) for r in log):
                raise ValueError("Nonfinite training record")
            rows = load_rows(path / mode / "test.jsonl", expected)
            summary = json.loads((path / mode / "summary.json").read_text())
            report["models"][label] = {**aggregate(rows), "parameters": checks["trainable_parameters"], "token_policy": policy,
                                      "training_seconds": summary["training_seconds"],
                                      "peak_allocated_gib": summary["peak_allocated_gib"]}
        report["runs"].append({"path": str(path), "data_sha256": manifest["data_sha256"],
                               "tokenized_input_answer_sha256": token_hash,
                               "steps": manifest["config"]["steps"],
                               "optimized_input_tokens": manifest["optimized_input_tokens_per_mode"],
                               "supervised_answer_tokens": manifest["supervised_answer_tokens_per_mode"],
                               "model": manifest["config"]["model"], "group_index": manifest["config"]["group_index"],
                               "token_policy": manifest["config"].get("token_policy", "all")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, model in report["models"].items():
        current = next(r for r in model["metrics"] if r["family"] == "copy" and r["role"] == "target_current")
        history = next(r for r in model["metrics"] if r["family"] == "copy" and r["role"] == "target_initial")
        joint = next(r for r in model["joint_all_three_correct"] if r["family"] == "copy")
        print(json.dumps({"model": name, "copy_current": f"{current['correct']}/{current['count']}",
                          "copy_initial": f"{history['correct']}/{history['count']}",
                          "copy_joint": f"{joint['correct']}/{joint['count']}"}))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names, values = list(report["models"]), []
        for name in names:
            model = report["models"][name]
            values.append([next(r["accuracy"] for r in model["metrics"] if r["family"] == "copy" and r["role"] == role)
                           for role in ("target_current", "target_initial")]
                          + [next(r["accuracy"] for r in model["joint_all_three_correct"] if r["family"] == "copy")])
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for index, label in enumerate(("Current balance", "Initial balance", "All three readouts")):
            ax.bar([i + (index - 1) * 0.24 for i in range(len(names))], [row[index] for row in values], width=0.23, label=label)
        ax.set_xticks(range(len(names)), [name.replace("_", " ") for name in names])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Greedy exact accuracy")
        ax.set_title("Frozen 2B controller pilot: COPY / 8 held-out cases, 2 orders")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
        fig.text(0.5, 0.01, "Small diagnostic sample; no significance claim. All methods retain protected-account accuracy.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=180, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
