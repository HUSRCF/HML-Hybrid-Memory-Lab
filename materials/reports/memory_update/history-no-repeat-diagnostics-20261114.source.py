"""Export the complete no-repeat history screen, all paired cases and figure."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.history_no_repeat_data import CELLS, ROLES, registered_inputs
from scripts.history_no_repeat_metrics import summarize
from scripts.run_controller_trial import write_json


def error_numeric_matches(metrics):
    counts = Counter()
    for pair in metrics["individual_pairs"]:
        worlds = pair["worlds"]
        named = {"initial_A": worlds[CELLS[0]]["roles"][ROLES[0]]["answer"],
                 "middle_A": worlds[CELLS[0]]["roles"][ROLES[1]]["answer"],
                 "new_A": worlds[CELLS[1]]["roles"][ROLES[1]]["answer"],
                 "current_C": worlds[CELLS[0]]["roles"][ROLES[2]]["answer"]}
        for cell in CELLS:
            for role, row in worlds[cell]["roles"].items():
                if row["greedy_exact"]:
                    continue
                match = re.fullmatch(r"\s*(-?\d+)\s*", row["generated"])
                value = match.group(1) if match else None
                counts[role, cell, tuple(k for k, v in named.items() if value == v)] += 1
    if sum(counts.values()) != metrics["examples"] - metrics["correct"]:
        raise ValueError("Error taxonomy lost an actual failed readout")
    return {"counts": [{"role": key[0], "world": key[1], "matches": list(key[2]), "errors": value}
                       for key, value in counts.items()],
            "interpretation": "Post-hoc numeric-output matches, not inferred mechanisms; all original primary decisions are unchanged."}


def complete_inputs(audit_path):
    report = json.loads(audit_path.read_text())
    run = Path(report["source_run"])
    manifest = json.loads((run / "manifest.json").read_text())
    registered, data, registry = registered_inputs(Path(manifest["dataset"]))
    if (report["status"] != "completed_and_audited" or report["records_audited"] != 384 or
        report["pairs_audited"] != 64 or report["actual_attention_checks"] != 16 or
        report["native_exact_replays"] != 16 or manifest["smoke"] or manifest["status"] != "completed" or
        streaming_hash(run / "manifest.json") != report["manifest_sha256"] or
        streaming_hash(Path(manifest["dataset"]) / "manifest.json") != report["dataset_manifest_sha256"] or
        registered["registry_sha256"] != report["registry_sha256"] or data_fingerprint(registry) != report["registry_sha256"] or
        streaming_hash(audit_path.with_suffix(".source.py")) != report["analysis_source_sha256"]):
        raise ValueError("Complete independently audited384-query native screen required")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / "sources" / name) != digest:
            raise ValueError("Actual/archived audited diagnostic implementation changed")
    raw_path = run / "validation.native.jsonl"
    if streaming_hash(raw_path) != manifest["result_sha256"]:
        raise ValueError("Actual native observations changed")
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    metrics = summarize(rows, data)
    if metrics != report["metrics"]:
        raise ValueError("Recomputed complete screen statistics differ from independent audit")
    return report, rows, metrics


def make_plot(metrics, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    groups = {(g["cohort"], g["active_accounts"]): g for g in metrics["cohort_load_results"]}
    roles = ROLES + ("all_roles_joint",)
    labels = ["Historical A", "Current A", "Current C", "All roles jointly"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for i, cohort in enumerate(("discovery", "replica")):
        for j, load in enumerate((32, 128)):
            group, ax = groups[cohort, load], axes[i, j]
            x = np.arange(4)
            for shift, cell, color, label in ((-.18, CELLS[0], "#0284c7", "Hold intermediate A"),
                                             (.18, CELLS[1], "#e58b1a", "Update A again")):
                values = [group["counts"][cell][role]["correct"] for role in roles]
                ax.bar(x + shift, values, width=.36, color=color, label=label)
                for place, value in zip(x + shift, values):
                    ax.text(place, value + .22, str(value), ha="center", fontsize=10)
            ax.set_xticks(x, labels, fontsize=9)
            ax.set_ylim(0, 18)
            ax.set_yticks(range(0, 17, 4))
            ax.set_title(f"{cohort.capitalize()}, {load} accounts ({'primary' if load==128 else 'secondary'})")
            if j == 0:
                ax.set_ylabel("Exact correct /16 related value cases")
            count, streams = len(group["commit_qualified_regression_cases"]), len(group["commit_qualified_regression_base_streams"])
            floor = "\nLimited hold-world history competence" if group["baseline_historical_floor"] else ""
            ax.text(.5, -.19, f"Qualified history regressions: {count}/16 across {streams} base streams{floor}",
                    transform=ax.transAxes, ha="center", fontsize=9, wrap=True)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .085), ncol=2)
    fig.suptitle("Historical-version retention without a recent initial-value repeat", fontsize=16)
    fig.text(.5, .018, "Native Qwen3.5-2B IT; identical common prefixes and distant initial evidence in both worlds.\nQualified regressions require current A and C correct in both worlds; four base streams per cohort, related value siblings.\nThe registered gate requires a two-stream effect at 128 accounts in both cohorts; functional errors do not isolate recurrent-state damage.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .14, 1, .94), h_pad=3.3)
    fig.savefig(destination, dpi=160)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    report, rows, metrics = complete_inputs(args.audit)
    result = {"status": "complete_native_no_repeat_history_export", "source_audit": str(args.audit),
        "source_audit_sha256": streaming_hash(args.audit), "source_manifest_sha256": report["manifest_sha256"],
        "dataset_manifest_sha256": report["dataset_manifest_sha256"], "registry_sha256": report["registry_sha256"],
        "analysis_source_sha256": streaming_hash(Path(__file__)), "weights_sha256": report["weights_sha256"],
        "metrics": metrics, "posthoc_error_numeric_matches": error_numeric_matches(metrics),
        "interpretation": report["interpretation"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        fields = ["cohort", "active_accounts", "base_stream_id", "case_id", "value_version", "variant", "role",
                  "answer", "generated", "greedy_exact", "answer_nll"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    paired_path = args.output.with_name(args.output.stem + "-pairs.csv")
    with paired_path.open("w", newline="") as stream:
        fields = ["cohort", "active_accounts", "base_stream_id", "case_id", "value_version", "history_regressed",
                  "history_rescued", "all_current_correct_both_worlds", "commit_qualified_history_regression",
                  "hold_joint", "update_joint", "hold_history", "update_history"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pair in metrics["individual_pairs"]:
            writer.writerow({**{key: pair[key] for key in fields[:9]},
                "hold_joint": pair["worlds"][CELLS[0]]["all_roles_joint"],
                "update_joint": pair["worlds"][CELLS[1]]["all_roles_joint"],
                "hold_history": pair["worlds"][CELLS[0]]["roles"][ROLES[0]]["generated"],
                "update_history": pair["worlds"][CELLS[1]]["roles"][ROLES[0]]["generated"]})
    make_plot(metrics, args.plot)
    print(json.dumps({"status": result["status"], "examples": metrics["examples"], "correct": metrics["correct"],
                      "primary_signal": metrics["primary_signal"]}))


if __name__ == "__main__":
    main()
