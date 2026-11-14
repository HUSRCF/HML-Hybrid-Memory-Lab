"""Export every factorial outcome, paired comparison and signed repeat cost."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.overwrite_history_data import CELLS, ROLES, registered_inputs
from scripts.overwrite_history_metrics import summarize
from scripts.run_controller_trial import write_json


def complete_inputs(audit_path):
    report = json.loads(audit_path.read_text())
    run = Path(report["source_run"])
    manifest = json.loads((run / "manifest.json").read_text())
    registered, data, registry = registered_inputs(Path(manifest["dataset"]))
    if (report["status"] != "completed_and_audited" or report["records_audited"] != 384 or
        report["case_values_audited"] != 32 or report["same_final_pairs_audited"] != 64 or
        report["actual_attention_checks"] != 16 or report["native_exact_replays"] != 16 or
        manifest["status"] != "completed" or manifest["smoke"] or
        streaming_hash(run / "manifest.json") != report["manifest_sha256"] or
        streaming_hash(Path(manifest["dataset"]) / "manifest.json") != report["dataset_manifest_sha256"] or
        registered["registry_sha256"] != report["registry_sha256"] or data_fingerprint(registry) != report["registry_sha256"] or
        streaming_hash(audit_path.with_suffix(".source.py")) != report["analysis_source_sha256"] or
        report["reference_gate"]["previous_history_primary_passed"]):
        raise ValueError("Complete independently audited counterbalanced native screen required")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / "sources" / name) != digest:
            raise ValueError("Actual/archived audited implementation changed")
    raw_path = run / "validation.native.jsonl"
    if streaming_hash(raw_path) != manifest["result_sha256"]:
        raise ValueError("Actual recorded native outcomes changed")
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    metrics = summarize(rows, data)
    if metrics != report["metrics"]:
        raise ValueError("Complete paired statistics no longer reproduce the independent audit")
    return report, rows, metrics


def make_plot(metrics, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={"width_ratios": [1.3, 1]})
    roles = ROLES + ("all_roles_joint",)
    labels = ["Historical A", "Current A", "Current C", "All jointly"]
    for i, group in enumerate(metrics["cohort_results"]):
        ax, heat = axes[i]
        x = np.arange(4)
        for shift, cells, color, label in ((-.18, ("YX", "XY"), "#0284c7", "Novel final write"),
                                           (.18, ("XX", "YY"), "#e58b1a", "Repeated final write")):
            values = [sum(group["counts"][cell][role]["correct"] for cell in cells) for role in roles]
            ax.bar(x + shift, values, width=.36, color=color, label=label)
            for place, value in zip(x + shift, values):
                ax.text(place, value + .4, str(value), ha="center", fontsize=10)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 36)
        ax.set_yticks(range(0, 33, 8))
        ax.set_ylabel("Exact correct /32 related final-value pairs")
        ax.set_title(group["cohort"].capitalize() + ": pooled counterbalanced values")
        raw, qual = group["raw"], group["guard_qualified"]
        ax.text(.5, -.23, f"Current A losses/rescues: {raw['losses']}/{raw['rescues']} (net {raw['signed_net_loss']})\n"
            f"Guard-qualified: {qual['losses']}/{qual['rescues']} (net {qual['signed_net_loss']}); "
            f"eligible {qual['pairs']}/32; positive streams {len(group['positive_qualified_base_streams'])}/4",
            transform=ax.transAxes, ha="center", fontsize=9)
        values = np.array([[group["counts"][cell][role]["correct"] for role in roles] for cell in CELLS])
        heat.imshow(values, vmin=0, vmax=16, cmap="Blues", aspect="auto")
        heat.set_xticks(range(4), labels, fontsize=9)
        heat.set_yticks(range(4), ["XX repeat", "XY novel", "YX novel", "YY repeat"])
        heat.set_title("Individual cells: exact correct /16")
        for row in range(4):
            for col in range(4):
                val = values[row, col]
                heat.text(col, row, str(val), ha="center", va="center", color="white" if val >= 11 else "#111827", fontsize=12)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .085), ncol=2)
    decision = ("met in both cohorts" if metrics["primary_signal"]["present_in_both_cohorts"]
                else "not met in both cohorts")
    fig.suptitle("Same final states with counterbalanced overwrite histories\n"
                 "Registered repeat-loss gate: " + decision, fontsize=15)
    fig.text(.5, .017, "Native Qwen3.5-2B IT, 128 accounts; compare YX→XX and XY→YY with identical final answers and executable states.\nQualified pairs require historical A and current C correct in both worlds; report losses and rescues.\nFour base streams per cohort with related value/final/cell outcomes; descriptive gate only. The earlier history-preservation primary remains failed.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .14, 1, .95), h_pad=4.1, w_pad=2)
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
    result = {"status": "complete_counterbalanced_overwrite_history_export", "source_audit": str(args.audit),
        "source_audit_sha256": streaming_hash(args.audit), "source_manifest_sha256": report["manifest_sha256"],
        "dataset_manifest_sha256": report["dataset_manifest_sha256"], "registry_sha256": report["registry_sha256"],
        "analysis_source_sha256": streaming_hash(Path(__file__)), "weights_sha256": report["weights_sha256"],
        "metrics": metrics, "reference_gate": report["reference_gate"], "interpretation": report["interpretation"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        fields = ["cohort", "base_stream_id", "case_id", "value_version", "variant", "role", "answer", "generated", "greedy_exact", "answer_nll"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    with args.output.with_name(args.output.stem + "-pairs.csv").open("w", newline="") as stream:
        fields = ["id", "cohort", "base_stream_id", "case_id", "value_version", "final_identity", "answer",
            "novel_cell", "repeat_cell", "novel_current_correct", "repeat_current_correct", "raw_repeat_loss",
            "raw_repeat_rescue", "preservation_guards_correct", "qualified_repeat_loss", "qualified_repeat_rescue",
            "current_nll_repeat_minus_novel"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pair in metrics["individual_same_final_pairs"]:
            writer.writerow({key: pair[key] for key in fields})
    make_plot(metrics, args.plot)
    print(json.dumps({"status": result["status"], "examples": metrics["examples"], "correct": metrics["correct"],
        "primary_signal": metrics["primary_signal"]}))


if __name__ == "__main__":
    main()
