"""Standalone descriptive figures from an independently audited capacity trial."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="PNG path; also writes PDF")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report["status"] != "completed_and_audited" or set(report["arms"]) != {"local", "gdn_query"}:
        raise ValueError("Plot requires the completed independent training audit")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), sharex=True, sharey=True)
    labels = {"local": "Local adapters (689,216)", "gdn_query": "GDN query adapters (688,128)"}
    colors = {"local": "#2563eb", "gdn_query": "#d97706"}
    for column, split in enumerate(("train", "validation")):
        for row, metric in enumerate(("family", "answer")):
            axis = axes[row, column]
            for arm in ("local", "gdn_query"):
                checkpoints = report["arms"][arm]
                points = [checkpoint["splits"][split] for checkpoint in checkpoints]
                values = [point["all_four_correct"] / point["families"] if metric == "family" else point["accuracy"] for point in points]
                axis.plot([checkpoint["step"] for checkpoint in checkpoints], values, marker="o", color=colors[arm], label=labels[arm])
            axis.axhline(0, color="#6b7280", linewidth=1, linestyle=":", label="Native, records blocked")
            axis.axhline(1, color="#6b7280", linewidth=1, linestyle="--", label="Native, full history")
            axis.set_xscale("log", base=2)
            axis.set_xticks([64, 256, 1024], ["64", "256", "1024"])
            axis.set_ylim(-.03, 1.05)
            axis.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100"])
            axis.grid(axis="y", alpha=.16)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_title(f"{split.capitalize()}: {'all four siblings' if metric == 'family' else 'individual answers'}")
            if column == 0:
                axis.set_ylabel("Accuracy (%)")
            if row == 1:
                axis.set_xlabel("Optimizer updates")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.015))
    figure.suptitle("Frozen Qwen3.5-2B IT: same-question, four-value carrier/readout control", fontsize=12)
    figure.text(.5, .087, "One initialization; 32 train / 16 validation families. Descriptive capacity diagnostic, not an architecture comparison.",
                ha="center", fontsize=8, color="#4b5563")
    figure.tight_layout(rect=(0, .11, 1, .96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    args.output.with_suffix(".plot-source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"png": str(args.output), "pdf": str(args.output.with_suffix('.pdf'))}))


if __name__ == "__main__":
    main()
