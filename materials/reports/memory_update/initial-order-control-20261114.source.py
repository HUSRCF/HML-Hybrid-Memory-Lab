"""Export canonical/permuted comparisons from an independently audited frozen run."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from scripts.run_controller_trial import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.audit.read_text())
    if report["status"] != "completed_and_audited" or file_hash(Path(report["source_run"]) / "manifest.json") != report["manifest_sha256"]:
        raise ValueError("A current full-cohort independent audit is required")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    names = ["native", "shared", "independent", "local", "local_question"]
    labels = ["Native", "Shared", "Independent", "Local", "Question local"]
    specifications = [("individual", "Individual answers (432)"), ("three_roles", "All 3 roles per context (144)"),
                      ("stable_unchanged_case_role", "Unchanged fact across 3 variants (96)")]
    figure, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    positions = np.arange(len(names))
    for axis, (metric, title) in zip(axes, specifications):
        pairs = [report["canonical_to_permuted"][name][metric] for name in names]
        for offset, (key, label, color) in enumerate((("reference_correct", "Canonical order", "#94a3b8"), ("candidate_correct", "Permuted order", "#2563eb"))):
            values = [100 * pair[key] / pair["units"] for pair in pairs]
            bars = axis.bar(positions + (offset - .5) * .36, values, .34, label=label, color=color)
            for bar, pair in zip(bars, pairs):
                axis.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, str(pair[key]), ha="center", va="bottom", fontsize=8)
        axis.set_ylim(0, 110)
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.set_title(title, fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("All required answers correct (%)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(.5, .06), ncol=2, frameon=False)
    figure.suptitle("Independent initial-record order: frozen Qwen3.5-2B IT controllers", fontsize=13)
    figure.text(.5, .015, "Reused validation facts; identical query positions; 12 base event families, one initialization. No isolated write-damage claim.", ha="center", fontsize=9, color="#475569")
    figure.tight_layout(rect=(0, .14, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    write_json(args.output.with_suffix(".provenance.json"), {"audit_report": str(args.audit), "audit_sha256": file_hash(args.audit), "plot_source_sha256": file_hash(Path(__file__)), "figure_sha256": file_hash(args.output)})
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(str(args.output))


if __name__ == "__main__":
    main()
