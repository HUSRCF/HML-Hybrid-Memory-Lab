"""Paired, nonvacuous functional comparisons from a completed independent audit."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from scripts.analyze_controller_trial import load_rows
from scripts.run_controller_trial import write_json


def functional_units(rows):
    groups = {name: defaultdict(list) for name in ("individual", "four_values", "three_roles", "twelve_value_variant",
               "stable_unchanged_case_role", "stable_unchanged_four_values", "updated_target", "updated_target_except_add")}
    for row in rows:
        case, base, variant, role = row["case_id"], row["base_case_id"], row["variant"], row["role"]
        groups["individual"][case, variant, role].append(row)
        groups["four_values"][base, variant, role].append(row)
        groups["three_roles"][case, variant].append(row)
        groups["twelve_value_variant"][base, variant].append(row)
        if role in ("protected_current", "target_initial"):
            groups["stable_unchanged_case_role"][case, role].append(row)
            groups["stable_unchanged_four_values"][base, role].append(row)
        if role == "target_current" and variant != "before":
            groups["updated_target"][case, variant].append(row)
            if row["family"] != "accumulate":
                groups["updated_target_except_add"][case, variant].append(row)
    expected_sizes = {"individual": 1, "four_values": 4, "three_roles": 3, "twelve_value_variant": 12,
                      "stable_unchanged_case_role": 3, "stable_unchanged_four_values": 12,
                      "updated_target": 1, "updated_target_except_add": 1}
    units = {}
    for metric, members in groups.items():
        if any(len(items) != expected_sizes[metric] for items in members.values()):
            raise ValueError("Incomplete paired functional unit")
        if metric.startswith("stable_") and any(len({row["answer"] for row in items}) != (4 if metric.endswith("four_values") else 1) for items in members.values()):
            raise ValueError("Unchanged-fact unit has invalid executable labels")
        units[metric] = {key: all(row["greedy_exact"] for row in items) for key, items in members.items()}
    return units


def paired_counts(reference, candidate):
    if reference.keys() != candidate.keys():
        raise ValueError("Paired metrics have different cases")
    rescued = sum(not reference[key] and candidate[key] for key in reference)
    regressed = sum(reference[key] and not candidate[key] for key in reference)
    return {"units": len(reference), "reference_correct": sum(reference.values()), "candidate_correct": sum(candidate.values()),
            "rescued": rescued, "regressed": regressed, "both_correct": sum(reference[key] and candidate[key] for key in reference),
            "both_wrong": sum(not reference[key] and not candidate[key] for key in reference), "net_correct_change": rescued - regressed}


def compare(audited):
    if audited["status"] != "completed_and_audited" or set(audited["arms"]) != {"shared", "independent", "local", "local_question"}:
        raise ValueError("Comparison requires all four independently audited trained arms")
    run = Path(audited["source_run"])
    if file_hash(run / "manifest.json") != audited["manifest_sha256"]:
        raise ValueError("Training manifest changed since audit")
    manifest = json.loads((run / "manifest.json").read_text())
    native_run = Path(manifest["native_reference"]["source_run"])
    native_manifest = json.loads((native_run / "manifest.json").read_text())
    if (file_hash(native_run / "manifest.json") != manifest["native_reference"]["manifest_sha256"] or
        file_hash(native_run / "validation.native.jsonl") != native_manifest["result_sha256"]):
        raise ValueError("Native reference changed since audit")
    native = [json.loads(line) for line in (native_run / "validation.native.jsonl").read_text().splitlines()]
    expected = {(row["case_id"], row["variant"], row["role"]): row["answer"] for row in native}
    rows = {"native": load_rows(native_run / "validation.native.jsonl", expected)}
    for arm, model in audited["arms"].items():
        evaluation = run / arm / "step-000432/evaluation"
        if file_hash(evaluation / "summary.json") != model["evaluation_summary_sha256"]:
            raise ValueError("Trained evaluation summary changed since audit")
        summary = json.loads((evaluation / "summary.json").read_text())
        if file_hash(evaluation / "validation.jsonl") != summary["file_sha256"]["validation.jsonl"]:
            raise ValueError("Trained free answers changed since audit")
        rows[arm] = load_rows(evaluation / "validation.jsonl", expected)
    units = {name: functional_units(members) for name, members in rows.items()}
    pairs = [("native", arm) for arm in audited["arms"]] + [(arm, "shared") for arm in ("independent", "local", "local_question")] + [("local", "local_question")]
    return {"status": "completed_paired_comparison", "source_run": str(run), "training_manifest_sha256": audited["manifest_sha256"],
            "models": {name: {metric: {"correct": sum(values.values()), "units": len(values)} for metric, values in model.items()} for name, model in units.items()},
            "comparisons": [{"reference": reference, "candidate": candidate,
                             "metrics": {metric: paired_counts(units[reference][metric], units[candidate][metric]) for metric in units[reference]}}
                            for reference, candidate in pairs],
            "target_update_by_family": {name: {family: {"correct": sum(row["greedy_exact"] for row in members if row["family"] == family and row["role"] == "target_current" and row["variant"] != "before"),
                                                      "examples": sum(row["family"] == family and row["role"] == "target_current" and row["variant"] != "before" for row in members)}
                                               for family in sorted({row["family"] for row in members})} for name, members in rows.items()},
            "interpretation": "Single initialization and 12 underlying value-paired event families. Counts share cases; equal averages are not equivalence and a small gap is not superiority. Stable unchanged units require correct before/control/treatment answers, so making the before answer wrong cannot improve this metric. ADD is reported separately from other target updates. No isolated recurrent-write or semantic-interface claim."}


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    names = ["native", "shared", "independent", "local", "local_question"]
    labels = ["Native", "Shared", "Independent", "Local", "Question local"]
    metrics = {"four_values": "4 values per query (108)", "three_roles": "3 roles per context (144)",
               "twelve_value_variant": "12 readouts per variant (36)"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.3), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(names))
    for offset, (metric, label) in enumerate(metrics.items()):
        values = [100 * report["models"][name][metric]["correct"] / report["models"][name][metric]["units"] for name in names]
        axes[0].bar(x + (offset - 1) * .24, values, .23, label=label)
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("All required answers correct (%)")
    axes[0].set_title("Joint semantic competence")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, frameon=False, fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(.5, .065))
    axes[0].spines[["top", "right"]].set_visible(False)
    families = sorted(report["target_update_by_family"]["native"])
    values = np.array([[100 * report["target_update_by_family"][name][family]["correct"] / report["target_update_by_family"][name][family]["examples"] for name in names] for family in families])
    axes[1].imshow(values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    axes[1].set_xticks(range(len(names)), labels, rotation=25, ha="right")
    axes[1].set_yticks(range(len(families)), [family.replace("_", " ") for family in families])
    axes[1].set_title("Current target after updates")
    for row, family in enumerate(families):
        for column, name in enumerate(names):
            cell = report["target_update_by_family"][name][family]
            axes[1].text(column, row, f"{cell['correct']}/{cell['examples']}", ha="center", va="center", color="white" if values[row, column] > 60 else "black", fontsize=9)
    figure.suptitle("Frozen Qwen3.5-2B IT: matched one-pass semantic-update adaptation", fontsize=12)
    figure.text(.5, .015, "One initialization, 12 underlying event families; descriptive results, no shared-interface or superiority claim.", ha="center", fontsize=9, color="#4b5563")
    figure.tight_layout(rect=(0, .13, 1, .94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    report = compare(json.loads(args.audit.read_text()))
    report.update(audit_report_sha256=file_hash(args.audit), analysis_source_sha256=file_hash(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    if args.plot:
        plot(report, args.plot)
    print(json.dumps(report["models"]))


if __name__ == "__main__":
    main()
