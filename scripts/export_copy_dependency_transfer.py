"""Export every fixed COPY readout and paired control comparison, with a figure."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.copy_transfer_data import MODELS, ROLES, read_plan, registered_inputs, selected_inputs
from scripts.copy_transfer_metrics import summarize
from scripts.run_controller_trial import write_json


def complete_inputs(audit_path):
    report = json.loads(audit_path.read_text())
    run = Path(report["source_run"])
    manifest = json.loads((run / "manifest.json").read_text())
    registered, cases, registry = registered_inputs(Path(manifest["dataset"]))
    if (report["status"] != "completed_and_audited" or report["physical_readouts_audited"] != 2688 or
        report["reported_bindings_audited"] != 3456 or report["actual_attention_checks"] != 48 or
        report["detached_native_replays"] != 48 or report["native_prefix_identity_checks"] != 12 or
        manifest["status"] != "completed" or manifest["smoke"] or
        streaming_hash(run / "manifest.json") != report["manifest_sha256"] or
        registered["registry_sha256"] != report["registry_sha256"] or
        streaming_hash(audit_path.with_suffix(".source.py")) != report["analysis_source_sha256"]):
        raise ValueError("Complete independently audited eight-arm COPY transfer required")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / "sources" / name) != digest:
            raise ValueError("Actual/archived audited COPY implementation changed")
    all_rows = {}
    for name in MODELS:
        path = run / (name + ".jsonl")
        if streaming_hash(path) != manifest["models"][name]["results_sha256"]:
            raise ValueError("Actual fixed-model readouts changed")
        all_rows[name] = [json.loads(line) for line in path.read_text().splitlines()]
    metrics = summarize(all_rows, cases, registry, read_plan())
    if metrics != report["metrics"]:
        raise ValueError("All complete COPY comparisons no longer reproduce the independent audit")
    return report, all_rows, registry, metrics


def make_plot(metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = ["Native", "Shared", "Independent", "Local all", "Local question", "Attention only", "GDN all", "GDN question"]
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), gridspec_kw={"width_ratios": [1.15, 1]})
    ax, heat, net, preserve = axes.flat
    y = np.arange(len(MODELS))
    for shift, field, color, label in ((-.19, "both_order_targets_correct", "#0284c7", "Current target correct in both orders"),
                                       (.19, "joint_six_after", "#f59e0b", "All six after-update answers correct")):
        values = [metrics["models"][m][field] for m in MODELS]
        ax.barh(y + shift, values, height=.38, color=color, label=label)
        for row, value in zip(y + shift, values):
            ax.text(value + .45, row, str(value), va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 53)
    ax.set_xticks(range(0, 49, 8))
    ax.set_xlabel("Complete depth/wording cases correct /48 (related)")
    ax.set_title("Snapshot semantics and preservation")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.27), fontsize=9)
    conditions = [(d, s) for d in (1, 2, 3) for s in ("original", "entity_first")]
    values = np.array([[sum(c["joint_after"] for c in metrics["models"][m]["case_outcomes"]
        if (c["dependency_depth"], c["question_style"]) == condition) for condition in conditions] for m in MODELS])
    heat.imshow(values, vmin=0, vmax=8, cmap="Blues", aspect="auto")
    heat.set_yticks(y, labels)
    heat.set_xticks(range(6), [f"D{d}\n{'Original' if s == 'original' else 'Entity first'}" for d, s in conditions], fontsize=9)
    heat.set_title("All six after-update answers correct /8 base streams")
    for row in range(8):
        for col in range(6):
            heat.text(col, row, str(values[row, col]), ha="center", va="center", color="white" if values[row, col] >= 5 else "#111827")
    colors = ["#0284c7", "#f59e0b", "#7c3aed"]
    net_values = []
    for shift, reference, color in zip((-.24, 0, .24), ("independent", "local_question", "gdn_question_only"), colors):
        comparison = next(c for c in metrics["comparisons"] if c["candidate"] == "shared" and c["reference"] == reference)
        values = [d["signed_net_gain"] for d in comparison["depth_results"]]
        net_values.extend(values)
        net.bar(np.arange(3) + shift, values, width=.24, color=color, label=reference.replace("_", " "))
        for x, value in zip(np.arange(3) + shift, values):
            net.text(x, value + (.25 if value >= 0 else -.25), str(value), ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    net.axhline(0, color="#64748b", linewidth=.8)
    net.set_xticks(range(3), ["Depth 1", "Depth 2", "Depth 3"])
    net.set_ylabel("Shared minus control: joint gains - losses /16")
    net.set_title("Paired joint differences; both question wordings retained")
    net.set_ylim(min([0] + net_values) - 2, max([0] + net_values) + 2)
    net.legend(fontsize=9)
    counts, totals = [], [48, 48, 96, 96]
    for m in MODELS:
        roles = metrics["models"][m]["role_variant_counts"]
        counts.append([roles[ROLES[2]]["before"]["correct"], roles[ROLES[1]]["before"]["correct"],
            sum(roles[ROLES[2]][v]["correct"] for v in ("control", "treatment")),
            sum(roles[ROLES[1]][v]["correct"] for v in ("control", "treatment"))])
    preserve.imshow(np.array(counts) / np.array(totals), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    preserve.set_yticks(y, labels)
    preserve.set_xticks(range(4), ["Initial\nbefore", "Unrelated\nbefore", "Initial\nafter", "Unrelated\nafter"], fontsize=9)
    preserve.set_title("All preservation outcomes, including incorrect baselines")
    for row in range(8):
        for col in range(4):
            value = counts[row][col]
            preserve.text(col, row, f"{value}/{totals[col]}", ha="center", va="center", color="white" if value / totals[col] >= .6 else "#111827", fontsize=10)
    status = "passed" if metrics["descriptive_shared_candidate_gate"]["passed"] else "not passed"
    fig.suptitle("Fixed-checkpoint transfer on snapshot-COPY dependencies\nShared candidate screen: " + status + " (reused development cohort)", fontsize=16, y=.99)
    fig.text(.5, .015, "Native Qwen3.5-2B IT + seven existing final adapters; no new training. 8 base streams × 3 depths × 2 wordings.\n"
        "432 related bindings /336 unique prompts per model; only identical before-context prompts are merged.\n"
        "A shared candidate must beat independent and both question-stage controls overall, in ≥3 streams and at every depth. Prior history/repeat primaries remain failed.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .075, 1, .93), h_pad=4, w_pad=3)
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    report, rows, registry, metrics = complete_inputs(args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "complete_eight_arm_copy_dependency_transfer_export", "source_audit": str(args.audit),
        "source_audit_sha256": streaming_hash(args.audit), "source_manifest_sha256": report["manifest_sha256"],
        "dataset_manifest_sha256": report["dataset_manifest_sha256"], "registry_sha256": report["registry_sha256"],
        "analysis_source_sha256": streaming_hash(Path(__file__)), "weights_sha256": report["weights_sha256"],
        "metrics": metrics, "references": report["references"], "interpretation": report["interpretation"]}
    write_json(args.output, result)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    physical_fields = ["model", "id", "answer", "generated", "greedy_exact", "answer_nll", "query_tokens"]
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=physical_fields)
        writer.writeheader()
        for name in MODELS:
            for row in rows[name]:
                writer.writerow({k: row[k] for k in physical_fields})
    _, bindings = selected_inputs(registry)
    with args.output.with_name(args.output.stem + "-bindings.csv").open("w", newline="") as stream:
        fields = list(bindings[0]) + physical_fields
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in MODELS:
            indexed = {r["id"]: r for r in rows[name]}
            for binding in bindings:
                row = indexed[binding["physical_id"]]
                writer.writerow({**binding, **{k: row[k] for k in physical_fields}})
    make_plot(metrics, args.plot)
    print(json.dumps({"status": result["status"], "shared_candidate_screen": metrics["descriptive_shared_candidate_gate"]}), flush=True)


if __name__ == "__main__":
    main()
