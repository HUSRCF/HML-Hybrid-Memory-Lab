"""Export the complete fixed-checkpoint transfer and every natural-world binding."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.locality_transfer_data import MODELS
from scripts.locality_transfer_metrics import summarize
from scripts.run_controller_trial import write_json


def complete_inputs(audit_path):
    report = json.loads(audit_path.read_text())
    run = Path(report["source_run"])
    manifest = json.loads((run / "manifest.json").read_text())
    data = Path(manifest["dataset"])
    registered = json.loads((data / "manifest.json").read_text())
    registry = json.loads((data / "registry.json").read_text())
    if (report["status"] != "completed_and_audited" or report["physical_readouts_audited"] != 768 or
        report["reported_bindings_audited"] != 896 or report["pairs_per_model_audited"] != 96 or
        manifest["status"] != "completed" or manifest["smoke"] or
        streaming_hash(run / "manifest.json") != report["manifest_sha256"] or
        streaming_hash(data / "manifest.json") != report["dataset_manifest_sha256"] or
        data_fingerprint(registry) != registered["registry_sha256"] or registered["registry_sha256"] != report["registry_sha256"] or
        streaming_hash(audit_path.with_suffix(".source.py")) != report["analysis_source_sha256"]):
        raise ValueError("Complete independently audited transfer required")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / "sources" / name) != digest:
            raise ValueError("Audited runtime sources changed")
    rows = {}
    for model in MODELS:
        raw = run / (model + ".jsonl")
        if streaming_hash(raw) != manifest["models"][model]["results_sha256"]:
            raise ValueError("Actual frozen-adapter observations changed")
        rows[model] = [json.loads(line) for line in raw.read_text().splitlines()]
    metrics = summarize(registry, rows)
    if metrics != report["metrics"]:
        raise ValueError("Recomputed complete transfer statistics disagree with independent audit")
    return report, manifest, registry, rows, metrics


def make_plot(metrics, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    strata = {"unrelated_B_at32": "Protected B, 32 accounts",
              "initial_A_at128": "Initial A, 128 accounts",
              "unrelated_C_on_initial_A_at128": "Initial A after C update, 128 accounts"}
    groups = {(g["cohort"], g["stratum"]): g for g in metrics["cohort_stratum_results"]}
    models = ("native",) + MODELS
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for i, cohort in enumerate(("discovery", "replica")):
        for j, (stratum, title) in enumerate(strata.items()):
            group = groups[(cohort, stratum)]
            ax = axes[i, j]
            x = np.arange(3)
            for shift, role, color, label in ((-.18, "preservation", "#0284c7", "Preservation"),
                                               (.18, "all_roles_joint", "#15986e", "Preservation + current updates")):
                values = [group["counts"][m]["right"][role]["correct"] for m in models]
                ax.bar(x + shift, np.array(values) * 100 / group["pairs"], width=.36, color=color, label=label)
                for place, count in zip(x + shift, values):
                    ax.text(place, count * 100 / group["pairs"] + 1.5, str(count), ha="center", fontsize=10)
            ax.set_xticks(x, ["Native", "GDN\nall tokens", "GDN\nquestion phase"])
            ax.set_ylim(0, 110)
            ax.set_title(f"{cohort.capitalize()}: {title}", fontsize=11)
            if j == 0:
                ax.set_ylabel("Exact accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .08), ncol=2)
    fig.suptitle("Fixed GDN checkpoints: transfer to natural long-prefix update worlds", fontsize=16)
    fig.text(.5, .018, "Updated natural endpoints, full-prefill execution for every model; correct own-world current A, and also current C in the C-axis stratum.\nCounts /16 related value readouts, four base streams per cohort. Fixed checkpoints trained on shorter eight-account tasks; no new training.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .15, 1, .94))
    fig.savefig(destination, dpi=160)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    report, manifest, registry, rows, metrics = complete_inputs(args.audit)
    indexed = {"native": {q["id"]: q["native_full_readout"] for q in registry["physical_queries"]},
               **{model: {r["id"]: r for r in values} for model, values in rows.items()}}
    result = {"status": "complete_fixed_checkpoint_transfer_export", "source_audit": str(args.audit),
              "source_audit_sha256": streaming_hash(args.audit), "source_manifest_sha256": report["manifest_sha256"],
              "analysis_source_sha256": streaming_hash(Path(__file__)), "weights_sha256": report["weights_sha256"],
              "adapter_state_sha256": report["adapter_state_sha256"], "metrics": metrics,
              "prefix_identity_checks": report["prefix_identity_checks"],
              "interpretation": report["interpretation"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    pair_index = {p["id"]: p for p in registry["pairs"]}
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        fields = ["cohort", "stratum", "base_stream_id", "case_id", "value_version", "side", "label", "physical_id",
                  "model", "answer", "generated", "greedy_exact", "answer_nll"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for binding in registry["bindings"]:
            pair = pair_index[binding["pair_id"]]
            for model, observations in indexed.items():
                row = observations[binding["physical_id"]]
                writer.writerow({**{key: pair[key] for key in ("cohort", "stratum", "base_stream_id", "case_id", "value_version")},
                                 **{key: binding[key] for key in ("side", "label", "physical_id")}, "model": model,
                                 **{key: row[key] for key in ("answer", "generated", "greedy_exact", "answer_nll")}})
    make_plot(metrics, args.plot)
    print(json.dumps({"status": result["status"], "unique_query_totals": metrics["unique_query_totals"]}))


if __name__ == "__main__":
    main()
