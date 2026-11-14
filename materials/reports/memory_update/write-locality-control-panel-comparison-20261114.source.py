"""Export complete panel outcomes beside the already audited fixed GDN controls."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.locality_panel_data import MODELS, read_plan, registered_inputs
from scripts.locality_panel_metrics import summarize
from scripts.export_write_locality_adapter_transfer import complete_inputs as timing_inputs
from scripts.run_controller_trial import write_json

LABELS = {"native": "Native", "gdn_all_tokens": "GDN all", "gdn_question_only": "GDN question",
          "shared": "Shared", "independent": "Independent", "local": "Local all",
          "local_question": "Local question", "attention_only": "Attention only"}


def unique_role_totals(registry, indexed):
    pairs = {p["id"]: p for p in registry["pairs"]}
    roles = {}
    for binding in registry["bindings"]:
        role = binding["label"]
        if role == "preservation":
            role = "protected_B" if pairs[binding["pair_id"]]["stratum"] == "unrelated_B_at32" else "historical_A"
        key = binding["physical_id"]
        if key in roles and roles[key] != role:
            raise ValueError("One physical query was assigned conflicting functional roles")
        roles[key] = role
    return {model: {role: {"correct": sum(rows[key]["greedy_exact"] for key, value in roles.items() if value == role),
                          "physical_queries": sum(value == role for value in roles.values())}
                    for role in ("protected_B", "historical_A", "current_A", "current_C")}
            for model, rows in indexed.items()}


def complete_inputs(audit_path):
    report = json.loads(audit_path.read_text())
    run = Path(report["source_run"])
    manifest = json.loads((run / "manifest.json").read_text())
    registered, _, registry = registered_inputs(Path(manifest["dataset"]))
    if (report["status"] != "completed_and_audited" or report["physical_readouts_audited"] != 1920 or
        report["reported_bindings_audited"] != 2240 or report["pairs_per_model_audited"] != 96 or
        manifest["status"] != "completed" or manifest["smoke"] or
        streaming_hash(run / "manifest.json") != report["manifest_sha256"] or
        streaming_hash(Path(manifest["dataset"]) / "manifest.json") != report["dataset_manifest_sha256"] or
        registered["registry_sha256"] != report["registry_sha256"] or
        data_fingerprint(registry) != report["registry_sha256"] or
        streaming_hash(audit_path.with_suffix(".source.py")) != report["analysis_source_sha256"]):
        raise ValueError("Complete independently audited five-model panel required")
    for name, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / "sources" / name) != digest:
            raise ValueError("Audited panel implementation changed")
    rows = {}
    for model in MODELS:
        raw = run / (model + ".jsonl")
        if streaming_hash(raw) != manifest["models"][model]["results_sha256"]:
            raise ValueError("Actual panel observations changed")
        rows[model] = [json.loads(line) for line in raw.read_text().splitlines()]
    metrics = summarize(registry, rows)
    if metrics != report["metrics"]:
        raise ValueError("Complete panel statistics no longer reproduce")
    return report, registry, rows, metrics


def make_plot(panel_metrics, timing_metrics, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    models = list(LABELS)
    groups = {(g["cohort"], g["stratum"]): g for g in panel_metrics["cohort_stratum_results"]}
    timing = {(g["cohort"], g["stratum"]): g for g in timing_metrics["cohort_stratum_results"]}
    strata = {"unrelated_B_at32": "Unrelated B, 32 accounts",
              "initial_A_at128": "Historical A, 128 accounts",
              "unrelated_C_on_initial_A_at128": "Historical A after C update"}
    roles = ("preservation", "current_A", "current_C", "all_roles_joint")
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#eeeeee")
    for i, cohort in enumerate(("discovery", "replica")):
        for j, (stratum, title) in enumerate(strata.items()):
            key = cohort, stratum
            counts = {**timing[key]["counts"], **groups[key]["counts"]}
            values = np.array([[counts[m]["right"][role]["correct"] if role in counts[m]["right"] else np.nan
                                for role in roles] for m in models])
            ax = axes[i, j]
            ax.imshow(values, vmin=0, vmax=16, cmap=cmap, aspect="auto")
            ax.set_xticks(range(4), ["Preserve", "Current A", "Current C", "Joint"])
            ax.set_yticks(range(len(models)), [LABELS[m] for m in models])
            for row in range(len(models)):
                for col in range(4):
                    value = values[row, col]
                    label = "n/a" if np.isnan(value) else str(int(value))
                    ax.text(col, row, label, ha="center", va="center", color="white" if value >= 11 else "#111827", fontsize=11)
            ax.set_title(f"{cohort.capitalize()}: {title}", fontsize=12)
    fig.suptitle("Fixed hybrid-block controls on the same natural update endpoints", fontsize=17)
    fig.text(.5, .018, "Exact counts /16 at the updated natural endpoint; Joint requires preservation and every current-value companion.\nFour base streams per cohort with related value variants; fixed final checkpoints trained on shorter tasks, no new training.\nGDN rows reuse the separate audited timing transfer. Different ranks, one initialization and weaker Attention-only teacher fit limit architecture claims.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .10, 1, .95))
    fig.savefig(destination, dpi=160)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    report, registry, rows, metrics = complete_inputs(args.audit)
    plan = read_plan()
    prior_path = Path(plan["prior_timing_audit"])
    if streaming_hash(prior_path) != plan["prior_timing_audit_sha256"]:
        raise ValueError("Fixed external GDN timing audit changed")
    timing_report, _, old_registry, old_rows, timing_metrics = timing_inputs(prior_path)
    if registry != old_registry or report["weights_sha256"] != timing_report["weights_sha256"]:
        raise ValueError("External timing comparison changed natural inputs or native checkpoint")
    indexed = {"native": {q["id"]: q["native_full_readout"] for q in registry["physical_queries"]},
               **{m: {r["id"]: r for r in items} for m, items in {**old_rows, **rows}.items()}}
    result = {"status": "complete_fixed_control_panel_export", "source_audit": str(args.audit),
        "source_audit_sha256": streaming_hash(args.audit), "source_manifest_sha256": report["manifest_sha256"],
        "external_timing_audit": str(prior_path), "external_timing_audit_sha256": streaming_hash(prior_path),
        "analysis_source_sha256": streaming_hash(Path(__file__)), "registry_sha256": report["registry_sha256"],
        "weights_sha256": report["weights_sha256"], "panel_metrics": metrics, "external_timing_metrics": timing_metrics,
        "combined_unique_query_totals": {**timing_metrics["unique_query_totals"], **metrics["unique_query_totals"]},
        "combined_unique_role_totals": unique_role_totals(registry, indexed),
        "posthoc_descriptive_output_agreement": {
            "shared_versus_local_question": {
                "physical_queries": len(indexed["shared"]),
                "identical_generated": sum(indexed["shared"][key]["generated"] == indexed["local_question"][key]["generated"] for key in indexed["shared"]),
                "same_correctness": sum(indexed["shared"][key]["greedy_exact"] == indexed["local_question"][key]["greedy_exact"] for key in indexed["shared"])},
            "interpretation": "Descriptive comparison added after observing equal totals; no representation equivalence, causal equivalence or population equivalence claim."},
        "teacher_train_fit": report["teacher_train_fit"], "interpretation": report["interpretation"]}
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
                writer.writerow({**{k: pair[k] for k in ("cohort", "stratum", "base_stream_id", "case_id", "value_version")},
                    **{k: binding[k] for k in ("side", "label", "physical_id")}, "model": model,
                    **{k: row[k] for k in ("answer", "generated", "greedy_exact", "answer_nll")}})
    make_plot(metrics, timing_metrics, args.plot)
    print(json.dumps({"status": result["status"], "combined_unique_query_totals": result["combined_unique_query_totals"]}))


if __name__ == "__main__":
    main()
