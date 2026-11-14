"""Export all whole-record necessity readouts and contrasts without replacing prior decisions."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_copy_evidence_necessity import audit_run
from scripts.copy_necessity_data import MODELS, REGIMES, ROLES, registered_inputs, selected_inputs
from scripts.copy_necessity_metrics import summarize
from scripts.run_controller_trial import write_json

DISPLAY = ["Native", "Shared", "Independent", "Local all", "Local question", "Attention only", "GDN all", "GDN question"]
ALL_REGIMES = ("full", *REGIMES)
COLORS = ("#64748b", "#e11d48", "#0284c7")
LABELS = ("Full history (sealed reference)", "Remove whole evidence SET", "Remove nearby irrelevant SET")


def complete_inputs(audit_path):
    report = json.loads(audit_path.read_text())
    run = Path(report["source_run"])
    if (report["status"] != "completed_and_audited" or report["new_physical_readouts_audited"] != 5376 or
        report["new_related_bindings_audited"] != 6912 or report["all_history_exact_controls"] != 80 or
        report["actual_masked_attention_checks"] != 160 or report["complete_prefix_and_historical_KV_proofs"] != 160 or
        report["detached_native_exact_replays"] != 80 or report["question_only_native_prefix_checks"] != 12 or
        streaming_hash(audit_path.with_suffix(".source.py")) != report["analysis_source_sha256"] or audit_run(run) != report):
        raise ValueError("Complete independently reproducible5376-query frozen-memory audit required")
    manifest = json.loads((run / "manifest.json").read_text())
    registered, source_plan, cases, registry, masks, full_rows = registered_inputs(Path(manifest["dataset"]))
    new_rows = {m: {} for m in MODELS}
    for m in MODELS:
        for r in REGIMES:
            path = run / f"{m}.{r}.jsonl"
            if streaming_hash(path) != manifest["models"][m]["results_sha256"][r]:
                raise ValueError("Actual readouts changed before export")
            new_rows[m][r] = [json.loads(line) for line in path.read_text().splitlines()]
    metrics = summarize(new_rows, full_rows, cases, registry, source_plan,
                        json.loads((run / "plan.json").read_text()))
    if metrics != report["metrics"]:
        raise ValueError("Complete three-regime metrics do not reproduce audit")
    rows = {r: {m: full_rows[m] if r == "full" else new_rows[m][r] for m in MODELS} for r in ALL_REGIMES}
    return report, manifest, registry, rows, metrics


def after_count(result, role):
    return sum(result["role_variant_counts"][role][v]["correct"] for v in ("control", "treatment"))


def make_plot(metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(17, 12), dpi=160)
    current, joint, depth, preserve = axes.flat
    y = np.arange(8)
    models = metrics["models_by_regime"]
    for shift, regime, color, label in zip((-.25, 0, .25), ALL_REGIMES, COLORS, LABELS):
        for ax, values in ((current, [after_count(models[regime][m], ROLES[0]) for m in MODELS]),
                           (joint, [models[regime][m]["joint_six_after"] for m in MODELS])):
            ax.barh(y + shift, values, height=.24, color=color, label=label)
            for row, value in zip(y + shift, values):
                ax.text(value + .6, row, str(value), va="center", fontsize=8)
    for ax, limit, title, xlabel in (
        (current, 96, "Current COPY target after updates", "Correct answers /96 related bindings"),
        (joint, 48, "Current target, history and unrelated fact in both orders", "All six answers correct /48 related cases")):
        ax.set_yticks(y, DISPLAY)
        ax.invert_yaxis()
        ax.set_xlim(0, limit * 1.10)
        ax.set_xticks(np.linspace(0, limit, 7))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
    current.legend(loc="upper center", bbox_to_anchor=(.5, -.19), fontsize=9)

    positions = np.arange(6)
    q_models = ("local_question", "gdn_question_only")
    for shift, regime, color in zip((-.25, 0, .25), ALL_REGIMES, COLORS):
        values = [sum(c["correct"] for c in models[regime][m]["cells"] if c["depth"] == d and
                      c["role"] == ROLES[0] and c["variant"] in ("control", "treatment"))
                  for m in q_models for d in (1, 2, 3)]
        depth.bar(positions + shift, values, width=.24, color=color)
        for x, value in zip(positions + shift, values):
            depth.text(x, value + .25, str(value), ha="center", fontsize=8)
    depth.set_xticks(positions, [f"Local Q\nD{d}" for d in (1, 2, 3)] + [f"GDN Q\nD{d}" for d in (1, 2, 3)], fontsize=9)
    depth.set_ylim(0, 35)
    depth.set_yticks(range(0, 33, 8))
    depth.set_ylabel("Correct current targets /32 related bindings")
    depth.set_title("Registered question-only models: every dependency depth", fontsize=11)

    counts = np.array([[after_count(models[r][m], role) for role in (ROLES[2], ROLES[1])
                        for r in ALL_REGIMES] for m in MODELS])
    preserve.imshow(counts / 96, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    preserve.set_yticks(y, DISPLAY)
    preserve.set_xticks(range(6), ["Full\nhistory", "Drop evidence\nhistory", "Drop other\nhistory",
                                  "Full\nunrelated", "Drop evidence\nunrelated", "Drop other\nunrelated"], fontsize=9)
    preserve.set_title("Preservation after updates: all bindings retained", fontsize=11)
    for i in range(8):
        for j in range(6):
            preserve.text(j, i, f"{counts[i, j]}/96", ha="center", va="center", fontsize=8,
                          color="white" if counts[i, j] >= 58 else "#111827")
    status = "passed" if metrics["whole_record_necessity"]["passed"] else "not passed"
    fig.suptitle("Whole-record evidence-necessity diagnostic\n"
                 f"Direct SET-leaf necessity screen: {status}", fontsize=16, y=.985)
    fig.text(.5, .014, "Same 8 reused base streams, 48 related cases, 8 fixed models; no new training or error filtering.\n"
             "Each intervention blocks exactly one whole17-token SET record; all other original history and prefix computation remain active.\n"
             "5,376 new readouts + 2,688 sealed full references. Sufficiency and prior failed architecture/history/repeat decisions remain unchanged.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .085, 1, .93), h_pad=4.5, w_pad=3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if current.get_legend().get_window_extent(renderer).overlaps(current.xaxis.label.get_window_extent(renderer)):
        raise ValueError("Figure legend overlaps current-target axis label")
    width, height = fig.canvas.get_width_height()
    outside = []
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.width > 0 and box.height > 0 and (box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1):
            outside.append({"text": text.get_text(), "bbox": list(box.bounds)})
    if outside:
        raise ValueError(f"Figure text outside image: {outside}")
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    write_json(path.with_suffix(".layout.json"), {"status": "all_visible_text_within_figure", "width": width, "height": height,
               "outside_text": outside, "source_sha256": streaming_hash(Path(__file__)),
               "png_sha256": streaming_hash(path), "pdf_sha256": streaming_hash(path.with_suffix(".pdf"))})
    plt.close(fig)


def export(audit, output, plot):
    report, manifest, registry, rows, metrics = complete_inputs(audit)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot.parent.mkdir(parents=True, exist_ok=True)
    source_sha = streaming_hash(Path(__file__))
    result = {"status": "complete_whole_record_copy_necessity_export", "source_audit": str(audit),
        "source_audit_sha256": streaming_hash(audit), "source_manifest_sha256": report["manifest_sha256"],
        "dataset_manifest_sha256": report["dataset_manifest_sha256"], "mask_registry_sha256": report["mask_registry_sha256"],
        "analysis_source_sha256": source_sha, "weights_sha256": report["weights_sha256"], "references": report["references"],
        "counts": {"new_physical": 5376, "reused_full_physical": 2688, "physical_export_rows": 8064, "related_binding_export_rows": 10368},
        "metrics": metrics, "interpretation": report["interpretation"]}
    write_json(output, result)
    output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    fields = ["model", "regime", "sealed_reference_reused", "id", "answer", "generated", "greedy_exact", "answer_nll", "query_tokens"]
    def record(model, regime, row):
        return {"model": model, "regime": regime, "sealed_reference_reused": regime == "full",
                **{k: row[k] for k in fields[3:]}}
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in ALL_REGIMES:
            for m in MODELS:
                writer.writerows(record(m, r, row) for row in rows[r][m])
    _, bindings = selected_inputs(registry)
    binding_path = output.with_name(output.stem + "-bindings.csv")
    with binding_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys([*bindings[0], *fields])))
        writer.writeheader()
        for r in ALL_REGIMES:
            for m in MODELS:
                indexed = {row["id"]: row for row in rows[r][m]}
                writer.writerows({**b, **record(m, r, indexed[b["physical_id"]])} for b in bindings)
    pair_path = output.with_name(output.stem + "-pairs.csv")
    pairs = [{"model": m, "candidate_regime": r, "reference_regime": "full", **p}
             for m in MODELS for r in REGIMES
             for p in metrics["current_comparisons_vs_full"][m][r]["individual_pairs"]]
    pairs += [{"model": m, "candidate_regime": "remove_matched_other", "reference_regime": "remove_support", **p}
              for m in MODELS for p in metrics["support_removal_effects"][m]["individual_pairs"]]
    with pair_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    make_plot(metrics, plot)
    artifacts = (output, output.with_suffix(".source.py"), output.with_suffix(".csv"), binding_path, pair_path,
                 plot, plot.with_suffix(".pdf"), plot.with_suffix(".layout.json"))
    write_json(output.with_suffix(".provenance.json"), {"status": "complete_export_source_and_artifact_closure",
        "source_audit_sha256": streaming_hash(audit), "source_manifest_sha256": report["manifest_sha256"],
        "exporter_sha256": source_sha, "artifacts_sha256": {str(p): streaming_hash(p) for p in artifacts},
        "run_sources_sha256": manifest["source_sha256"], "physical_rows": 8064, "related_bindings": 10368,
        "current_target_pairs": len(pairs), "visual_inspection": "pending separate inspection"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.audit, args.output, args.plot)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      **{k: result["metrics"][k] for k in ("whole_record_necessity",)}}), flush=True)


if __name__ == "__main__":
    main()
