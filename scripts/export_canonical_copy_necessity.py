"""Export all8064new canonical observations and separate legacy-profile differences."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_canonical_copy_necessity import audit_run
from scripts.canonical_necessity_common import MASKED_REGIMES, REGIMES, canonical_metrics
from scripts.copy_necessity_data import MODELS, ROLES, read_plan, registered_inputs, selected_inputs
from scripts.copy_necessity_metrics import after_count
from scripts.copy_readout_runtime import readout
from scripts.run_controller_trial import write_json

DISPLAY = ["Native", "Shared", "Independent", "Local all", "Local question", "Attention only", "GDN all", "GDN question"]
COLORS = ("#64748b", "#e11d48", "#0284c7")
LABELS = ("Fresh full history", "Remove whole evidence SET", "Remove nearby irrelevant SET")


def complete_inputs(path):
    report = json.loads(path.read_text())
    run = Path(report["source_run"])
    if (report["status"] != "completed_and_audited_canonical" or report["new_physical_readouts_audited"] != 8064 or
        report["fresh_full_reference_readouts_audited"] != 2688 or report["new_related_bindings_audited"] != 10368 or
        streaming_hash(path.with_suffix(".source.py")) != report["analysis_source_sha256"] or audit_run(run) != report):
        raise ValueError("Complete independent8064-readout same-profile audit required")
    manifest = json.loads((run / "manifest.json").read_text())
    _, source_plan, cases, registry, masks, legacy = registered_inputs(Path(manifest["dataset"]))
    rows = {r: {m: [json.loads(line) for line in (run / f"{m}.{r}.jsonl").read_text().splitlines()]
                for m in MODELS} for r in REGIMES}
    masked = {m: {r: rows[r][m] for r in MASKED_REGIMES} for m in MODELS}
    metrics = canonical_metrics(masked, rows["full"], cases, registry, source_plan, read_plan(), report["arithmetic_profile"])
    if metrics != report["metrics"]:
        raise ValueError("Fresh-reference three-regime metrics differ from the independent audit")
    return report, manifest, registry, rows, legacy, metrics


def legacy_comparisons(rows, legacy):
    """Include all2688pairs, including zero differences; never include in primary."""
    result = []
    for model in MODELS:
        if [r["id"] for r in rows["full"][model]] != [r["id"] for r in legacy[model]]:
            raise ValueError("All old-versus-new full-history query pairs required")
        for old, new in zip(legacy[model], rows["full"][model]):
            a, b = readout(old), readout(new)
            if a["answer"] != b["answer"] or a["query_tokens"] != b["query_tokens"]:
                raise ValueError("Legacy-versus-canonical query truth or boundary changed")
            changed = [key for key in a if a[key] != b[key]]
            result.append({"model": model, "id": new["id"], "descriptive_only": True,
                "changed": bool(changed), "changed_fields": ",".join(changed),
                "gold_answer": b["answer"], "query_tokens": b["query_tokens"],
                "legacy_generated": a["generated"], "canonical_generated": b["generated"],
                "legacy_greedy_exact": a["greedy_exact"], "canonical_greedy_exact": b["greedy_exact"],
                "legacy_answer_nll": a["answer_nll"], "canonical_answer_nll": b["answer_nll"],
                "canonical_minus_legacy_nll": b["answer_nll"] - a["answer_nll"]})
    if len(result) != 2688:
        raise ValueError("Legacy descriptive comparison is incomplete")
    return result


def make_plot(metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    import numpy as np
    models = metrics["models_by_regime"]
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), dpi=160)
    current, joint, effect, preserve = axes.flat
    y = np.arange(8)
    for shift, regime, color, label in zip((-.25, 0, .25), REGIMES, COLORS, LABELS):
        for ax, values in ((current, [after_count(models[regime][m], ROLES[0]) for m in MODELS]),
                           (joint, [models[regime][m]["joint_six_after"] for m in MODELS])):
            ax.barh(y + shift, values, height=.24, color=color, label=label)
            for row, value in zip(y + shift, values):
                ax.text(value + .6, row, str(value), va="center", fontsize=8)
    for ax, limit, title, xlabel in (
        (current, 96, "Current COPY target after updates", "Correct answers /96 related bindings"),
        (joint, 48, "Current target, history and unrelated fact in both orders", "All six answers correct /48 related cases")):
        ax.set_yticks(y, DISPLAY); ax.invert_yaxis(); ax.set_xlim(0, limit * 1.10)
        ax.set_xticks(np.linspace(0, limit, 7)); ax.set_title(title, fontsize=11); ax.set_xlabel(xlabel)
    current.legend(loc="upper center", bbox_to_anchor=(.5, -.19), fontsize=9)
    groups = ("local_question", "gdn_question_only")
    contrasts = metrics["support_removal_effects"]
    xpos = np.arange(2)
    for shift, key, color, label in ((-.22, "gains", "#e11d48", "Support-removal harm"),
                                      (0, "losses", "#0284c7", "Support-removal rescue"),
                                      (.22, "signed_net_gain", "#64748b", "Net harm")):
        values = [contrasts[m][key] for m in groups]
        effect.bar(xpos + shift, values, width=.21, color=color, label=label)
        for x, value in zip(xpos + shift, values):
            effect.annotate(str(value), (x, value), xytext=(0, 4 if value >= 0 else -12), textcoords="offset points", ha="center", fontsize=9)
    effect.set_xticks(xpos, ["Local question", "GDN question"])
    effect.axhline(0, color="#334155", linewidth=.8)
    effect.set_ylim(-100, 105)
    effect.set_ylabel("Paired current-answer differences /96 bindings")
    effect.set_title("Remove support versus matched-other removal", fontsize=11)
    effect.legend(loc="lower center", fontsize=8)
    counts = np.array([[after_count(models[r][m], role) for role in (ROLES[2], ROLES[1]) for r in REGIMES] for m in MODELS])
    preserve.imshow(counts / 96, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    preserve.set_yticks(y, DISPLAY)
    preserve.set_xticks(range(6), ["Full\nhistory", "Drop evidence\nhistory", "Drop other\nhistory",
                                  "Full\nunrelated", "Drop evidence\nunrelated", "Drop other\nunrelated"], fontsize=9)
    preserve.set_title("Preservation after updates: every binding retained", fontsize=11)
    for i in range(8):
        for j in range(6):
            preserve.text(j, i, f"{counts[i,j]}/96", ha="center", va="center", fontsize=8,
                          color="white" if counts[i,j] >= 58 else "#111827")
    status = "passed" if metrics["whole_record_necessity"]["passed"] else "not passed"
    fig.suptitle(f"Whole-record evidence-necessity diagnostic\nRegistered necessity screen: {status}", fontsize=16, y=.985)
    fig.text(.5, .014, "8,064 NEW readouts: 8 fixed models x336queries x3regimes; 48 related cases from8reused base streams.\n"
             "Each mask removes access to one whole17-token SET at all6softmax layers; original prefix computation remains active.\n"
             "Frozen adapter affines: FP64 accumulation, round toFP32; all full references fresh. Prior profile decisions remain recorded separately.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .085, 1, .93), h_pad=4.5, w_pad=3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if current.get_legend().get_window_extent(renderer).overlaps(current.xaxis.label.get_window_extent(renderer)):
        raise ValueError("Figure legend overlaps current-target axis label")
    width, height = fig.canvas.get_width_height()
    outside = []
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text().strip(): continue
        box = item.get_window_extent(renderer)
        if box.width > 0 and box.height > 0 and (box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1):
            outside.append({"text": item.get_text(), "bbox": list(box.bounds)})
    if outside: raise ValueError(f"Figure text outside image: {outside}")
    fig.savefig(path, dpi=160); fig.savefig(path.with_suffix(".pdf"))
    write_json(path.with_suffix(".layout.json"), {"status": "all_visible_text_within_figure",
        "width": width, "height": height, "outside_text": outside, "source_sha256": streaming_hash(Path(__file__)),
        "png_sha256": streaming_hash(path), "pdf_sha256": streaming_hash(path.with_suffix(".pdf"))})
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def export(audit_path, output, plot):
    report, manifest, registry, rows, legacy, metrics = complete_inputs(audit_path)
    output.parent.mkdir(parents=True, exist_ok=True); plot.parent.mkdir(parents=True, exist_ok=True)
    old_comparison = legacy_comparisons(rows, legacy)
    summary = {m: {"pairs": sum(p["model"] == m for p in old_comparison),
                   "any_changed": sum(p["model"] == m and p["changed"] for p in old_comparison),
                   "greedy_changed": sum(p["model"] == m and p["legacy_greedy_exact"] != p["canonical_greedy_exact"] for p in old_comparison),
                   "text_changed": sum(p["model"] == m and p["legacy_generated"] != p["canonical_generated"] for p in old_comparison)} for m in MODELS}
    result = {"status": "complete_canonical_whole_record_necessity_export", "source_audit": str(audit_path),
        "source_audit_sha256": streaming_hash(audit_path), "source_manifest_sha256": report["manifest_sha256"],
        "dataset_manifest_sha256": report["dataset_manifest_sha256"], "mask_registry_sha256": report["mask_registry_sha256"],
        "precision_addendum_sha256": report["precision_addendum_sha256"], "arithmetic_profile": report["arithmetic_profile"],
        "analysis_source_sha256": streaming_hash(Path(__file__)), "weights_sha256": report["weights_sha256"], "references": report["references"],
        "counts": {"new_physical": 8064, "fresh_full_physical": 2688, "new_masked_physical": 5376,
                   "reused_full_physical_in_primary": 0, "related_binding_export_rows": 10368, "legacy_descriptive_pairs": 2688},
        "metrics": metrics, "legacy_full_comparison_descriptive_only": summary, "interpretation": report["interpretation"]}
    write_json(output, result)
    output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    physical_rows = [{"model": m, "regime": r, "arithmetic_profile": report["arithmetic_profile"],
                      "fresh_observation": True, "id": row["id"], **readout(row)} for r in REGIMES for m in MODELS for row in rows[r][m]]
    write_csv(output.with_suffix(".csv"), physical_rows)
    _, bindings = selected_inputs(registry)
    binding_rows = []
    for r in REGIMES:
        for m in MODELS:
            indexed = {row["id"]: row for row in physical_rows if row["model"] == m and row["regime"] == r}
            binding_rows.extend({**b, **indexed[b["physical_id"]]} for b in bindings)
    binding_path = output.with_name(output.stem + "-bindings.csv")
    write_csv(binding_path, binding_rows)
    pairs = [{"model": m, "candidate_regime": r, "reference_regime": "full", **p}
             for m in MODELS for r in MASKED_REGIMES for p in metrics["current_comparisons_vs_full"][m][r]["individual_pairs"]]
    pairs += [{"model": m, "candidate_regime": "remove_matched_other", "reference_regime": "remove_support", **p}
              for m in MODELS for p in metrics["support_removal_effects"][m]["individual_pairs"]]
    pair_path = output.with_name(output.stem + "-pairs.csv"); write_csv(pair_path, pairs)
    legacy_path = output.with_name(output.stem + "-legacy-full.csv"); write_csv(legacy_path, old_comparison)
    if len(physical_rows) != 8064 or len(binding_rows) != 10368 or len(pairs) != 2304:
        raise ValueError("Complete canonical export row/binding/paired coverage changed")
    make_plot(metrics, plot)
    artifacts = (output, output.with_suffix(".source.py"), output.with_suffix(".csv"), binding_path, pair_path, legacy_path,
                 plot, plot.with_suffix(".pdf"), plot.with_suffix(".layout.json"))
    write_json(output.with_suffix(".provenance.json"), {"status": "complete_export_source_and_artifact_closure",
        "source_audit_sha256": streaming_hash(audit_path), "source_manifest_sha256": report["manifest_sha256"],
        "exporter_sha256": streaming_hash(Path(__file__)), "artifacts_sha256": {str(p): streaming_hash(p) for p in artifacts},
        "run_sources_sha256": manifest["source_sha256"], "physical_rows": len(physical_rows), "related_bindings": len(binding_rows),
        "current_target_pairs": len(pairs), "legacy_descriptive_pairs": len(old_comparison), "cross_profile_rows_in_primary": 0,
        "visual_inspection": "pending separate inspection"})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.audit, args.output, args.plot)
    print(json.dumps({"status": result["status"], "counts": result["counts"], "gate": result["metrics"]["whole_record_necessity"]}), flush=True)
