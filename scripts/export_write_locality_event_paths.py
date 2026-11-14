"""Export all audited native path effects and legitimate-update tradeoffs."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_write_locality_event_paths import paired_outcomes, row_key, summarize
from scripts.run_controller_trial import write_json


def load_complete(audit_path):
    report = json.loads(audit_path.read_text())
    run = Path(report["source_run"])
    manifest = json.loads((run / "manifest.json").read_text())
    data = Path(manifest["dataset"])
    registered = json.loads((data / "manifest.json").read_text())
    registry = json.loads((data / "registry.json").read_text())
    if (report["status"] != "completed_and_audited" or report["pairs_audited"] != 96 or
        report["readouts_audited"] != 1200 or report["calibrations_audited"] != 448 or manifest["smoke"] or
        manifest["status"] != "completed" or streaming_hash(run / "manifest.json") != report["manifest_sha256"] or
        streaming_hash(data / "manifest.json") != report["dataset_manifest_sha256"] or
        data_fingerprint(registry) != registered["registry_sha256"] or registered["registry_sha256"] != report["registry_sha256"] or
        streaming_hash(audit_path.with_suffix(".source.py")) != report["analysis_source_sha256"]):
        raise ValueError("Complete independently audited source cohort required")
    for relative, digest in manifest["source_sha256"].items():
        if streaming_hash(ROOT / relative) != digest or streaming_hash(run / "sources" / relative) != digest:
            raise ValueError("Audited implementation changed")
    for relative, digest in manifest["results_sha256"].items():
        if streaming_hash(run / relative) != digest:
            raise ValueError("Audited observations changed")
    rows = [json.loads(line) for line in (run / "readouts.jsonl").read_text().splitlines()]
    calibrations = [json.loads(line) for line in (run / "calibrations.jsonl").read_text().splitlines()]
    indexed = {row_key(row): row for row in rows}
    metrics = summarize(registry["pairs"], indexed, calibrations)
    if metrics != report["metrics"]:
        raise ValueError("Recomputed complete path and calibration statistics differ from independent audit")
    return report, manifest, registry, rows, metrics


def plots(metrics, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    titles = {"unrelated_B_at32": "Protected B, 32 accounts",
              "initial_A_at128": "Initial A, 128 accounts",
              "unrelated_C_on_initial_A_at128": "Initial A after C update, 128 accounts"}
    groups = metrics["by_cohort_and_stratum"]
    cohorts = ("discovery", "replica")
    strata = list(titles)
    indexed = {(r["cohort"], r["stratum"]): r for r in groups}
    labels = ["L / L", "R / R", "L / R", "R / L", "Oracle L", "Oracle R"]
    colors = ["#94a3b8", "#334155", "#0284c7", "#d97706", "#a7d8be", "#15986e"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for i, cohort in enumerate(cohorts):
        for j, stratum in enumerate(strata):
            group = indexed[(cohort, stratum)]
            counts = [r["versus_natural_right"]["preservation"]["candidate_correct"] for r in group["conditions"]]
            counts += [r["candidate_correct"] for r in group["oracle_vs_cached_natural"]]
            ax = axes[i, j]
            ax.bar(range(6), np.array(counts) * 100 / group["pairs"], color=colors)
            ax.set_xticks(range(6), labels, rotation=35, ha="right")
            ax.set_ylim(0, 110)
            ax.set_title(f"{cohort.capitalize()}: {titles[stratum]}", fontsize=11)
            for x, count in enumerate(counts):
                ax.text(x, count * 100 / group["pairs"] + 1.5, str(count), ha="center", fontsize=10)
            if j == 0:
                ax.set_ylabel("Preservation exact accuracy (%)")
    fig.suptitle("Aligned recurrent / KV source interventions and oracle evidence access", fontsize=16)
    fig.text(.5, .025, "L/R = matched no-op/update worlds at identical ledger positions. State includes recurrent and convolution caches; KV includes all six Attention layers.\nCounts /16 related value readouts, four base streams per cohort. Mixed caches are diagnostic interventions, not repair methods.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, .94))
    fig.savefig(destination, dpi=160)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for i, cohort in enumerate(cohorts):
        for j, stratum in enumerate(strata):
            group = indexed[(cohort, stratum)]
            preservation = [r["versus_natural_right"]["preservation"]["candidate_correct"] for r in group["conditions"]]
            joint = [r["versus_natural_right"]["all_roles_joint"]["candidate_correct"] for r in group["conditions"]]
            ax = axes[i, j]
            x = np.arange(4)
            for shift, values, color, label in ((-.18, preservation, "#0284c7", "Preservation"),
                                                 (.18, joint, "#15986e", "Preservation + current updates")):
                ax.bar(x + shift, np.array(values) * 100 / group["pairs"], width=.36, color=color, label=label)
                for place, count in zip(x + shift, values):
                    ax.text(place, count * 100 / group["pairs"] + 1.5, str(count), ha="center", fontsize=9)
            ax.set_xticks(x, labels[:4])
            ax.set_ylim(0, 110)
            ax.set_title(f"{cohort.capitalize()}: {titles[stratum]}", fontsize=11)
            if j == 0:
                ax.set_ylabel("Exact accuracy (%)")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .085), ncol=2)
    fig.suptitle("Does preservation retain the legitimate right-world update?", fontsize=16)
    fig.text(.5, .018, "Columns are recurrent-state / KV origins; L/R = matched no-op/update worlds at identical ledger positions. State includes convolution caches.\nEvery bar uses fixed right-world targets; L/L is a counterfactual current-update score. Own-world endpoint calibration is reported separately.\nJoint means preservation + current A, and also current C in the C-axis stratum. Counts /16 related readouts; no best-of-two truth.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .15, 1, .92))
    commit = destination.with_name(destination.stem + "-commit" + destination.suffix)
    fig.savefig(commit, dpi=160)
    fig.savefig(commit.with_suffix(".pdf"))
    plt.close(fig)


def detailed_transitions(registry, rows):
    indexed = {row_key(row): row for row in rows}
    groups, transitions = {}, []
    for pair in registry["pairs"]:
        groups.setdefault((pair["cohort"], pair["stratum"], pair["base_stream_id"]), []).append(pair)
        reference = indexed[(pair["id"], "cache_condition", 1, "preservation", "")]["world_scores"]["right"]
        left_natural = indexed[(pair["id"], "cache_condition", 0, "preservation", "")]["world_scores"]["left"]
        for index in range(4):
            candidate = indexed[(pair["id"], "cache_condition", index, "preservation", "")]["world_scores"]["right"]
            updates = {label: indexed[(pair["id"], "cache_condition", index, label, "")]["world_scores"]["right"]["greedy_exact"]
                       for label in pair["queries"] if label != "preservation"}
            transitions.append({"pair_id": pair["id"], "cohort": pair["cohort"], "stratum": pair["stratum"],
                                "base_stream_id": pair["base_stream_id"], "case_id": pair["case_id"], "condition": index,
                                "left_natural_correct": left_natural["greedy_exact"],
                                "cached_natural_update_regression": left_natural["greedy_exact"] and not reference["greedy_exact"],
                                "reference_correct": reference["greedy_exact"], "candidate_correct": candidate["greedy_exact"],
                                "rescued": not reference["greedy_exact"] and candidate["greedy_exact"],
                                "rescued_cached_update_regression": left_natural["greedy_exact"] and not reference["greedy_exact"] and candidate["greedy_exact"],
                                "regressed": reference["greedy_exact"] and not candidate["greedy_exact"],
                                "right_world_current_updates": updates, "all_roles_joint": candidate["greedy_exact"] and all(updates.values())})
    streams = []
    for key, pairs in groups.items():
        conditions = []
        for index in range(4):
            labels = list(pairs[0]["queries"])
            scores = {}
            for label in labels:
                before = [indexed[(p["id"], "cache_condition", 1, label, "")]["world_scores"]["right"]["greedy_exact"] for p in pairs]
                after = [indexed[(p["id"], "cache_condition", index, label, "")]["world_scores"]["right"]["greedy_exact"] for p in pairs]
                scores[label] = paired_outcomes(before, after)
            members = [r for r in transitions if r["base_stream_id"] == key[2] and r["cohort"] == key[0] and r["stratum"] == key[1] and r["condition"] == index]
            conditions.append({"condition": index, "versus_natural_right": scores,
                               "rescued_cached_update_regressions": sum(r["rescued_cached_update_regression"] for r in members),
                               "rescued_preexisting_cached_baseline_errors": sum(r["rescued"] and not r["left_natural_correct"] for r in members)})
        streams.append({**dict(zip(("cohort", "stratum", "base_stream_id"), key)), "value_cases": len(pairs),
                        "conditions": conditions})
    if len(streams) != 24 or any(s["value_cases"] != 4 for s in streams) or len(transitions) != 384:
        raise ValueError("Complete base-stream/value transition export required")
    return streams, transitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    report, manifest, registry, rows, metrics = load_complete(args.audit)
    streams, transitions = detailed_transitions(registry, rows)
    result = {"status": "completed_native_event_path_export", "source_audit": str(args.audit),
              "source_audit_sha256": streaming_hash(args.audit), "source_manifest_sha256": report["manifest_sha256"],
              "analysis_source_sha256": streaming_hash(Path(__file__)), "weights_sha256": report["weights_sha256"],
              "readouts": len(rows), "calibrations": report["calibrations_audited"], "metrics": metrics,
              "base_stream_effects": streams, "individual_preservation_transitions": transitions,
              "interpretation": "All pre-registered units, two separate cohorts. Mixed paths may be incompatible; oracle recovery is not proof against recurrent damage. No shared-interface or training advantage is claimed."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    pair_index = {p["id"]: p for p in registry["pairs"]}
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        fields = ["cohort", "stratum", "case_id", "seed", "value_version", "kind", "condition", "clone_kind", "label", "world", "answer", "generated", "greedy_exact", "answer_nll"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            pair = pair_index[row["pair_id"]]
            for side, score in row["world_scores"].items():
                writer.writerow({**{key: pair[key] for key in ("cohort", "stratum", "case_id", "seed", "value_version")},
                                 **{key: row[key] for key in ("kind", "condition", "label", "generated")},
                                 "clone_kind": row.get("clone_kind", ""), "world": side, **score})
    plots(metrics, args.plot)
    print(json.dumps({"status": result["status"], "readouts": len(rows), "cohort_strata": len(metrics["by_cohort_and_stratum"])}))


if __name__ == "__main__":
    main()
