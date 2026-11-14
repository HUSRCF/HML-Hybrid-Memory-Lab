"""Audit frozen blocked-reader generalization without changing the test cohort."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.counterbalance_data import file_digest
from sas_memory.counterbalance_tasks import ORDERS, validate_counterbalance_cases
from sas_memory.record_kv_ablation import build_record_registry
from sas_memory.training_data import data_fingerprint
from scripts.analyze_blocked_reader_adaptation import audit_run, tokenizer_runtime
from scripts.analyze_controller_confirmation import scalar_metrics
from scripts.analyze_controller_trial import load_rows
from scripts.analyze_readout_counterbalance import interval, order_consistency
from scripts.run_controller_trial import write_json


def plot_report(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(report["models"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, key, label in zip(axes, ("accuracy", "answer_nll"), ("Current-target greedy accuracy", "Gold answer NLL (nats/token)")):
        values = [report["models"][name]["overall"][key] for name in names]
        bars = ax.bar(names, values, color=["#2681b5", "#d5634f"])
        ax.set_xticks(range(2), ["Question-only", "All-token"])
        ax.set_ylabel(label)
        if key == "accuracy":
            ax.set_ylim(0, 1.05)
            ax.bar_label(bars, [f"{report['models'][name]['overall']['correct']}/288" for name in names], padding=3)
        else:
            ax.bar_label(bars, [f"{value:.3f}" for value in values], padding=3)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Matched local adaptation with question-time record KV blocked")
    fig.text(.5, .02, "One initialization; 24 reused base facts. Prefix hybrid processing retained; final checkpoints only.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    plan = json.loads((args.run / "plan.json").read_text())
    training_plan = json.loads((args.run / "training_plan.json").read_text())
    cases = json.loads((args.run / "cases.json").read_text())
    registry = json.loads((args.run / "registry.json").read_text())
    if (manifest["status"] != "completed" or manifest["training"] or
        manifest["completed_models"] != list(plan["checkpoints"]) or
        manifest["plan_sha256"] != data_fingerprint(plan)):
        raise ValueError("Incomplete frozen evaluation or changed plan")
    for key, value in (("cases", cases), ("registry", registry)):
        if data_fingerprint(value) != plan[key + "_sha256"] or plan[key + "_sha256"] != manifest[key + "_sha256"]:
            raise ValueError("Cohort or masks changed")
    if validate_counterbalance_cases(cases)["contexts"] != plan["contexts_per_model"]:
        raise ValueError("Wrong context count")
    for relative, fingerprint in manifest["source_sha256"].items():
        if file_digest(args.run / "sources" / relative) != fingerprint:
            raise ValueError(f"Source snapshot changed: {relative}")
    for relative in ("sas_memory/runtime.py", "sas_memory/controllers.py", "sas_memory/kv_readout.py", "sas_memory/record_kv_ablation.py"):
        if file_digest(ROOT / relative) != manifest["source_sha256"][relative]:
            raise ValueError("Actual core implementation differs from evaluation")
    runtime = tokenizer_runtime(manifest["model_path"])
    if build_record_registry(runtime, cases) != registry:
        raise ValueError("Mask registry does not reconstruct from original inputs")
    indexed = {(row["case_id"], row["variant"]): row for row in registry}
    lookup = {case["id"]: case for case in cases}
    expected = {(case["id"], order, "target_current"): case["variants"][order]["queries"][0]["answer"] for case in cases for order in ORDERS}
    all_rows, models = {}, {}
    for label, path in plan["checkpoints"].items():
        training, _ = audit_run(Path(path), training_plan)
        if training != manifest["checkpoint_sources"][label] or not manifest["native_replay_after_detach"][label]:
            raise ValueError("Checkpoint provenance changed or native restoration failed")
        checks = manifest["safety_checks"][label]
        if [row["case_id"] for row in checks] != [cases[0]["id"], cases[-1]["id"]]:
            raise ValueError("Missing short/long intervention checks")
        for check in checks:
            for key in ("all_history_full_cache_and_logits_exact", "all_history_scoring_and_greedy_exact",
                        "pre_question_kv_recurrent_conv_and_logits_exact", "historical_kv_exact_in_full_prefill"):
                if check[key] is not True:
                    raise ValueError("Prefix/all-history/historical-KV identity check failed")
            if check["historical_kv_fields_checked"] != 12 or check["unrestricted_replay_max_nll_error"] != 0:
                raise ValueError("Replay or KV-field coverage failed")
            if set(check["blocked_probabilities"]) != {str(i) for i in (3, 7, 11, 15, 19, 23)}:
                raise ValueError("Missing inspected attention layers")
            for layer, stat in check["blocked_probabilities"].items():
                if (stat["max_record_probability"] != 0 or stat["max_record_mass"] != 0 or
                    stat["prefill_calls"] != 1 or stat["cached_calls"] < 1 or
                    check["unmasked_probabilities"][layer]["max_record_mass"] <= 0):
                    raise ValueError("Attention probability safety check failed")
        if training["token_policy"] == "question":
            prefix_checks = manifest["native_prefix_checks"][label]
            if ([row["case_id"] for row in prefix_checks] != [cases[0]["id"], cases[-1]["id"]] or
                not all(row["kv_recurrent_conv_and_logits_exact"] for row in prefix_checks)):
                raise ValueError("Question-only final checkpoint prefix identity not verified")
        rows = load_rows(args.run / f"{label}.jsonl", expected)
        for row in rows:
            case = lookup[row["case_id"]]
            mask = {key: value for key, value in indexed[(row["case_id"], row["variant"])].items() if key not in ("case_id", "variant")}
            if (row["reader_mask"] != mask or row["readout_calls"]["prefill"] != 1 or row["readout_calls"]["cached"] < 1 or
                any(row[key] != case[key] for key in ("base_case_id", "dependency_depth", "question_style"))):
                raise ValueError("Mask, cached calls, or condition metadata invalid")
        all_rows[label] = rows
        models[label] = {"overall": scalar_metrics(rows), "training": training,
                         "by_condition": [{"depth": depth, "style": style, **scalar_metrics([
                             row for row in rows if row["dependency_depth"] == depth and row["question_style"] == style])}
                             for depth in (1, 2, 3) for style in ("original", "entity_first")],
                         "late_order_consistency": order_consistency(rows),
                         "correct_by_base_case": dict(Counter(row["base_case_id"] for row in rows if row["greedy_exact"])),
                         "prediction_counts": dict(Counter(row["generated"].strip() for row in rows)),
                         "integer_format_outputs": sum(bool(re.fullmatch(r"-?\d+", row["generated"].strip())) for row in rows)}
    clusters = {case["id"]: case["base_case_id"] for case in cases}
    report = {"status": "completed_and_audited", "source_run": str(args.run), "manifest_sha256": file_digest(args.run / "manifest.json"),
              "analysis_source_sha256": file_digest(Path(__file__)), "models": models,
              "records_audited": sum(len(rows) for rows in all_rows.values()),
              "hindsight_best_constant_answer_correct": max(Counter(expected.values()).values()),
              "constant_reference_note": "Descriptive ceiling for a single constant answer chosen with hindsight on these labels; not a trained or deployable baseline, and not a random-chance calibration.",
              "all_minus_question_accuracy": interval(all_rows["all_tokens"], all_rows["question_only"], clusters),
              "interpretation": plan["interpretation"] + " One fixed initialization; reused diagnostics, not untouched confirmation. Degenerate zero-effect intervals do not prove equivalence. Missing task success can reflect optimization/adapter expressivity and the intervention's distribution shift; it does not establish absent information."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_report(report, args.plot)
    print(json.dumps({label: model["overall"] for label, model in models.items()}))


if __name__ == "__main__":
    main()
