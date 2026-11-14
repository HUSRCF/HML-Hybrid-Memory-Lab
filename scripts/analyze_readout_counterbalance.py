"""Audit fresh counterbalanced readout controls and estimate paired effects."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.counterbalance_data import build_counterbalance_registry, file_digest, locked_training_sources, tokenizer_runtime
from sas_memory.counterbalance_tasks import build_counterbalance_cases, validate_counterbalance_cases, MASKS, ORDERS
from sas_memory.training_data import data_fingerprint
from scripts.analyze_controller_trial import load_rows
from scripts.analyze_controller_confirmation import scalar_metrics, paired_interval, unary_cluster_scores


def interval(left, right, lookup, unit="base_case_id, averaging depth/wording/late-order variants"):
    result = paired_interval(unary_cluster_scores(left, lookup), unary_cluster_scores(right, lookup))
    result["unit"] = unit
    return result


def paired_changes(left, right):
    baseline = {(row["case_id"], row["variant"]): row for row in right}
    return {"right_correct": sum(row["greedy_exact"] for row in right),
            "regressions": sum(baseline[(row["case_id"], row["variant"])]["greedy_exact"] and not row["greedy_exact"] for row in left),
            "recoveries": sum(not baseline[(row["case_id"], row["variant"])]["greedy_exact"] and row["greedy_exact"] for row in left)}


def order_consistency(members):
    pairs = defaultdict(dict)
    for row in members:
        pairs[row["case_id"]][row["variant"]] = row
    if any(set(pair) != set(ORDERS) for pair in pairs.values()):
        raise ValueError("Missing late-order consistency pair")
    return {"pairs": len(pairs),
            "both_correct": sum(all(row["greedy_exact"] for row in pair.values()) for pair in pairs.values()),
            "both_incorrect": sum(not any(row["greedy_exact"] for row in pair.values()) for pair in pairs.values()),
            "correctness_disagreements": sum(pair[ORDERS[0]]["greedy_exact"] != pair[ORDERS[1]]["greedy_exact"] for pair in pairs.values()),
            "generated_string_disagreements": sum(pair[ORDERS[0]]["generated"].strip() != pair[ORDERS[1]]["generated"].strip() for pair in pairs.values()),
            "mean_absolute_answer_nll_difference": mean(abs(pair[ORDERS[0]]["answer_nll"] - pair[ORDERS[1]]["answer_nll"]) for pair in pairs.values())}


def summarize(rows, cases):
    lookup = {case["id"]: case for case in cases}
    clusters = {case["id"]: case["base_case_id"] for case in cases}
    source, unrelated = rows["extra_source"], rows["extra_unrelated"]
    conditions = sorted({(case["dependency_depth"], case["question_style"]) for case in cases})
    error_types = {}
    for regime, members in rows.items():
        labels = []
        for row in members:
            case = lookup[row["case_id"]]
            initial = {event["target"]: event["value"] for event in case["prefix"][:8]}
            prediction = row["generated"].strip()
            labels.append("correct" if row["greedy_exact"] else "late_written_value" if prediction == str(case["new_value"])
                          else "target_initial" if prediction == str(initial[case["target_entity"]])
                          else "unrelated_initial" if prediction == str(initial[case["unrelated_entity"]]) else "other")
        error_types[regime] = dict(Counter(labels))
    return {"regimes": {name: scalar_metrics(members) for name, members in rows.items()},
            "source_minus_unrelated": interval(source, unrelated, clusters),
            "source_vs_unrelated_paired": paired_changes(source, unrelated),
            "extra_vs_oracle_paired": {name: paired_changes(rows[name], rows["oracle"]) for name in ("extra_source", "extra_unrelated")},
            "error_types": error_types,
            "independent_late_order_consistency": {name: order_consistency(members) for name, members in rows.items()},
            "by_condition": [{"dependency_depth": depth, "question_style": style,
                              "regimes": {name: scalar_metrics([row for row in members if lookup[row["case_id"]]["dependency_depth"] == depth and lookup[row["case_id"]]["question_style"] == style]) for name, members in rows.items()}}
                             for depth, style in conditions],
            "by_late_order": {order: {"regimes": {name: scalar_metrics([row for row in members if row["variant"] == order]) for name, members in rows.items()},
                                      "source_minus_unrelated": interval([row for row in source if row["variant"] == order], [row for row in unrelated if row["variant"] == order], clusters)}
                              for order in ORDERS},
            "by_extra_position": {position: interval([row for row in source if row["variant"] == source_order],
                                                      [row for row in unrelated if row["variant"] == unrelated_order], clusters,
                                                      "base_case_id; same extra-token interval, swapping full-input late SET order")
                                  for position, source_order, unrelated_order in (("first_late_SET", "source_first", "unrelated_first"),
                                                                                ("second_late_SET", "unrelated_first", "source_first"))}}


def plot_report(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    modes = list(report["models"])
    labels = ["Native" if name == "native" else "Q-only " + name.split("seed")[-1] for name in modes]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    regimes = [("raw_full", "Full historical access", "#7b8794"), ("oracle", "Necessary evidence", "#2681b5"),
               ("extra_source", "+ source SET", "#d5634f"), ("extra_unrelated", "+ unrelated SET", "#57a584")]
    positions = np.arange(len(modes))
    for offset, (regime, label, color) in enumerate(regimes):
        values = [report["models"][mode]["regimes"][regime] for mode in modes]
        bars = axes[0].bar(positions + (offset - 1.5) * .2, [v["accuracy"] for v in values], .19, label=label, color=color)
        axes[0].bar_label(bars, [str(v["correct"]) for v in values], fontsize=8, padding=2)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(0, 1.07)
    axes[0].set_ylabel("Current-target accuracy (bar labels: correct / 288)")
    axes[0].grid(axis="y", alpha=.2)
    axes[0].set_axisbelow(True)
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, fontsize=8, loc="lower center", bbox_to_anchor=(.29, .065), ncol=2, frameon=False)
    estimates = [report["models"][mode]["source_minus_unrelated"] for mode in modes]
    values = np.array([item["difference"] * 100 for item in estimates])
    low = np.array([item["ci95"][0] * 100 for item in estimates])
    high = np.array([item["ci95"][1] * 100 for item in estimates])
    axes[1].errorbar(values, positions, xerr=[values - low, high - values], fmt="o", capsize=4, color="#7b3f8c")
    axes[1].axvline(0, color="#777", lw=1)
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Source SET minus unrelated SET (percentage points)")
    axes[1].set_title("Paired base-fact bootstrap, 95% intervals")
    axes[1].grid(axis="x", alpha=.2)
    fig.suptitle("Equal-valued late updates with swapped positions — frozen Qwen3.5-2B IT")
    fig.text(.5, .025, "24 base facts; depth 1/2/3 × 2 questions × 2 late orders. Three fixed adapter initializations; oracle masks, dense eager kernels.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .17, 1, .95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--allow-smoke", action="store_true", help="Only audit operational checks, without effect estimates")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    data = json.loads((args.run / "data_manifest.json").read_text())
    plan = json.loads((args.run / "plan.json").read_text())
    all_cases = json.loads((args.run / "cases.json").read_text())
    registry = json.loads((args.run / "registry.json").read_text())
    if manifest["status"] != "completed" or manifest["completed_modes"] != plan["checkpoints"] or manifest["training"]:
        raise ValueError("Frozen evaluation is incomplete")
    if manifest["smoke"] and not args.allow_smoke:
        raise ValueError("Smoke evaluation cannot provide cohort evidence")
    if file_digest(args.run / "data_manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Preparation manifest changed")
    if datetime.fromisoformat(data["registered_utc"]) >= datetime.fromisoformat(manifest["started_utc"]):
        raise ValueError("Data/masks were not registered before inference")
    for key, value in (("plan", plan), ("cases", all_cases), ("registry", registry)):
        if data_fingerprint(value) != data[key + "_sha256"]:
            raise ValueError(f"Registered {key} changed")
    validation = validate_counterbalance_cases(all_cases)
    if validation != data["task_validation"]:
        raise ValueError("Task validation changed")
    regenerated = build_counterbalance_cases(range(plan["new_base_seeds"]["start_inclusive"], plan["new_base_seeds"]["stop_exclusive"]),
                                             plan["dependency_depths"], plan["question_styles"])
    if regenerated != all_cases:
        raise ValueError("Registered cases differ from the locked generator/plan")
    sources = locked_training_sources(plan, ROOT, {case["seed"] for case in all_cases})
    if sources != data["checkpoint_sources"] or sources != manifest["checkpoint_sources"]:
        raise ValueError("Frozen source checkpoints changed")
    for info, directory in ((manifest, args.run / "sources"), (data, Path(manifest["data_path"]) / "sources")):
        for relative, fingerprint in info["source_sha256"].items():
            if file_digest(directory / relative) != fingerprint:
                raise ValueError(f"Source snapshot changed: {relative}")
    for relative in ("sas_memory/counterbalance_tasks.py", "sas_memory/counterbalance_data.py", "sas_memory/kv_readout.py"):
        if manifest["source_sha256"][relative] != data["source_sha256"][relative] or file_digest(ROOT / relative) != data["source_sha256"][relative]:
            raise ValueError("Registered and evaluated token-control implementations differ")
    for name, fingerprint in data["tokenizer_sha256"].items():
        if file_digest(Path(data["model_path"]) / name) != fingerprint:
            raise ValueError("Tokenizer changed")
    if build_counterbalance_registry(tokenizer_runtime(data["model_path"]), all_cases) != registry:
        raise ValueError("Registered masks/token positions no longer reproduce")
    cases = [all_cases[0], all_cases[-1]] if manifest["smoke"] else all_cases
    if manifest["evaluated_case_ids"] != [case["id"] for case in cases] or manifest["regimes"] != plan["evaluation_regimes"] or manifest["modes"] != plan["checkpoints"]:
        raise ValueError("Evaluation coverage differs from plan")
    expected = {(case["id"], order, "target_current"): case["variants"][order]["queries"][0]["answer"] for case in cases for order in ORDERS}
    registered = {(row["case_id"], row["variant"]): row for row in registry}
    all_rows = {}
    safety_cases = {cases[0]["id"], cases[-1]["id"]}
    expected_checks = {(case, ORDERS[0], regime) for case in safety_cases for regime in MASKS}
    for model in manifest["modes"]:
        checks = manifest["safety_checks"][model]
        if {(check["case_id"], check["variant"], check["regime"]) for check in checks} != expected_checks or len(checks) != len(expected_checks):
            raise ValueError("Missing/duplicate safety samples")
        for check in checks:
            for field in ("all_history_full_cache_and_logits_exact", "all_history_scoring_and_greedy_exact", "pre_question_kv_recurrent_conv_and_logits_exact", "historical_kv_exact_in_full_prefill"):
                if not check[field]:
                    raise ValueError("Mask identity verification failed")
            if check["historical_kv_fields_checked"] != 12 or check["unrestricted_replay_max_nll_error"] > 1e-5 or not all(check["all_history_hook_calls"].values()):
                raise ValueError("Incomplete all-history/historical-KV verification")
        if model != "native":
            prefix = manifest["native_prefix_checks"][model]
            if {row["case_id"] for row in prefix} != safety_cases or len(prefix) != len(safety_cases) or any(not row["kv_recurrent_conv_and_logits_exact"] for row in prefix):
                raise ValueError("Question-only checkpoint changed native prefix")
            replay = manifest["native_replay_after_detach"][model]
            if replay["max_nll_error"] > 1e-5 or not replay["greedy_exact_match"] or not replay["adapter_weights_unchanged"]:
                raise ValueError("Frozen/restoration verification failed")
        rows = {}
        for regime in manifest["regimes"]:
            members = load_rows(args.run / f"{model}.{regime}.jsonl", expected)
            for row in members:
                record = registered[(row["case_id"], row["variant"])]
                if row["regime"] != regime or any(row[key] != record[key] for key in ("ledger_tokens", "prompt_tokens", "prompt_ids_sha256")):
                    raise ValueError("Result changed the original input")
                if regime != "raw_full":
                    mask = record["masks"][regime]
                    if any(row[key] != mask[key] for key in ("selected_event_indices", "kept_history_tokens", "keep_sha256")):
                        raise ValueError("Actual event selection/budget differs from registration")
                    if row["hook_calls"]["prefill"] != 1 or row["hook_calls"]["cached"] < 1:
                        raise ValueError("Restriction was not active on prefill and cached decoding")
            rows[regime] = members
        all_rows[model] = rows
    report = {"status": "audited_smoke" if manifest["smoke"] else "completed", "source_run": str(args.run),
              "manifest_sha256": file_digest(args.run / "manifest.json"), "cases_sha256": data["cases_sha256"], "registry_sha256": data["registry_sha256"],
              "base_cases": len({case["base_case_id"] for case in cases}), "contexts_per_regime": len(expected),
              "rows_audited": len(expected) * len(manifest["modes"]) * len(manifest["regimes"]),
              "checks": {"source_snapshots_and_checkpoints": True, "executable_truth_and_balanced_conditions": True,
                         "original_prompt_token_identity": True, "retained_evidence_and_equal_extra_token_budgets": True,
                         "same_extra_token_positions_after_role_swap": True, "equal_extra_SET_values": True,
                         "sampled_cache_logit_and_greedy_identity": True, "all_rows_prefill_and_cached_masks": True},
              "interpretation": "Oracle event selection at all six full-attention layers during question/answer processing, with original prefix intact. Both extra SETs write the same new value; source/unrelated roles swap at exact matching late token intervals. Effects describe this COPY-template family, not irreversible GDN damage, a learned router, architecture superiority, or efficient sparse kernels. Bootstrap units are base facts; initialization averages are conditional on the three fixed trained seeds. Zero empirical paired accuracy differences can yield degenerate bootstrap intervals; these do not establish equivalence or rule out effects on unseen facts. This fresh post-COPY, single-extra-record protocol also differs from the earlier two-extra-record/both-COPY-order screen; a changed result cannot be attributed only to position/value balancing."}
    if not manifest["smoke"]:
        report["models"] = {model: summarize(rows, cases) for model, rows in all_rows.items()}
        clusters = {case["id"]: case["base_case_id"] for case in cases}
        scores = {kind: [unary_cluster_scores(all_rows[model][kind], clusters) for model in manifest["modes"] if model != "native"] for kind in ("extra_source", "extra_unrelated")}
        averages = {kind: {base: mean(values[base] for values in seeds) for base in seeds[0]} for kind, seeds in scores.items()}
        fixed_effect = paired_interval(averages["extra_source"], averages["extra_unrelated"])
        fixed_effect["unit"] = "base_case_id, averaging all depth/wording/orders and the three fixed initializations"
        report["fixed_three_initialization_average"] = {"source_minus_unrelated": fixed_effect,
                                                       "source_accuracy": mean(averages["extra_source"].values()), "unrelated_accuracy": mean(averages["extra_unrelated"].values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.plot:
        if manifest["smoke"]:
            raise ValueError("No effect plots for operational smoke checks")
        plot_report(report, args.plot)
    print(json.dumps({key: report[key] for key in ("status", "base_cases", "contexts_per_regime", "rows_audited")}))
    for model, result in report.get("models", {}).items():
        print(model, {name: f"{value['correct']}/{value['count']}" for name, value in result["regimes"].items()}, result["source_minus_unrelated"])


if __name__ == "__main__":
    main()
