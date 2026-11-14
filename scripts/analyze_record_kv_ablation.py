"""Audit record-blocked KV inference and compare paired full/oracle readouts."""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.counterbalance_data import file_digest, tokenizer_runtime
from sas_memory.counterbalance_tasks import ORDERS
from sas_memory.record_kv_ablation import build_record_registry, reference_protocol
from sas_memory.training_data import data_fingerprint
from scripts.analyze_controller_trial import load_rows
from scripts.analyze_controller_confirmation import scalar_metrics, unary_cluster_scores, paired_interval
from scripts.analyze_readout_counterbalance import interval, paired_changes, order_consistency


def summarize(rows, cases):
    lookup = {case["id"]: case for case in cases}
    clusters = {case["id"]: case["base_case_id"] for case in cases}
    conditions = sorted({(case["dependency_depth"], case["question_style"]) for case in cases})
    errors = {}
    for regime, members in rows.items():
        labels = []
        for row in members:
            case = lookup[row["case_id"]]
            initial = {event["target"]: event["value"] for event in case["prefix"][:8]}
            prediction = row["generated"].strip()
            labels.append("correct" if row["greedy_exact"] else "late_written_value" if prediction == str(case["new_value"])
                          else "target_initial_value" if prediction == str(initial[case["target_entity"]])
                          else "unrelated_initial_value" if prediction == str(initial[case["unrelated_entity"]]) else "other")
        errors[regime] = dict(Counter(labels))
    return {"regimes": {name: scalar_metrics(members) for name, members in rows.items()},
            "blocked_minus_full": interval(rows["records_blocked"], rows["raw_full"], clusters),
            "blocked_minus_oracle": interval(rows["records_blocked"], rows["oracle"], clusters),
            "blocked_vs_full_paired": paired_changes(rows["records_blocked"], rows["raw_full"]),
            "blocked_vs_oracle_paired": paired_changes(rows["records_blocked"], rows["oracle"]),
            "error_types": errors,
            "blocked_prediction_counts": dict(Counter(row["generated"].strip() for row in rows["records_blocked"])),
            "blocked_query_target_id_echoes": sum(row["generated"].strip() == lookup[row["case_id"]]["target_entity"][1:] for row in rows["records_blocked"]),
            "blocked_integer_format_outputs": sum(bool(re.fullmatch(r"-?\d+", row["generated"].strip())) for row in rows["records_blocked"]),
            "independent_late_order_consistency": {name: order_consistency(members) for name, members in rows.items()},
            "by_condition": [{"dependency_depth": depth, "question_style": style,
                              "regimes": {name: scalar_metrics([row for row in members if lookup[row["case_id"]]["dependency_depth"] == depth and lookup[row["case_id"]]["question_style"] == style]) for name, members in rows.items()}}
                             for depth, style in conditions],
            "by_late_order": {order: {name: scalar_metrics([row for row in members if row["variant"] == order]) for name, members in rows.items()} for order in ORDERS}}


def plot_report(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    models = list(report["models"])
    positions = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, (regime, label, color) in enumerate((("raw_full", "Full historical access", "#7b8794"),
                                                   ("oracle", "Oracle evidence", "#2681b5"),
                                                   ("records_blocked", "Historical records blocked", "#d5634f"))):
        values = [report["models"][model]["regimes"][regime] for model in models]
        bars = ax.bar(positions + (offset - 1) * .25, [value["accuracy"] for value in values], .24, label=label, color=color)
        ax.bar_label(bars, [str(value["correct"]) for value in values], fontsize=9, padding=3)
    ax.set_xticks(positions, ["Native" if model == "native" else "Q-only " + model.split("seed")[-1] for model in models])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Current-target accuracy (labels: correct / 288)")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.set_title("Question-time historical record KV access — frozen Qwen3.5-2B IT")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .075), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .025, "24 base facts; complete prefix retained. Only instructions and causal question/answer KV remain accessible in the blocked condition.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .17, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--allow-smoke", action="store_true", help="Operational audit only; no cohort effect estimates")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    data = json.loads((args.run / "data_manifest.json").read_text())
    plan = json.loads((args.run / "plan.json").read_text())
    all_cases = json.loads((args.run / "cases.json").read_text())
    registry = json.loads((args.run / "registry.json").read_text())
    reference, original, _, reference_cases, sources, hashes = reference_protocol(plan, ROOT)
    if manifest["status"] != "completed" or manifest["completed_models"] != plan["models"] or manifest["models"] != plan["models"] or manifest["training"]:
        raise ValueError("Ablation evaluation is incomplete")
    if manifest["smoke"] and not args.allow_smoke:
        raise ValueError("Smoke cannot provide full-cohort evidence")
    if file_digest(args.run / "data_manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Preparation manifest changed")
    for key, value in (("plan", plan), ("cases", all_cases), ("registry", registry)):
        if data_fingerprint(value) != data[key + "_sha256"]:
            raise ValueError(f"Registered {key} changed")
    if all_cases != reference_cases or data["checkpoint_sources"] != sources or manifest["checkpoint_sources"] != sources:
        raise ValueError("Cohort/checkpoints differ from reference")
    if manifest["model_path"] != data["model_path"] or data["model_path"] != original["model_path"] or any(manifest[key] != data[key] for key in ("cases_sha256", "registry_sha256")):
        raise ValueError("Model/cohort identity metadata differs")
    if data["reference_manifest_sha256"] != file_digest(reference / "manifest.json") or hashes != data["reference_result_sha256"]:
        raise ValueError("Reference results changed")
    if data["reference_audit_sha256"] != file_digest(args.run / "reference.audit.json"):
        raise ValueError("Reference audit changed")
    audit = json.loads((args.run / "reference.audit.json").read_text())
    if audit["status"] != "completed" or audit["manifest_sha256"] != data["reference_manifest_sha256"] or not all(audit["checks"].values()):
        raise ValueError("Reference audit does not establish complete coverage")
    if datetime.fromisoformat(data["registered_utc"]) >= datetime.fromisoformat(manifest["started_utc"]):
        raise ValueError("Masks were not registered before inference")
    for info, directory in ((data, Path(manifest["data_path"]) / "sources"), (manifest, args.run / "sources")):
        for relative, fingerprint in info["source_sha256"].items():
            if file_digest(directory / relative) != fingerprint:
                raise ValueError(f"Source snapshot changed: {relative}")
    for relative in ("sas_memory/record_kv_ablation.py", "sas_memory/kv_readout.py", "sas_memory/counterbalance_data.py", "sas_memory/runtime.py", "sas_memory/controllers.py"):
        if data["source_sha256"][relative] != manifest["source_sha256"][relative] or file_digest(ROOT / relative) != data["source_sha256"][relative]:
            raise ValueError("Registered and actual intervention implementations differ")
    for name, fingerprint in data["tokenizer_sha256"].items():
        if file_digest(Path(data["model_path"]) / name) != fingerprint:
            raise ValueError("Tokenizer changed")
    if build_record_registry(tokenizer_runtime(data["model_path"]), all_cases) != registry:
        raise ValueError("Boundary-only masks do not reproduce")
    if manifest["packages"] != original["packages"]:
        raise ValueError("Runtime package mismatch")
    cases = [all_cases[0], all_cases[-1]] if manifest["smoke"] else all_cases
    if manifest["evaluated_case_ids"] != [case["id"] for case in cases]:
        raise ValueError("Evaluated case coverage changed")
    expected = {(case["id"], order, "target_current"): case["variants"][order]["queries"][0]["answer"] for case in cases for order in ORDERS}
    all_expected = {(case["id"], order, "target_current"): case["variants"][order]["queries"][0]["answer"] for case in all_cases for order in ORDERS}
    registered = {(row["case_id"], row["variant"]): row for row in registry}
    checks_expected = {(cases[0]["id"], ORDERS[0]), (cases[-1]["id"], ORDERS[0])}
    rows_by_model = {}
    for model in plan["models"]:
        for name in ("safety_checks", "oracle_replay"):
            checks = manifest[name][model]
            if {(row["case_id"], row["variant"]) for row in checks} != checks_expected or len(checks) != len(checks_expected):
                raise ValueError("Missing/duplicate short/long replay checks")
        for check in manifest["safety_checks"][model]:
            for name in ("all_history_full_cache_and_logits_exact", "all_history_scoring_and_greedy_exact", "pre_question_kv_recurrent_conv_and_logits_exact", "historical_kv_exact_in_full_prefill"):
                if not check[name]:
                    raise ValueError("Record-blocking identity proof failed")
            if check["historical_kv_fields_checked"] != 12 or check["unrestricted_replay_max_nll_error"] > 1e-5 or not all(check["all_history_hook_calls"].values()):
                raise ValueError("Incomplete full-history/prefix verification")
        if any(row["max_nll_error"] > 1e-5 or not row["greedy_exact_match"] for row in manifest["oracle_replay"][model]):
            raise ValueError("Oracle reference replay failed")
        if model != "native":
            checks = manifest["native_prefix_checks"][model]
            if len(checks) != len(checks_expected) or {(row["case_id"], row["variant"]) for row in checks} != checks_expected or any(not row["kv_recurrent_conv_and_logits_exact"] for row in checks):
                raise ValueError("Question-only adaptation changed native prefix")
            replay = manifest["native_replay_after_detach"][model]
            if replay["max_nll_error"] > 1e-5 or not replay["greedy_exact_match"] or not replay["adapter_weights_unchanged"]:
                raise ValueError("Frozen/restoration verification failed")
        blocked = load_rows(args.run / f"{model}.records_blocked.jsonl", expected)
        for row in blocked:
            record = registered[(row["case_id"], row["variant"])]
            if row["regime"] != "records_blocked" or any(row[key] != record[key] for key in ("prelude_tokens", "ledger_tokens", "prompt_tokens", "prompt_ids_sha256", "keep_sha256", "blocked_record_tokens")):
                raise ValueError("Actual record mask/input differs from registration")
            if row["hook_calls"]["prefill"] != 1 or row["hook_calls"]["cached"] < 1:
                raise ValueError("Missing prefill/decode restrictions")
        rows = {regime: [row for row in load_rows(reference / f"{model}.{regime}.jsonl", all_expected) if (row["case_id"], row["variant"], row["role"]) in expected] for regime in ("raw_full", "oracle")}
        rows_by_model[model] = {**rows, "records_blocked": blocked}
    report = {"status": "audited_smoke" if manifest["smoke"] else "completed", "source_run": str(args.run),
              "analysis_source_sha256": file_digest(Path(__file__).resolve()),
              "manifest_sha256": file_digest(args.run / "manifest.json"), "cases_sha256": data["cases_sha256"], "registry_sha256": data["registry_sha256"],
              "base_cases": len({case["base_case_id"] for case in cases}), "contexts_per_model": len(expected),
              "new_records_audited": len(expected) * len(plan["models"]),
              "checks": {"registered_before_inference": True, "boundary_only_masks_regenerated": True, "all_ledger_record_columns_blocked": True,
                         "original_input_and_prefix_retained": True, "short_and_long_raw_oracle_replays": True,
                         "cached_generation_always_masked": True, "source_tokenizer_checkpoint_and_result_hashes": True},
              "interpretation": "Historical ledger-record KV is blocked only for question/answer rows in all six attention layers. Normal hybrid prefix computation, recurrent/convolution state, instruction KV and causal question/answer KV remain. This is not isolated GDN-matrix performance. Success can indicate usable information in remaining carriers; failure under a frozen perturbed readout does not establish absent information or memory damage. Dense eager masking is not an efficiency result. All queries here are post-COPY current-value queries; semantic-order and historical-version generalization remain separate requirements."}
    if not manifest["smoke"]:
        report["models"] = {model: summarize(rows, cases) for model, rows in rows_by_model.items()}
        answer_counts = Counter(value for value in expected.values())
        report["target_answer_counts"] = dict(answer_counts)
        report["largest_gold_answer_frequency"] = max(answer_counts.values())
        clusters = {case["id"]: case["base_case_id"] for case in cases}
        scores = {regime: [unary_cluster_scores(rows[regime], clusters) for model, rows in rows_by_model.items() if model != "native"] for regime in ("raw_full", "oracle", "records_blocked")}
        averages = {regime: {base: mean(seed[base] for seed in seeds) for base in seeds[0]} for regime, seeds in scores.items()}
        report["fixed_three_initialization_average"] = {"accuracy": {regime: mean(values.values()) for regime, values in averages.items()},
                                                       "blocked_minus_full": paired_interval(averages["records_blocked"], averages["raw_full"]),
                                                       "blocked_minus_oracle": paired_interval(averages["records_blocked"], averages["oracle"])}
        for name in ("blocked_minus_full", "blocked_minus_oracle"):
            report["fixed_three_initialization_average"][name]["unit"] = "base_case_id, averaging depth/wording/orders and three fixed trained initializations"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.plot:
        if manifest["smoke"]:
            raise ValueError("No full-cohort plot for smoke")
        plot_report(report, args.plot)
    print(json.dumps({key: report[key] for key in ("status", "base_cases", "contexts_per_model", "new_records_audited")}))
    for model, values in report.get("models", {}).items():
        print(model, {regime: f"{value['correct']}/{value['count']}" for regime, value in values["regimes"].items()})


if __name__ == "__main__":
    main()
