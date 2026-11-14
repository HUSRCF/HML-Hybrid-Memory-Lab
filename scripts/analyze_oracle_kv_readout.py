"""Audit original-prefix KV filtering against dense and rebuilt-slice controls."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.generalization_tasks import validate_generalization_cases
from sas_memory.training_data import data_fingerprint
from scripts.analyze_controller_trial import load_rows
from scripts.analyze_controller_confirmation import scalar_metrics, paired_interval, unary_cluster_scores
from scripts.run_oracle_kv_readout import digest, selection


def plot_report(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = {"native": "Native", "local_all": "Full-token adaptation", "local_question": "Question-only adaptation"}
    colors = ["#6f7b8c", "#2579b2", "#dfa52f"]
    selected_label = {"oracle": "Same prefix, oracle KV", "competing_versions": "Same prefix, + version records",
                      "matched_unrelated": "Same prefix, + unrelated records", "target_initial_only": "Same prefix, + initial target",
                      "source_other_only": "Same prefix, + unused source SET"}[report.get("selection_kind", "oracle")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), sharey=True)
    for ax, (model, result) in zip(axes, report["models"].items()):
        conditions = result["by_condition"]
        positions = np.arange(len(conditions))
        for offset, (regime, legend) in enumerate((("raw_full", "Complete prefix"), ("oracle_kv", selected_label), ("slice_rebuilt", "Rebuilt minimal prefix"))):
            values = [item["regimes"][regime]["correct"] for item in conditions]
            bars = ax.bar(positions + (offset - 1) * .25, values, .24, label=legend, color=colors[offset])
            ax.bar_label(bars, fontsize=7, padding=2)
        ax.set_title(labels[model])
        ax.set_xticks(positions, [f"{item['dependency_depth']} / {'O' if item['question_style']=='original' else 'E'}" for item in conditions])
        ax.set_xlabel("COPY depth / question style")
        ax.set_ylim(0, 18)
        ax.set_yticks([0, 4, 8, 12, 16])
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Current-target answers correct / 16")
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="lower center", ncol=3, bbox_to_anchor=(.5, .085))
    fig.suptitle("Question-time evidence access with the original hybrid prefix retained", fontsize=13)
    fig.text(.5, .025, "8 shared base cases; one trained initialization. O = original, E = entity-first. Oracle dependencies are not a learned selector.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .17, 1, .93))
    path.parent.mkdir(parents=True, exist_ok=True)
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
    config = manifest["config"]
    if manifest["status"] != "completed" or manifest["completed_modes"] != config["modes"] or manifest["smoke"] or manifest["training"]:
        raise ValueError("Expected a complete, full-cohort frozen diagnostic")
    cases = json.loads((args.run / "cases.json").read_text())
    validation = validate_generalization_cases(cases)
    if data_fingerprint(cases) != manifest["cases_sha256"] or manifest["cases_sha256"] != config["expected_cases_sha256"]:
        raise ValueError("Cohort changed")
    if validation != manifest["task_validation"]:
        raise ValueError("Task validation changed")
    reference = Path(config["reference_run"])
    if digest(reference / "manifest.json") != manifest["reference_manifest_sha256"]:
        raise ValueError("Reference manifest changed")
    original = json.loads((reference / "manifest.json").read_text())
    for relative, fingerprint in manifest["source_sha256"].items():
        if digest(args.run / "sources" / relative) != fingerprint:
            raise ValueError(f"Source snapshot changed: {relative}")
    for relative in ("sas_memory/runtime.py", "sas_memory/controllers.py"):
        if manifest["source_sha256"][relative] != original["source_sha256"][relative]:
            raise ValueError("Reference/runtime computation differs")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(manifest["model_path"], local_files_only=True)
    marker = "SAS_UNIQUE_CONTENT_MARKER"
    template = tokenizer.apply_chat_template([{"role": "user", "content": marker}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    start, end = template.split(marker)
    runtime = SimpleNamespace(tokenizer=tokenizer, prompt_start=start,
                              query_suffix=lambda q: f"Question: {q}\nAnswer:" + end,
                              ids=lambda text: tokenizer.encode(text, add_special_tokens=False))
    lookup = {case["id"]: case for case in cases}
    clusters = {case["id"]: case["base_case_id"] for case in cases}
    expected = {(case["id"], variant, "target_current"): next(q["answer"] for q in case["variants"][variant]["queries"] if q["role"] == "target_current")
                for case in cases for variant in config["variants"]}
    all_expected = {(case["id"], variant, q["role"]): q["answer"] for case in cases
                    for variant, item in case["variants"].items() for q in item["queries"]}
    report = {"status": "completed", "cases": len(cases), "base_cases": validation["base_cases"], "models": {},
              "selection_kind": config.get("selection_kind", "oracle"),
              "source_run": str(args.run), "manifest_sha256": digest(args.run / "manifest.json"),
              "interpretation": "Exploratory oracle intervention at all full-attention layers during questions/answers, with full prefix computation retained. It is not a learned/efficient selector or evidence that recurrence alone lost the answer. One initialization and eight base cases; clustered intervals are conditional and unadjusted."}
    conditions = sorted({(case["dependency_depth"], case["question_style"]) for case in cases})
    for mode in config["modes"]:
        for kind, suffix in (("raw", "jsonl"), ("oracle", "oracle.jsonl")):
            if digest(reference / f"{mode}.{suffix}") != manifest["reference_results_sha256"][mode][kind]:
                raise ValueError("Reference results changed")
        if mode != "native":
            spec = original["checkpoint_sources"][mode]
            checkpoint = Path(spec["training_run"]) / spec["mode"] / "controller.safetensors"
            if digest(checkpoint) != manifest["checkpoint_sha256"][mode] or manifest["checkpoint_sha256"][mode] != original["checkpoint_sha256"][mode]:
                raise ValueError("Controller changed")
        safety = manifest["safety_checks"].get(mode, {})
        for field in ("all_history_full_cache_and_logits_exact", "all_history_scoring_and_greedy_exact",
                      "pre_question_kv_recurrent_conv_and_logits_exact", "historical_kv_exact_in_full_prefill"):
            if not safety.get(field):
                raise ValueError(f"Missing safety check: {mode}/{field}")
        if safety["historical_kv_fields_checked"] != 12 or safety["unrestricted_replay_max_nll_error"] > 1e-5:
            raise ValueError("Incomplete identity/replay verification")
        kv_rows = load_rows(args.run / f"{mode}.jsonl", expected)
        for row in kv_rows:
            case = lookup[row["case_id"]]
            query = next(q for q in case["variants"][row["variant"]]["queries"] if q["role"] == "target_current")
            _, prompt, keep, indices = selection(runtime, case, row["variant"], query, config.get("selection_kind", "oracle"))
            if row["selected_event_indices"] != indices or row["ledger_tokens"] != len(keep) or row["kept_history_tokens"] != sum(keep):
                raise ValueError("Oracle token selection changed")
            if row["history_keep_sha256"] != hashlib.sha256(bytes(keep)).hexdigest() or row["prompt_ids_sha256"] != hashlib.sha256(json.dumps(runtime.ids(prompt)).encode()).hexdigest():
                raise ValueError("Original prompt or token keep mask changed")
            if row["hook_calls"]["prefill"] != 1 or row["hook_calls"]["cached"] < 1:
                raise ValueError("Missing prefill/decode intervention")
        raw_rows = [row for row in load_rows(reference / f"{mode}.jsonl", all_expected)
                    if (row["case_id"], row["variant"], row["role"]) in expected]
        regimes = {"raw_full": raw_rows, "oracle_kv": kv_rows,
                   "slice_rebuilt": load_rows(reference / f"{mode}.oracle.jsonl", expected)}
        indexed_raw = {(row["case_id"], row["variant"]): row for row in raw_rows}
        paired = {"raw_correct": sum(row["greedy_exact"] for row in raw_rows),
                  "rescued": sum(not indexed_raw[(row["case_id"], row["variant"])]["greedy_exact"] and row["greedy_exact"] for row in kv_rows),
                  "regressed": sum(indexed_raw[(row["case_id"], row["variant"])]["greedy_exact"] and not row["greedy_exact"] for row in kv_rows)}
        interval = paired_interval(unary_cluster_scores(kv_rows, clusters), unary_cluster_scores(raw_rows, clusters))
        interval["unit"] = "base_case_id, averaging all six depth/wording conditions and both orders"
        report["models"][mode] = {"regimes": {name: scalar_metrics(rows) for name, rows in regimes.items()},
                                  "kv_vs_raw_paired": paired, "kv_minus_raw": interval,
                                  "by_variant": {variant: {name: scalar_metrics([row for row in rows if row["variant"] == variant]) for name, rows in regimes.items()} for variant in config["variants"]},
                                  "by_condition": [{"dependency_depth": depth, "question_style": style,
                                                    "regimes": {name: scalar_metrics([row for row in rows if lookup[row["case_id"]]["dependency_depth"] == depth and lookup[row["case_id"]]["question_style"] == style]) for name, rows in regimes.items()}}
                                                   for depth, style in conditions]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.plot:
        plot_report(report, args.plot)
    for mode, result in report["models"].items():
        print(mode, {name: f"{value['correct']}/{value['count']}" for name, value in result["regimes"].items()}, result["kv_vs_raw_paired"])


if __name__ == "__main__":
    main()
