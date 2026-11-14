"""Independently recompute final-checkpoint training-fit summaries from raw rows."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.counterbalance_data import file_digest
from scripts.analyze_blocked_reader_adaptation import audit_run
from scripts.analyze_controller_trial import load_rows
from scripts.run_controller_trial import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["training"] or not manifest["posthoc"] or set(manifest["arms"]) != {"question", "all"}:
        raise ValueError("A complete, frozen, post-hoc two-arm fit probe is required")
    if file_digest(args.run / "source.py") != manifest["source_sha256"]:
        raise ValueError("Fit probe source snapshot changed")
    summaries = {}
    for label, saved in manifest["arms"].items():
        source, _ = audit_run(Path(saved["source"]["path"]))
        if source != saved["source"] or source["token_policy"] != label:
            raise ValueError("Fit probe used a different final checkpoint")
        path = Path(source["path"])
        data = json.loads((path / "data.json").read_text())
        encoded = json.loads((path / "tokenized_train.json").read_text())
        indexed = {(row["case_id"], row["variant"], row["role"]): row for row in encoded}
        expected = {(item["case_id"], item["variant"], item["role"]): example["query"]["answer"]
                    for item, example in zip(encoded, data["train"])}
        row_path = args.run / f"{label}.jsonl"
        if file_digest(row_path) != saved["rows_sha256"]:
            raise ValueError("Fit rows changed")
        rows = load_rows(row_path, expected)
        for row in rows:
            original = indexed[(row["case_id"], row["variant"], row["role"])]
            if (row["reader_mask"] != original["reader_mask"] or row["supervised_ids"] != original["answer_ids"] or
                row["readout_calls"]["prefill"] != 1 or row["readout_calls"]["cached"] < 1 or
                len(row["teacher_prediction_ids"]) != len(row["supervised_ids"]) or
                any(type(value) is not int or value < 0 for value in row["teacher_prediction_ids"]) or
                not math.isfinite(row["teacher_loss_including_newline"]) or row["teacher_loss_including_newline"] < 0):
                raise ValueError("Fit mask, teacher labels or cached access invalid")
            if len(row["supervised_ids"]) != 4:
                raise ValueError("This diagnostic's three-digit plus newline assumption is not satisfied")
        tokens = sum(len(row["supervised_ids"]) for row in rows)
        correct_tokens = sum(sum(left == right for left, right in zip(row["supervised_ids"], row["teacher_prediction_ids"])) for row in rows)
        result = {"examples": len(rows), "free_greedy_correct": sum(row["greedy_exact"] for row in rows),
                  "gold_answer_nll_excluding_newline": mean(row["answer_nll"] for row in rows),
                  "teacher_loss_including_newline": sum(row["teacher_loss_including_newline"] * len(row["supervised_ids"]) for row in rows) / tokens,
                  "teacher_correct_tokens": correct_tokens, "teacher_tokens": tokens,
                  "teacher_all_tokens_correct_examples": sum(row["supervised_ids"] == row["teacher_prediction_ids"] for row in rows),
                  "empirical_most_common_gold_answer_count": max(Counter(row["answer"] for row in rows).values()),
                  "prediction_counts": dict(Counter(row["generated"].strip() for row in rows))}
        if any(result[key] != saved[key] for key in result):
            raise ValueError("Stored fit summaries disagree with raw predictions")
        result.update(teacher_digit_correct=sum(sum(left == right for left, right in zip(row["supervised_ids"][:-1], row["teacher_prediction_ids"][:-1])) for row in rows),
                      teacher_digit_tokens=3 * len(rows),
                      teacher_newline_correct=sum(row["supervised_ids"][-1] == row["teacher_prediction_ids"][-1] for row in rows),
                      source=source, rows_sha256=file_digest(row_path))
        summaries[label] = result
    report = {"status": "completed_and_audited", "source_run": str(args.run),
              "manifest_sha256": file_digest(args.run / "manifest.json"), "records_audited": sum(arm["examples"] for arm in summaries.values()),
              "analysis_source_sha256": file_digest(Path(__file__)), "arms": summaries,
              "interpretation": manifest["interpretation"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({label: {key: arm[key] for key in ("free_greedy_correct", "examples", "teacher_digit_correct", "teacher_digit_tokens", "teacher_newline_correct")} for label, arm in summaries.items()}))


if __name__ == "__main__":
    main()
