"""Post-hoc final-checkpoint fit diagnostic; no optimization or checkpoint selection."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.counterbalance_data import file_digest
from sas_memory.training_data import data_fingerprint
from scripts.analyze_blocked_reader_adaptation import audit_run
from scripts.analyze_controller_trial import load_rows
from scripts.run_controller_trial import prompt_for, reader_context, supervised_loss, write_json
from scripts.run_oracle_kv_readout import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs=2, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [audit_run(path)[0] for path in args.runs]
    if {source["token_policy"] for source in sources} != {"question", "all"}:
        raise ValueError("Question-only and all-token final checkpoints required")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "training": False, "posthoc": True, "source_sha256": file_digest(Path(__file__)), "arms": {},
              "interpretation": "Fit of the final step-64 checkpoints on the exact 128 original training examples; no extra updates or checkpoint selection. Teacher-forced token accuracy is distinct from unrestricted greedy answer accuracy. Poor fit leaves optimizer budget and adapter expressivity unresolved and cannot prove absent memory information."}
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(args.output / "manifest.json", report)
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    from sas_memory.runtime import TextRuntime
    torch.set_num_threads(4)
    model_path = json.loads((ROOT / "configs/models.json").read_text())["it"]
    try:
        runtime = TextRuntime(model_path, "it")
        for path, source in zip(args.runs, sources):
            manifest = json.loads((path / "manifest.json").read_text())
            if file_digest(path / "manifest.json") != source["manifest_sha256"]:
                raise ValueError("Training manifest changed")
            config = manifest["config"]
            data = json.loads((path / "data.json").read_text())
            encoded = json.loads((path / "tokenized_train.json").read_text())
            checkpoint = path / "local/controller.safetensors"
            if file_digest(checkpoint) != source["checkpoint_sha256"]:
                raise ValueError("Final checkpoint changed")
            weights = load_file(str(checkpoint), device=str(runtime.model.device))
            controller = GroupController(runtime.model, config["group_index"], config["modes"]["local"], "local",
                                         adapt_gdn_keys=config["adapt_gdn_keys"])
            controller.load_state_dict(weights, strict=True)
            rows = []
            label = config["token_policy"]
            result_path = args.output / f"{label}.jsonl"
            try:
                with torch.inference_mode(), result_path.open("w", buffering=1) as output:
                    for item, example in zip(encoded, data["train"]):
                        calls = []
                        loss, logits = supervised_loss(runtime, item, controller, config["token_policy"], "records_blocked", calls)
                        prediction = logits.argmax(-1)[0].tolist()
                        ledger, _ = prompt_for(runtime, example["case"], example["variant"], example["query"])
                        with reader_context(runtime, "records_blocked", item["reader_mask"]) as reader:
                            actual = measure(runtime, ledger, example["query"])
                        if calls != [{"prefill": 1, "cached": 0}] or reader.calls["prefill"] != 1 or reader.calls["cached"] < 1:
                            raise ValueError("Fit probe missed a masked forward")
                        row = {"case_id": item["case_id"], "variant": item["variant"], "role": item["role"],
                               "family": item["family"], "supervised_ids": item["answer_ids"], "teacher_prediction_ids": prediction,
                               "teacher_loss_including_newline": loss.item(), "reader_mask": item["reader_mask"],
                               "readout_calls": dict(reader.calls), **actual}
                        rows.append(row)
                        output.write(json.dumps(row) + "\n")
                if any(not torch.equal(value, weights[key]) for key, value in controller.state_dict().items()):
                    raise ValueError("Frozen adapter changed during fit probe")
            finally:
                controller.close()
            expected = {(item["case_id"], item["variant"], item["role"]): example["query"]["answer"]
                        for item, example in zip(encoded, data["train"])}
            if load_rows(result_path, expected) != rows or len(rows) != 128 or data_fingerprint(data) != manifest["data_sha256"]:
                raise ValueError("Training fit coverage/gold/provenance changed")
            correct_tokens = sum(sum(left == right for left, right in zip(row["supervised_ids"], row["teacher_prediction_ids"])) for row in rows)
            total_tokens = sum(len(row["supervised_ids"]) for row in rows)
            answer_tokens = sum(len(runtime.continuation_ids(prompt_for(runtime, example["case"], example["variant"], example["query"])[1], example["query"]["answer"])) for example in data["train"])
            report["arms"][label] = {"source": source, "rows_sha256": file_digest(result_path), "examples": len(rows),
                                     "free_greedy_correct": sum(row["greedy_exact"] for row in rows),
                                     "gold_answer_nll_excluding_newline": mean(row["answer_nll"] for row in rows),
                                     "teacher_loss_including_newline": sum(row["teacher_loss_including_newline"] * len(row["supervised_ids"]) for row in rows) / total_tokens,
                                     "teacher_correct_tokens": correct_tokens, "teacher_tokens": total_tokens,
                                     "answer_tokens_excluding_newline": answer_tokens,
                                     "teacher_all_tokens_correct_examples": sum(row["supervised_ids"] == row["teacher_prediction_ids"] for row in rows),
                                     "empirical_most_common_gold_answer_count": max(Counter(row["answer"] for row in rows).values()),
                                     "prediction_counts": dict(Counter(row["generated"].strip() for row in rows))}
            write_json(args.output / "manifest.json", report)
            print(json.dumps({"event": "fit_checked", "policy": label, "greedy_correct": report["arms"][label]["free_greedy_correct"], "examples": len(rows)}), flush=True)
        report.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output / "manifest.json", report)


if __name__ == "__main__":
    main()
