"""Semantic mutation checks for committed query-capacity checkpoint audits."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from scripts.analyze_gdn_query_capacity import audit_checkpoint
from scripts.analyze_single_fact_probe import registered_dataset
from scripts.run_controller_trial import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.run / "plan.json").read_text())
    _, data, _ = registered_dataset(Path(plan["data_source"]))
    encoded = json.loads((args.run / "tokenized_train.json").read_text())
    source = args.run / "local/step-000064"
    audit_checkpoint(args.run, "local", 64, plan, data, encoded)
    import torch
    results = {}
    for mutation in ("wrong_adamw_step_with_updated_hash", "opened_record_column_with_updated_hash"):
        with tempfile.TemporaryDirectory(prefix="sas-query-auditor-") as temporary:
            trial = Path(temporary)
            destination = trial / "local/step-000064"
            shutil.copytree(source, destination)
            shutil.copy2(args.run / "local/training.jsonl", trial / "local/training.jsonl")
            if mutation.startswith("wrong_adamw"):
                path = destination / "optimizer.pt"
                saved = torch.load(path, map_location="cpu", weights_only=True)
                next(iter(saved["optimizer"]["state"].values()))["step"].add_(1)
                torch.save(saved, path)
                record_path = destination / "checkpoint.json"
                record = json.loads(record_path.read_text())
                record["optimizer_sha256"] = file_hash(path)
                write_json(record_path, record)
                expected_error = "AdamW"
            else:
                path = destination / "evaluation/train.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                mask = rows[0]["reader_mask"]
                index = mask["prelude_tokens"]
                mask["keep_bits"] = mask["keep_bits"][:index] + "1" + mask["keep_bits"][index + 1:]
                import hashlib
                mask["keep_sha256"] = hashlib.sha256(bytes(bit == "1" for bit in mask["keep_bits"])).hexdigest()
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                summary_path = destination / "evaluation/summary.json"
                summary = json.loads(summary_path.read_text())
                summary["file_sha256"]["train.jsonl"] = file_hash(path)
                write_json(summary_path, summary)
                expected_error = "mask/labels"
            try:
                audit_checkpoint(trial, "local", 64, plan, data, encoded)
            except ValueError as error:
                if expected_error not in str(error):
                    raise ValueError(f"Mutation failed for an unexpected reason: {error}") from error
                results[mutation] = {"status": "rejected", "reason": str(error)}
            else:
                raise ValueError(f"Auditor accepted semantic corruption: {mutation}")
    report = {"status": "passed", "source_run": str(args.run), "checkpoint_sha256": file_hash(source / "checkpoint.json"),
              "positive_control_audited": True, "mutations": results, "source_sha256": file_hash(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(results))


if __name__ == "__main__":
    main()
