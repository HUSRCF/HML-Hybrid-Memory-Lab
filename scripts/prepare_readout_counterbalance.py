"""Freeze fresh executable cases and every token mask before model inference."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.counterbalance_data import build_counterbalance_registry, file_digest, locked_training_sources, tokenizer_runtime
from sas_memory.counterbalance_tasks import build_counterbalance_cases, validate_counterbalance_cases
from sas_memory.training_data import data_fingerprint
from scripts.run_controller_trial import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/readout_counterbalance_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    seeds = range(plan["new_base_seeds"]["start_inclusive"], plan["new_base_seeds"]["stop_exclusive"])
    cases = build_counterbalance_cases(seeds, plan["dependency_depths"], plan["question_styles"])
    validation = validate_counterbalance_cases(cases)
    if validation["base_cases"] != plan["base_case_clusters"]:
        raise ValueError("Wrong number of fresh base facts")
    sources = locked_training_sources(plan, ROOT, seeds)
    model_path = json.loads((ROOT / "configs/models.json").read_text())["it"]
    registry = build_counterbalance_registry(tokenizer_runtime(model_path), cases)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "cases.json", cases)
    write_json(args.output / "registry.json", registry)
    write_json(args.output / "plan.json", plan)
    source_paths = sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__).resolve()]
    manifest = {"status": "validated_before_inference", "registered_utc": datetime.now(timezone.utc).isoformat(),
                "plan_sha256": data_fingerprint(plan), "cases_sha256": data_fingerprint(cases),
                "registry_sha256": data_fingerprint(registry), "task_validation": validation,
                "checkpoint_sources": sources, "model_path": model_path,
                "tokenizer_sha256": {name: file_digest(Path(model_path) / name) for name in ("tokenizer.json", "tokenizer_config.json")},
                "max_prompt_tokens": max(row["prompt_tokens"] for row in registry),
                "all_extra_token_budgets_equal": True, "all_swapped_extra_token_positions_equal": True,
                "source_sha256": {str(path.relative_to(ROOT)): file_digest(path) for path in source_paths}}
    for relative in manifest["source_sha256"]:
        path = args.output / "sources" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "cases_sha256", "registry_sha256", "task_validation", "max_prompt_tokens")}))


if __name__ == "__main__":
    main()
