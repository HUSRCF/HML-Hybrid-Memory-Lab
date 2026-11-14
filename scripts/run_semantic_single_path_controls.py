"""Run GDN-only and Attention-only adapters under the frozen semantic protocol."""

import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from scripts.analyze_order_augmented_semantics import audit_order_evaluation, order_inputs
from scripts.analyze_paired_semantic_probe import identity
from scripts.analyze_semantic_single_path_controls import gated_reference
from scripts.prepare_order_augmented_semantics import registered_augmented_dataset
from scripts.run_controller_trial import write_json
from scripts.run_order_augmented_semantics import evaluate_order
from scripts.run_paired_semantic_controllers import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/semantic_single_path_controls_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.resume:
        raise ValueError("Safety smoke cannot resume scheduled training")
    plan = json.loads(args.plan.read_text())
    references = gated_reference(plan)
    registered, data, registry = registered_augmented_dataset(Path(plan["data_source"]))
    for key in ("data_sha256", "registry_sha256", "tokenized_input_answer_sha256"):
        if plan[key] != registered[key]:
            raise ValueError("Actual component-control data differ from registration")
    if file_hash(Path(plan["data_source"]) / "manifest.json") != plan["data_manifest_sha256"]:
        raise ValueError("Pinned component data manifest changed")
    order_examples, _ = order_inputs(plan)
    if [identity(example) for example in data["validation"]] != [identity(example) for example in order_examples]:
        raise ValueError("Canonical/permuted component-control cohorts differ")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with (args.output / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = args.output / "reference_audits.json"
        if args.resume:
            if json.loads(source.read_text()) != references:
                raise ValueError("Matched augmentation reference audits changed before resume")
        else:
            write_json(source, references)
        extras = [Path(__file__).resolve()] + [ROOT / "scripts" / name for name in (
            "analyze_semantic_single_path_controls.py", "prepare_order_augmented_semantics.py", "analyze_order_augmented_semantics.py",
            "run_order_augmented_semantics.py", "analyze_initial_order_control.py", "prepare_initial_order_control.py", "compare_paired_semantic_controllers.py")]
        execute(args, plan, registered, data, registry, references["native"], extra_sources=extras,
                extra_validation=evaluate_order, extra_validation_audit=audit_order_evaluation)


if __name__ == "__main__":
    main()
