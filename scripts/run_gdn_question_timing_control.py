"""Run paired-initialization GDN adapters only during question/answer processing."""

import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.calibration import file_hash
from scripts.analyze_gdn_question_timing_control import gated_references
from scripts.analyze_order_augmented_semantics import audit_order_evaluation, order_inputs
from scripts.analyze_paired_semantic_probe import identity
from scripts.prepare_order_augmented_semantics import registered_augmented_dataset
from scripts.run_controller_trial import write_json
from scripts.run_order_augmented_semantics import evaluate_order
from scripts.run_paired_semantic_controllers import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,default=ROOT/'configs/gdn_question_timing_control_plan.json')
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    if args.smoke and args.resume:
        raise ValueError("Safety smoke cannot resume scheduled training")
    plan = json.loads(args.plan.read_text())
    started = time.monotonic()
    references = gated_references(plan)
    print(json.dumps({"event":"timing_reference_gates_passed","cpu_threads":4,"seconds":time.monotonic()-started}),flush=True)
    registered,data,registry = registered_augmented_dataset(Path(plan["data_source"]))
    if (any(plan[key] != registered[key] for key in ("data_sha256","registry_sha256","tokenized_input_answer_sha256")) or
        file_hash(Path(plan["data_source"])/"manifest.json") != plan["data_manifest_sha256"]):
        raise ValueError("Timing data differ from registration")
    examples,_ = order_inputs(plan)
    if [identity(e) for e in examples] != [identity(e) for e in data["validation"]]:
        raise ValueError("Timing canonical/permuted cohorts differ")
    args.output.mkdir(parents=True,exist_ok=args.resume)
    with (args.output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = args.output/'reference_audits.json'
        if args.resume:
            if json.loads(path.read_text()) != references:
                raise ValueError("Timing reference audits changed before resume")
        else:
            write_json(path,references)
        relative_sources = {"scripts/run_gdn_question_timing_control.py","scripts/analyze_gdn_question_timing_control.py"}
        for field in ("component_reference_run","native_path_reference_run"):
            relative_sources.update(json.loads((Path(plan[field])/'manifest.json').read_text())["source_sha256"])
        execute(args,plan,registered,data,registry,references['native'],extra_sources=[ROOT/relative for relative in sorted(relative_sources)],
                extra_validation=evaluate_order,extra_validation_audit=audit_order_evaluation)


if __name__ == '__main__':
    main()
