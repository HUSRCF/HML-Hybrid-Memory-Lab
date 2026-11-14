"""Gate the unchanged scientific training engine on the complete safety inventory."""

import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_matched_shared_training_timing_verified import audit_run,runtime_sources
from scripts.analyze_order_augmented_semantics import audit_order_evaluation
from scripts.matched_shared_timing_common import PROTOCOL,read_plan,references
from scripts.prepare_order_augmented_semantics import registered_augmented_dataset
from scripts.run_controller_trial import write_json
from scripts.run_matched_shared_training_timing import final_guard
from scripts.run_paired_semantic_controllers import execute


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');parser.add_argument('--safety-audit',type=Path);args=parser.parse_args()
    plan=read_plan();refs=references(plan);path=args.safety_audit or Path(plan['training_smoke_audit']);report=json.loads(path.read_text())
    if (audit_run(Path(plan['training_smoke_output']),True)!=report or
        streaming_hash(path.with_suffix('.source.py'))!=report['analysis_source_sha256']):
        raise ValueError('Independently verified complete146-source safety smoke required')
    args.plan=PROTOCOL;args.smoke=False;args.output=Path(plan['training_output'])
    if args.output.exists() and not args.resume:raise ValueError('Preserve prior matched training')
    registered,data,registry=registered_augmented_dataset(Path(plan['data_source']))
    args.output.mkdir(parents=True,exist_ok=args.resume)
    with (args.output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=args.output/'reference_audits.json'
        if args.resume:
            if json.loads(path.read_text())!=refs:raise ValueError('Matched training references changed')
        else:write_json(path,refs)
        gate={'audit_path':str(args.safety_audit or Path(plan['training_smoke_audit'])),
            'audit_sha256':streaming_hash(args.safety_audit or Path(plan['training_smoke_audit'])),
            'smoke_manifest_sha256':report['manifest_sha256'],
            'source_inventory_addendum_sha256':report['source_inventory_addendum_sha256']}
        if args.resume:
            if json.loads((args.output/'safety_gate.json').read_text())!=gate:raise ValueError('Original verified safety gate changed')
        else:write_json(args.output/'safety_gate.json',gate)
        execute(args,plan,registered,data,registry,refs['native'],extra_sources=[ROOT/n for n in sorted(runtime_sources())],
            extra_validation=final_guard,extra_validation_audit=audit_order_evaluation)


if __name__=='__main__':main()
