"""Freeze complete program-executed factorial text before any new model inference."""

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json
from scripts.text_competence_factorial_data import (PROTOCOL,DESTINATION,DATA_SOURCES,read_plan,source_inputs,
    build_transformations,collision_registry,control_ids,queries_for,unique_prefixes,REGIMES,tokenizer_runtime)


def prepare(output):
    plan=read_plan()
    if output!=DESTINATION or output.exists():raise ValueError('New registered destination required; no overwrite/resampling')
    started=datetime.now(timezone.utc).isoformat();cases,registry,_,refs=source_inputs(plan)
    # Reexecute the complete previous independent audit, without historical novelty rescans.
    from scripts.analyze_trained_timing_evidence_access import audit_run
    if audit_run(Path(plan['source_run']))!=json.loads(Path(plan['source_audit']).read_text()):
        raise ValueError('Previous full5040-readout audit no longer reproduces')
    transforms=build_transformations(cases,registry,tokenizer_runtime(plan['model_path']))
    collisions=collision_registry(transforms,registry)
    if len(DATA_SOURCES)!=169:raise ValueError('Complete169-file data closure required')
    output.mkdir(parents=True,exist_ok=False)
    for name,value in [('plan.json',plan),('cases.json',cases),('registry.json',registry),
                       ('transformations.json',transforms),('collisions.json',collisions)]:write_json(output/name,value)
    hashes={n:streaming_hash(ROOT/n) for n in sorted(DATA_SOURCES)}
    for name in hashes:
        target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    manifest={'status':'registered_before_inference','generation_started_utc':started,'registered_utc':datetime.now(timezone.utc).isoformat(),
        'training':False,'model_inference':False,'protocol_sha256':streaming_hash(PROTOCOL),'counts':plan['counts'],
        'references':refs,'cases_sha256':data_fingerprint(cases),'registry_sha256':data_fingerprint(registry),
        'transformations_sha256':data_fingerprint(transforms),'collisions_sha256':data_fingerprint(collisions),
        'control_query_ids':control_ids(cases,registry),'source_sha256':hashes,'complete_previous_audit_reexecuted':True,
        'distinct_prefixes_per_regime':{r:len(unique_prefixes(queries_for(transforms,cases,registry,r))) for r in REGIMES}}
    write_json(output/'manifest.json',manifest)
    return {'status':manifest['status'],'transformed_registrations':4*len(transforms),'sources':len(hashes),
            'unique_prompts':{r:v['unique_prompts'] for r,v in collisions['by_regime'].items()},
            'prefix_counts':manifest['distinct_prefixes_per_regime']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DESTINATION)
    args=parser.parse_args();print(json.dumps(prepare(args.output)),flush=True)


if __name__=='__main__':main()
