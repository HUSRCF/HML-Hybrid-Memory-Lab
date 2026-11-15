"""Seal complete paired input/control provenance without constructing a model."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.version_query_data import (ROOT,DESTINATION,PROTOCOL_SHA256,read_plan,reference_inputs,sources,build_registry)
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def main():
    if DESTINATION.exists():raise ValueError('Never overwrite a registered packet')
    plan=read_plan();cases,old_registry,old_queries,refs=reference_inputs()
    registry,queries=build_registry(tokenizer_runtime(plan['model_path']),plan,old_registry,old_queries)
    digests={n:streaming_hash(ROOT/n) for n in sorted(sources())}
    if len(digests)!=246:raise ValueError('Complete246source closure required')
    DESTINATION.mkdir(parents=True,exist_ok=False)
    payload={'plan':plan,'cases':cases,'registry':registry,'queries':queries,'references':refs}
    for name,value in payload.items():write_json(DESTINATION/(name+'.json'),value)
    for name in digests:
        target=DESTINATION/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    write_json(DESTINATION/'manifest.json',{'status':'registered','created_utc':datetime.now(timezone.utc).isoformat(),
        'model_inference':False,'training':False,'fresh_confirmation':False,'protocol_sha256':PROTOCOL_SHA256,
        'counts':plan['counts'],'source_sha256':digests,
        'payload_sha256':{n+'.json':streaming_hash(DESTINATION/(n+'.json')) for n in payload}})
    print(json.dumps({'status':'registered_without_inference','sources':246,
        'manifest_sha256':streaming_hash(DESTINATION/'manifest.json'),'new_readouts':1728,'archived_readouts':1728}),flush=True)


if __name__=='__main__':main()
