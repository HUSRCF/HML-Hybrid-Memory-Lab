"""Seal the complete reused/fresh grid without constructing or loading a model."""
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.version_address_data import ROOT,DESTINATION,PROTOCOL_SHA256,read_plan,build_inputs,sources
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def main():
    if DESTINATION.exists():raise ValueError('Never overwrite input registration')
    plan=read_plan();cases,registry,queries,refs=build_inputs(tokenizer_runtime(plan['model_path']))
    digests={n:streaming_hash(ROOT/n) for n in sorted(sources())}
    if len(digests)!=268:raise ValueError('Full268source closure required')
    DESTINATION.mkdir(parents=True,exist_ok=False)
    payload={'plan':plan,'cases':cases,'registry':registry,'queries':queries,'references':refs}
    for name,value in payload.items():write_json(DESTINATION/(name+'.json'),value)
    for name in digests:
        target=DESTINATION/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    manifest={'status':'registered','created_utc':datetime.now(timezone.utc).isoformat(),'model_inference':False,'training':False,
        'protocol_sha256':PROTOCOL_SHA256,'counts':plan['counts'],'source_sha256':digests,
        'payload_sha256':{n+'.json':streaming_hash(DESTINATION/(n+'.json')) for n in payload}}
    write_json(DESTINATION/'manifest.json',manifest)
    print(json.dumps({'status':'registered_without_inference','sources':268,'questions':1728,
        'manifest_sha256':streaming_hash(DESTINATION/'manifest.json')}),flush=True)


if __name__=='__main__':main()
