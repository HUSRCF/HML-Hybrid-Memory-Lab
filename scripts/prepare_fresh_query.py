"""Seal fresh inputs and all222measurement sources without model inference."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fresh_query_data import (read_plan,prior_inventory,reference_inputs,build_registry,select_seeds,
    tokenizer_runtime,sources,DESTINATION,PROTOCOL_SHA256)
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def main():
    if DESTINATION.exists():raise ValueError('Never overwrite a registered dataset')
    p=read_plan();index=prior_inventory();refs=reference_inputs()
    cases,registry,queries=build_registry(tokenizer_runtime(p['model_path']),p,index)
    digests={name:streaming_hash(ROOT/name) for name in sorted(sources())}
    if len(digests)!=222:raise ValueError('All222sources required')
    DESTINATION.mkdir(parents=True,exist_ok=False)
    for name,value in (('plan',p),('prior_inputs',index),('cases',cases),('registry',registry),('queries',queries),('references',refs)):
        write_json(DESTINATION/(name+'.json'),value)
    for name in digests:
        target=DESTINATION/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    m={'status':'registered','created_utc':datetime.now(timezone.utc).isoformat(),'model_inference':False,'training':False,'fresh_streams':True,
        'protocol_sha256':PROTOCOL_SHA256,'seeds':select_seeds(index,p),'counts':p['counts'],'source_sha256':digests,
        'payload_sha256':{name+'.json':streaming_hash(DESTINATION/(name+'.json')) for name in ('plan','prior_inputs','cases','registry','queries','references')}}
    write_json(DESTINATION/'manifest.json',m)
    print(json.dumps({'status':'fresh_query_inputs_registered_without_inference','manifest_sha256':streaming_hash(DESTINATION/'manifest.json'),
        'sources':222,'seeds':m['seeds'],'contexts':len(registry['contexts']),'questions':len(queries),'primary_readouts':3264}),flush=True)


if __name__=='__main__':main()
