"""Register reused inputs/endpoints and all214sources before crossover inference."""
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.post_tail_path_data import read_plan,reference_inputs,sources,DESTINATION,PROTOCOL_SHA256
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json


def main():
    plan=read_plan()
    if DESTINATION.exists():raise ValueError('Never overwrite a registered packet')
    values=reference_inputs();digests={n:streaming_hash(ROOT/n) for n in sorted(sources())}
    if len(digests)!=214:raise ValueError('All214sources required before registration')
    DESTINATION.mkdir(parents=True,exist_ok=False)
    for name,value in zip(('cases','registry','queries','endpoints','references'),values):write_json(DESTINATION/(name+'.json'),value)
    write_json(DESTINATION/'plan.json',plan)
    for name in digests:
        target=DESTINATION/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    m={'status':'registered','created_utc':datetime.now(timezone.utc).isoformat(),'model_inference':False,'training':False,'fresh_confirmation':False,
        'protocol_sha256':PROTOCOL_SHA256,'counts':plan['counts'],'source_sha256':digests,
        'payload_sha256':{name+'.json':streaming_hash(DESTINATION/(name+'.json')) for name in ('cases','registry','queries','endpoints','references','plan')}}
    write_json(DESTINATION/'manifest.json',m)
    print(json.dumps({'status':'registered_reused_crossover_packet','source_files':214,'primary_readouts':4608,
        'old_endpoint_references':1536,'new_inference':False,'manifest_sha256':streaming_hash(DESTINATION/'manifest.json')}),flush=True)

if __name__=='__main__':main()
