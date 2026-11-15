"""Lock all fresh cases and complete source snapshots before model inference."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.post_encoding_data import read_plan,reference_inputs,build_registry,select_seeds,sources,DESTINATION,PROTOCOL_SHA256
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json


def main():
    plan=read_plan();reference_inputs();index_path=Path(plan['prior_input_inventory']);index=json.loads(index_path.read_text())
    if DESTINATION.exists():raise ValueError('Preserve prior dataset outputs')
    cases,registry,queries=build_registry(tokenizer_runtime(plan['model_path']),plan,index)
    digests={name:streaming_hash(ROOT/name) for name in sorted(sources())}
    if len(digests)!=206:raise ValueError('All206sources must exist before registration')
    DESTINATION.mkdir(parents=True,exist_ok=False)
    for name,value in (('cases',cases),('registry',registry),('queries',queries),('plan',plan)):
        write_json(DESTINATION/(name+'.json'),value)
    (DESTINATION/'prior_inputs.json').write_bytes(index_path.read_bytes())
    for name in digests:
        out=DESTINATION/'sources'/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((ROOT/name).read_bytes())
    m={'status':'registered','created_utc':datetime.now(timezone.utc).isoformat(),'model_inference':False,'training':False,
        'protocol_sha256':PROTOCOL_SHA256,'seeds':select_seeds(index,plan),'counts':plan['counts'],'source_sha256':digests,
        'payload_sha256':{name+'.json':streaming_hash(DESTINATION/(name+'.json')) for name in ('cases','registry','queries','plan','prior_inputs')},
        'novelty_scope':'Exact seed/identifier/prefix/prompt disjointness from the immutable snapshot of25prior registered dataset directories; not a pretraining-corpus claim.'}
    write_json(DESTINATION/'manifest.json',m)
    print(json.dumps({'status':'fresh_post_encoding_data_registered','seeds':m['seeds'],'source_files':len(digests),'contexts':len(registry['contexts']),
        'questions':len(queries),'primary_readouts':plan['counts']['primary_readouts'],'manifest_sha256':streaming_hash(DESTINATION/'manifest.json')}),flush=True)

if __name__=='__main__':main()
