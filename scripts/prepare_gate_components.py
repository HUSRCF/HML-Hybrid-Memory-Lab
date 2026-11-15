"""Copy exact original inputs after registering the gate-component diagnostic."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from sas_memory.training_data import data_fingerprint
from scripts.run_controller_trial import write_json
from scripts.gate_component_data import (read_plan,previous_inputs,DATA_SOURCES,PROTOCOL,DESTINATION,control_ids)

def main():
    p=read_plan()
    if DESTINATION.exists():raise ValueError('Never overwrite a registered dataset')
    start=datetime.now(timezone.utc).isoformat();cases,registry,queries,policies,_,refs=previous_inputs(p)
    from scripts.analyze_oracle_write_hold import audit_run
    if audit_run(Path(p['source_run']))!=json.loads(Path(p['source_audit']).read_text()):raise ValueError('Preceding full audit no longer reproduces')
    DESTINATION.mkdir(parents=True,exist_ok=False);write_json(DESTINATION/'plan.json',p)
    original={}
    for name in ('cases.json','registry.json','queries.json','policies.json'):
        src=Path(p['source_dataset'])/name;shutil.copy2(src,DESTINATION/name);original[name]=streaming_hash(src)
    hashes={n:streaming_hash(ROOT/n) for n in sorted(DATA_SOURCES)}
    for n in hashes:
        target=DESTINATION/'sources'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,target)
    m={'status':'registered_before_inference','training':False,'model_inference':False,'copy_started_utc':start,
        'registered_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':streaming_hash(PROTOCOL),
        'previous_full_audit_reexecuted':True,'original_file_sha256':original,'references':refs,'counts':p['counts'],
        'queries_sha256':data_fingerprint(queries),'policies_sha256':data_fingerprint(policies),
        'control_query_ids':control_ids(cases,registry),'source_sha256':hashes}
    write_json(DESTINATION/'manifest.json',m);print(json.dumps(m,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
