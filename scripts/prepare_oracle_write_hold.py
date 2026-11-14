"""Register fixed original tokens and oracle hold positions before inference."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json
from scripts.oracle_write_hold_data import (read_plan,previous_inputs,build_policies,DESTINATION,
    DATA_SOURCES,PROTOCOL,control_ids)

def main():
    p=read_plan()
    if DESTINATION.exists():raise ValueError('Never overwrite a registered dataset')
    start=datetime.now(timezone.utc).isoformat()
    cases,registry,queries,masks,_,refs=previous_inputs(p)
    from scripts.analyze_text_competence_factorial import audit_run
    report=audit_run(Path(p['source_run']))
    if report!=json.loads(Path(p['source_audit']).read_text()):raise ValueError('Previous full audit does not reproduce')
    policies=build_policies(queries,masks)
    DESTINATION.mkdir(parents=True,exist_ok=False)
    for filename,value in [('plan.json',p),('cases.json',cases),('registry.json',registry),('queries.json',queries),('policies.json',policies)]:
        write_json(DESTINATION/filename,value)
    source_sha={n:streaming_hash(ROOT/n) for n in sorted(DATA_SOURCES)}
    for n in source_sha:
        out=DESTINATION/'sources'/n;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((ROOT/n).read_bytes())
    m={'status':'registered_before_inference','training':False,'model_inference':False,
        'generation_started_utc':start,'registered_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':streaming_hash(PROTOCOL),'previous_full_audit_reexecuted':True,
        'queries_sha256':data_fingerprint(queries),'policies_sha256':data_fingerprint(policies),
        'counts':p['counts'],'control_query_ids':control_ids(cases,registry),'references':refs,'source_sha256':source_sha,
        'held_token_range':[min(x['held_token_count'] for x in policies.values()),max(x['held_token_count'] for x in policies.values())]}
    write_json(DESTINATION/'manifest.json',m);print(json.dumps(m,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
