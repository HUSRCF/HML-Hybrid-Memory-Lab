"""Register exact whole-record count-matched masks before native task inference."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json
from scripts.update_budget_data import read_plan,previous_inputs,build_policies,DATA_SOURCES,PROTOCOL,DESTINATION,control_ids

def main():
    plan=read_plan()
    if DESTINATION.exists():raise ValueError('Preserve registered dataset; never overwrite')
    start=datetime.now(timezone.utc).isoformat();cases,registry,queries,oracle,_,refs=previous_inputs(plan)
    from scripts.analyze_gate_components import audit_run
    if audit_run(Path(plan['source_run']))!=json.loads(Path(plan['source_audit']).read_text()):raise ValueError('Previous full audit must reexecute exactly')
    policies,summary=build_policies(queries,oracle,plan['model_path'])
    if len(policies)!=336 or summary['unique_prefixes']!=56 or summary['prefix_budget_groups']!=104:raise ValueError('Complete original coverage required')
    DESTINATION.mkdir(parents=True,exist_ok=False);write_json(DESTINATION/'plan.json',plan);original={}
    for name in ('cases','registry','queries','oracle_policies'):
        src=Path(plan['source_dataset'])/('policies.json' if name=='oracle_policies' else name+'.json')
        target=DESTINATION/(name+'.json');shutil.copy2(src,target);original[target.name]=streaming_hash(src)
    write_json(DESTINATION/'policies.json',policies);write_json(DESTINATION/'mask_summary.json',summary)
    hashes={n:streaming_hash(ROOT/n) for n in sorted(DATA_SOURCES)}
    for n in hashes:
        out=DESTINATION/'sources'/n;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,out)
    m={'status':'registered_before_inference','training':False,'model_inference':False,'generation_started_utc':start,
        'registered_utc':datetime.now(timezone.utc).isoformat(),'previous_full_audit_reexecuted':True,
        'protocol_sha256':streaming_hash(PROTOCOL),'original_file_sha256':original,'references':refs,'counts':plan['counts'],
        'queries_sha256':data_fingerprint(queries),'policies_sha256':data_fingerprint(policies),
        'control_query_ids':control_ids(cases,registry),'mask_summary':summary,'source_sha256':hashes}
    write_json(DESTINATION/'manifest.json',m);print(json.dumps(m,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
