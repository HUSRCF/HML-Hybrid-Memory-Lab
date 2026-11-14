"""Freeze all original-cohort oracle/nonslice masks before inference."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json
from scripts.trained_timing_evidence_access_data import (PROTOCOL,PROTOCOL_SHA256,DESTINATION,DATA_SOURCES,
    read_plan,source_inputs,build_masks,sample_ids,safety_samples,selected_inputs,tokenizer_runtime)


def main():
    if DESTINATION.exists():raise ValueError('Preserve mask registration; no overwrite/resampling')
    plan=read_plan();cases,registry,rows,refs=source_inputs(plan);DESTINATION.mkdir(parents=True,exist_ok=False)
    m={'status':'preparing_fixed_masks','generation_started_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':PROTOCOL_SHA256,'training':False,'model_inference':False}
    write_json(DESTINATION/'manifest.json',m)
    try:
        masks=build_masks(cases,registry,tokenizer_runtime(plan['model_path']));physical,bindings=selected_inputs(registry);smoke,smoke_bindings=selected_inputs(registry,True)
        if (len(physical),len(bindings),len(smoke),len(smoke_bindings),len(masks),len(sample_ids(cases,registry)))!=(336,432,18,18,336,10):raise ValueError('Complete fixed cohort or safety controls changed')
        for name,value in [('plan',plan),('masks',masks)]:write_json(DESTINATION/f'{name}.json',value)
        m.update(status='registered_before_inference',registered_utc=datetime.now(timezone.utc).isoformat(),counts=plan['counts'],
            references=refs,cases_sha256=data_fingerprint(cases),registry_sha256=data_fingerprint(registry),
            mask_registry_sha256=data_fingerprint(masks),sample_query_ids=sample_ids(cases,registry),
            source_sha256={name:streaming_hash(ROOT/name) for name in sorted(DATA_SOURCES)})
        for name in m['source_sha256']:
            p=DESTINATION/'sources'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
        print(json.dumps({'status':m['status'],'masks':len(masks),'data_sources':len(DATA_SOURCES),'mask_registry_sha256':m['mask_registry_sha256']}),flush=True)
    except BaseException as error:m.update(status='failed_before_model_inference',error=f'{type(error).__name__}: {error}');raise
    finally:write_json(DESTINATION/'manifest.json',m)


if __name__=='__main__':main()
