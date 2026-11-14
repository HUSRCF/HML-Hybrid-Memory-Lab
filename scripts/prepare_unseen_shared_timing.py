"""Generate exactly the registered unseen cohort; never resample on outcomes."""

from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.run_controller_trial import write_json
from scripts.unseen_shared_timing_data import (PROTOCOL,PROTOCOL_SHA256,DESTINATION,DATA_SOURCES,
    read_plan,references,make_cases,build_registry,prior_index,check_novelty,sample_ids,
    safety_samples,selected_inputs,unique_prefix_queries)


def main():
    if DESTINATION.exists():raise ValueError('Preserve the fixed cohort; no overwrite or resampling')
    plan=read_plan();refs=references(plan);DESTINATION.mkdir(parents=True,exist_ok=False)
    manifest={'status':'preparing_fixed_registered_cohort','generation_started_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':PROTOCOL_SHA256,'training':False,'model_inference':False}
    write_json(DESTINATION/'manifest.json',manifest)
    try:
        runtime=tokenizer_runtime(plan['model_path']);index=prior_index(runtime,excluded=DESTINATION)
        cases=make_cases(plan);registry=build_registry(cases,runtime);novelty=check_novelty(cases,registry,index)
        smoke,bindings=selected_inputs(registry,True);physical,_=selected_inputs(registry)
        if len(smoke)!=18 or len(bindings)!=18 or len(sample_ids(cases,registry))!=10 or len(safety_samples(registry))!=6:
            raise ValueError('All-role/variant/depth/style operational smoke coverage changed')
        for name,value in [('plan',plan),('cases',cases),('registry',registry),('prior_inputs',index)]:write_json(DESTINATION/f'{name}.json',value)
        manifest.update(status='registered_before_inference',registered_utc=datetime.now(timezone.utc).isoformat(),
            counts=plan['counts'],model_path=plan['model_path'],references=refs,
            cases_sha256=data_fingerprint(cases),registry_sha256=data_fingerprint(registry),prior_inputs_sha256=data_fingerprint(index),
            freshness=novelty,sample_query_ids=sample_ids(cases,registry),native_prefix_sample_ids=safety_samples(registry),
            all_distinct_prefix_ids=[q['source_key'] for q in unique_prefix_queries(physical)],
            source_sha256={name:streaming_hash(ROOT/name) for name in sorted(DATA_SOURCES)})
        for name in manifest['source_sha256']:
            target=DESTINATION/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
        print(json.dumps({'status':manifest['status'],'freshness':novelty,'distinct_ledger_prefixes':len(manifest['all_distinct_prefix_ids']),
            'runtime_sources':len(DATA_SOURCES),'registry_sha256':manifest['registry_sha256']}),flush=True)
    except BaseException as error:
        manifest.update(status='failed_before_model_inference_no_resampling',error=f'{type(error).__name__}: {error}');raise
    finally:write_json(DESTINATION/'manifest.json',manifest)


if __name__=='__main__':main()
