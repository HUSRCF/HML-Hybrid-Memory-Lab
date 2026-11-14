"""Freeze exactly one preregistered new cohort and all prior-input novelty evidence."""

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.fresh_copy_data import (DATA_SOURCES, DESTINATION, PROTOCOL, read_plan, prior_references,
    prior_index, make_cases, build_registry, make_masks, check_novelty, sample_ids, safety_samples, selected_inputs)
from scripts.run_controller_trial import write_json


def main():
    if DESTINATION.exists(): raise ValueError('Preserve the fixed fresh dataset; no resampling or overwrite')
    plan = read_plan(); _, refs = prior_references(plan)
    started = datetime.now(timezone.utc).isoformat()
    DESTINATION.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'preparing_fixed_registered_cohort', 'generation_started_utc': started,
                'protocol_sha256': streaming_hash(PROTOCOL), 'training': False, 'model_inference': False}
    write_json(DESTINATION / 'manifest.json', manifest)
    try:
        runtime = tokenizer_runtime(plan['model_path'])
        index = prior_index(runtime)
        cases = make_cases(plan); registry = build_registry(cases, runtime)
        novelty = check_novelty(cases, registry, index)
        masks = make_masks(cases, registry, runtime)
        physical, bindings = selected_inputs(registry, True)
        if len(physical) != 18 or len(bindings) != 18 or len(sample_ids(cases, registry)) != 10 or len(safety_samples(registry)) != 6:
            raise ValueError('Prescribed fresh smoke/proof coverage changed')
        manifest.update(status='registered_before_inference', registered_utc=datetime.now(timezone.utc).isoformat(),
            counts=plan['counts'], model_path=plan['model_path'], references=refs,
            cases_sha256=data_fingerprint(cases), registry_sha256=data_fingerprint(registry),
            mask_registry_sha256=data_fingerprint(masks), prior_inputs_sha256=data_fingerprint(index), freshness=novelty,
            sample_query_ids=sample_ids(cases, registry), native_prefix_sample_ids=safety_samples(registry),
            source_sha256={n: streaming_hash(ROOT / n) for n in DATA_SOURCES})
        for name, value in (('plan',plan),('cases',cases),('registry',registry),('masks',masks),('prior_inputs',index)):
            write_json(DESTINATION / f'{name}.json', value)
        for name in DATA_SOURCES:
            target = DESTINATION / 'sources' / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        print(json.dumps({'status': manifest['status'], 'freshness': novelty,
            'mask_registry_sha256': manifest['mask_registry_sha256'],
            'oracle_retained_record_tokens': sorted({m['oracle_registration']['retained_record_tokens'] for m in masks.values()}),
            'removed_record_tokens': sorted({m['removal_registration']['removed_record_tokens'] for m in masks.values()})}), flush=True)
    except BaseException as error:
        manifest.update(status='failed_before_model_inference_no_resampling', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(DESTINATION / 'manifest.json', manifest)


if __name__ == '__main__': main()
