"""Lock all completed training dependencies before constructing fixed new cohorts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.mixture_confirmation_data import (read_plan, source_seal, training_gate, observed_inputs,
    build_inputs, prior_index, check_novelty, PROTOCOL_SHA256)
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def main():
    p = read_plan()
    seal = source_seal()
    # Complete audited weights must precede tokenizer construction and all new cases.
    training = training_gate()
    observed, _ = observed_inputs()
    destination = Path(p['dataset'])
    if destination.exists():
        raise ValueError('Never overwrite or resample confirmation')
    destination.mkdir(parents=True, exist_ok=False)
    lock = {'locked_utc': datetime.now(timezone.utc).isoformat(), 'training': training}
    write_json(destination / 'training-lock.json', lock)
    write_json(destination / 'plan.json', p)
    for name in seal['source_sha256']:
        target = destination / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    base = json.loads((ROOT / 'configs/readout_adapter_training_plan.json').read_text())
    runtime = tokenizer_runtime(base['model_path'])
    index = prior_index(runtime, excluded=destination)
    write_json(destination / 'prior-inputs.json', index)
    started = datetime.now(timezone.utc).isoformat()
    fresh = build_inputs(runtime, p['set_seeds'], p['copy_seeds'], index)
    novelty = check_novelty(fresh, index)
    write_json(destination / 'inputs.json', {'observed': observed, 'fresh': fresh})
    m = {'status': 'registered_mixture_confirmation_after_complete_training',
         'protocol_sha256': PROTOCOL_SHA256, 'source_sha256': seal['source_sha256'],
         'source_seal_sha256': streaming_hash(Path(p['source_seal'])),
         'generation_started_utc': started, 'registered_utc': datetime.now(timezone.utc).isoformat(),
         'novelty': novelty, 'counts': p['counts'], 'training': False, 'model_inference': False,
         'payload_sha256': {name: streaming_hash(destination / name)
                            for name in ('plan.json', 'training-lock.json', 'inputs.json', 'prior-inputs.json')}}
    write_json(destination / 'manifest.json', m)
    print(json.dumps({'status': m['status'], 'novelty': novelty, 'sources': 303}), flush=True)


if __name__ == '__main__':
    main()
