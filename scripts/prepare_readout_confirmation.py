"""Register fixed SET/COPY confirmation only after complete training audit."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.readout_confirmation_data import (read_plan, training_gate, source_seal,
    prior_index, build_inputs, check_novelty, PROTOCOL_SHA256)
from scripts.run_controller_trial import write_json
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from sas_memory.write_locality_data import streaming_hash


def main():
    p = read_plan()
    seal = source_seal()
    # These gates precede even tokenizer construction or future case generation.
    training = training_gate()
    destination = Path(p['dataset'])
    if destination.exists():
        raise ValueError('Never overwrite confirmation registration or resample')
    destination.mkdir(parents=True, exist_ok=False)
    lock = {'locked_utc': datetime.now(timezone.utc).isoformat(), 'training': training}
    write_json(destination / 'training-lock.json', lock)
    write_json(destination / 'plan.json', p)
    for name, digest in seal['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest:
            raise ValueError('Sealed confirmation code changed')
        target = destination / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    runtime = tokenizer_runtime(json.loads((ROOT / p['parent_training_protocol']).read_text())['model_path'])
    index = prior_index(runtime)
    write_json(destination / 'prior-inputs.json', index)
    started = datetime.now(timezone.utc).isoformat()
    inputs = build_inputs(runtime, p['set_seeds'], p['copy_seeds'], index)
    novelty = check_novelty(inputs, index)
    write_json(destination / 'inputs.json', inputs)
    m = {'status': 'registered_fixed_confirmation_before_inference', 'protocol_sha256': PROTOCOL_SHA256,
         'generation_started_utc': started, 'registered_utc': datetime.now(timezone.utc).isoformat(),
         'source_sha256': seal['source_sha256'], 'source_seal_sha256': streaming_hash(Path(p['source_seal'])),
         'counts': p['counts'], 'novelty': novelty, 'model_inference': False, 'training': False,
         'payload_sha256': {name: streaming_hash(destination / name)
                            for name in ('inputs.json', 'training-lock.json', 'prior-inputs.json', 'plan.json')}}
    write_json(destination / 'manifest.json', m)
    print(json.dumps({'status': m['status'], 'novelty': novelty, 'sources': 289,
                      'manifest_sha256': streaming_hash(destination / 'manifest.json')}), flush=True)


if __name__ == '__main__':
    main()
