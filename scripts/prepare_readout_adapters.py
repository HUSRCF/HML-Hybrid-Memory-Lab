"""Register observed adaptation inputs without constructing a model."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.readout_adapter_data import PROTOCOL_SHA256, load_sources, sources, build_examples
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def main():
    p, old, cases, registry, queries, rows = load_sources()
    destination = Path(p['registration_destination'])
    if destination.exists():
        raise ValueError('Never overwrite fit registration')
    examples = build_examples(tokenizer_runtime(p['model_path']), cases, registry, queries)
    digests = {name: streaming_hash(ROOT / name) for name in sorted(sources())}
    if len(digests) != 277:
        raise ValueError('Complete277source closure required')
    newline = {e['answer_ids'][-1] for e in examples}
    if len(newline) != 1:
        raise ValueError('Canonical four-token answers required without filtering')
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / 'examples.json', examples)
    write_json(destination / 'plan.json', p)
    for name in digests:
        target = destination / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    m = {'status': 'registered_observed_fit_inputs_before_training',
         'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': PROTOCOL_SHA256,
         'source_manifest_sha256': p['source_manifest_sha256'], 'source_sha256': digests,
         'payload_sha256': {name: streaming_hash(destination / name) for name in ('examples.json', 'plan.json')},
         'examples': len(examples), 'newline_token_id': next(iter(newline)),
         'training': False, 'model_inference': False, 'all_observed_inputs_are_fit_data': True,
         'new_confirmation_inputs_generated': False}
    write_json(destination / 'manifest.json', m)
    print(json.dumps({'status': m['status'], 'examples': len(examples), 'sources': 277,
                      'packet_manifest_sha256': streaming_hash(destination / 'manifest.json')}), flush=True)


if __name__ == '__main__':
    main()
