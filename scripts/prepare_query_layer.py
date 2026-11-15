"""Seal all reused inputs and230sources without native model inference."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.query_layer_data import read_plan, reference_inputs, sources, DESTINATION, PROTOCOL_SHA256
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def main():
    if DESTINATION.exists():
        raise ValueError('Never overwrite a registered packet')
    plan = read_plan()
    values = reference_inputs()
    digests = {name: streaming_hash(ROOT / name) for name in sorted(sources())}
    if len(digests) != 230:
        raise ValueError('Complete230source closure required')
    DESTINATION.mkdir(parents=True, exist_ok=False)
    names = ('cases', 'registry', 'queries', 'references')
    for name, value in zip(names, values):
        write_json(DESTINATION / (name + '.json'), value)
    write_json(DESTINATION / 'plan.json', plan)
    for name in digests:
        target = DESTINATION / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    m = {'status': 'registered', 'created_utc': datetime.now(timezone.utc).isoformat(),
         'model_inference': False, 'training': False, 'fresh_confirmation': False,
         'protocol_sha256': PROTOCOL_SHA256, 'counts': plan['counts'], 'source_sha256': digests,
         'payload_sha256': {name + '.json': streaming_hash(DESTINATION / (name + '.json')) for name in (*names, 'plan')}}
    write_json(DESTINATION / 'manifest.json', m)
    print(json.dumps({'status': 'reused_single_layer_packet_registered_without_inference',
        'sources': 230, 'manifest_sha256': streaming_hash(DESTINATION / 'manifest.json'),
        'primary_readouts': 4608, 'archived_full_references': 1536}), flush=True)


if __name__ == '__main__':
    main()
