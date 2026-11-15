"""Seal all normal-update inputs and240sources without constructing a model."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.natural_load_data import (read_plan, reference_inputs, sources, build_registry,
    ROOT, DESTINATION, PROTOCOL_SHA256)
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def main():
    if DESTINATION.exists():
        raise ValueError('Never overwrite a registered natural-load packet')
    p = read_plan(); refs = reference_inputs()
    values = build_registry(tokenizer_runtime(p['model_path']), p, refs)
    digests = {n: streaming_hash(ROOT/n) for n in sorted(sources())}
    if len(digests) != 240:
        raise ValueError('Complete240source closure required')
    DESTINATION.mkdir(parents=True, exist_ok=False)
    payload = dict(zip(('cases', 'registry', 'queries'), values)); payload.update(plan=p, references=refs)
    for name, value in payload.items():
        write_json(DESTINATION/(name+'.json'), value)
    for name in digests:
        target = DESTINATION/'sources'/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    write_json(DESTINATION/'manifest.json', {'status':'registered', 'created_utc':datetime.now(timezone.utc).isoformat(),
        'model_inference':False, 'training':False, 'protocol_sha256':PROTOCOL_SHA256, 'counts':p['counts'],
        'source_sha256':digests, 'payload_sha256':{n+'.json':streaming_hash(DESTINATION/(n+'.json')) for n in payload}})
    print(json.dumps({'status':'registered_without_inference', 'sources':240,
        'manifest_sha256':streaming_hash(DESTINATION/'manifest.json'), 'readouts':3456}), flush=True)


if __name__ == '__main__':
    main()
