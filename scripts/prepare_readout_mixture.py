"""Register only complete previously observed Gate training inputs after source tests."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.readout_mixture_data import observed_gate, verify_seal, build_inputs, PROTOCOL_SHA256
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def main():
    p = observed_gate(reexecute=True)
    seal = verify_seal()
    path = Path(p['dataset'])
    if path.exists():
        raise ValueError('Never overwrite or regenerate an input packet')
    data = build_inputs(tokenizer_runtime(p['model_path']))
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / 'plan.json', p)
    write_json(path / 'inputs.json', data)
    for name in seal['source_sha256']:
        target = path / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    manifest = {'status': 'registered_observed_mixture_inputs_before_native',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': PROTOCOL_SHA256, 'source_sha256': seal['source_sha256'],
                'source_seal_sha256': streaming_hash(Path(p['source_seal'])),
                'payload_sha256': {n: streaming_hash(path / n) for n in ('plan.json', 'inputs.json')},
                'physical_questions': 1536, 'normal_histories': 256,
                'exposures_per_regime': 1728, 'training': False,
                'future_confirmation_generated': False}
    write_json(path / 'manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'packet': str(path)}), flush=True)


if __name__ == '__main__':
    main()
