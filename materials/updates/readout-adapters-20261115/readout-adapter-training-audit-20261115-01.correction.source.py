"""Extend the frozen training audit with isolated-cache correction provenance."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_readout_adapter_training import audit_run as original_audit
from scripts.readout_adapter_cache_commit import verify_correction_seal, ROOT
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def correction_audit(path, result):
    path = Path(path)
    m = json.loads((path / 'manifest.json').read_text())
    proof = verify_correction_seal()
    if (m['cache_commit_correction'] != proof
        or json.loads((path / 'cache-commit-correction.json').read_text()) != proof
        or not datetime.fromisoformat(proof['created_utc']) < datetime.fromisoformat(m['started_utc'])):
        raise ValueError('Correction seal must precede every corrected native run')
    for name, digest in proof['source_sha256'].items():
        if streaming_hash(path / 'sources' / name) != digest:
            raise ValueError('Corrected282source snapshot changed')
    if m['smoke']:
        for arm in ('query', 'gate', 'query_gate'):
            checks = m['arms'][arm]['safety']['cache_commit_equivalence']
            if set(checks) != {'before_update', 'after_update'}:
                raise ValueError('Actual zero and nonzero correction comparisons required')
            for record in checks.values():
                if (not all(record[k] for k in ('all_logits_bit_exact', 'loss_bit_exact',
                    'all48final_cache_tensors_bit_exact', 'native_no_grad_corrected_with_grad',
                    'corrected_loss_requires_grad'))
                    or any(len(record[k]) != 64 for k in ('logits_sha256', 'loss_sha256'))):
                    raise ValueError('A cache-only correction changed native forward values')
    else:
        gate = json.loads(Path(m['smoke_gate']['path']).read_text())
        correction_audit(Path(gate['source_run']), gate)
    return {'status': 'audited_copy_on_write_teacher_cache_correction', 'source_run': str(path),
            'base_audit': result, 'all_source_files_verified': 282,
            'correction_seal_sha256': proof['seal_sha256'],
            'analysis_source_sha256': streaming_hash(Path(__file__)),
            'native_model_and_fla_kernels_unchanged': True, 'source_caches_unchanged': True}


def audit_run(path, allow_smoke=False):
    result = original_audit(path, allow_smoke)
    correction_audit(path, result)
    # Preserve the original audit schema so its full-run smoke recursion can
    # independently reproduce it. Additional282source proof is a required sidecar.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    side = args.output.with_suffix('.correction.json')
    if args.output.exists() or side.exists():
        raise ValueError('Never overwrite either independent audit')
    result = audit_run(args.run, args.allow_smoke)
    correction = correction_audit(args.run, result)
    write_json(args.output, result)
    write_json(side, correction)
    args.output.with_suffix('.source.py').write_bytes((ROOT / 'scripts/analyze_readout_adapter_training.py').read_bytes())
    args.output.with_suffix('.correction.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status': result['status'], 'verified_sources': 282,
                      'arms': result['arms'], 'correction_status': correction['status']}), flush=True)


if __name__ == '__main__':
    main()
