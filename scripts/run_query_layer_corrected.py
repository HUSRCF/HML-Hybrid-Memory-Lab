"""Audit-only gate correction; original sealed model execution is unchanged."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_query_layer as original
from scripts.query_layer_data import DESTINATION
from sas_memory.write_locality_data import streaming_hash

CORRECTION = ROOT / 'configs/query_layer_audit_correction.json'
CORRECTION_SHA256 = '471837b3a109dc5183431ab2d069441e907dd3b447b60a693bc59057664b7c91'
SEAL = Path('/media/990Pro/Code/SAS/artifacts/memory_update/query-layer-audit-correction-seal-20261115-01.json')


def correction_descriptor(require_seal=True):
    if streaming_hash(CORRECTION) != CORRECTION_SHA256:
        raise ValueError('Registered audit-only correction changed')
    p = json.loads(CORRECTION.read_text())
    if (p['measurement_changes'] or p['new_accuracy_gate'] or p['tolerances_changed'] or
            p['scientific_outcomes_inspected_before_correction'] or len(p['additional_sources']) != 4 or
            streaming_hash(Path(p['diagnosis'])) != p['diagnosis_sha256'] or
            streaming_hash(Path(p['failed_audit_log'])) != p['failed_audit_log_sha256'] or
            streaming_hash(Path(p['base_protocol'])) != p['base_protocol_sha256'] or
            streaming_hash(Path(p['base_packet']) / 'manifest.json') != p['base_packet_manifest_sha256'] or
            streaming_hash(Path(p['allowlisted_pre_correction_run']) / 'manifest.json') != p['allowlisted_manifest_sha256']):
        raise ValueError('Preserve original failure, registration and completed forward run')
    base = json.loads((Path(p['base_packet']) / 'manifest.json').read_text())
    for name, digest in base['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest:
            raise ValueError('Keep all230base sources unchanged')
    descriptor = {'profile': p['profile'], 'protocol_sha256': CORRECTION_SHA256,
                  'base_sources_unchanged': 230,
                  'source_sha256': {n: streaming_hash(ROOT / n) for n in p['additional_sources']}}
    if require_seal:
        seal = json.loads(SEAL.read_text())
        if (seal['status'] != 'sealed_audit_only_correction_before_further_native_inference' or
                seal['descriptor'] != descriptor or not datetime.fromisoformat(p['registered_utc']) < datetime.fromisoformat(seal['created_utc'])):
            raise ValueError('Exact correction source seal required before further native inference')
    return descriptor


def check_annotation(manifest, path, allow_unsealed=False):
    p = json.loads(CORRECTION.read_text())
    descriptor = correction_descriptor(not allow_unsealed)
    if (str(path) == p['allowlisted_pre_correction_run'] and
            streaming_hash(path / 'manifest.json') == p['allowlisted_manifest_sha256']):
        if not manifest['smoke'] or manifest['reverse_model_order'] or manifest['status'] != 'completed':
            raise ValueError('Only exact completed forward smoke can predate correction')
    else:
        if manifest.get('audit_correction') != descriptor:
            raise ValueError('Every subsequent run must declare complete audit correction provenance')
        seal = json.loads(SEAL.read_text())
        if not datetime.fromisoformat(seal['created_utc']) < datetime.fromisoformat(manifest['started_utc']):
            raise ValueError('Correction must be sealed before subsequent native execution')
        for name, digest in descriptor['source_sha256'].items():
            if streaming_hash(path / 'audit-correction-sources' / name) != digest:
                raise ValueError('Actual correction source snapshot differs')
    return descriptor


def checked_gate(path, digest):
    from scripts.analyze_query_layer_corrected import audit_equivalence
    p = Path(path)
    g = json.loads(p.read_text())
    if (g != audit_equivalence(Path(g['forward_audit']), Path(g['reverse_audit'])) or
            g['packet_manifest_sha256'] != digest or
            streaming_hash(p.with_suffix('.source.py')) != g['analysis_source_sha256']):
        raise ValueError('Exact independently audited opposite288readout smokes required')
    return {'path': str(p), 'sha256': streaming_hash(p)}


def execute_run(args):
    descriptor = correction_descriptor()
    previous_gate, previous_writer = original.checked_gate, original.write_json
    def annotated_write(path, value):
        if Path(path) == args.output / 'manifest.json':
            value['audit_correction'] = descriptor
            for name, digest in descriptor['source_sha256'].items():
                source, target = ROOT / name, args.output / 'audit-correction-sources' / name
                if streaming_hash(source) != digest:
                    raise ValueError('Correction source changed during execution')
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if streaming_hash(target) != digest: raise ValueError('Preserve previous correction snapshot')
                else: target.write_bytes(source.read_bytes())
        return previous_writer(path, value)
    try:
        original.checked_gate, original.write_json = checked_gate, annotated_write
        return original.execute_run(args)
    finally:
        original.checked_gate, original.write_json = previous_gate, previous_writer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, default=DESTINATION)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--reverse-model-order', action='store_true')
    parser.add_argument('--smoke-equivalence', type=Path)
    execute_run(parser.parse_args())


if __name__ == '__main__':
    main()
