"""Serialization-only proof comparison; preserve all246sealed execution sources."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_version_query as original
from scripts.version_query_data import DESTINATION
from sas_memory.write_locality_data import streaming_hash

CORRECTION=ROOT/'configs/version_query_serialization_correction.json'
CORRECTION_SHA256='b7cbf4942851666eb829a884c36444518963720529d779e14d37addfe7e61856'
SEAL=Path('/media/990Pro/Code/SAS/artifacts/memory_update/version-query-serialization-seal-20261115-01.json')
BASE_VERIFY=original.verify_model_sources


def correction_descriptor(require_seal=True):
    if streaming_hash(CORRECTION)!=CORRECTION_SHA256:raise ValueError('Registered serialization correction changed')
    p=json.loads(CORRECTION.read_text());failed=Path(p['failed_run'])
    if (any(p[k] for k in ('measurement_changes','new_accuracy_gate','tolerances_changed','scientific_outcomes_inspected_before_correction')) or
        p['base_sources_immutable']!=246 or len(p['additional_sources'])!=4 or p['total_verified_sources']!=250 or
        streaming_hash(Path(p['diagnosis']))!=p['diagnosis_sha256'] or
        streaming_hash(failed/'manifest.json')!=p['failed_manifest_sha256'] or
        streaming_hash(failed/'native.jsonl')!=p['failed_partial_readout_sha256'] or
        streaming_hash(Path(p['failed_log']))!=p['failed_log_sha256'] or
        streaming_hash(ROOT/p['base_protocol'])!=p['base_protocol_sha256'] or
        streaming_hash(Path(p['base_packet'])/'manifest.json')!=p['base_packet_manifest_sha256']):
        raise ValueError('Preserve failed partial smoke, diagnosis and registration')
    failed_manifest=json.loads((failed/'manifest.json').read_text())
    base=json.loads((Path(p['base_packet'])/'manifest.json').read_text())
    if (failed_manifest['status']!='failed' or failed_manifest['completed_models'] or
        list(failed_manifest['models'])!=['native'] or failed_manifest['models']['native']['completed_primary']!=54 or
        len(base['source_sha256'])!=246 or failed_manifest['source_sha256']!=base['source_sha256']):
        raise ValueError('Preserve the original54readout native failure')
    for name,digest in base['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(failed/'sources'/name)!=digest:
            raise ValueError('Keep all246base sources unchanged')
    descriptor={'profile':p['profile'],'protocol_sha256':CORRECTION_SHA256,'base_sources_unchanged':246,
        'source_sha256':{name:streaming_hash(ROOT/name) for name in p['additional_sources']}}
    if require_seal:
        seal=json.loads(SEAL.read_text())
        if (seal['status']!='sealed_serialization_only_correction_before_further_native_inference' or
            seal['descriptor']!=descriptor or not datetime.fromisoformat(p['registered_utc'])<datetime.fromisoformat(seal['created_utc'])):
            raise ValueError('Exact correction seal required before further inference')
    return descriptor


def verify_model_sources(state,reference):
    """Normalize only JSON layer-key representation; do not change any source."""
    normalized={**state,'sources':{}}
    for key,source in state['sources'].items():
        proof=source['tail_proof'];gates=proof['gates'];layers={}
        for index,stat in gates['layers'].items():
            if type(index) not in (int,str):raise ValueError('Unsupported layer-key type')
            canonical=str(index)
            if canonical in layers:raise ValueError('Colliding layer keys')
            layers[canonical]=stat
        normalized['sources'][key]={**source,'tail_proof':{**proof,'gates':{**gates,'layers':layers}}}
    return BASE_VERIFY(normalized,reference)


def check_annotation(manifest,path,allow_unsealed=False):
    descriptor=correction_descriptor(not allow_unsealed)
    if manifest.get('serialization_correction')!=descriptor:raise ValueError('Every corrected run must declare source provenance')
    if not allow_unsealed:
        seal=json.loads(SEAL.read_text())
        if not datetime.fromisoformat(seal['created_utc'])<datetime.fromisoformat(manifest['started_utc']):
            raise ValueError('Correction seal must precede model execution')
    for name,digest in descriptor['source_sha256'].items():
        if streaming_hash(path/'serialization-correction-sources'/name)!=digest:
            raise ValueError('Actual correction snapshot differs')
    return descriptor


def checked_gate(path,packet_digest):
    from scripts.analyze_version_query_corrected import audit_equivalence
    path=Path(path);gate=json.loads(path.read_text())
    if (gate!=audit_equivalence(Path(gate['forward_run']),Path(gate['reverse_run'])) or
        gate['packet_manifest_sha256']!=packet_digest or
        streaming_hash(path.with_suffix('.source.py'))!=gate['analysis_source_sha256']):
        raise ValueError('Exact independently audited corrected opposite-order smokes required')
    return {'path':str(path),'sha256':streaming_hash(path)}


def execute_run(args):
    descriptor=correction_descriptor()
    saved={key:getattr(original,key) for key in ('verify_model_sources','checked_gate','write_json')}
    def annotated_write(path,value):
        if Path(path)==args.output/'manifest.json':
            value['serialization_correction']=descriptor
            for name,digest in descriptor['source_sha256'].items():
                source=ROOT/name;target=args.output/'serialization-correction-sources'/name
                if streaming_hash(source)!=digest:raise ValueError('Correction source changed during execution')
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if streaming_hash(target)!=digest:raise ValueError('Preserve correction snapshot bytes')
                else:target.write_bytes(source.read_bytes())
        return saved['write_json'](path,value)
    try:
        original.verify_model_sources=verify_model_sources;original.checked_gate=checked_gate;original.write_json=annotated_write
        return original.execute_run(args)
    finally:
        for key,value in saved.items():setattr(original,key,value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,default=DESTINATION);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--reverse-model-order',action='store_true')
    parser.add_argument('--smoke-equivalence',type=Path)
    execute_run(parser.parse_args())


if __name__=='__main__':main()
