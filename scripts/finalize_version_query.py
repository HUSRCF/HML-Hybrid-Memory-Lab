"""Pure CPU validation recovery; retain complete failed native run unchanged."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_version_query_corrected import correction_descriptor
from scripts.run_version_query import primary_subset_replay
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash

PROTOCOL=ROOT/'configs/version_query_terminal_validation_recovery.json'
PROTOCOL_SHA256='992dd5a17738a40df80485df70e649ece733385bcfb21569f8fdedf7d22f2ad1'
SEAL=Path('/media/990Pro/Code/SAS/artifacts/memory_update/version-query-terminal-validation-seal-20261115-01.json')
ERROR='ValueError: Shared smoke/primary source bytes or guard bookkeeping differ'


def inventory(path):
    if path.is_symlink():raise ValueError('Real parent directory required')
    files={}
    for p in sorted(path.rglob('*')):
        if p.is_symlink():raise ValueError('No symlink in immutable snapshot')
        if p.is_file():files[p.relative_to(path).as_posix()]=streaming_hash(p)
    return files


def recovery_descriptor(require_seal=True):
    if streaming_hash(PROTOCOL)!=PROTOCOL_SHA256:raise ValueError('Registered terminal recovery changed')
    p=json.loads(PROTOCOL.read_text());parent=Path(p['failed_run']);m=json.loads((parent/'manifest.json').read_text())
    base=correction_descriptor()
    if (any(p[k] for k in ('new_native_inference','measurement_changes','new_accuracy_gate','tolerances_changed','scientific_outcomes_inspected_before_registration')) or
        p['base_sources_immutable']!=250 or len(p['additional_sources'])!=4 or p['total_verified_sources']!=254 or
        inventory(parent)!=p['failed_snapshot_sha256'] or streaming_hash(parent/'manifest.json')!=p['failed_manifest_sha256'] or
        streaming_hash(Path(p['failed_log']))!=p['failed_log_sha256'] or streaming_hash(Path(p['diagnosis']))!=p['diagnosis_sha256'] or
        m['status']!='failed' or m['error']!=ERROR or m['smoke'] or 'completed_utc' in m or 'primary_same_subset_replay' in m or
        m['completed_models']!=['native','gdn_all_tokens'] or [s['completed_primary'] for s in m['models'].values()]!=[864,864] or
        m['weights_after']!=m['weights_before'] or not m['hooks_restored'] or m['serialization_correction']!=base):
        raise ValueError('Preserve complete failed parent and exact terminal error')
    descriptor={'profile':p['profile'],'protocol_sha256':PROTOCOL_SHA256,'parent_run':str(parent),
        'parent_manifest_sha256':p['failed_manifest_sha256'],'original_native_execution_status':'failed_at_terminal_replay_after_all1728_readouts',
        'original_error':ERROR,'new_native_inference':False,'base_sources_unchanged':250,
        'source_sha256':{name:streaming_hash(ROOT/name) for name in p['additional_sources']}}
    if require_seal:
        seal=json.loads(SEAL.read_text())
        if (seal['status']!='sealed_cpu_only_terminal_validation_recovery' or seal['descriptor']!=descriptor or
            not datetime.fromisoformat(p['registered_utc'])<datetime.fromisoformat(seal['created_utc'])):
            raise ValueError('Exact CPU recovery seal required')
    return descriptor


def check_recovery(path,allow_unsealed=False):
    path=Path(path);descriptor=recovery_descriptor(not allow_unsealed);parent=Path(descriptor['parent_run'])
    before=json.loads((parent/'manifest.json').read_text());after=json.loads((path/'manifest.json').read_text())
    replay=primary_subset_replay(parent,before,before['smoke_equivalence_gate'])
    allowed={'status','completed_utc','primary_same_subset_replay','terminal_validation_recovery'}
    if ({k:v for k,v in after.items() if k not in allowed}!={k:v for k,v in before.items() if k not in allowed} or
        set(after)!=set(before)|allowed or after['status']!='completed' or
        after['primary_same_subset_replay']!=replay or after['terminal_validation_recovery']!=descriptor):
        raise ValueError('Only registered CPU completion fields may differ')
    if not allow_unsealed:
        seal=json.loads(SEAL.read_text())
        if not datetime.fromisoformat(seal['created_utc'])<datetime.fromisoformat(after['completed_utc']):
            raise ValueError('CPU completion must follow recovery seal')
    actual=inventory(path);expected=inventory(parent);expected['manifest.json']=streaming_hash(path/'manifest.json')
    for name,digest in descriptor['source_sha256'].items():expected['terminal-validation-sources/'+name]=digest
    if actual!=expected:raise ValueError('Derived raw, cache or source bytes differ from immutable parent')
    return descriptor


def materialize(output,allow_unsealed=False):
    output=Path(output);descriptor=recovery_descriptor(not allow_unsealed)
    if output.exists():raise ValueError('Never overwrite recovery output')
    parent=Path(descriptor['parent_run']);before=json.loads((parent/'manifest.json').read_text())
    replay=primary_subset_replay(parent,before,before['smoke_equivalence_gate'])
    shutil.copytree(parent,output)
    for name in descriptor['source_sha256']:
        target=output/'terminal-validation-sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    recovered={**before,'status':'completed','completed_utc':datetime.now(timezone.utc).isoformat(),
        'primary_same_subset_replay':replay,'terminal_validation_recovery':descriptor}
    write_json(output/'manifest.json',recovered)
    check_recovery(output,allow_unsealed)
    return recovered


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(PROTOCOL.read_text())
    if str(args.output)!=plan['validated_output']:raise ValueError('Use the registered derived output path')
    materialize(args.output)
    print(json.dumps({'status':'cpu_only_terminal_validation_materialized','new_native_inference':False,'output':str(args.output)}),flush=True)


if __name__=='__main__':main()
