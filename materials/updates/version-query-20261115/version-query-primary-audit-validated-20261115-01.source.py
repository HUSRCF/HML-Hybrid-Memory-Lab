"""Audit exact derived recovery, then unchanged250source independent auditor."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.finalize_version_query import check_recovery
from scripts.analyze_version_query_corrected import audit_run as prior_audit
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def audit_run(path,allow_smoke=False,allow_unsealed=False):
    if allow_smoke:raise ValueError('Terminal recovery is primary-only')
    descriptor=check_recovery(Path(path),allow_unsealed)
    result=prior_audit(Path(path))
    return {**result,'terminal_validation_recovery':descriptor,'terminal_validation_sources_verified':4,
        'total_source_files_verified':254,'native_inference_repeated':False,
        'prior_analysis_source_sha256':result['analysis_source_sha256'],
        'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Never overwrite a recovery audit')
    report=audit_run(args.run);write_json(args.output,report)
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:report[k] for k in ('status','primary_readouts_audited','archived_full_references_audited','total_source_files_verified','native_inference_repeated')}),flush=True)


if __name__=='__main__':main()
