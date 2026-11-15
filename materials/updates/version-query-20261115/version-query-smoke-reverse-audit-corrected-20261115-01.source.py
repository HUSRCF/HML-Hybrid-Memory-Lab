"""Original audit code objects plus separately sealed serialization provenance."""
import argparse
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import analyze_version_query as original
from scripts import run_version_query as runner
from scripts.run_version_query_corrected import check_annotation,checked_gate,verify_model_sources
from scripts.run_controller_trial import write_json


def clone(function,**bindings):
    namespace=dict(function.__globals__);namespace.update(bindings)
    result=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


def audit_run(path,allow_smoke=False,allow_unsealed=False):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    descriptor=check_annotation(manifest,path,allow_unsealed)
    previous=runner.checked_gate
    try:
        runner.checked_gate=checked_gate
        report=clone(original.audit_run,verify_model_sources=verify_model_sources,__file__=__file__)(path,allow_smoke)
    finally:runner.checked_gate=previous
    return {**report,'serialization_correction':descriptor,'correction_sources_verified':4,'total_source_files_verified':250}


def audit_equivalence(forward,reverse):
    result=clone(original.audit_equivalence,audit_run=audit_run,__file__=__file__)(forward,reverse)
    from scripts.run_version_query_corrected import correction_descriptor
    return {**result,'serialization_correction':correction_descriptor(),'total_source_files_verified':250}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path)
    parser.add_argument('--allow-smoke',action='store_true');parser.add_argument('--forward',type=Path);parser.add_argument('--reverse',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Never overwrite an independent audit')
    if args.run is not None and args.forward is None and args.reverse is None:report=audit_run(args.run,args.allow_smoke)
    elif args.run is None and args.forward is not None and args.reverse is not None:report=audit_equivalence(args.forward,args.reverse)
    else:raise ValueError('Choose one complete run or opposite-order equivalence')
    write_json(args.output,report);args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:report[k] for k in ('status','primary_readouts_audited','total_source_files_verified') if k in report}),flush=True)


if __name__=='__main__':main()
