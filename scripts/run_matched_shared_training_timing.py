"""Reuse the unchanged432-step engine for two paired shared training schedules."""

import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.calibration import assert_rng_equal,rng_state
from scripts.analyze_matched_shared_training_timing import audit_guard
from scripts.analyze_order_augmented_semantics import audit_order_evaluation,order_inputs
from scripts.matched_shared_timing_common import (PROTOCOL,read_plan,references,sources,scope_samples,query)
from scripts.prepare_order_augmented_semantics import registered_augmented_dataset
from scripts.run_controller_trial import state_fingerprint,write_json,prompt_for
from scripts.run_order_augmented_semantics import evaluate_order
from scripts.run_paired_semantic_controllers import execute
from scripts.run_unseen_shared_timing import measured
from scripts.run_write_locality_load_screen import hook_state,weights_fingerprint
from scripts.unseen_shared_timing_runtime import verify_prefix


def final_guard(runtime,controller,arm,policy,plan,checkpoint):
    import torch
    state,rng=state_fingerprint(controller),rng_state(controller);hooks=hook_state(runtime.model)
    permuted=evaluate_order(runtime,controller,arm,policy,plan,checkpoint)
    _,data,registry=registered_augmented_dataset(Path(plan['data_source']))
    guard={'arm':arm,'policy':policy,'native_weights':weights_fingerprint(runtime.model),
        'adapter_state_before':state,'adapter_linear_profile':'original_training_fp32_adapter_linears',
        'all_adapter_linear_weights_fp32':all(p.dtype==torch.float32 for p in controller.parameters()),
        'no_adapter_arithmetic_hooks':all(not m._forward_hooks for m in controller.modules()),
        'description':controller.description(),'enabled_outputs':sorted(controller._enabled_outputs),'scope_checks':[],'complete_prefix_checks':[]}
    with torch.inference_mode():
        for i in scope_samples(data):
            ledger,q=query(runtime,data['validation'][i],registry['validation'][i])
            actual,calls,targets=measured(runtime,controller,ledger,q,policy)
            guard['scope_checks'].append({'index':i,'query':q,'readout':actual,'policy_calls':calls,'projection_targets':targets})
        if policy=='question':
            examples,permuted_registry=order_inputs(plan)
            for label,examples,reg in [('canonical',data['validation'],registry['validation']),('permuted',examples,permuted_registry)]:
                for i in [min(range(len(reg)),key=lambda i:(reg[i]['ledger_tokens'],i)),max(range(len(reg)),key=lambda i:(reg[i]['ledger_tokens'],i))]:
                    ledger,q=query(runtime,examples[i],reg[i])
                    guard['complete_prefix_checks'].append({'condition':label,'index':i,**verify_prefix(runtime,controller,ledger,q)})
    guard.update(adapter_state_after=state_fingerprint(controller),hooks_restored=hook_state(runtime.model)==hooks)
    assert_rng_equal(rng,rng_state(controller));guard['rng_unchanged']=True
    if guard['adapter_state_after']!=state:raise ValueError('Final complete guard changed trained weights')
    audit_guard(guard,arm,plan,data,registry,references(plan),runtime)
    write_json(checkpoint/'matched_timing_guard.json',guard)
    return permuted


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();plan=read_plan();refs=references(plan)
    if args.smoke and args.resume:raise ValueError('Discarded scratch smoke cannot resume training')
    args.plan=PROTOCOL;args.output=Path(plan['training_smoke_output'] if args.smoke else plan['training_output'])
    if args.output.exists() and not args.resume:raise ValueError('Preserve prior matched-timing runs')
    if not args.smoke:
        from scripts.analyze_matched_shared_training_timing import audit_run
        path=Path(plan['training_smoke_audit']);report=json.loads(path.read_text())
        if audit_run(Path(plan['training_smoke_output']),True)!=report:raise ValueError('Independently audited two-arm safety smoke required')
    registered,data,registry=registered_augmented_dataset(Path(plan['data_source']))
    args.output.mkdir(parents=True,exist_ok=args.resume)
    with (args.output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'reference_audits.json'
        if args.resume:
            if json.loads(path.read_text())!=refs:raise ValueError('Matched training references changed')
        else:write_json(path,refs)
        execute(args,plan,registered,data,registry,refs['native'],extra_sources=[ROOT/n for n in sorted(sources())],
            extra_validation=final_guard,extra_validation_audit=audit_order_evaluation)


if __name__=='__main__':main()
