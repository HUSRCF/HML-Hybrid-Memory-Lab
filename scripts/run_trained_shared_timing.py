"""Full3x3 shared training/deployment timing on the single untouched cohort."""

import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.runtime import TextRuntime
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.canonical_necessity_common import check_affine_scope
from scripts.copy_readout_runtime import readout
from scripts.run_controller_trial import state_fingerprint,write_json
from scripts.run_copy_dependency_transfer import actual_prompt
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_load_screen import hook_state,weights_fingerprint
from scripts.stable_adapter_linears import StableAdapterLinears
from scripts.trained_shared_timing_data import (DESTINATION,PROTOCOL,registered_inputs,selected_inputs,
    sample_ids,safety_samples,unique_prefix_queries,recheck_prior)
from scripts.unseen_shared_timing_runtime import PolicyTrace,audit_policy,verify_prefix
from scripts.trained_shared_timing_data import sources,SHARED_MODELS

def checked_gate(path,dataset_hash):
    from scripts.analyze_trained_shared_timing import audit_equivalence
    report=json.loads(Path(path).read_text());actual=audit_equivalence(Path(report["forward_audit"]),Path(report["reverse_audit"]))
    if (actual!=report or report["dataset_manifest_sha256"]!=dataset_hash or
        streaming_hash(Path(path).with_suffix(".source.py"))!=report["analysis_source_sha256"]):
        raise ValueError("Both288-readout smoke audits and exact order equivalence required")
    return {"path":str(path),"sha256":streaming_hash(Path(path))}


def measured(runtime,controller,ledger,q,policy):
    before=hook_state(runtime.model)
    if controller:controller.set_token_policy(policy,q['ledger_tokens'])
    with PolicyTrace(runtime.model,controller,q,policy) as trace:
        result=measure(runtime,ledger,q)
    if hook_state(runtime.model)!=before:raise ValueError('Passive policy observers leaked hooks')
    audit_policy(trace.calls,q,policy,trace.targets)
    return result,trace.calls,trace.targets


def execute(args,registered,plan,cases,registry,profile,novelty,gate):
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    torch.set_num_threads(4)
    physical,bindings=selected_inputs(registry,args.smoke)
    samples=sample_ids(cases,registry);safety=safety_samples(registry)
    case_map={c['id']:c for c in cases}
    order=plan['confirmation']['evaluation_order'][::-1] if args.reverse_order else plan['confirmation']['evaluation_order']
    refs=registered['references']
    manifest={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'training':False,'new_adapters':False,'newly_trained_checkpoints':2,'smoke':args.smoke,'reverse_order':args.reverse_order,'model_order':order,
        'dataset':str(DESTINATION),'dataset_manifest_sha256':streaming_hash(DESTINATION/'manifest.json'),
        'protocol_sha256':streaming_hash(PROTOCOL),'arithmetic_profile':profile['profile'],
        'precision_addendum_sha256':plan['precision_addendum_sha256'],'references':refs,
        'independent_novelty_recheck':novelty,'model_path':plan['model_path'],
        'generation':plan['generation'] if 'generation' in plan else {'max_tokens':12,'prefill':'full','score_batch':False},
        'selected_physical_ids':[q['id'] for q in physical],'related_bindings_per_arm':len(bindings),
        'sample_query_ids':samples,'prefix_query_ids':{},'smoke_equivalence_gate':gate,
        'source_sha256':{n:streaming_hash(ROOT/n) for n in sorted(sources())},
        'installed_modeling_source_sha256':refs['installed_modeling_source_sha256'],
        'generation_config_sha256':refs['generation_config_sha256'],
        'models':{},'completed_models':[],'native_anchors':[]}
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'plan.json',plan);write_json(args.output/'precision_addendum.json',profile)
    for name in manifest['source_sha256']:
        target=args.output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    (args.output/'installed_modeling_source.py').write_bytes(Path(refs['installed_modeling_source']).read_bytes())
    path=args.output/'manifest.json';write_json(path,manifest)
    try:
        runtime=TextRuntime(plan['model_path'],'it')
        manifest.update(python=sys.executable,gpu=torch.cuda.get_device_name(0),
            actual_eos_token_id=runtime.model.generation_config.eos_token_id,
            packages={n:importlib.metadata.version(n) for n in ('torch','transformers','fla-core')})
        native_hooks=hook_state(runtime.model);before=weights_fingerprint(runtime.model);manifest['weights_before']=before
        if (manifest['packages']!=plan['packages'] or runtime.model.config.layer_types!=(['linear_attention']*3+['full_attention'])*6 or
            before['sha256']!=plan['native_loaded_weights_sha256'] or not before['all_parameters_frozen'] or
            not before['no_parameter_gradients'] or manifest['actual_eos_token_id']!=refs['actual_eos_token_id'] or
            {p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')}!=refs['generation_config_sha256']):
            raise ValueError('Actual frozen native model/runtime changed')
        anchors={}
        with torch.inference_mode():
            for q in physical:
                if q['id'] in samples:
                    anchors[q['id']]=measure(runtime,actual_prompt(runtime,case_map,q),q)
                    manifest['native_anchors'].append({'id':q['id'],'readout':anchors[q['id']]})
        if list(anchors)!=samples or hook_state(runtime.model)!=native_hooks:raise ValueError('Native anchor coverage changed')
        write_json(path,manifest)
        for name in order:
            arm=plan['confirmation']['evaluation_arms'][name];spec=registered['models'][arm['model']];policy=arm['token_policy']
            prefix_q=(unique_prefix_queries(physical) if arm['model'] in SHARED_MODELS and policy=='question' else
                [q for q in physical if q['id'] in safety] if policy=='question' else [])
            manifest['prefix_query_ids'][name]=[q['id'] for q in prefix_q]
            state={'model_key':arm['model'],'token_policy':policy,'completed_queries':0,'prefix_checks':[],
                'none_identity_checks':[],'native_before_attach_checks':[],'native_restore_checks':[]}
            manifest['models'][name]=state;controller=arithmetic=None
            try:
                with torch.inference_mode():
                    for q in physical:
                        if q['id'] in anchors:
                            actual=measure(runtime,actual_prompt(runtime,case_map,q),q)
                            if actual!=anchors[q['id']]:raise ValueError('Native pre-attach anchor differs')
                            state['native_before_attach_checks'].append({'id':q['id'],'exact':True,'readout':actual})
                if spec:
                    controller=GroupController(runtime.model,spec['group_index'],spec['rank'],spec['mode'],adapt_gdn_keys=spec['adapt_gdn_keys'])
                    controller.load_state_dict(load_file(spec['checkpoint'],device=str(runtime.model.device)),strict=True)
                    controller.eval().requires_grad_(False)
                    training=json.loads((Path(spec['training_run'])/'manifest.json').read_text())
                    state.update(description=controller.description(),adapter_parameters=sum(p.numel() for p in controller.parameters()),
                        checkpoint_sha256=streaming_hash(Path(spec['checkpoint'])),adapter_state_sha256_before=state_fingerprint(controller),
                        enabled_outputs=sorted(controller._enabled_outputs),projection_targets={n:p.out_features for p,n,_,_ in controller._specs})
                    if (state['description']!=training['arms'][spec['training_arm']]['description'] or state['adapter_parameters']!=spec['parameters'] or
                        state['checkpoint_sha256']!=spec['checkpoint_sha256'] or
                        state['adapter_state_sha256_before']!=refs['training'][arm['model']]['checkpoint_state_sha256']):
                        raise ValueError('Actual frozen controller checkpoint or allocation differs')
                    state['affine_scope']=check_affine_scope(controller,runtime.model)
                    if not state['affine_scope']['all_adapter_linear_weights_fp32']:raise ValueError('Original FP32 adapter storage required')
                    arithmetic=StableAdapterLinears(controller);state['arithmetic']=arithmetic.description()
                else:state.update(enabled_outputs=[],projection_targets={})
                with torch.inference_mode(),(args.output/f'{name}.jsonl').open('w',buffering=1) as output:
                    if controller:
                        for q in physical:
                            if q['id'] in anchors:
                                actual,calls,_=measured(runtime,controller,actual_prompt(runtime,case_map,q),q,'none')
                                if actual!=anchors[q['id']]:raise ValueError('Attached none-policy differs from original native model')
                                state['none_identity_checks'].append({'id':q['id'],'readout':actual,'policy_calls':calls})
                        for q in prefix_q:state['prefix_checks'].append(verify_prefix(runtime,controller,actual_prompt(runtime,case_map,q),q))
                    for number,q in enumerate(physical,1):
                        actual,calls,targets=measured(runtime,controller,actual_prompt(runtime,case_map,q),q,policy)
                        if name=='native' and q['id'] in anchors and actual!=anchors[q['id']]:raise ValueError('Native arm differs from anchor')
                        if targets!=state['projection_targets']:raise ValueError('Actual controller projection scope differs')
                        output.write(json.dumps({'id':q['id'],'model':name,'token_policy':policy,'prefill':'full','regime':'full',
                            'registration_sha256':data_fingerprint(q),'arithmetic_profile':profile['profile'],
                            'policy_calls':calls,**actual})+'\n')
                        state['completed_queries']=number
                        if number%(6 if args.smoke else 16)==0:
                            write_json(path,manifest);print(json.dumps({'event':'trained_shared_timing','arm':name,'queries':number,'total':len(physical)}),flush=True)
                if controller:
                    state.update(adapter_state_sha256_after=state_fingerprint(controller),
                        all_adapter_parameters_frozen=all(not p.requires_grad and p.grad is None for p in controller.parameters()),actual_affine_calls=dict(arithmetic.calls))
                    if (state['adapter_state_sha256_before']!=state['adapter_state_sha256_after'] or not state['all_adapter_parameters_frozen'] or
                        any(v<1 for v in arithmetic.calls.values())):raise ValueError('Frozen controller bytes or affine execution changed')
            finally:
                if arithmetic:
                    arithmetic.close();state['arithmetic_hooks_removed']=all(not m._forward_hooks for m in controller.modules())
                if controller:controller.close()
            if hook_state(runtime.model)!=native_hooks:raise ValueError('Controller detach failed to restore original hooks')
            with torch.inference_mode():
                for q in physical:
                    if q['id'] in anchors:
                        actual=measure(runtime,actual_prompt(runtime,case_map,q),q)
                        if actual!=anchors[q['id']]:raise ValueError('Detached native readout differs from anchor')
                        state['native_restore_checks'].append({'id':q['id'],'exact':True,'readout':actual})
            state.update(hooks_restored=True,results_sha256=streaming_hash(args.output/f'{name}.jsonl'))
            manifest['completed_models'].append(name);write_json(path,manifest)
            print(json.dumps({'event':'trained_shared_timing_arm_complete','arm':name,'readouts':len(physical)}),flush=True)
        manifest['weights_after']=weights_fingerprint(runtime.model);manifest['hooks_restored']=hook_state(runtime.model)==native_hooks
        if manifest['weights_after']!=before or not manifest['hooks_restored']:raise ValueError('Frozen native tensors/gradients/hooks changed')
        manifest.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:write_json(path,manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--reverse-order',action='store_true');parser.add_argument('--gate',type=Path)
    args=parser.parse_args()
    if args.output.exists() or args.smoke==bool(args.gate) or (args.reverse_order and not args.smoke):
        raise ValueError('New operational smoke or gated forward primary run required')
    registered,plan,cases,registry=registered_inputs();profile=json.loads((ROOT/plan['precision_addendum']).read_text())
    novelty=recheck_prior(DESTINATION,registered)
    gate=None if args.smoke else checked_gate(args.gate,streaming_hash(DESTINATION/'manifest.json'))
    execute(args,registered,plan,cases,registry,profile,novelty,gate)


if __name__=='__main__':main()
