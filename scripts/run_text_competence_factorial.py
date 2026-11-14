"""Run registered text competence controls with unchanged frozen model arithmetic."""

import argparse
from contextlib import ExitStack
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
from scripts.text_competence_factorial_data import (MODELS,REGIMES,PROTOCOL,DESTINATION,read_plan,registered_inputs,
    sources,control_ids,queries_for,unique_prefixes,actual_prompt,selected_inputs,require_exact_full)
from scripts.run_controller_trial import write_json,state_fingerprint
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_load_screen import hook_state,weights_fingerprint
from scripts.canonical_necessity_common import check_affine_scope
from scripts.check_record_attention_weights import ObservedRecordWeights
from scripts.stable_adapter_linears import StableAdapterLinears
from scripts.unseen_shared_timing_runtime import verify_prefix


def checked_gate(path,digest):
    from scripts.analyze_text_competence_factorial import audit_equivalence
    p=Path(path);g=json.loads(p.read_text())
    if (g!=audit_equivalence(Path(g['forward_audit']),Path(g['reverse_audit'])) or
        g['dataset_manifest_sha256']!=digest or streaming_hash(p.with_suffix('.source.py'))!=g['analysis_source_sha256']):
        raise ValueError('Both independently audited exact opposite360-readout runs required')
    return {'path':str(p),'sha256':streaming_hash(p)}


def execute_run(args):
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    torch.set_num_threads(4)
    if args.output.exists() or args.smoke==bool(args.smoke_equivalence) or (args.reverse_model_order and not args.smoke):
        raise ValueError('New opposite-order smoke or gated complete forward run required')
    plan=read_plan();registered,cases,registry,transforms,references=registered_inputs(args.dataset)
    gate=None if args.smoke else checked_gate(args.smoke_equivalence,streaming_hash(args.dataset/'manifest.json'))
    physical,bindings=selected_inputs(registry,args.smoke);controls=control_ids(cases,registry)
    order=list(MODELS[::-1] if args.reverse_model_order else MODELS)
    profile=json.loads(Path(plan['precision_addendum']).read_text())
    ref={n:{r['id']:r for r in rows} for n,rows in references.items()}
    m={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'training':False,'new_adapters':False,
       'smoke':args.smoke,'reverse_model_order':args.reverse_model_order,'model_order':order,'dataset':str(args.dataset),
       'dataset_manifest_sha256':streaming_hash(args.dataset/'manifest.json'),'protocol_sha256':streaming_hash(PROTOCOL),
       'transformations_sha256':registered['transformations_sha256'],'collisions_sha256':registered['collisions_sha256'],
       'references':registered['references'],'model_path':plan['model_path'],'arithmetic_profile':profile['profile'],
       'precision_addendum_sha256':streaming_hash(Path(plan['precision_addendum'])),'generation':plan['generation'],
       'selected_physical_ids':[q['id'] for q in physical],'bindings_per_model_regime':len(bindings),
       'control_query_ids':controls,'smoke_equivalence_gate':gate,'native_anchors':{r:[] for r in REGIMES},
       'models':{},'completed_models':[],'source_sha256':{n:streaming_hash(ROOT/n) for n in sorted(sources())},
       'generation_config_sha256':{p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')}}
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'plan.json',plan);write_json(args.output/'precision_addendum.json',profile)
    for name in m['source_sha256']:
        p=args.output/'sources'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    mp=args.output/'manifest.json';write_json(mp,m)
    try:
        runtime=TextRuntime(plan['model_path'],'it');m.update(python=sys.executable,gpu=torch.cuda.get_device_name(0),
            packages={n:importlib.metadata.version(n) for n in ('torch','transformers','fla-core')},
            actual_eos_token_id=runtime.model.generation_config.eos_token_id)
        if (m['packages']!=plan['packages'] or m['actual_eos_token_id']!=registered['references']['actual_eos_token_id'] or
            m['generation_config_sha256']!=registered['references']['generation_config_sha256'] or
            runtime.model.config.layer_types!=(['linear_attention']*3+['full_attention'])*6):raise ValueError('Native runtime or layout changed')
        native_source=Path(registered['references']['installed_modeling_source'])
        if streaming_hash(native_source)!=registered['references']['installed_modeling_source_sha256']:raise ValueError('Installed native source changed')
        (args.output/'installed_modeling_source.py').write_bytes(native_source.read_bytes())
        native_hooks=hook_state(runtime.model);m['weights_before']=before=weights_fingerprint(runtime.model)
        if before['sha256']!=plan['native_loaded_weights_sha256'] or not before['all_parameters_frozen'] or not before['no_parameter_gradients']:
            raise ValueError('Actual loaded native weights or freeze changed')
        anchors={r:{} for r in REGIMES}
        with torch.inference_mode():
            for r in REGIMES:
                for key in controls:
                    q=transforms[key][r];value=measure(runtime,actual_prompt(runtime,q),q)
                    if r=='full':require_exact_full(value,ref['native'][key],'native_anchor',key)
                    anchors[r][key]=value;m['native_anchors'][r].append({'id':key,'readout':value})
        write_json(mp,m)
        for name in order:
            spec=plan['models'][name];controller=None;arithmetic=None
            state={'token_policy':spec['token_policy'] if spec else 'none','completed_queries_per_regime':{r:0 for r in REGIMES},
                   'native_before_attach_checks':{r:[] for r in REGIMES},'attached_none_checks':{r:[] for r in REGIMES},
                   'native_restore_checks':{r:[] for r in REGIMES},'attention_controls':{r:[] for r in REGIMES},
                   'prefix_checks':{r:[] for r in REGIMES}}
            m['models'][name]=state
            try:
                with torch.inference_mode():
                    for r in REGIMES:
                        for key in controls:
                            q=transforms[key][r];value=measure(runtime,actual_prompt(runtime,q),q)
                            if value!=anchors[r][key]:raise ValueError('Native before-attach exact replay failed')
                            state['native_before_attach_checks'][r].append({'id':key,'readout':value})
                if spec:
                    controller=GroupController(runtime.model,spec['group_index'],spec['rank'],spec['mode'],adapt_gdn_keys=spec['adapt_gdn_keys'])
                    controller.load_state_dict(load_file(spec['checkpoint'],device=str(runtime.model.device)),strict=True)
                    controller.eval().requires_grad_(False)
                    training=json.loads((Path(spec['training_run'])/'manifest.json').read_text())
                    state.update(description=controller.description(),adapter_parameters=sum(p.numel() for p in controller.parameters()),
                        checkpoint_sha256=streaming_hash(Path(spec['checkpoint'])),adapter_state_sha256_before=state_fingerprint(controller))
                    if (state['description']!=training['arms'][spec['training_arm']]['description'] or state['adapter_parameters']!=spec['parameters'] or
                        state['checkpoint_sha256']!=spec['checkpoint_sha256'] or
                        state['adapter_state_sha256_before']!=registered['references']['training'][name]['checkpoint_state_sha256']):
                        raise ValueError('Actual frozen checkpoint, scope or allocation changed')
                    state['affine_scope']=check_affine_scope(controller,runtime.model);arithmetic=StableAdapterLinears(controller);state['arithmetic']=arithmetic.description()
                seen={}
                with torch.inference_mode(),ExitStack() as stack:
                    outputs={r:stack.enter_context((args.output/f'{name}.{r}.jsonl').open('w',buffering=1)) for r in REGIMES}
                    for r in REGIMES:
                        if controller:
                            for key in controls:
                                q=transforms[key][r];controller.set_token_policy('none',q['ledger_tokens'])
                                value=measure(runtime,actual_prompt(runtime,q),q)
                                if value!=anchors[r][key]:raise ValueError('Attached-none exact identity failed')
                                state['attached_none_checks'][r].append({'id':key,'readout':value})
                        if controller and spec['token_policy']=='question':
                            for q in unique_prefixes(queries_for(transforms,cases,registry,r,args.smoke,True)):
                                state['prefix_checks'][r].append(verify_prefix(runtime,controller,actual_prompt(runtime,q),q))
                        values={}
                        for key in controls:
                            q=transforms[key][r]
                            if controller:controller.set_token_policy(spec['token_policy'],q['ledger_tokens'])
                            active=hook_state(runtime.model)
                            with ObservedRecordWeights(runtime.model,q['prelude_tokens'],q['ledger_tokens']) as observer:
                                value=measure(runtime,actual_prompt(runtime,q),q)
                            if (set(observer.stats)!={3,7,11,15,19,23} or hook_state(runtime.model)!=active or any(
                                s['prefill_calls']!=1 or s['cached_calls']<1 or s['max_record_probability']<=0 or s['max_record_mass']<=0 for s in observer.stats.values())):
                                raise ValueError('Full historical Attention or observer cleanup failed')
                            if r=='full':require_exact_full(value,ref[name][key],name,key)
                            if name=='native' and value!=anchors[r][key]:raise ValueError('Native control differs from independent anchor')
                            values[key]=value;state['attention_controls'][r].append({'id':key,'readout':value,'stats':observer.stats})
                        for i,q in enumerate(queries_for(transforms,cases,registry,r,args.smoke),1):
                            if controller:controller.set_token_policy(spec['token_policy'],q['ledger_tokens'])
                            value=values[q['id']] if q['id'] in values else measure(runtime,actual_prompt(runtime,q),q)
                            if r=='full':require_exact_full(value,ref[name][q['id']],name,q['id'])
                            if value['query_tokens']!=q['question_tokens']:raise ValueError('Actual question boundary changed')
                            key=(q['prompt_ids_sha256'],q['ledger_tokens'])
                            if key in seen and value!=seen[key]:raise ValueError('Identical transformed input/boundary failed exact replay')
                            seen[key]=value
                            outputs[r].write(json.dumps({'id':q['id'],'model':name,'regime':r,'prefill':'full','reader_policy':'full',
                                'token_policy':state['token_policy'],'arithmetic_profile':profile['profile'],'registration_sha256':data_fingerprint(q),
                                'actual_prompt_sha256':q['prompt_ids_sha256'],'ledger_source_key':q['source_key'],**value})+'\n')
                            state['completed_queries_per_regime'][r]=i
                            if i%(6 if args.smoke else 32)==0:
                                write_json(mp,m);print(json.dumps({'event':'text_competence_factorial','model':name,'regime':r,'queries':i,'total':len(physical)}),flush=True)
                if controller:
                    state.update(adapter_state_sha256_after=state_fingerprint(controller),actual_affine_calls=dict(arithmetic.calls),
                        all_adapter_parameters_frozen=all(not p.requires_grad and p.grad is None for p in controller.parameters()))
                    if state['adapter_state_sha256_after']!=state['adapter_state_sha256_before'] or not state['all_adapter_parameters_frozen']:
                        raise ValueError('Frozen adapter tensors or gradients changed')
            finally:
                if arithmetic:arithmetic.close();state['arithmetic_hooks_removed']=all(not p._forward_hooks for p in controller.modules())
                if controller:controller.close()
            if hook_state(runtime.model)!=native_hooks:raise ValueError('Controller cleanup failed')
            with torch.inference_mode():
                for r in REGIMES:
                    for key in controls:
                        q=transforms[key][r];value=measure(runtime,actual_prompt(runtime,q),q)
                        if value!=anchors[r][key]:raise ValueError('Detached native replay failed')
                        state['native_restore_checks'][r].append({'id':key,'readout':value})
            state.update(hooks_restored=True,results_sha256={r:streaming_hash(args.output/f'{name}.{r}.jsonl') for r in REGIMES})
            m['completed_models'].append(name);write_json(mp,m)
        m['weights_after']=weights_fingerprint(runtime.model);m['hooks_restored']=hook_state(runtime.model)==native_hooks
        if m['weights_after']!=before or not m['hooks_restored']:raise ValueError('Native weights or hooks changed')
        m.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as e:m.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:write_json(mp,m)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=DESTINATION);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--reverse-model-order',action='store_true');p.add_argument('--smoke-equivalence',type=Path)
    execute_run(p.parse_args())


if __name__=='__main__':main()
