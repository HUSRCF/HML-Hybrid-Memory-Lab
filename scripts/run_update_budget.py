"""Whole-record update-budget controls with independent fixed semantic reading."""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.runtime import TextRuntime
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.update_budget_data import (MODELS,REGIMES,PROTOCOL,DESTINATION,read_plan,registered_inputs,
    sources,control_ids,selected_inputs,unique_prefixes,actual_prompt,ARCHIVED)
from scripts.update_budget_runtime import (measure_condition,component_kernel_contract,SETTINGS,
    validation_addendum,VALIDATION_ADDENDUM_SHA256)
from scripts.run_controller_trial import write_json,state_fingerprint
from scripts.run_oracle_kv_readout import measure
from scripts.run_write_locality_load_screen import hook_state,weights_fingerprint
from scripts.canonical_necessity_common import check_affine_scope
from scripts.stable_adapter_linears import StableAdapterLinears
from scripts.unseen_shared_timing_runtime import verify_prefix
from scripts.copy_readout_runtime import readout


def checked_gate(path,digest):
    from scripts.analyze_update_budget import audit_equivalence
    p=Path(path);g=json.loads(p.read_text())
    if (g!=audit_equivalence(Path(g['forward_audit']),Path(g['reverse_audit'])) or
        g['dataset_manifest_sha256']!=digest or streaming_hash(p.with_suffix('.source.py'))!=g['analysis_source_sha256']):
        raise ValueError('Exact independently audited opposite720budget-control smokes required')
    return {'path':str(p),'sha256':streaming_hash(p)}


def execute_run(args):
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    torch.set_num_threads(4)
    if args.output.exists() or args.smoke==bool(args.smoke_equivalence) or (args.reverse_model_order and not args.smoke):
        raise ValueError('New operational smoke or gated complete forward run required')
    plan=read_plan();registered,cases,registry,queries,policies,references=registered_inputs(args.dataset)
    correction=validation_addendum()
    gate=None if args.smoke else checked_gate(args.smoke_equivalence,streaming_hash(args.dataset/'manifest.json'))
    physical,bindings=selected_inputs(registry,args.smoke);controls=control_ids(cases,registry)
    order=list(MODELS[::-1] if args.reverse_model_order else MODELS)
    ref={n:{r:{v['id']:readout(v) for v in rows} for r,rows in cell.items()} for n,cell in references.items()}
    profile=json.loads(Path(plan['precision_addendum']).read_text())
    m={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'training':False,'new_adapters':False,
        'smoke':args.smoke,'reverse_model_order':args.reverse_model_order,'model_order':order,'dataset':str(args.dataset),
        'dataset_manifest_sha256':streaming_hash(args.dataset/'manifest.json'),'protocol_sha256':streaming_hash(PROTOCOL),
        'policies_sha256':registered['policies_sha256'],'queries_sha256':registered['queries_sha256'],
        'references':registered['references'],'model_path':plan['model_path'],'arithmetic_profile':profile['profile'],
        'precision_addendum_sha256':streaming_hash(Path(plan['precision_addendum'])),'generation':plan['generation'],
        'selected_physical_ids':[q['id'] for q in physical],'bindings_per_model_regime':len(bindings),
        'control_query_ids':controls,'smoke_equivalence_gate':gate,'native_anchors':[],'models':{},'completed_models':[],
        'validation_addendum_sha256':VALIDATION_ADDENDUM_SHA256,
        'source_sha256':{n:streaming_hash(ROOT/n) for n in sorted(sources())},
        'generation_config_sha256':{p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')}}
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'plan.json',plan);write_json(args.output/'precision_addendum.json',profile)
    write_json(args.output/'kernel_validation_addendum.json',correction)
    for n in m['source_sha256']:
        out=args.output/'sources'/n;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((ROOT/n).read_bytes())
    mp=args.output/'manifest.json';write_json(mp,m)
    try:
        runtime=TextRuntime(plan['model_path'],'it');m.update(python=sys.executable,gpu=torch.cuda.get_device_name(0),
            packages={n:importlib.metadata.version(n) for n in ('torch','transformers','fla-core')},
            actual_eos_token_id=runtime.model.generation_config.eos_token_id)
        if (m['packages']!=plan['packages'] or m['actual_eos_token_id']!=registered['references']['actual_eos_token_id'] or
            m['generation_config_sha256']!=registered['references']['generation_config_sha256'] or
            runtime.model.config.layer_types!=(['linear_attention']*3+['full_attention'])*6):raise ValueError('Native runtime changed')
        native_source=Path(registered['references']['installed_modeling_source'])
        if streaming_hash(native_source)!=registered['references']['installed_modeling_source_sha256']:raise ValueError('Installed native source changed')
        (args.output/'installed_modeling_source.py').write_bytes(native_source.read_bytes())
        native_hooks=hook_state(runtime.model);m['weights_before']=before=weights_fingerprint(runtime.model)
        if before['sha256']!=plan['native_loaded_weights_sha256'] or not before['all_parameters_frozen'] or not before['no_parameter_gradients']:
            raise ValueError('Actual loaded weights/freeze changed')
        m['native_component_kernel_checks']=component_kernel_contract(runtime)
        anchors={}
        with torch.inference_mode():
            for key in controls:
                q=queries[key];value=measure(runtime,actual_prompt(runtime,q),q)
                if value!=ref['native']['full'][key]:raise ValueError('Native independent anchor replay failed')
                anchors[key]=value;m['native_anchors'].append({'id':key,'readout':value})
        write_json(mp,m)
        for name in order:
            spec=plan['models'][name];controller=None;arithmetic=None
            state={'token_policy':spec['token_policy'] if spec else 'none','completed_queries_per_regime':{r:0 for r in REGIMES},
                'native_before_attach_checks':[],'attached_none_checks':[],'native_restore_checks':[],
                'endpoint_controls':[],'prefix_checks':[],'condition_controls':{r:[] for r in REGIMES}}
            m['models'][name]=state
            try:
                with torch.inference_mode():
                    for key in controls:
                        q=queries[key];value=measure(runtime,actual_prompt(runtime,q),q)
                        if value!=anchors[key]:raise ValueError('Native preattach replay failed')
                        state['native_before_attach_checks'].append({'id':key,'readout':value})
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
                        raise ValueError('Fixed checkpoint/scope changed')
                    state['affine_scope']=check_affine_scope(controller,runtime.model);arithmetic=StableAdapterLinears(controller);state['arithmetic']=arithmetic.description()
                def activate(q,policy=None):
                    if controller:controller.set_token_policy(policy or spec['token_policy'],q['ledger_tokens'])
                with torch.inference_mode(),ExitStack() as stack:
                    outputs={r:stack.enter_context((args.output/f'{name}.{r}.jsonl').open('w',buffering=1)) for r in REGIMES}
                    if controller:
                        for key in controls:
                            q=queries[key];activate(q,'none');value=measure(runtime,actual_prompt(runtime,q),q)
                            if value!=anchors[key]:raise ValueError('Attached-none native identity failed')
                            state['attached_none_checks'].append({'id':key,'readout':value})
                    if spec and spec['token_policy']=='question':
                        control_set={q['id'] for q in physical}|set(controls)
                        for q in unique_prefixes([v for v in queries.values() if v['id'] in control_set]):
                            state['prefix_checks'].append(verify_prefix(runtime,controller,actual_prompt(runtime,q),q))
                    for key in controls:
                        q=queries[key];activate(q)
                        for r in ARCHIVED:
                            value,proof=measure_condition(runtime,actual_prompt(runtime,q),q,policies[key],r,observe=True)
                            if value!=ref[name][r][key]:raise ValueError('New wrapper must exactly reproduce every old endpoint control')
                            state['endpoint_controls'].append({'id':key,'regime':r,'readout':value,'proof':proof})
                    for r in REGIMES:
                        values={}
                        for key in controls:
                            q=queries[key];activate(q)
                            value,proof=measure_condition(runtime,actual_prompt(runtime,q),q,policies[key],r,observe=True)
                            values[key]=(value,proof);state['condition_controls'][r].append({'id':key,'readout':value,'proof':proof})
                        for i,physical_query in enumerate(physical,1):
                            q=queries[physical_query['id']];activate(q)
                            value,proof=values[q['id']] if q['id'] in values else measure_condition(runtime,actual_prompt(runtime,q),q,policies[q['id']],r)
                            family,restricted,_=SETTINGS[r]
                            collision=policies[q['id']]['support_overlap'][family]['exact_oracle_write_mask']
                            if collision and value!=ref[name]['delta_oracle' if restricted else 'delta_full'][q['id']]:
                                raise ValueError('Accidental original-mask collision must reproduce all5archived readout fields exactly')
                            if value['query_tokens']!=q['question_tokens']:raise ValueError('Actual input boundary changed')
                            outputs[r].write(json.dumps({'id':q['id'],'model':name,'regime':r,'prefill':'full',
                                'reader_policy':'oracle' if SETTINGS[r][1] else 'full','write_family':SETTINGS[r][0],'gate_component':SETTINGS[r][2],'token_policy':state['token_policy'],
                                'oracle_write_mask_collision':collision,
                                'arithmetic_profile':profile['profile'],'registration_sha256':data_fingerprint(q),
                                'policy_sha256':data_fingerprint(policies[q['id']]),'prompt_ids_sha256':q['prompt_ids_sha256'],
                                'proof':proof,**value})+'\n')
                            state['completed_queries_per_regime'][r]=i
                            if i%(6 if args.smoke else 32)==0:
                                write_json(mp,m);print(json.dumps({'event':'update_budget','model':name,'regime':r,'queries':i,'total':len(physical)}),flush=True)
                if controller:
                    state.update(adapter_state_sha256_after=state_fingerprint(controller),actual_affine_calls=dict(arithmetic.calls),
                        all_adapter_parameters_frozen=all(not p.requires_grad and p.grad is None for p in controller.parameters()))
                    if state['adapter_state_sha256_after']!=state['adapter_state_sha256_before'] or not state['all_adapter_parameters_frozen']:
                        raise ValueError('Frozen adapter tensors changed')
            finally:
                if arithmetic:arithmetic.close();state['arithmetic_hooks_removed']=all(not p._forward_hooks for p in controller.modules())
                if controller:controller.close()
            if hook_state(runtime.model)!=native_hooks:raise ValueError('Native hooks not restored')
            with torch.inference_mode():
                for key in controls:
                    q=queries[key];value=measure(runtime,actual_prompt(runtime,q),q)
                    if value!=anchors[key]:raise ValueError('Detached native replay failed')
                    state['native_restore_checks'].append({'id':key,'readout':value})
            state.update(hooks_restored=True,results_sha256={r:streaming_hash(args.output/f'{name}.{r}.jsonl') for r in REGIMES})
            m['completed_models'].append(name);write_json(mp,m)
        m['weights_after']=weights_fingerprint(runtime.model);m['hooks_restored']=hook_state(runtime.model)==native_hooks
        if m['weights_after']!=before or not m['hooks_restored']:raise ValueError('Native weights/hooks changed')
        m.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as e:m.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:write_json(mp,m)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=DESTINATION);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--reverse-model-order',action='store_true');p.add_argument('--smoke-equivalence',type=Path)
    execute_run(p.parse_args())

if __name__=='__main__':main()
