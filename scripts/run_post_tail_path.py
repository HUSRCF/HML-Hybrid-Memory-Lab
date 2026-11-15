"""Frozen post-tail cache crossovers, gated by opposite-order operational smokes."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.runtime import TextRuntime
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.post_tail_path_data import (read_plan,registered_inputs,sources,selected_contexts,context_queries,
    DESTINATION,PROTOCOL,MODELS,READERS,CONDITIONS,PRIMARY,ENDPOINTS)
from scripts.post_tail_path_runtime import compose_cache,compare_components,memory_identity
from scripts.post_encoding_runtime import CacheGuard,advance_tail,measure_cached,historical_kv_proof,assert_isolated
from scripts.analyze_post_encoding import core_cache
from scripts.gate_component_runtime import component_kernel_contract,validation_addendum,VALIDATION_ADDENDUM_SHA256
from scripts.run_controller_trial import write_json,state_fingerprint
from scripts.run_oracle_kv_readout import measure
from scripts.update_budget_data import actual_prompt
from scripts.run_write_locality_load_screen import hook_state,weights_fingerprint
from scripts.canonical_necessity_common import check_affine_scope
from scripts.stable_adapter_linears import StableAdapterLinears


def checked_gate(path,digest):
    from scripts.analyze_post_tail_path import audit_equivalence
    p=Path(path);g=json.loads(p.read_text())
    if (g!=audit_equivalence(Path(g['forward_audit']),Path(g['reverse_audit'])) or
        g['packet_manifest_sha256']!=digest or streaming_hash(p.with_suffix('.source.py'))!=g['analysis_source_sha256']):
        raise ValueError('Exact independently audited opposite288readout smokes required')
    return {'path':str(p),'sha256':streaming_hash(p)}


def run_model(runtime,name,plan,registry,queries,endpoints,refs,smoke,output,state,anchors,progress):
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    spec=plan['models'][name];controller=None;arithmetic=None;native_hooks=hook_state(runtime.model)
    controls=refs['legacy_ids'];legacy=refs['legacy_queries'];oldstate=refs['model_states'][name]
    expected={label:{(v['id'],v['reader']):readout_fields(v) for v in rows} for label,rows in endpoints[name].items()}
    def activate(policy,boundary=0):
        if controller:controller.set_token_policy(policy,boundary)
    try:
        for key in controls:
            q=legacy[key];v=measure(runtime,actual_prompt(runtime,q),q)
            if v!=anchors[key]:raise ValueError('Native preattach replay failed')
            state['native_before_attach_checks'].append({'id':key,'readout':v})
        if spec:
            controller=GroupController(runtime.model,spec['group_index'],spec['rank'],spec['mode'],adapt_gdn_keys=spec['adapt_gdn_keys'])
            controller.load_state_dict(load_file(spec['checkpoint'],device=str(runtime.model.device)),strict=True)
            controller.eval().requires_grad_(False)
            state.update(description=controller.description(),adapter_parameters=sum(p.numel() for p in controller.parameters()),
                checkpoint_sha256=streaming_hash(Path(spec['checkpoint'])),adapter_state_sha256_before=state_fingerprint(controller))
            if (state['description']!=oldstate['description'] or state['adapter_parameters']!=spec['parameters'] or
                state['checkpoint_sha256']!=spec['checkpoint_sha256'] or state['adapter_state_sha256_before']!=oldstate['adapter_state_sha256_before']):
                raise ValueError('Preserve fixed frozen GDN adapter')
            state['affine_scope']=check_affine_scope(controller,runtime.model);arithmetic=StableAdapterLinears(controller);state['arithmetic']=arithmetic.description()
            for key in controls:
                q=legacy[key];activate('none',q['ledger_tokens']);v=measure(runtime,actual_prompt(runtime,q),q)
                if v!=anchors[key]:raise ValueError('Attached-none native replay failed')
                state['attached_none_checks'].append({'id':key,'readout':v})
        activate('all' if spec else 'none')
        for key in controls:
            q=legacy[key];v=measure(runtime,actual_prompt(runtime,q),q)
            if v!=refs['legacy_full'][name][key]:raise ValueError('Frozen legacy endpoint changed')
            state['legacy_full_checks'].append({'id':key,'readout':v})
        contexts=selected_contexts(registry,smoke);seeds=list(dict.fromkeys(c['seed'] for c in contexts))
        with (output/f'{name}.primary.jsonl').open('w',buffering=1) as primary,(output/f'{name}.endpoints.jsonl').open('w',buffering=1) as replay:
            for seed in seeds:
                group=[c for c in contexts if c['seed']==seed];first=group[0]
                prefix,logits,count=runtime.advance(first['prefix_text'])
                if count!=first['prefix_tokens']:raise ValueError('Original prefix boundary changed')
                parent=CacheGuard(prefix,logits);initial=parent.record()
                if core_cache(initial)!=core_cache(refs['prefix_caches'][name][first['prefix_key']]):raise ValueError('Normal prefix cache/logits differ from published source')
                for context in group:
                    origin={};guards={};source_proofs={}
                    for label,mode in ENDPOINTS.items():
                        short=label[0];cache,last,proof=advance_tail(runtime,prefix,context['prefix_text'],context['tail_text'],context['prefix_tokens'],mode)
                        parent.check();guard=CacheGuard(cache,last);actual=guard.record();proof=json.loads(json.dumps(proof))
                        old=refs['source_caches'][name][context['id']][label]
                        if core_cache(actual)!=core_cache(old['cache']) or proof!=old['tail_proof']:
                            raise ValueError('Complete N/H source cache, native logits or tail gates changed')
                        origin[short]=cache;guards[short]=guard;source_proofs[short]=proof
                    differences=compare_components(origin,context['prefix_tokens']);memories={};memory_guards={};identities={};kv={}
                    for condition in CONDITIONS:
                        memory=compose_cache(origin,condition);g=CacheGuard(memory);record=g.record()
                        for prior_memory in memories.values():assert_isolated(prior_memory,memory)
                        identities[condition]=memory_identity(record);kv[condition]=historical_kv_proof(prefix,memory,context['prefix_tokens'])
                        memories[condition]=memory;memory_guards[condition]=g
                        for guard in guards.values():guard.check()
                        parent.check()
                    # Every condition exists before a question is supplied.
                    equivalent={}
                    for condition in CONDITIONS:
                        memory_id=context['id']+'.'+condition;destination=primary if condition in PRIMARY else replay
                        for q in context_queries(context,queries,smoke):
                            for reader in READERS:
                                v,proof=measure_cached(runtime,context,memories[condition],q,reader)
                                memory_guards[condition].check();parent.check()
                                for guard in guards.values():guard.check()
                                key=(identities[condition],q['id'],reader)
                                prior=equivalent.get(key)
                                if prior and v!=prior['readout']:raise ValueError('Identical composed caches gave different five-field readouts')
                                equivalent.setdefault(key,{'condition':condition,'readout':v})
                                if condition in ENDPOINTS and v!=expected[condition][(q['id'],reader)]:raise ValueError('Published original endpoint readout changed')
                                destination.write(json.dumps({'id':q['id'],'model':name,'condition':condition,'reader':reader,
                                    'memory_id':memory_id,'memory_identity':identities[condition],'prefill':'cached',
                                    'registration_sha256':data_fingerprint(q),'source_key':q['source_key'],'prompt_ids_sha256':q['prompt_ids_sha256'],
                                    'measurement_origin':'new_primary' if condition in PRIMARY else 'endpoint_operational_replay',
                                    'equivalent_condition':equivalent[key]['condition'],'proof':proof,**v})+'\n')
                                state['completed_primary' if condition in PRIMARY else 'completed_endpoint_replays']+=1
                        state['memories'][memory_id]={'context_id':context['id'],'condition':condition,'source_key':context['source_key'],
                            'cache':memory_guards[condition].record(),'memory_identity':identities[condition],'historical_kv_sha256':kv[condition]}
                    state['sources'][context['id']]={'caches':{short:{'cache':guards[short].record(),'tail_proof':source_proofs[short]} for short in 'NH'},
                        'component_differences':differences,'all_conditions_built_before_queries':True,'pairwise_storage_isolated':True}
                    del origin,guards,memories,memory_guards
                state['prefixes'][first['prefix_key']]=parent.record();del prefix,logits,parent
                progress();print(json.dumps({'event':'post_tail_path','model':name,'seed':seed,'primary':state['completed_primary'],
                    'endpoint_replays':state['completed_endpoint_replays']}),flush=True)
        if controller:
            state.update(adapter_state_sha256_after=state_fingerprint(controller),actual_affine_calls=dict(arithmetic.calls),
                all_adapter_parameters_frozen=all(not p.requires_grad and p.grad is None for p in controller.parameters()))
            if state['adapter_state_sha256_after']!=state['adapter_state_sha256_before'] or not state['all_adapter_parameters_frozen']:raise ValueError('Adapter changed')
    finally:
        if arithmetic:arithmetic.close();state['arithmetic_hooks_removed']=all(not p._forward_hooks for p in controller.modules())
        if controller:controller.close()
    if hook_state(runtime.model)!=native_hooks:raise ValueError('Native hooks not restored')
    for key in controls:
        q=legacy[key];v=measure(runtime,actual_prompt(runtime,q),q)
        if v!=anchors[key]:raise ValueError('Detached native replay failed')
        state['native_restore_checks'].append({'id':key,'readout':v})
    state.update(hooks_restored=True,results_sha256={kind:streaming_hash(output/f'{name}.{kind}.jsonl') for kind in ('primary','endpoints')})


def readout_fields(value):return {k:value[k] for k in ('answer','answer_nll','greedy_exact','generated','query_tokens')}


def execute_run(args):
    import torch
    torch.set_num_threads(4)
    if args.output.exists() or args.smoke==bool(args.smoke_equivalence) or (args.reverse_model_order and not args.smoke):raise ValueError('New smoke or gated primary required')
    plan=read_plan();registered,cases,registry,queries,endpoints,refs=registered_inputs(args.packet)
    gate=None if args.smoke else checked_gate(args.smoke_equivalence,streaming_hash(args.packet/'manifest.json'))
    correction=validation_addendum();profile=json.loads(Path(plan['precision_addendum']).read_text());order=list(MODELS[::-1] if args.reverse_model_order else MODELS)
    m={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'training':False,'new_adapters':False,'fresh_confirmation':False,
        'smoke':args.smoke,'reverse_model_order':args.reverse_model_order,'model_order':order,'packet':str(args.packet),
        'packet_manifest_sha256':streaming_hash(args.packet/'manifest.json'),'protocol_sha256':streaming_hash(PROTOCOL),
        'source_sha256':{n:streaming_hash(ROOT/n) for n in sorted(sources())},'models':{},'completed_models':[],'native_anchors':[],
        'model_path':plan['model_path'],'generation':plan['generation'],'arithmetic_profile':profile['profile'],
        'precision_addendum_sha256':streaming_hash(Path(plan['precision_addendum'])),'validation_addendum_sha256':VALIDATION_ADDENDUM_SHA256,
        'smoke_equivalence_gate':gate,'generation_config_sha256':{p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')}}
    args.output.mkdir(parents=True,exist_ok=False);mp=args.output/'manifest.json'
    for filename,value in (('plan',plan),('precision_addendum',profile),('kernel_validation_addendum',correction)):write_json(args.output/(filename+'.json'),value)
    for name in m['source_sha256']:
        p=args.output/'sources'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    write_json(mp,m)
    try:
        runtime=TextRuntime(plan['model_path'],'it');m.update(python=sys.executable,gpu=torch.cuda.get_device_name(0),
            packages={n:importlib.metadata.version(n) for n in ('torch','transformers','fla-core')},actual_eos_token_id=runtime.model.generation_config.eos_token_id)
        if (m['packages']!=plan['packages'] or m['actual_eos_token_id']!=refs['runtime']['actual_eos_token_id'] or
            m['generation_config_sha256']!=refs['runtime']['generation_config_sha256'] or
            runtime.model.config.layer_types!=(['linear_attention']*3+['full_attention'])*6):raise ValueError('Native runtime changed')
        path=Path(refs['runtime']['installed_modeling_source'])
        if streaming_hash(path)!=refs['runtime']['installed_modeling_source_sha256']:raise ValueError('Native installed source changed')
        (args.output/'installed_modeling_source.py').write_bytes(path.read_bytes())
        native_hooks=hook_state(runtime.model);m['weights_before']=before=weights_fingerprint(runtime.model)
        if before['sha256']!=plan['native_loaded_weights_sha256'] or not before['all_parameters_frozen'] or not before['no_parameter_gradients']:raise ValueError('Native weights/freeze changed')
        m['native_component_kernel_checks']=component_kernel_contract(runtime);anchors={}
        with torch.inference_mode():
            for key in refs['legacy_ids']:
                q=refs['legacy_queries'][key];v=measure(runtime,actual_prompt(runtime,q),q)
                if v!=refs['legacy_full']['native'][key]:raise ValueError('Native old anchor changed')
                anchors[key]=v;m['native_anchors'].append({'id':key,'readout':v})
            for name in order:
                state={'token_policy':'all' if plan['models'][name] else 'none','completed_primary':0,'completed_endpoint_replays':0,
                    'native_before_attach_checks':[],'attached_none_checks':[],'legacy_full_checks':[],'native_restore_checks':[],
                    'prefixes':{},'sources':{},'memories':{}}
                m['models'][name]=state;write_json(mp,m)
                run_model(runtime,name,plan,registry,queries,endpoints,refs,args.smoke,args.output,state,anchors,lambda:write_json(mp,m))
                m['completed_models'].append(name);write_json(mp,m)
        m['weights_after']=weights_fingerprint(runtime.model);m['hooks_restored']=hook_state(runtime.model)==native_hooks
        if m['weights_after']!=before or not m['hooks_restored']:raise ValueError('Native weights/hooks changed')
        m.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as e:m.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:write_json(mp,m)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,default=DESTINATION);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--reverse-model-order',action='store_true');p.add_argument('--smoke-equivalence',type=Path)
    execute_run(p.parse_args())

if __name__=='__main__':main()
