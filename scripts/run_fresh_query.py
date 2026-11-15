"""Fresh state/Query-source factorial, gated by exact opposite-order smokes."""
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
from scripts.fresh_query_data import (read_plan,registered_inputs,sources,selected_contexts,context_queries,
    DESTINATION,PROTOCOL,MODELS,READERS,CONDITIONS,ORIGINS)
from scripts.post_tail_path_runtime import memory_identity
from scripts.fresh_query_runtime import capture_question,measure_query,ReaderContext,restore_recurrent
from scripts.post_encoding_runtime import CacheGuard,advance_tail,assert_isolated,ATTENTION_LAYERS,cache_tensors,cache_record,tensor_layouts
from scripts.analyze_post_encoding import core_cache
from scripts.gate_component_runtime import component_kernel_contract,validation_addendum,VALIDATION_ADDENDUM_SHA256
from scripts.run_controller_trial import write_json,state_fingerprint
from scripts.run_oracle_kv_readout import measure
from scripts.update_budget_data import actual_prompt
from scripts.run_write_locality_load_screen import hook_state,weights_fingerprint
from scripts.canonical_necessity_common import check_affine_scope
from scripts.stable_adapter_linears import StableAdapterLinears


def checked_gate(path,digest):
    from scripts.analyze_fresh_query import audit_equivalence
    p=Path(path);g=json.loads(p.read_text())
    if (g!=audit_equivalence(Path(g['forward_audit']),Path(g['reverse_audit'])) or
        g['packet_manifest_sha256']!=digest or streaming_hash(p.with_suffix('.source.py'))!=g['analysis_source_sha256']):
        raise ValueError('Exact independently audited opposite204readout smokes required')
    return {'path':str(p),'sha256':streaming_hash(p)}


def run_model(runtime,name,plan,registry,queries,refs,smoke,output,state,anchors,progress):
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    spec=plan['models'][name];controller=None;arithmetic=None;native_hooks=hook_state(runtime.model)
    controls=refs['legacy_ids'];legacy=refs['legacy_queries'];oldstate=refs['model_states'][name]
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
        with (output/f'{name}.primary.jsonl').open('w',buffering=1) as primary, \
             (output/f'{name}.same-source.jsonl').open('w',buffering=1) as replay, \
             (output/f'{name}.donors.jsonl').open('w',buffering=1) as donor_file:
            for seed in seeds:
                group=[c for c in contexts if c['seed']==seed];before=next(c for c in group if c['episode']=='before')
                after=[c for c in group if c['episode']!='before'];prefix,logits,count=runtime.advance(before['prefix_text'])
                if count!=before['prefix_tokens']:raise ValueError('Normal fresh prefix boundary changed')
                parent=CacheGuard(prefix,logits);parent.record();pairs={};guards={};tail_proofs={};identities={};banks={};captures={};donor_storages=set()
                # All after-tail source caches exist before any question/answer is supplied.
                for context in after:
                    normal,last,tail_proof=advance_tail(runtime,prefix,context['prefix_text'],context['tail_text'],context['prefix_tokens'],'none')
                    parent.check();ng=CacheGuard(normal,last);ng.record()
                    held,restore_proof=restore_recurrent(prefix,normal);ng.check();parent.check();hg=CacheGuard(held);hg.record()
                    pairs[context['id']]={'N':normal,'H':held};guards[context['id']]={'N':ng,'H':hg}
                    tail_proofs[context['id']]={'normal_tail':json.loads(json.dumps(tail_proof)),'restoration':restore_proof}
                actual_sources=[prefix]+[v for pair in pairs.values() for v in pair.values()]
                for i,left in enumerate(actual_sources):
                    for right in actual_sources[i+1:]:assert_isolated(left,right)
                def check_context(cid=None,active_banks=()):
                    parent.check()
                    for c,values in guards.items():
                        if cid is None or c==cid:
                            for g in values.values():g.check()
                    for bank in active_banks:bank.check()
                # Capture every donor for this stream before ANY fresh answer scoring/generation.
                cache_storages={t.untyped_storage().data_ptr() for cache in actual_sources for _,t in cache_tensors(cache)}
                for context in after:
                    cid=context['id'];identities[cid]={origin:memory_identity({**cache_record(cache),'layouts':tensor_layouts(cache)}) for origin,cache in pairs[cid].items()}
                    for q in context_queries(context,queries,smoke):
                        for reader in READERS:
                            active_banks=[]
                            for origin in ORIGINS:
                                bank,proof=capture_question(runtime,context['ledger'],pairs[cid][origin],q['question'],
                                    ReaderContext(reader,q['history_keep_bits'],q['prelude_tokens']))
                                pointers={t.untyped_storage().data_ptr() for t in bank.queries.values()}
                                if len(pointers)!=6 or pointers&(cache_storages|donor_storages):raise ValueError('Donor mutable storage aliases another donor/cache')
                                donor_storages.update(pointers);active_banks.append(bank);check_context(cid,active_banks)
                                key=(q['id'],reader,origin);banks[key]=bank;captures[key]=proof;state['completed_captures']+=1
                def save(handle,q,condition,reader,value,proof,origin):
                    handle.write(json.dumps({'id':q['id'],'model':name,'condition':condition,'reader':reader,
                        'registration_sha256':data_fingerprint(q),'source_key':q['source_key'],'prompt_ids_sha256':q['prompt_ids_sha256'],
                        'prefill':'cached','measurement_origin':origin,'proof':proof,**value})+'\n')
                for q in context_queries(before,queries,smoke):
                    for reader in READERS:
                        value,proof=measure_query(runtime,before,prefix,q,reader);check_context()
                        save(primary,q,'before',reader,value,proof,'before_baseline');state['completed_primary']+=1
                for context in after:
                    cid=context['id']
                    for q in context_queries(context,queries,smoke):
                        for reader in READERS:
                            active_banks=[banks[(q['id'],reader,o)] for o in ORIGINS];baseline={};baseline_proofs={}
                            for origin in ORIGINS:
                                value,proof=measure_query(runtime,context,pairs[cid][origin],q,reader);check_context(cid,active_banks)
                                baseline[origin]=value;baseline_proofs[origin]=proof
                                save(primary,q,origin*2,reader,value,proof,'unpatched_after_baseline');state['completed_primary']+=1
                            for origin in ORIGINS:
                                value,proof=measure_query(runtime,context,pairs[cid][origin],q,reader,banks[(q['id'],reader,origin)])
                                check_context(cid,active_banks)
                                if value!=baseline[origin]:raise ValueError('Actual same-source Query replacement failed exact native replay')
                                if any(proof[k]!=baseline_proofs[origin][k] for k in ('reader_calls','attention','reader_keep_sha256')):
                                    raise ValueError('Same-source actual Attention reader probabilities changed')
                                if any(v['native_query']!=captures[(q['id'],reader,origin)]['donor']['tensors'][str(i)]
                                    for i,v in proof['query_exchange']['layers'].items()):raise ValueError('Same-source original Query differs from captured donor')
                                save(replay,q,origin*2,reader,value,proof,'same_source_operational_replay');state['completed_same_source_replays']+=1
                            for condition in ('NH','HN'):
                                recipient,donor=condition
                                value,proof=measure_query(runtime,context,pairs[cid][recipient],q,reader,banks[(q['id'],reader,donor)])
                                check_context(cid,active_banks)
                                if all(torch.equal(active_banks[0].queries[i],active_banks[1].queries[i]) for i in ATTENTION_LAYERS) and value!=baseline[recipient]:
                                    raise ValueError('Equivalent donor Queries must preserve recipient output')
                                save(primary,q,condition,reader,value,proof,'cross_source_query');state['completed_primary']+=1
                            for origin in ORIGINS:
                                key=(q['id'],reader,origin);bank=banks[key];record=bank.record()
                                donor_file.write(json.dumps({'id':q['id'],'model':name,'reader':reader,'origin':origin,
                                    'registration_sha256':data_fingerprint(q),'source_key':q['source_key'],'source_cache_identity':identities[cid][origin],
                                    'bank':record,'capture_proof':captures[key],'isolated_from_all_cache_and_donor_storage':True})+'\n')
                    state['sources'][cid]={'caches':{o:guards[cid][o].record() for o in ORIGINS},'proofs':tail_proofs[cid],
                        'identities':identities[cid],'all_stream_caches_and_donors_built_before_any_fresh_answer':True,'pairwise_storage_isolated':True}
                state['prefixes'][before['prefix_key']]=parent.record()
                del pairs,guards,banks,captures,actual_sources,prefix,logits,parent
                progress();print(json.dumps({'event':'fresh_query','model':name,'seed':seed,'primary':state['completed_primary'],
                    'same_source_replays':state['completed_same_source_replays'],'question_captures':state['completed_captures']}),flush=True)
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
    state.update(hooks_restored=True,results_sha256={kind:streaming_hash(output/f'{name}.{kind}.jsonl') for kind in ('primary','same-source','donors')})


def readout_fields(value):return {k:value[k] for k in ('answer','answer_nll','greedy_exact','generated','query_tokens')}


def execute_run(args):
    import torch
    torch.set_num_threads(4)
    if args.output.exists() or args.smoke==bool(args.smoke_equivalence) or (args.reverse_model_order and not args.smoke):raise ValueError('New smoke or gated primary required')
    plan=read_plan();registered,cases,registry,queries,refs=registered_inputs(args.packet)
    gate=None if args.smoke else checked_gate(args.smoke_equivalence,streaming_hash(args.packet/'manifest.json'))
    correction=validation_addendum();profile=json.loads(Path(plan['precision_addendum']).read_text());order=list(MODELS[::-1] if args.reverse_model_order else MODELS)
    m={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'training':False,'new_adapters':False,'fresh_streams':True,
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
                state={'token_policy':'all' if plan['models'][name] else 'none','completed_primary':0,'completed_same_source_replays':0,'completed_captures':0,
                    'native_before_attach_checks':[],'attached_none_checks':[],'legacy_full_checks':[],'native_restore_checks':[],
                    'prefixes':{},'sources':{}}
                m['models'][name]=state;write_json(mp,m)
                run_model(runtime,name,plan,registry,queries,refs,args.smoke,args.output,state,anchors,lambda:write_json(mp,m))
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
