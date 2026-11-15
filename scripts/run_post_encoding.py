"""Normally encode shared prefixes, intervene only on subsequent update chunks."""
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
from scripts.post_encoding_data import (read_plan,registered_inputs,sources,selected_contexts,context_queries,
    DESTINATION,PROTOCOL,MODES,READERS)
from scripts.post_encoding_runtime import (CacheGuard,advance_tail,measure_cached,exact_cache,assert_isolated)
from scripts.gate_component_runtime import component_kernel_contract,validation_addendum,VALIDATION_ADDENDUM_SHA256
from scripts.run_controller_trial import write_json,state_fingerprint
from scripts.run_oracle_kv_readout import measure
from scripts.update_budget_data import actual_prompt
from scripts.run_write_locality_load_screen import hook_state,weights_fingerprint
from scripts.canonical_necessity_common import check_affine_scope
from scripts.stable_adapter_linears import StableAdapterLinears


def checked_gate(path,digest):
    from scripts.analyze_post_encoding import audit_equivalence
    p=Path(path);value=json.loads(p.read_text())
    if (value!=audit_equivalence(Path(value['forward_audit']),Path(value['reverse_audit'])) or
        value['dataset_manifest_sha256']!=digest or streaming_hash(p.with_suffix('.source.py'))!=value['analysis_source_sha256']):
        raise ValueError('Exact independently audited opposite510readout smokes required')
    return {'path':str(p),'sha256':streaming_hash(p)}


def run_model(runtime,name,plan,registry,queries,references,smoke,output,state,anchors):
    import torch
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    spec=plan['models'][name];controller=None;arithmetic=None;native_hooks=hook_state(runtime.model)
    legacy=references['legacy_queries'];controls=references['legacy_ids']
    def activate(policy,boundary=0):
        if controller:controller.set_token_policy(policy,boundary)
    def encode():activate('none' if spec is None or spec['token_policy']=='question' else 'all')
    def query_activate(q):activate(spec['token_policy'] if spec else 'none',q['ledger_tokens'])
    try:
        for key in controls:
            q=legacy[key];v=measure(runtime,actual_prompt(runtime,q),q)
            if v!=anchors[key]:raise ValueError('Native preattach exact replay failed')
            state['native_before_attach_checks'].append({'id':key,'readout':v})
        if spec:
            controller=GroupController(runtime.model,spec['group_index'],spec['rank'],spec['mode'],adapt_gdn_keys=spec['adapt_gdn_keys'])
            controller.load_state_dict(load_file(spec['checkpoint'],device=str(runtime.model.device)),strict=True)
            controller.eval().requires_grad_(False)
            training=json.loads((Path(spec['training_run'])/'manifest.json').read_text())
            oldrefs=references['predecessor_manifest']['references']
            state.update(description=controller.description(),adapter_parameters=sum(p.numel() for p in controller.parameters()),
                checkpoint_sha256=streaming_hash(Path(spec['checkpoint'])),adapter_state_sha256_before=state_fingerprint(controller))
            if (state['description']!=training['arms'][spec['training_arm']]['description'] or state['adapter_parameters']!=spec['parameters'] or
                state['checkpoint_sha256']!=spec['checkpoint_sha256'] or
                state['adapter_state_sha256_before']!=oldrefs['training'][name]['checkpoint_state_sha256']):
                raise ValueError('Frozen fixed step432 checkpoint changed')
            state['affine_scope']=check_affine_scope(controller,runtime.model);arithmetic=StableAdapterLinears(controller)
            state['arithmetic']=arithmetic.description()
            for key in controls:
                q=legacy[key];activate('none',q['ledger_tokens']);v=measure(runtime,actual_prompt(runtime,q),q)
                if v!=anchors[key]:raise ValueError('Attached-none exact replay failed')
                state['attached_none_checks'].append({'id':key,'readout':v})
        for key in controls:
            q=legacy[key];query_activate(q);v=measure(runtime,actual_prompt(runtime,q),q)
            if v!=references['legacy_full'][name][key]:raise ValueError('Legacy full endpoint changed')
            state['legacy_full_checks'].append({'id':key,'readout':v})
        contexts=selected_contexts(registry,smoke);seeds=list(dict.fromkeys(c['seed'] for c in contexts))
        with (output/f'{name}.jsonl').open('w',buffering=1) as raw:
            def probes(context,cache,logits,mode,prefix_guard,tail_proof=None,noop=None):
                guard=CacheGuard(cache,logits);memory_id=context['id']+'.'+mode
                for q in context_queries(context,queries,smoke):
                    for reader in READERS:
                        query_activate(q);v,proof=measure_cached(runtime,context,cache,q,reader)
                        guard.check();prefix_guard.check()
                        if noop is not None and q['role']=='target_current' and q['question_style']=='original':
                            if v!=noop['readouts'][reader]:raise ValueError('Unwrapped and none-held exact cached readouts differ')
                            noop['matched_primary_readouts']+=1
                        raw.write(json.dumps({'id':q['id'],'model':name,'mode':mode,'reader':reader,'memory_id':memory_id,
                            'prefill':'cached','registration_sha256':data_fingerprint(q),'prompt_ids_sha256':q['prompt_ids_sha256'],
                            'source_key':q['source_key'],'proof':proof,**v})+'\n')
                        state['completed_readouts']+=1
                record=guard.record()
                if len(record['tensors'])!=48:raise ValueError('Complete48tensor mixed cache required')
                state['memories'][memory_id]={'context_id':context['id'],'mode':mode,'prefix_key':context['prefix_key'],
                    'source_key':context['source_key'],'cache':record,'tail_proof':tail_proof}
            for seed in seeds:
                group=[c for c in contexts if c['seed']==seed];before=group[0]
                if before['episode']!='before':raise ValueError('Normal prefix must precede all branches')
                encode();prefix,first,count=runtime.advance(before['prefix_text'])
                if count!=before['prefix_tokens'] or prefix.get_seq_length()!=count:raise ValueError('Prefix token count changed')
                parent=CacheGuard(prefix,first)
                probes(before,prefix,first,'none',parent)
                for context in group[1:]:
                    encode();unwrapped,unwrapped_logits,count=runtime.advance(context['tail_text'],prefix,context['prefix_text'])
                    parent.check();assert_isolated(prefix,unwrapped)
                    if count!=context['tail_tokens'] or unwrapped.get_seq_length()!=context['ledger_tokens']:raise ValueError('Tail token count changed')
                    unwrapped_guard=CacheGuard(unwrapped,unwrapped_logits)
                    control={'context_id':context['id'],'readouts':{},'matched_primary_readouts':0}
                    current=next(q for q in context_queries(context,queries,smoke) if q['role']=='target_current' and q['question_style']=='original')
                    for reader in READERS:
                        query_activate(current);value,proof=measure_cached(runtime,context,unwrapped,current,reader)
                        unwrapped_guard.check();parent.check();control['readouts'][reader]=value
                        control.setdefault('reader_proofs',{})[reader]=proof
                    control['unwrapped_cache']=unwrapped_guard.record()
                    for mode in MODES:
                        encode();cache,logits,proof=advance_tail(runtime,prefix,context['prefix_text'],context['tail_text'],context['prefix_tokens'],mode)
                        parent.check()
                        if mode=='none':
                            exact_cache(unwrapped,cache)
                            if not torch.equal(unwrapped_logits,logits):raise ValueError('None-held tail changed logits')
                            control.update(complete_cache_and_logits_exact=True)
                        probes(context,cache,logits,mode,parent,proof,control if mode=='none' else None)
                        del cache,logits
                    if control['matched_primary_readouts']!=2:raise ValueError('Both no-op reader comparisons required')
                    state['noop_controls'].append(control);del unwrapped,unwrapped_logits,unwrapped_guard
                state['prefixes'][before['prefix_key']]=parent.record()
                del prefix,first,parent
                print(json.dumps({'event':'post_encoding','model':name,'seed':seed,'readouts':state['completed_readouts']}),flush=True)
        if controller:
            state.update(adapter_state_sha256_after=state_fingerprint(controller),actual_affine_calls=dict(arithmetic.calls),
                all_adapter_parameters_frozen=all(not p.requires_grad and p.grad is None for p in controller.parameters()))
            if state['adapter_state_sha256_before']!=state['adapter_state_sha256_after'] or not state['all_adapter_parameters_frozen']:
                raise ValueError('Frozen adapter parameters changed')
    finally:
        if arithmetic:
            arithmetic.close();state['arithmetic_hooks_removed']=all(not p._forward_hooks for p in controller.modules())
        if controller:controller.close()
    if hook_state(runtime.model)!=native_hooks:raise ValueError('Native hook restoration failed')
    for key in controls:
        q=legacy[key];v=measure(runtime,actual_prompt(runtime,q),q)
        if v!=anchors[key]:raise ValueError('Detached native replay failed')
        state['native_restore_checks'].append({'id':key,'readout':v})
    state.update(hooks_restored=True,results_sha256=streaming_hash(output/f'{name}.jsonl'))


def execute_run(args):
    import torch
    torch.set_num_threads(4)
    if args.output.exists() or args.smoke==bool(args.smoke_equivalence) or (args.reverse_model_order and not args.smoke):
        raise ValueError('New operational smoke or gated primary required')
    plan=read_plan();registered,cases,registry,queries,references=registered_inputs(args.dataset)
    correction=validation_addendum();gate=None if args.smoke else checked_gate(args.smoke_equivalence,streaming_hash(args.dataset/'manifest.json'))
    order=plan['model_order'][::-1] if args.reverse_model_order else plan['model_order']
    profile=json.loads(Path(plan['precision_addendum']).read_text());oldrefs=references['predecessor_manifest']['references']
    m={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'training':False,'new_adapters':False,
        'smoke':args.smoke,'reverse_model_order':args.reverse_model_order,'model_order':order,'dataset':str(args.dataset),
        'dataset_manifest_sha256':streaming_hash(args.dataset/'manifest.json'),'protocol_sha256':streaming_hash(PROTOCOL),
        'source_sha256':{n:streaming_hash(ROOT/n) for n in sorted(sources())},'native_anchors':[],'models':{},'completed_models':[],
        'smoke_equivalence_gate':gate,'arithmetic_profile':profile['profile'],'precision_addendum_sha256':streaming_hash(Path(plan['precision_addendum'])),
        'validation_addendum_sha256':VALIDATION_ADDENDUM_SHA256,'model_path':plan['model_path'],'generation':plan['generation'],
        'generation_config_sha256':{p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')}}
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'plan.json',plan);write_json(args.output/'precision_addendum.json',profile)
    write_json(args.output/'kernel_validation_addendum.json',correction)
    for n in m['source_sha256']:
        out=args.output/'sources'/n;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((ROOT/n).read_bytes())
    mp=args.output/'manifest.json';write_json(mp,m)
    try:
        runtime=TextRuntime(plan['model_path'],'it');m.update(python=sys.executable,gpu=torch.cuda.get_device_name(0),
            packages={n:importlib.metadata.version(n) for n in ('torch','transformers','fla-core')},actual_eos_token_id=runtime.model.generation_config.eos_token_id)
        if (m['packages']!=plan['packages'] or m['actual_eos_token_id']!=oldrefs['actual_eos_token_id'] or
            m['generation_config_sha256']!=oldrefs['generation_config_sha256'] or
            runtime.model.config.layer_types!=(['linear_attention']*3+['full_attention'])*6):raise ValueError('Frozen runtime changed')
        native_source=Path(plan['installed_modeling_source'])
        if streaming_hash(native_source)!=plan['installed_modeling_source_sha256']:raise ValueError('Native installed source changed')
        (args.output/'installed_modeling_source.py').write_bytes(native_source.read_bytes())
        native_hooks=hook_state(runtime.model);m['weights_before']=before=weights_fingerprint(runtime.model)
        if before['sha256']!=plan['native_loaded_weights_sha256'] or not before['all_parameters_frozen'] or not before['no_parameter_gradients']:
            raise ValueError('Native weights/freeze changed')
        m['native_component_kernel_checks']=component_kernel_contract(runtime)
        anchors={}
        with torch.inference_mode():
            for key in references['legacy_ids']:
                q=references['legacy_queries'][key];value=measure(runtime,actual_prompt(runtime,q),q)
                if value!=references['legacy_full']['native'][key]:raise ValueError('Native original anchors changed')
                anchors[key]=value;m['native_anchors'].append({'id':key,'readout':value})
            for name in order:
                state={'token_policy':plan['models'][name]['token_policy'] if plan['models'][name] else 'none',
                    'completed_readouts':0,'native_before_attach_checks':[],'attached_none_checks':[],'legacy_full_checks':[],
                    'native_restore_checks':[],'memories':{},'prefixes':{},'noop_controls':[]}
                m['models'][name]=state;write_json(mp,m)
                run_model(runtime,name,plan,registry,queries,references,args.smoke,args.output,state,anchors)
                m['completed_models'].append(name);write_json(mp,m)
        a,b=[m['models'][n] for n in ('native','shared_question_trained_question')]
        if a['prefixes']!=b['prefixes'] or a['memories']!=b['memories']:
            raise ValueError('Question-only controller changed prefix/tail memories before any question')
        m['question_only_native_memory_exact']=True
        m['weights_after']=weights_fingerprint(runtime.model);m['hooks_restored']=hook_state(runtime.model)==native_hooks
        if m['weights_after']!=before or not m['hooks_restored']:raise ValueError('Native weights/hooks changed')
        m.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as e:m.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:write_json(mp,m)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=DESTINATION)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--reverse-model-order',action='store_true');p.add_argument('--smoke-equivalence',type=Path)
    execute_run(p.parse_args())

if __name__=='__main__':main()
