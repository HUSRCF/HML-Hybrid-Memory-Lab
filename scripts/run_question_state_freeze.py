"""Eight question-only policies on one normally encoded, frozen GDN model."""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.question_state_freeze_data import (PROTOCOL,DESTINATION,MODEL,ARMS,read_plan,
    registered_inputs,sources,selected_contexts,context_queries,verified_seal)
from scripts.question_state_freeze_runtime import canonical,measure_question,freeze_kernel_contract
from scripts import run_natural_load as base
from scripts.run_version_query import verify_model_sources
from scripts.copy_readout_runtime import readout
from scripts.run_controller_trial import write_json,state_fingerprint
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash


def source_replay(state,reference):
    return verify_model_sources(canonical({k:state[k] for k in ('prefixes','sources')}),canonical(reference))


def primary_subset_replay(output,manifest,gate):
    g=json.loads(Path(gate['path']).read_text());path=Path(g['forward_run'])
    sm=json.loads((path/'manifest.json').read_text())
    if streaming_hash(path/'manifest.json')!=g['forward_manifest_sha256']:raise ValueError('Completed smoke changed')
    counts={}
    for arm in ARMS:
        rows=[json.loads(l) for l in (path/f'{arm}.jsonl').read_text().splitlines()]
        actual=[json.loads(l) for l in (output/f'{arm}.jsonl').read_text().splitlines()]
        ids={r['id'] for r in rows}
        if len(rows)!=54 or [r for r in actual if r['id'] in ids]!=rows:
            raise ValueError('Every shared432primary/smoke readout must replay exactly')
        counts[arm]=54
    for field in ('prefixes','sources'):
        for key,old in sm['model_state'][field].items():
            if canonical(manifest['model_state'][field][key])!=canonical(old):
                raise ValueError('Shared canonical source or guard bookkeeping differs')
    return {'readouts_by_arm':counts,'prefix_caches':1,'after_caches':8,'all_exact':True}


def checked_gate(path,packet_digest):
    from scripts.analyze_question_state_freeze import audit_equivalence
    path=Path(path);g=json.loads(path.read_text())
    if (g!=audit_equivalence(Path(g['forward_run']),Path(g['reverse_run'])) or
        g['packet_manifest_sha256']!=packet_digest or
        streaming_hash(path.with_suffix('.source.py'))!=g['analysis_source_sha256']):
        raise ValueError('Exact independent opposite-arm-order432readout smokes required')
    return {'path':str(path),'sha256':streaming_hash(path)}


def run_model(runtime,p,registry,queries,refs,smoke,arm_order,output,state,anchors,progress):
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    spec=p['model_spec'];controller=None;arithmetic=None;native_hooks=base.hook_state(runtime.model)
    controls=refs['legacy_ids'];legacy=refs['legacy_queries'];normal={r['id']:r for r in refs['normal_full']}
    try:
        for key in controls:
            value=base.measure(runtime,base.actual_prompt(runtime,legacy[key]),legacy[key])
            if value!=anchors[key]:raise ValueError('Native preattach endpoint changed')
            state['native_before_attach_checks'].append({'id':key,'readout':value})
        controller=GroupController(runtime.model,spec['group_index'],spec['rank'],spec['mode'],adapt_gdn_keys=spec['adapt_gdn_keys'])
        controller.load_state_dict(load_file(spec['checkpoint'],device=str(runtime.model.device)),strict=True)
        controller.eval().requires_grad_(False)
        state.update(description=controller.description(),adapter_parameters=sum(v.numel() for v in controller.parameters()),
            checkpoint_sha256=streaming_hash(Path(spec['checkpoint'])),adapter_state_sha256_before=state_fingerprint(controller))
        if (state['description']!=refs['model_state']['description'] or state['adapter_parameters']!=spec['parameters'] or
            state['checkpoint_sha256']!=spec['checkpoint_sha256'] or
            state['adapter_state_sha256_before']!=refs['model_state']['adapter_state_sha256_before']):
            raise ValueError('Existing frozen controller changed')
        state['affine_scope']=base.check_affine_scope(controller,runtime.model)
        arithmetic=base.StableAdapterLinears(controller);state['arithmetic']=arithmetic.description()
        for key in controls:
            controller.set_token_policy('none',legacy[key]['ledger_tokens'])
            value=base.measure(runtime,base.actual_prompt(runtime,legacy[key]),legacy[key])
            if value!=anchors[key]:raise ValueError('Attached-none native endpoint changed')
            state['attached_none_checks'].append({'id':key,'readout':value})
        controller.set_token_policy('all',0)
        for key in controls:
            value=base.measure(runtime,base.actual_prompt(runtime,legacy[key]),legacy[key])
            if value!=refs['legacy_full'][MODEL][key]:raise ValueError('Frozen GDN legacy endpoint changed')
            state['legacy_full_checks'].append({'id':key,'readout':value})
        contexts=selected_contexts(registry,smoke)
        with ExitStack() as stack:
            execution=stack.enter_context((output/'execution.jsonl').open('w',buffering=1))
            files={arm:stack.enter_context((output/f'{arm}.jsonl').open('w',buffering=1)) for arm in ARMS}
            for case_id in dict.fromkeys(c['case_id'] for c in contexts):
                group=[c for c in contexts if c['case_id']==case_id];before=group[0]
                if before['episode']!='before' or len(group)!=9:raise ValueError('All normal histories required per case')
                prefix,logits,count=runtime.advance(before['prefix_text'])
                if count!=before['prefix_tokens']:raise ValueError('Normal prefix boundary changed')
                parent=base.CacheGuard(prefix,logits);prefix_record=canonical(parent.record())
                children={};guards={};proofs={};initial_records={}
                for context in group[1:]:
                    cache,last,proof=base.advance_tail(runtime,prefix,context['prefix_text'],context['tail_text'],context['prefix_tokens'],'none')
                    parent.check();guard=base.CacheGuard(cache,last)
                    children[context['id']]=cache;guards[context['id']]=guard
                    initial_records[context['id']]=canonical(guard.record());proofs[context['id']]=canonical(proof)
                actual=[prefix]+list(children.values())
                for i,left in enumerate(actual):
                    for right in actual[i+1:]:base.assert_isolated(left,right)
                for context in group:
                    cid=context['id'];cache=prefix if context['episode']=='before' else children[cid]
                    record=prefix_record if context['episode']=='before' else initial_records[cid]
                    for q in context_queries(context,queries,smoke):
                        for arm in arm_order:
                            started=time.monotonic();value,proof=measure_question(runtime,context,cache,q,ARMS[arm],record)
                            state['readout_wall_seconds_with_observers'][arm]+=time.monotonic()-started
                            parent.check()
                            if cid in guards:guards[cid].check()
                            if arm=='normal':
                                old=normal[q['id']]
                                if value!=readout(old) or {k:proof[k] for k in ('reader_calls','attention','reader_keep_sha256')}!=old['proof']:
                                    raise ValueError('Actual normal readout and passive weights differ from predecessor')
                                state['normal_reference_replays']+=1
                            row={'id':q['id'],'model':MODEL,'reader':'full','arm':arm,
                                'registration_sha256':data_fingerprint(q),'source_key':q['source_key'],
                                'prompt_ids_sha256':q['prompt_ids_sha256'],'prefill':'cached',
                                'measurement_origin':'normally_encoded_history_question_only_gate_policy','proof':proof,**value}
                            line=json.dumps(canonical(row))+'\n';execution.write(line);files[arm].write(line)
                            state['completed_primary']+=1;state['completed_by_arm'][arm]+=1
                    if cid in guards:
                        state['sources'][cid]={'cache':canonical(guards[cid].record()),'tail_proof':proofs[cid],
                            'all_sources_built_before_any_question':True,'pairwise_storage_isolated':True}
                state['prefixes'][before['prefix_key']]=canonical(parent.record())
                del actual,children,guards,proofs,initial_records,prefix,logits,parent,cache,guard,last
                progress();print(json.dumps({'event':'question_state_freeze','case_id':case_id,'readouts':state['completed_primary']}),flush=True)
        state.update(adapter_state_sha256_after=state_fingerprint(controller),actual_affine_calls=dict(arithmetic.calls),
            all_adapter_parameters_frozen=all(not v.requires_grad and v.grad is None for v in controller.parameters()))
        if state['adapter_state_sha256_before']!=state['adapter_state_sha256_after'] or not state['all_adapter_parameters_frozen']:
            raise ValueError('Frozen adapter bytes or gradients changed')
    finally:
        if arithmetic:
            arithmetic.close();state['arithmetic_hooks_removed']=all(not v._forward_hooks for v in controller.modules())
        if controller:controller.close()
    if base.hook_state(runtime.model)!=native_hooks:raise ValueError('Controller or query hooks not restored')
    for key in controls:
        value=base.measure(runtime,base.actual_prompt(runtime,legacy[key]),legacy[key])
        if value!=anchors[key]:raise ValueError('Detached native endpoint changed')
        state['native_restore_checks'].append({'id':key,'readout':value})
    state.update(hooks_restored=True,predecessor_sources=source_replay(state,refs['source_caches']),
        file_sha256={n+'.jsonl':streaming_hash(output/(n+'.jsonl')) for n in ('execution',*ARMS)})
    with (output/'references.jsonl').open('w') as f:
        for row in refs['normal_full']:
            f.write(json.dumps({**row,'archived_reference':True,'source_run':p['source_run'],
                'source_manifest_sha256':p['source_manifest_sha256']})+'\n')
    state.update(archived_references=864,references_sha256=streaming_hash(output/'references.jsonl'))


def execute_run(args):
    import torch
    torch.set_num_threads(4)
    if args.output.exists() or args.smoke==bool(args.smoke_equivalence) or (args.reverse_arm_order and not args.smoke):
        raise ValueError('New smoke or gated complete primary required')
    p=read_plan();registered,cases,registry,queries,refs=registered_inputs(args.packet)
    digest=streaming_hash(args.packet/'manifest.json');seal=verified_seal(registered,digest)
    gate=None if args.smoke else checked_gate(args.smoke_equivalence,digest)
    order=list(ARMS)[::-1] if args.reverse_arm_order else list(ARMS)
    m={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'training':False,'new_adapters':False,
        'fresh_confirmation':False,'oracle_evidence':False,'donor_forward':False,'smoke':args.smoke,
        'reverse_arm_order':args.reverse_arm_order,'arm_order':order,'packet':str(args.packet),'packet_manifest_sha256':digest,
        'protocol_sha256':streaming_hash(PROTOCOL),'registration_seal':seal,'source_sha256':registered['source_sha256'],
        'model':MODEL,'model_path':p['model_path'],'generation':p['generation'],'arithmetic_profile':p['arithmetic_profile'],
        'precision_addendum_sha256':streaming_hash(Path(p['precision_addendum'])),
        'validation_addendum_sha256':base.VALIDATION_ADDENDUM_SHA256,'smoke_equivalence_gate':gate,
        'native_anchors':[],'model_state':{'token_policy':'all','completed_primary':0,'normal_reference_replays':0,
            'completed_by_arm':{a:0 for a in ARMS},'readout_wall_seconds_with_observers':{a:0. for a in ARMS},
            'native_before_attach_checks':[],'attached_none_checks':[],'legacy_full_checks':[],
            'native_restore_checks':[],'prefixes':{},'sources':{}}}
    args.output.mkdir(parents=True,exist_ok=False);mp=args.output/'manifest.json'
    profile=json.loads(Path(p['precision_addendum']).read_text());correction=base.validation_addendum()
    for name,value in (('plan',p),('precision_addendum',profile),('kernel_validation_addendum',correction)):
        write_json(args.output/(name+'.json'),value)
    for name,digest in m['source_sha256'].items():
        target=args.output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    write_json(mp,m);started=time.monotonic()
    try:
        runtime=base.TextRuntime(p['model_path'],'it')
        m.update(python=sys.executable,gpu=torch.cuda.get_device_name(0),
            packages={n:importlib.metadata.version(n) for n in ('torch','transformers','fla-core')},
            actual_eos_token_id=runtime.model.generation_config.eos_token_id,
            generation_config_sha256={f.name:streaming_hash(f) for f in Path(p['model_path']).glob('generation_config.json')})
        if (m['packages']!=p['packages'] or m['actual_eos_token_id']!=refs['runtime']['actual_eos_token_id'] or
            m['generation_config_sha256']!=refs['runtime']['generation_config_sha256'] or
            runtime.model.config.layer_types!=(['linear_attention']*3+['full_attention'])*6):
            raise ValueError('Native runtime changed')
        installed=Path(p['installed_modeling_source'])
        if streaming_hash(installed)!=p['installed_modeling_source_sha256']:raise ValueError('Installed model source changed')
        (args.output/'installed_modeling_source.py').write_bytes(installed.read_bytes())
        native_hooks=base.hook_state(runtime.model);m['weights_before']=base.weights_fingerprint(runtime.model)
        if m['weights_before']['sha256']!=p['native_loaded_weights_sha256'] or not m['weights_before']['all_parameters_frozen'] or not m['weights_before']['no_parameter_gradients']:
            raise ValueError('Native weights or frozen status changed')
        with torch.inference_mode():
            m['native_component_kernel_checks']=base.component_kernel_contract(runtime)
            m['native_freeze_kernel_checks']=freeze_kernel_contract(runtime)
        anchors={}
        with torch.inference_mode():
            for key in refs['legacy_ids']:
                q=refs['legacy_queries'][key];value=base.measure(runtime,base.actual_prompt(runtime,q),q)
                if value!=refs['legacy_full']['native'][key]:raise ValueError('Native anchor changed')
                anchors[key]=value;m['native_anchors'].append({'id':key,'readout':value})
            run_model(runtime,p,registry,queries,refs,args.smoke,order,args.output,m['model_state'],anchors,lambda:write_json(mp,m))
        m['weights_after']=base.weights_fingerprint(runtime.model);m['hooks_restored']=base.hook_state(runtime.model)==native_hooks
        if m['weights_after']!=m['weights_before'] or not m['hooks_restored']:raise ValueError('Final native weights or hooks changed')
        m['primary_same_subset_replay']=None if args.smoke else primary_subset_replay(args.output,m,gate)
        m.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as e:m.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:
        m['execution_wall_seconds_with_controls']=time.monotonic()-started;write_json(mp,m)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packet',type=Path,default=DESTINATION)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--reverse-arm-order',action='store_true');parser.add_argument('--smoke-equivalence',type=Path)
    execute_run(parser.parse_args())


if __name__=='__main__':main()
