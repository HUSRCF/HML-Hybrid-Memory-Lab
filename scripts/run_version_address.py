"""Passive original-record observations with complete reused-reference replay."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import numpy as np
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.version_address_data import (PROTOCOL,DESTINATION,MODELS,read_plan,registered_inputs,
    selected_contexts,verified_seal)
from scripts.version_address_runtime import measure_observed,summarize_record_array
from scripts.question_state_freeze_runtime import canonical
from scripts import run_natural_load as base
from scripts.run_version_query import verify_model_sources
from scripts.copy_readout_runtime import readout
from scripts.run_controller_trial import write_json,state_fingerprint
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash


def replay_sources(state,refs):
    selected={field:{k:v for k,v in state[field].items() if k in refs[field]} for field in ('prefixes','sources')}
    return verify_model_sources(canonical(selected),canonical(refs))


def subset_replay(output,m,gate):
    g=json.loads(Path(gate['path']).read_text());run=Path(g['forward_run']);sm=json.loads((run/'manifest.json').read_text())
    if streaming_hash(run/'manifest.json')!=g['forward_manifest_sha256']:raise ValueError('Completed smoke changed')
    for name in MODELS:
        old=[json.loads(l) for l in (run/(name+'.jsonl')).read_text().splitlines()]
        actual=[json.loads(l) for l in (output/(name+'.jsonl')).read_text().splitlines()];ids={r['id'] for r in old}
        if len(old)!=108 or [r for r in actual if r['id'] in ids]!=old:raise ValueError('All216primary/smoke readouts must replay exactly')
        for field in ('prefixes','sources'):
            for key,value in sm['models'][name][field].items():
                if canonical(m['models'][name][field][key])!=canonical(value):raise ValueError('Shared normal source changed')
        for relative,digest in sm['models'][name]['record_mass_files'].items():
            if streaming_hash(output/relative)!=digest or streaming_hash(run/relative)!=digest:raise ValueError('Shared raw FP32 observations changed')
    return {'readouts_exact':216,'complete_raw_mass_files_exact':36,'all_shared_sources_exact':True}


def checked_gate(path,digest):
    from scripts.analyze_version_address import audit_equivalence
    path=Path(path);g=json.loads(path.read_text())
    if (g!=audit_equivalence(Path(g['forward_run']),Path(g['reverse_run'])) or g['packet_manifest_sha256']!=digest or
        streaming_hash(path.with_suffix('.source.py'))!=g['analysis_source_sha256']):
        raise ValueError('Exact independently audited opposite-model-order controls required')
    return {'path':str(path),'sha256':streaming_hash(path)}


def run_model(runtime,name,p,cases,registry,queries,refs,smoke,output,state,anchors,progress):
    from safetensors.torch import load_file
    from sas_memory.controllers import GroupController
    spec=p['models'][name];controller=None;arithmetic=None;native_hooks=base.hook_state(runtime.model)
    state['token_policy']='all' if spec else 'none'
    controls=refs['legacy_ids'];legacy=refs['legacy_queries'];normal={r['id']:r for r in refs['normal_full'][name]}
    case_by_id={c['id']:c for c in cases};contexts=selected_contexts(registry,smoke)
    try:
        for key in controls:
            value=base.measure(runtime,base.actual_prompt(runtime,legacy[key]),legacy[key])
            if value!=anchors[key]:raise ValueError('Native preattach endpoint changed')
            state['native_before_attach_checks'].append({'id':key,'readout':value})
        if spec:
            controller=GroupController(runtime.model,spec['group_index'],spec['rank'],spec['mode'],adapt_gdn_keys=spec['adapt_gdn_keys'])
            controller.load_state_dict(load_file(spec['checkpoint'],device=str(runtime.model.device)),strict=True)
            controller.eval().requires_grad_(False)
            state.update(description=controller.description(),adapter_parameters=sum(v.numel() for v in controller.parameters()),
                checkpoint_sha256=streaming_hash(Path(spec['checkpoint'])),adapter_state_sha256_before=state_fingerprint(controller))
            if (state['description']!=refs['model_states'][name]['description'] or state['adapter_parameters']!=spec['parameters'] or
                state['checkpoint_sha256']!=spec['checkpoint_sha256'] or
                state['adapter_state_sha256_before']!=refs['model_states'][name]['adapter_state_sha256_before']):
                raise ValueError('Existing frozen GDN adapter changed')
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
            if value!=refs['legacy_full'][name][key]:raise ValueError('Frozen legacy endpoint changed')
            state['legacy_full_checks'].append({'id':key,'readout':value})
        massdir=output/(name+'.record-masses');massdir.mkdir(exist_ok=False)
        with (output/(name+'.jsonl')).open('w',buffering=1) as raw:
            for case_id in dict.fromkeys(c['case_id'] for c in contexts):
                group=[c for c in contexts if c['case_id']==case_id];before=group[0]
                if before['episode']!='before' or len(group)!=9:raise ValueError('All normal histories required')
                prefix,logits,count=runtime.advance(before['prefix_text'])
                if count!=before['prefix_tokens']:raise ValueError('Normal prefix boundary changed')
                parent=base.CacheGuard(prefix,logits);parent.record();children={};guards={};proofs={}
                for c in group[1:]:
                    cache,last,proof=base.advance_tail(runtime,prefix,c['prefix_text'],c['tail_text'],c['prefix_tokens'],'none')
                    parent.check();guard=base.CacheGuard(cache,last);guard.record()
                    children[c['id']]=cache;guards[c['id']]=guard;proofs[c['id']]=canonical(proof)
                actual=[prefix]+list(children.values())
                for i,left in enumerate(actual):
                    for right in actual[i+1:]:base.assert_isolated(left,right)
                for c in group:
                    cid=c['id'];cache=prefix if c['episode']=='before' else children[cid];rows=[];arrays=[]
                    relative=name+'.record-masses/'+data_fingerprint(cid)+'.npy';path=output/relative
                    for qi,key in enumerate(c['query_ids']):
                        q=queries[key];start=time.monotonic();value,proof,array,capture=measure_observed(runtime,c,cache,q)
                        state['readout_wall_seconds_with_observers']+=time.monotonic()-start
                        parent.check()
                        if cid in guards:guards[cid].check()
                        cohort=registry['cohort_by_context'][cid]
                        if cohort=='reused':
                            old=normal[key]
                            if value!=readout(old) or proof!=old['proof']:raise ValueError('Actual reused readout or old passive proof differs')
                            state['reused_reference_replays']+=1
                        summary=summarize_record_array(array,capture,q,c,case_by_id[case_id],value['generated'])
                        arrays.append(array);rows.append({'id':key,'model':name,'reader':'full','cohort':cohort,
                            'registration_sha256':data_fingerprint(q),'source_key':q['source_key'],'prompt_ids_sha256':q['prompt_ids_sha256'],
                            'prefill':'cached','measurement_origin':'normal_history_read_only_record_observation','proof':proof,
                            'capture':capture,'address_rows':summary,'record_mass_file':relative,'record_mass_index':qi,**value})
                    block=np.stack(arrays)
                    if block.shape!=(6,6,2,len(c['records'])) or block.dtype!=np.float32:raise ValueError('Every actual raw record weight required')
                    with path.open('xb') as f:np.save(f,block,allow_pickle=False)
                    loaded=np.load(path,allow_pickle=False)
                    if not np.array_equal(loaded,block) or loaded.dtype!=block.dtype:raise ValueError('Raw FP32 serialization changed bytes')
                    digest=streaming_hash(path);state['record_mass_files'][relative]=digest
                    for row in rows:
                        row['record_mass_file_sha256']=digest;raw.write(json.dumps(canonical(row))+'\n');state['completed_primary']+=1
                    if cid in guards:state['sources'][cid]={'cache':canonical(guards[cid].record()),'tail_proof':proofs[cid],
                        'all_sources_built_before_any_question':True,'pairwise_storage_isolated':True}
                state['prefixes'][before['prefix_key']]=canonical(parent.record())
                del actual,children,guards,proofs,prefix,logits,parent,cache,guard,last
                progress();print(json.dumps({'event':'passive_version_address','model':name,'case_id':case_id,'readouts':state['completed_primary']}),flush=True)
        if controller:
            state.update(adapter_state_sha256_after=state_fingerprint(controller),actual_affine_calls=dict(arithmetic.calls),
                all_adapter_parameters_frozen=all(not v.requires_grad and v.grad is None for v in controller.parameters()))
            if state['adapter_state_sha256_before']!=state['adapter_state_sha256_after'] or not state['all_adapter_parameters_frozen']:
                raise ValueError('Frozen adapter weights or gradients changed')
    finally:
        if arithmetic:
            arithmetic.close();state['arithmetic_hooks_removed']=all(not v._forward_hooks for v in controller.modules())
        if controller:controller.close()
    if base.hook_state(runtime.model)!=native_hooks:raise ValueError('Observation/controller hooks not restored')
    for key in controls:
        value=base.measure(runtime,base.actual_prompt(runtime,legacy[key]),legacy[key])
        if value!=anchors[key]:raise ValueError('Detached native endpoint changed')
        state['native_restore_checks'].append({'id':key,'readout':value})
    state.update(hooks_restored=True,predecessor_sources=replay_sources(state,refs['source_caches'][name]),
        results_sha256=streaming_hash(output/(name+'.jsonl')))
    with (output/(name+'.references.jsonl')).open('w') as f:
        for row in refs['normal_full'][name]:f.write(json.dumps({**row,'archived_reference':True,'source_run':p['wording_run'],
            'source_manifest_sha256':p['wording_manifest_sha256']})+'\n')
    state.update(archived_references=864,references_sha256=streaming_hash(output/(name+'.references.jsonl')))


def execute_run(args):
    import torch
    torch.set_num_threads(4)
    if args.output.exists() or args.smoke==bool(args.smoke_equivalence) or (args.reverse_model_order and not args.smoke):
        raise ValueError('New smoke or exact-gated complete primary required')
    p=read_plan();packet,cases,registry,queries,refs=registered_inputs(args.packet);digest=streaming_hash(args.packet/'manifest.json')
    seal=verified_seal(packet,digest);gate=None if args.smoke else checked_gate(args.smoke_equivalence,digest)
    order=list(MODELS)[::-1] if args.reverse_model_order else list(MODELS)
    state=lambda:{'completed_primary':0,'reused_reference_replays':0,'readout_wall_seconds_with_observers':0.,
        'native_before_attach_checks':[],'attached_none_checks':[],'legacy_full_checks':[],'native_restore_checks':[],
        'prefixes':{},'sources':{},'record_mass_files':{}}
    m={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'training':False,'new_adapters':False,
        'oracle_reading':False,'donor_forward':False,'changes_recurrent_updates':False,'smoke':args.smoke,
        'reverse_model_order':args.reverse_model_order,'model_order':order,'packet':str(args.packet),'packet_manifest_sha256':digest,
        'protocol_sha256':streaming_hash(PROTOCOL),'source_sha256':packet['source_sha256'],'registration_seal':seal,
        'model_path':p['model_path'],'generation':p['generation'],'arithmetic_profile':p['arithmetic_profile'],
        'precision_addendum_sha256':streaming_hash(Path(p['precision_addendum'])),'validation_addendum_sha256':base.VALIDATION_ADDENDUM_SHA256,
        'smoke_equivalence_gate':gate,'serialization_numpy_version':np.__version__,'native_anchors':[],'models':{n:state() for n in MODELS}}
    args.output.mkdir(parents=True,exist_ok=False);mp=args.output/'manifest.json'
    for name,value in (('plan',p),('precision_addendum',json.loads(Path(p['precision_addendum']).read_text())),('kernel_validation_addendum',base.validation_addendum())):
        write_json(args.output/(name+'.json'),value)
    for name in m['source_sha256']:
        target=args.output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    write_json(mp,m);start=time.monotonic()
    try:
        runtime=base.TextRuntime(p['model_path'],'it')
        m.update(python=sys.executable,gpu=torch.cuda.get_device_name(0),packages={n:importlib.metadata.version(n) for n in ('torch','transformers','fla-core')},
            actual_eos_token_id=runtime.model.generation_config.eos_token_id,
            generation_config_sha256={f.name:streaming_hash(f) for f in Path(p['model_path']).glob('generation_config.json')})
        if (m['packages']!=p['packages'] or m['actual_eos_token_id']!=refs['runtime']['actual_eos_token_id'] or
            m['generation_config_sha256']!=refs['runtime']['generation_config_sha256'] or
            runtime.model.config.layer_types!=(['linear_attention']*3+['full_attention'])*6 or
            streaming_hash(Path(p['installed_modeling_source']))!=p['installed_modeling_source_sha256']):
            raise ValueError('Native runtime changed')
        (args.output/'installed_modeling_source.py').write_bytes(Path(p['installed_modeling_source']).read_bytes())
        hooks=base.hook_state(runtime.model);m['weights_before']=base.weights_fingerprint(runtime.model)
        if m['weights_before']['sha256']!=p['native_loaded_weights_sha256'] or not m['weights_before']['all_parameters_frozen'] or not m['weights_before']['no_parameter_gradients']:
            raise ValueError('Native weights or frozen status changed')
        anchors={}
        with torch.inference_mode():
            m['native_component_kernel_checks']=base.component_kernel_contract(runtime)
            for key in refs['legacy_ids']:
                q=refs['legacy_queries'][key];value=base.measure(runtime,base.actual_prompt(runtime,q),q)
                if value!=refs['legacy_full']['native'][key]:raise ValueError('Native legacy anchor changed')
                anchors[key]=value;m['native_anchors'].append({'id':key,'readout':value})
            for name in order:run_model(runtime,name,p,cases,registry,queries,refs,args.smoke,args.output,m['models'][name],anchors,lambda:write_json(mp,m))
        m['weights_after']=base.weights_fingerprint(runtime.model);m['hooks_restored']=base.hook_state(runtime.model)==hooks
        if m['weights_after']!=m['weights_before'] or not m['hooks_restored']:raise ValueError('Final native weights/hooks changed')
        m['primary_same_subset_replay']=None if args.smoke else subset_replay(args.output,m,gate)
        m.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as e:m.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:m['execution_wall_seconds_with_controls']=time.monotonic()-start;write_json(mp,m)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,default=DESTINATION);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--reverse-model-order',action='store_true');p.add_argument('--smoke-equivalence',type=Path)
    execute_run(p.parse_args())


if __name__=='__main__':main()
