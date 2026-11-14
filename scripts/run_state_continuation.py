"""Replay all32 path observations, then run20 registered state-origin continuations."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_question_path_mediation import audit_query
from scripts.canonical_necessity_common import check_affine_scope
from scripts.fresh_layer_controls import tensor_hash
from scripts.run_controller_trial import state_fingerprint, write_json
from scripts.run_copy_dependency_transfer import actual_prompt
from scripts.run_fresh_token_timing import advance_ids
from scripts.run_oracle_kv_readout import measure
from scripts.run_question_path_mediation import condition
from scripts.run_write_locality_adapter_transfer import verify_question_prefix
from scripts.run_write_locality_load_screen import hook_state, weights_fingerprint
from scripts.semantic_order_path_common import cache_fingerprint
from scripts.stable_adapter_linears import StableAdapterLinears
from scripts.state_continuation_controls import (PROTOCOL_SHA256, CONDITIONS, ForwardTrace,
    input_scope, sources, cache_record, verify_sources, make_condition, gold_losses, decode_vectors)


def observe_continuation(runtime,query,prompt,cache,first,initial_key,key,vectors,origin_caches):
    """Call existing scoring/decoding implementations, with passive hooks only."""
    hooks=hook_state(runtime.model);before=cache_record(cache)
    origin_before={k:cache_record(v) for k,v in origin_caches.items()}
    answer_ids=runtime.continuation_ids(prompt,query['answer'])
    if answer_ids!=query['answer_ids']:raise ValueError('Original gold tokenization changed')
    with ForwardTrace(runtime.model) as gold:
        score=runtime.candidate_scores(prompt,cache,first,[query['answer']],batch=False)[query['answer']]
    after_gold=cache_record(cache)
    if after_gold!=before or hook_state(runtime.model)!=hooks:raise ValueError('Gold scoring changed start cache or hooks')
    origins_after_gold={k:cache_record(v) for k,v in origin_caches.items()}
    if origins_after_gold!=origin_before:raise ValueError('Gold scoring changed reused origin caches')
    n=len(answer_ids);expected_calls=int(n>1)
    if len(gold.calls)!=expected_calls:raise ValueError('Singleton gold forward shape changed')
    if n>1 and (gold.calls[0]['input_ids']!=[answer_ids[:-1]] or gold.calls[0]['logits_to_keep']!=n-1):
        raise ValueError('Gold scoring did not use original singleton teacher-forced shape')
    losses=gold_losses(first,gold.vectors[0] if n>1 else None,answer_ids,runtime.model.device)
    if score!={k:losses[k] for k in ('nll_sum','tokens','nll_mean')}:
        raise ValueError('Archived logits do not exactly reconstruct actual gold score')
    with ForwardTrace(runtime.model) as greedy:
        generated=runtime.greedy_answer(cache,first,max_tokens=12)
    after_greedy=cache_record(cache)
    if after_greedy!=before or hook_state(runtime.model)!=hooks:raise ValueError('Greedy decode changed start cache or hooks')
    origins_after_greedy={k:cache_record(v) for k,v in origin_caches.items()}
    if origins_after_greedy!=origin_before:raise ValueError('Greedy decode changed reused origin caches')
    decoded=decode_vectors(runtime.tokenizer,first.detach().cpu(),greedy.vectors,
        runtime.model.generation_config.eos_token_id)
    if generated!=decoded['generated']:raise ValueError('Actual greedy text contradicts saved vectors')
    for i,call in enumerate(greedy.calls):
        if call['input_ids']!=[[decoded['generated_ids'][i]]] or call['logits_to_keep']!=1:
            raise ValueError('Observed greedy forward input disagrees with actual decision')
    for label,trace in (('gold',gold),('greedy',greedy)):
        for i,(call,vector) in enumerate(zip(trace.calls,trace.vectors,strict=True)):
            vector_key=f'{key}--{label}--{i}'
            if vector_key in vectors:raise ValueError('Duplicate actual vector key')
            vectors[vector_key]=vector;call['logits_key']=vector_key
    match=re.fullmatch(r'\s*(-?\d+)\s*',generated)
    readout={'answer':query['answer'],'answer_nll':score['nll_mean'],
        'greedy_exact':bool(match and match.group(1)==query['answer']),
        'generated':generated,'query_tokens':query['question_tokens']}
    return {'readout':readout,'gold_ids':answer_ids,'gold_losses':losses,
        'initial_logits_key':initial_key,'initial_logits_sha256':tensor_hash(first),
        'gold_forwards':gold.calls,'greedy_forwards':greedy.calls,'decode':decoded,
        'starting_cache_sha256':before['sha256'],'after_gold_cache_sha256':after_gold['sha256'],
        'after_greedy_cache_sha256':after_greedy['sha256'],
        'origin_cache_fingerprints':{stage:{k:v['sha256'] for k,v in records.items()} for stage,records in
            (('before',origin_before),('after_gold',origins_after_gold),('after_greedy',origins_after_greedy))},
        'origin_cache_metadata_unchanged_after_each_operation':True,
        'start_cache_unchanged_after_each_operation':True,'hooks_restored':True}


def execute(scope,output):
    import torch
    from safetensors.torch import load_file,save_file
    from sas_memory.controllers import GroupController
    from sas_memory.runtime import TextRuntime
    torch.set_num_threads(4);plan=scope['plan'];old=scope['mediation_scope']['plan']
    manifest={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'training':False,'new_adapters':False,'new_data':False,'posthoc':True,'selection_is_posthoc':True,
        'independent_scientific_confirmation':False,'can_change_primary_gates':False,'no_accuracy_gate':True,
        'full_generation_scored':True,'save_full_logits':True,'protocol_sha256':PROTOCOL_SHA256,
        **{k:plan[k] for k in ('model_path','dataset_manifest_sha256','mask_registry_sha256','arithmetic_profile',
            'precision_addendum_sha256','prior_mediation_audit_sha256','prior_mediation_manifest_sha256',
            'installed_modeling_source_sha256')},'original_primary_gates_unchanged':scope['primary_gates'],
        'source_sha256':{n:streaming_hash(ROOT/n) for n in sorted(sources())},
        'model_order':plan['models'],'completed_models':[],'models':{},'native_anchors':[]}
    output.mkdir(parents=True,exist_ok=False);path=output/'manifest.json'
    write_json(output/'plan.json',plan);write_json(output/'precision_addendum.json',scope['profile'])
    (output/'installed_modeling_source.py').write_bytes(Path(plan['installed_modeling_source']).read_bytes())
    for name in manifest['source_sha256']:
        target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    write_json(path,manifest);runtime=native_hooks=None
    try:
        runtime=TextRuntime(plan['model_path'],'it')
        manifest.update(python=sys.executable,gpu=torch.cuda.get_device_name(0),
            packages={n:importlib.metadata.version(n) for n in plan['packages']},
            actual_vocab_size=runtime.model.get_output_embeddings().weight.shape[0],
            actual_eos_token_id=runtime.model.generation_config.eos_token_id,
            generation_config_sha256={p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')})
        if (manifest['packages']!=plan['packages'] or
            manifest['actual_eos_token_id']!=scope['primary_manifest']['actual_eos_token_id'] or
            manifest['generation_config_sha256']!=scope['primary_manifest']['generation_config_sha256'] or
            runtime.model.config.layer_types!=(['linear_attention']*3+['full_attention'])*6):
            raise ValueError('Frozen runtime,hybrid layout or generation configuration changed')
        native_hooks=hook_state(runtime.model);before=manifest['weights_before']=weights_fingerprint(runtime.model)
        if (before['sha256']!=plan['native_loaded_weights_sha256'] or
            not before['all_parameters_frozen'] or not before['no_parameter_gradients']):
            raise ValueError('Native checkpoint or freeze changed')
        cases={c['id']:c for c in scope['cases']};anchors={}
        with torch.inference_mode():
            for entry in plan['queries']:
                q=scope['queries'][entry['id']];actual=measure(runtime,actual_prompt(runtime,cases,q),q)
                manifest['native_anchors'].append({'id':q['id'],'readout':actual,
                    'exact_completed_fresh_native_replay':actual==scope['native_references'][q['id']]})
                write_json(path,manifest)
                if actual!=scope['native_references'][q['id']]:raise ValueError('Original native anchor drifted')
                anchors[q['id']]=actual
        prior={(r['model'],r['id'],r['context'],r['patch']):r for r in plan['source_references']}
        for name in plan['models']:
            spec=scope['source_plan']['models'][name];entries=[e for e in plan['queries'] if e['model']==name]
            state=manifest['models'][name]={'token_policy':'question','proofs':[],'prefix_checks':[],
                'native_restore_checks':[],'completed_queries':0,'source_rows':0,'recorded_rows':0,'state_source_checks':[]}
            controller=arithmetic=None;vectors={}
            result_path=output/f'{name}.jsonl';source_path=output/f'{name}.sources.jsonl';logits_path=output/f'{name}.logits.safetensors'
            old_vectors=load_file(str(Path(plan['prior_mediation_run'])/f'{name}.logits.safetensors'))
            try:
                controller=GroupController(runtime.model,spec['group_index'],spec['rank'],spec['mode'],adapt_gdn_keys=spec['adapt_gdn_keys'])
                controller.load_state_dict(load_file(spec['checkpoint'],device=str(runtime.model.device)),strict=True)
                controller.eval().requires_grad_(False)
                state.update(description=controller.description(),checkpoint_sha256=streaming_hash(Path(spec['checkpoint'])),
                    adapter_state_sha256_before=state_fingerprint(controller),adapter_parameters=sum(p.numel() for p in controller.parameters()))
                if ({k:state[k] for k in plan['fixed_checkpoints'][name]}!=plan['fixed_checkpoints'][name] or state['adapter_parameters']!=spec['parameters']):
                    raise ValueError('Frozen controller or allocation changed')
                state['affine_scope']=check_affine_scope(controller,runtime.model)
                arithmetic=StableAdapterLinears(controller);state['arithmetic']=arithmetic.description()
                with torch.inference_mode(),result_path.open('w',buffering=1) as stream,source_path.open('w',buffering=1) as source_stream:
                    for entry in entries:
                        q=scope['queries'][entry['id']]
                        state['prefix_checks'].append({'id':q['id'],**verify_question_prefix(runtime,controller,actual_prompt(runtime,cases,q),q)})
                    for entry in entries:
                        q=scope['queries'][entry['id']];ledger=actual_prompt(runtime,cases,q);prompt=ledger+runtime.query_suffix(q['question']);ids=runtime.ids(prompt)
                        if data_fingerprint(ids)!=q['prompt_ids_sha256'] or len(ids)!=q['prompt_tokens']:
                            raise ValueError('Original full input IDs changed')
                        controller.set_token_policy('none');native_prefix,native_logits=advance_ids(runtime,ids[:q['ledger_tokens']])
                        controller.set_token_policy('question',q['ledger_tokens'])
                        prefix_ref={'cache':native_prefix,'logits':native_logits,'sha256':cache_fingerprint(native_prefix)}
                        if prefix_ref['sha256']!=next(p['prefix_sha256'] for p in state['prefix_checks'] if p['id']==q['id']):
                            raise ValueError('Actual native ledger-prefix reference changed')
                        keep=[v=='1' for v in scope['masks'][q['id']]['keep_bits'][entry['regime']]]
                        state['active_query']=q['id'];write_json(path,manifest)
                        unmasked=restored=None;source_rows=[];source_proofs=[];source_vectors={}
                        fixed=prior[(name,q['id'],'unmasked','none')]['fixed_competitor_id']
                        for context in old['context_order']:
                            ctx=gdn_states=None
                            for patch in old['patch_order']:
                                result=condition(runtime,q,ids,keep,context,patch,prefix_ref,unmasked,ctx)
                                if unmasked is None:unmasked=result
                                if ctx is None:ctx=result
                                if patch=='gdn_current_outputs':gdn_states=result['states']
                                expected=ctx['states'] if patch=='question_kv' else gdn_states if patch=='both' else None
                                if expected is not None and result['states']!=expected:raise ValueError('KV patch changed preceding linear states')
                                proof=result['proof'];proof['kv_patch_recurrent_states_unchanged']=True if expected is not None else None
                                closure=context=='earlier_only_layer19' and patch=='both'
                                proof['combined_paths_logits_exact']=bool(torch.equal(result['logits'],unmasked['logits'])) if closure else None
                                key=f"{q['id']}--{context}--{patch}";vector=result['logits'].detach().cpu().contiguous().clone()
                                vectors[key]=source_vectors[key]=vector
                                row={'model':name,'id':q['id'],'context':context,'patch':patch,'source_regime':entry['regime'],
                                    'original_direction':entry['direction'],'token_policy':'question','prefill':'full','arithmetic_profile':plan['arithmetic_profile'],
                                    'registration_sha256':data_fingerprint(q),'prompt_ids_sha256':q['prompt_ids_sha256'],
                                    'mask_registration_sha256':entry['mask_registration_sha256'],'keep_bits_sha256':entry['keep_bits_sha256'],
                                    'first_token':result['first_token'],'logits_key':key,'fixed_competitor_id':fixed,
                                    'fixed_pair_margin':result['logits'][0,q['answer_ids'][0]].item()-result['logits'][0,fixed].item()}
                                ref=prior[(name,q['id'],context,patch)]
                                if (any(row[k]!=ref[k] for k in ('first_token','logits_key','fixed_competitor_id','fixed_pair_margin')) or
                                    any(proof[k]!=ref[k] for k in ('prefix_sha256','full_question_cache_sha256','attention23_input_sha256','recurrent_conv_sha256','earlier_position_kv_sha256')) or
                                    not torch.equal(vector,old_vectors[key])):
                                    raise ValueError('Actual source vector/cache/input reference drifted')
                                source_stream.write(json.dumps(row)+'\n');source_rows.append(row);source_proofs.append(proof)
                                state['proofs'].append(proof);state['source_rows']+=1;write_json(path,manifest)
                                if closure:restored=result
                                print(json.dumps({'event':'source_replay','model':name,'id':q['id'],'context':context,'patch':patch,'rows':state['source_rows']}),flush=True)
                            del ctx
                        audit_query(scope['mediation_scope'],entry,source_rows,source_proofs,source_vectors,prefix_ref['sha256'],runtime.tokenizer,manifest['actual_vocab_size'])
                        if restored is None or not torch.equal(unmasked['logits'],restored['logits']):raise ValueError('Source logit closure missing')
                        source_check=verify_sources(unmasked['cache'],restored['cache']);source_check['id']=q['id']
                        state['state_source_checks'].append(source_check)
                        origins={'unmasked':unmasked['cache'],'both_paths_restored':restored['cache']}
                        origin_before={k:cache_record(v) for k,v in origins.items()};native_readout=None
                        for label in plan['condition_order']:
                            work,origin_proof=make_condition(origins,label)
                            key=f"{q['id']}--{label}";initial_key=f"{q['id']}--unmasked--none"
                            observed=observe_continuation(runtime,q,prompt,work,unmasked['logits'],initial_key,key,vectors,origins)
                            if label=='native_reference':
                                native_readout=observed['readout']
                                if native_readout!=entry['expected_full']:raise ValueError('Original full five-field reference drifted')
                            if label=='restore_both' and observed['readout']!=native_readout:raise ValueError('Restored complete cache failed full readout closure')
                            if any(cache_record(v)!=origin_before[k] for k,v in origins.items()):raise ValueError('Continuation changed reused source caches')
                            row={'model':name,'id':q['id'],'condition':label,'source_regime':entry['regime'],
                                'original_direction':entry['direction'],'token_policy':'question','arithmetic_profile':plan['arithmetic_profile'],
                                'registration_sha256':data_fingerprint(q),'prompt_ids_sha256':q['prompt_ids_sha256'],
                                **observed,'origin_proof':origin_proof,'source_caches_unchanged_after_continuation':True}
                            stream.write(json.dumps(row)+'\n');state['recorded_rows']+=1;write_json(path,manifest)
                            print(json.dumps({'event':'state_continuation','model':name,'id':q['id'],'condition':label,'readout':observed['readout'],'rows':state['recorded_rows']}),flush=True)
                            del work
                        del unmasked,restored,result,origins,prefix_ref,native_prefix
                        state['completed_queries']+=1;state.pop('active_query');write_json(path,manifest)
            finally:
                if vectors:
                    save_file(vectors,str(logits_path));state['logits_file_sha256']=streaming_hash(logits_path);state['logits_tensors']=len(vectors)
                if controller is not None:
                    state.update(adapter_state_sha256_after=state_fingerprint(controller),all_adapter_parameters_frozen=all(not p.requires_grad and p.grad is None for p in controller.parameters()))
                if arithmetic is not None:
                    state['actual_affine_calls']=dict(arithmetic.calls);arithmetic.close();state['arithmetic_hooks_removed']=all(not mod._forward_hooks for mod in controller.modules())
                if controller is not None:controller.close()
                for p,k in ((result_path,'results_sha256'),(source_path,'source_results_sha256')):
                    if p.exists():state[k]=streaming_hash(p)
                state['hooks_restored']=hook_state(runtime.model)==native_hooks
            if (state['completed_queries']!=2 or state['source_rows']!=16 or state['recorded_rows']!=10 or
                not state['hooks_restored'] or not state['arithmetic_hooks_removed'] or not state['all_adapter_parameters_frozen'] or
                state['adapter_state_sha256_after']!=state['adapter_state_sha256_before'] or any(v<1 for v in state['actual_affine_calls'].values())):
                raise ValueError('Complete execution, source/vector coverage or cleanup failed')
            with torch.inference_mode():
                for key,expected in anchors.items():
                    q=scope['queries'][key];actual=measure(runtime,actual_prompt(runtime,cases,q),q)
                    state['native_restore_checks'].append({'id':key,'readout':actual,'exact':actual==expected});write_json(path,manifest)
                    if actual!=expected:raise ValueError('Detached native anchor drifted')
            manifest['completed_models'].append(name);write_json(path,manifest)
        manifest['weights_after']=weights_fingerprint(runtime.model)
        if manifest['weights_after']!=before or hook_state(runtime.model)!=native_hooks:raise ValueError('Native weights,gradients or hooks changed')
        manifest.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        if runtime is not None and native_hooks is not None:manifest['hooks_restored']=hook_state(runtime.model)==native_hooks
        write_json(path,manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve all previous runs')
    execute(input_scope(),args.output)
