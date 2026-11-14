"""Independently audit raw source/continuation vectors and all20 cache origins."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_fresh_copy_confirmation import audit_readout
from scripts.analyze_question_path_mediation import audit_query
from scripts.canonical_necessity_common import audit_affines
from scripts.fresh_layer_controls import tensor_hash
from scripts.run_controller_trial import write_json
from scripts.state_continuation_controls import (PROTOCOL_SHA256,CONDITIONS,TARGETS,
    input_scope,sources,gold_losses,decode_vectors)


def audit_cache_records(source_check,rows,source_refs,query):
    from scripts.analyze_fresh_layer_localization import sha
    targets={str(k) for k in TARGETS}
    expected={str((i,f,0)) for i in range(24) for f in
        (('keys','values') if i%4==3 else ('recurrent_states','conv_states'))}
    origins={k:source_check[k] for k in ('unmasked','both_paths_restored')}
    if (source_check['id']!=query['id'] or source_check['complete_KV_exact_fields']!=12 or
        not source_check['all_other_tensors_exact'] or not source_check['complete_bookkeeping_exact'] or
        origins['unmasked']['metadata']!=origins['both_paths_restored']['metadata']):
        raise ValueError('Source complete K/V or bookkeeping proof incomplete')
    for name,record in origins.items():
        ref=source_refs[('unmasked','none') if name=='unmasked' else ('earlier_only_layer19','both')]
        if record['sha256']!=ref['full_question_cache_sha256'] or set(record['tensors'])!=expected:
            raise ValueError('Actual source cache coverage/fingerprint changed')
        for key,tensor in record['tensors'].items():
            sha(tensor['sha256'])
            if key in ref['recurrent_conv_sha256'] and tensor['sha256']!=ref['recurrent_conv_sha256'][key]:
                raise ValueError('Source state differs from registered actual observation')
            if any(key==str((i,f,0)) for i in (3,7,11,15,19,23) for f in ('keys','values')):
                if tensor['shape'][2]!=query['prompt_tokens']:raise ValueError('Full K/V must include final question row')
    changed=[k for k in origins['unmasked']['tensors'] if origins['unmasked']['tensors'][k]!=origins['both_paths_restored']['tensors'][k]]
    if set(changed)-targets or changed!=source_check['changed_tensor_keys']:
        raise ValueError('Unregistered source state or K/V difference')
    for row in rows:
        label=row['condition'];proof=row['origin_proof'];rule=CONDITIONS[label];record=proof['starting_cache']
        if (proof['condition']!=label or set(proof['origins'])!=expected or set(record['tensors'])!=expected or
            record['metadata']!=origins['unmasked']['metadata'] or
            not proof['source_caches_unchanged_after_transplant'] or not proof['all_tensors_unaliased']):
            raise ValueError('Starting cache origin, layout or immutability proof incomplete')
        for key,receipt in proof['origins'].items():
            field=next((f for i,f,j in TARGETS if key==str((i,f,j))),None)
            name=rule[field] if field else 'unmasked'
            if receipt!={'origin':name,**origins[name]['tensors'][key]} or record['tensors'][key]!=origins[name]['tensors'][key]:
                raise ValueError('Cache tensor came from an unregistered source')
        if label in ('native_reference','restore_both') and record!=origins['unmasked']:
            raise ValueError('Complete native/restored cache identity failed')
        sha(record['sha256'])
        if any(row[k]!=record['sha256'] for k in
            ('starting_cache_sha256','after_gold_cache_sha256','after_greedy_cache_sha256')):
            raise ValueError('Cached scoring or decoding mutated its start cache')
        expected_fingerprints={k:v['sha256'] for k,v in origins.items()}
        if (row['origin_cache_fingerprints']!={k:expected_fingerprints for k in ('before','after_gold','after_greedy')} or
            not row['origin_cache_metadata_unchanged_after_each_operation']):
            raise ValueError('Reused origin-cache immutability not proved after each operation')


def audit_continuation(row,query,entry,vectors,tokenizer,vocab,eos,device):
    import torch
    initial_key=f"{query['id']}--unmasked--none";first=vectors[initial_key]
    if (row['model']!=entry['model'] or row['id']!=query['id'] or row['source_regime']!=entry['regime'] or
        row['original_direction']!=entry['direction'] or row['token_policy']!='question' or
        row['registration_sha256']!=data_fingerprint(query) or row['prompt_ids_sha256']!=query['prompt_ids_sha256'] or
        row['initial_logits_key']!=initial_key or row['initial_logits_sha256']!=tensor_hash(first) or
        row['gold_ids']!=query['answer_ids'] or not row['hooks_restored'] or
        not row['start_cache_unchanged_after_each_operation'] or not row['source_caches_unchanged_after_continuation']):
        raise ValueError('Original query, first logits or immutability scope changed')
    used={initial_key};forward_vectors=[]
    for kind in ('gold','greedy'):
        calls=row[f'{kind}_forwards']
        if kind=='gold' and len(calls)!=int(len(query['answer_ids'])>1):raise ValueError('Gold forward coverage changed')
        for i,call in enumerate(calls):
            ids=query['answer_ids'][:-1] if kind=='gold' else row['decode']['generated_ids'][i:i+1]
            start=query['prompt_tokens']+(0 if kind=='gold' else i);length=len(ids)
            key=f"{query['id']}--{row['condition']}--{kind}--{i}";vector=vectors[call['logits_key']]
            if (call['logits_key']!=key or call['input_ids']!=[ids] or call['start']!=start or
                call['position_ids']!=[[list(range(start,start+length))]]*3 or call['logits_to_keep']!=length or
                call['logits_sha256']!=tensor_hash(vector) or call['logits_shape']!=[1,length,vocab] or
                list(vector.shape)!=[1,length,vocab] or vector.dtype!=torch.float32 or not torch.isfinite(vector).all()):
                raise ValueError('Actual continuation vector, decode IDs, forward shape or RoPE changed')
            used.add(key)
            if kind=='greedy':forward_vectors.append(vector)
    decoded=decode_vectors(tokenizer,first,forward_vectors,eos)
    if decoded!=row['decode'] or row['readout']['generated']!=decoded['generated']:
        raise ValueError('Greedy answer disagrees with raw step logits')
    for i,call in enumerate(row['greedy_forwards']):
        if call['input_ids']!=[[decoded['generated_ids'][i]]]:raise ValueError('Actual greedy input differs from argmax decision')
    continuation=vectors[row['gold_forwards'][0]['logits_key']] if row['gold_forwards'] else None
    loss=gold_losses(first,continuation,query['answer_ids'],device)
    if loss!=row['gold_losses'] or row['readout']['answer_nll']!=loss['nll_mean']:
        raise ValueError('Exact original-shape GPU gold loss reconstruction failed')
    audit_readout(row['readout'],query)
    if row['condition'] in ('native_reference','restore_both') and row['readout']!=entry['expected_full']:
        raise ValueError('Original complete full-answer reference not replayed')
    return used


def audit_run(path):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    path=Path(path);scope=input_scope();plan=scope['plan'];manifest=json.loads((path/'manifest.json').read_text())
    after=manifest['weights_after']
    if (manifest['status']!='completed' or
        not all(manifest[k] for k in ('posthoc','selection_is_posthoc','no_accuracy_gate','full_generation_scored','save_full_logits')) or
        any(manifest[k] for k in ('training','new_adapters','new_data','independent_scientific_confirmation','can_change_primary_gates')) or
        manifest['protocol_sha256']!=PROTOCOL_SHA256 or json.loads((path/'plan.json').read_text())!=plan or
        streaming_hash(path/'installed_modeling_source.py')!=plan['installed_modeling_source_sha256'] or
        json.loads((path/'precision_addendum.json').read_text())!=scope['profile'] or
        manifest['original_primary_gates_unchanged']!=scope['primary_gates'] or
        any(manifest[k]!=plan[k] for k in ('model_path','packages','dataset_manifest_sha256','mask_registry_sha256',
            'arithmetic_profile','precision_addendum_sha256','prior_mediation_audit_sha256','prior_mediation_manifest_sha256','installed_modeling_source_sha256')) or
        {n:importlib_version(n) for n in plan['packages']}!=plan['packages'] or
        manifest['actual_eos_token_id']!=scope['primary_manifest']['actual_eos_token_id'] or
        manifest['generation_config_sha256']!=scope['primary_manifest']['generation_config_sha256'] or
        manifest['generation_config_sha256']!={p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')} or
        manifest['actual_vocab_size']!=json.loads((Path(plan['model_path'])/'config.json').read_text())['text_config']['vocab_size'] or
        manifest['model_order']!=plan['models'] or manifest['completed_models']!=plan['models'] or list(manifest['models'])!=plan['models'] or
        set(manifest['source_sha256'])!=sources() or manifest['weights_before']!=after or after['sha256']!=plan['native_loaded_weights_sha256'] or
        not after['all_parameters_frozen'] or not after['no_parameter_gradients'] or not manifest['hooks_restored'] or
        not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(manifest['started_utc'])<datetime.fromisoformat(manifest['completed_utc'])):
        raise ValueError('Frozen continuation provenance, complete runtime, chronology or weights changed')
    for name,digest in manifest['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Preserve all sealed runtime sources')
    ids=[e['id'] for e in plan['queries']];anchors={}
    if [r['id'] for r in manifest['native_anchors']]!=ids:raise ValueError('Four native anchors required')
    for r in manifest['native_anchors']:
        if not r['exact_completed_fresh_native_replay'] or r['readout']!=scope['native_references'][r['id']]:raise ValueError('Native anchor drifted')
        anchors[r['id']]=r['readout']
    tokenizer=AutoTokenizer.from_pretrained(plan['model_path'],local_files_only=True)
    torch.set_num_threads(4);device='cuda:0';cases=[];sources_total=readouts=raw_vectors=greedy_forwards=gold_forwards=0
    refs={(r['model'],r['id'],r['context'],r['patch']):r for r in plan['source_references']}
    for name in plan['models']:
        state=manifest['models'][name];spec=scope['source_plan']['models'][name];selected=[e for e in plan['queries'] if e['model']==name]
        if ({k:state[k] for k in plan['fixed_checkpoints'][name]}!=plan['fixed_checkpoints'][name] or
            state['adapter_parameters']!=spec['parameters'] or state['adapter_state_sha256_after']!=state['adapter_state_sha256_before'] or
            not all(state[k] for k in ('all_adapter_parameters_frozen','arithmetic_hooks_removed','hooks_restored')) or
            state['token_policy']!='question' or (state['completed_queries'],state['source_rows'],state['recorded_rows'])!=(2,16,10) or
            'active_query' in state or len(state['proofs'])!=16 or len(state['state_source_checks'])!=2 or
            [p['id'] for p in state['prefix_checks']]!=[e['id'] for e in selected]):
            raise ValueError('Frozen controller, complete readout coverage or cleanup changed')
        audit_affines(state,plan['arithmetic_profile'])
        for suffix,key in (('.jsonl','results_sha256'),('.sources.jsonl','source_results_sha256'),('.logits.safetensors','logits_file_sha256')):
            if streaming_hash(path/f'{name}{suffix}')!=state[key]:raise ValueError('Actual continuation/source/vector file changed')
        rows=[json.loads(x) for x in (path/f'{name}.jsonl').read_text().splitlines()]
        source_rows=[json.loads(x) for x in (path/f'{name}.sources.jsonl').read_text().splitlines()]
        vectors=load_file(str(path/f'{name}.logits.safetensors'));old_vectors=load_file(str(Path(plan['prior_mediation_run'])/f'{name}.logits.safetensors'))
        if len(rows)!=10 or len(source_rows)!=16 or len(vectors)!=state['logits_tensors']:raise ValueError('Full actual observation coverage required')
        used=set()
        for i,entry in enumerate(selected):
            q=scope['queries'][entry['id']];sl=slice(i*8,(i+1)*8);src=source_rows[sl];proofs=state['proofs'][sl];prefix=state['prefix_checks'][i]
            if (prefix['source_key']!=q['source_key'] or prefix['ledger_tokens']!=q['ledger_tokens'] or
                not prefix['complete_KV_recurrent_conv_bookkeeping_exact'] or not prefix['last_prefix_logits_exact']):
                raise ValueError('Original native ledger-prefix identity failed')
            audit_query(scope['mediation_scope'],entry,src,proofs,{r['logits_key']:vectors[r['logits_key']] for r in src},prefix['prefix_sha256'],tokenizer,manifest['actual_vocab_size'])
            source_refs={}
            for row,proof in zip(src,proofs,strict=True):
                ref=refs[(name,q['id'],row['context'],row['patch'])];source_refs[(row['context'],row['patch'])]=ref
                if (any(row[k]!=ref[k] for k in ('first_token','logits_key','fixed_competitor_id','fixed_pair_margin')) or
                    any(proof[k]!=ref[k] for k in ('prefix_sha256','full_question_cache_sha256','attention23_input_sha256','recurrent_conv_sha256','earlier_position_kv_sha256')) or
                    not torch.equal(vectors[row['logits_key']],old_vectors[row['logits_key']])):
                    raise ValueError('Exact raw source reference drifted')
                used.add(row['logits_key'])
            sub=rows[i*5:(i+1)*5]
            if [r['condition'] for r in sub]!=list(CONDITIONS):raise ValueError('All five registered conditions required in order')
            audit_cache_records(state['state_source_checks'][i],sub,source_refs,q)
            for row in sub:
                if row['arithmetic_profile']!=plan['arithmetic_profile']:raise ValueError('Arithmetic profile changed')
                used|=audit_continuation(row,q,entry,vectors,tokenizer,manifest['actual_vocab_size'],manifest['actual_eos_token_id'],device)
                greedy_forwards+=len(row['greedy_forwards']);gold_forwards+=len(row['gold_forwards'])
            native=sub[0];restored=sub[-1]
            if any(native[k]!=restored[k] for k in ('readout','gold_losses','decode')):raise ValueError('Full state restoration failed exact functional closure')
            for kind in ('gold','greedy'):
                left,right=native[f'{kind}_forwards'],restored[f'{kind}_forwards']
                if len(left)!=len(right) or any(not torch.equal(vectors[x['logits_key']],vectors[y['logits_key']]) for x,y in zip(left,right,strict=True)):
                    raise ValueError('Complete-cache restoration did not exactly replay all later logits')
            losses={r['condition']:r['gold_losses']['nll_mean'] for r in sub}
            cases.append({'model':name,'id':q['id'],'source_regime':entry['regime'],'original_direction':entry['direction'],
                'bindings':entry['bindings'],'readouts':{r['condition']:r['readout'] for r in sub},'mean_gold_losses':losses,
                'convolution_restore_effect':losses['restore_convolution']-losses['intervened_states'],
                'recurrent_restore_effect':losses['restore_recurrent']-losses['intervened_states'],
                'four_point_loss_contrast':losses['restore_both']-losses['restore_convolution']-losses['restore_recurrent']+losses['intervened_states'],
                'changed_answers_vs_native':{r['condition']:r['readout']['generated']!=native['readout']['generated'] for r in sub},
                'gold_losses':{r['condition']:r['gold_losses'] for r in sub},
                'gold_later_vector_exact_to_native':{r['condition']:all(torch.equal(vectors[c['logits_key']],vectors[n['logits_key']]) for c,n in zip(r['gold_forwards'],native['gold_forwards'],strict=True)) for r in sub}})
        if used!=set(vectors):raise ValueError('Missing or unreferenced raw vectors')
        if [r['id'] for r in state['native_restore_checks']]!=ids or any(not r['exact'] or r['readout']!=anchors[r['id']] for r in state['native_restore_checks']):
            raise ValueError('All native anchors must replay after each detach')
        sources_total+=len(source_rows);readouts+=len(rows);raw_vectors+=len(vectors)
    if (sources_total,readouts,len(cases))!=(32,20,4):raise ValueError('Complete registered source and continuation panel required')
    return {'status':'completed_and_audited_posthoc_state_continuation','operational_controls_passed':True,
        'posthoc':True,'selection_is_posthoc':True,'independent_scientific_confirmation':False,'can_change_primary_gates':False,
        'no_accuracy_gate':True,'full_generation_scored':True,'source_run':str(path),'manifest_sha256':streaming_hash(path/'manifest.json'),
        'protocol_sha256':PROTOCOL_SHA256,'arithmetic_profile':plan['arithmetic_profile'],'weights_sha256':after['sha256'],
        'original_primary_gates_unchanged':scope['primary_gates'],'counts':{'exact_source_first_readouts':32,
            'cached_continuation_readouts':20,'archived_full_vectors':raw_vectors,'actual_greedy_forwards':greedy_forwards,
            'actual_gold_forwards':gold_forwards,'complete_KV_source_identity_fields':48,'complete_cache_origin_receipts':20*48,
            'exact_original_full_readout_references':4,'exact_restore_both_all_step_closures':4,'native_prefix_checks':4,
            'precontroller_native_anchors':4,'detached_native_replays':8},'cases':cases,
        'interpretation':'Conditional six-state cache cuts after oracle branch/KV first-logit closure in four reused cases. Complete K/V, all other tensors, bookkeeping and initial logits are fixed. Full-answer/loss differences concern these query-time states only; no original-prefix write damage, learned semantic scope, deployable method or architecture superiority is established. End this four-case localization series; further effectiveness claims require separately registered unseen streams.',
        'analysis_source_sha256':streaming_hash(Path(__file__))}


def importlib_version(name):
    import importlib.metadata
    return importlib.metadata.version(name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous audit')
    report=audit_run(args.run);write_json(args.output,report);source=args.output.with_suffix('.source.py');source.write_bytes(Path(__file__).read_bytes())
    rows=[json.loads(x) for name in ('local_question','gdn_question_only') for x in (args.run/f'{name}.jsonl').read_text().splitlines()]
    table=args.output.with_suffix('.csv')
    with table.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','id','condition','source_regime','original_direction','generated','greedy_exact','nll_sum','nll_mean','first_token_nll','continuation_nll'])
        writer.writeheader();writer.writerows({k:(r[k] if k in r else r['readout'][k] if k in ('generated','greedy_exact') else r['gold_losses'][k]) for k in writer.fieldnames} for r in rows)
    write_json(args.output.with_suffix('.provenance.json'),{'source_manifest_sha256':report['manifest_sha256'],'counts':report['counts'],
        'artifacts_sha256':{str(p):streaming_hash(p) for p in (args.output,source,table)},'runtime_sources_sha256':json.loads((args.run/'manifest.json').read_text())['source_sha256']})
    print(json.dumps({'status':report['status'],'counts':report['counts']}),flush=True)
