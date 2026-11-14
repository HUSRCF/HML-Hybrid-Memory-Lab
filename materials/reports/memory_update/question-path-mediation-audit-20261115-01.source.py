"""Audit raw first-logit vectors and every registered oracle path condition."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_fresh_copy_confirmation import audit_readout
from scripts.analyze_fresh_layer_localization import audit_first, sha
from scripts.canonical_necessity_common import audit_affines
from scripts.fresh_layer_controls import tensor_hash
from scripts.fresh_timing_controls import audit_operator, audit_weights
from scripts.question_path_controls import (PROTOCOL_SHA256, PATCHES, CONTEXTS,
    input_scope, sources, audit_receipts)
from scripts.run_controller_trial import write_json


def audit_vector(vector, row, query, tokenizer, vocab_size):
    import torch
    if vector.dtype != torch.float32 or list(vector.shape) != [1, vocab_size] or not torch.isfinite(vector).all():
        raise ValueError('Complete actual FP32 stored first-logit vector required')
    first = row['first_token']; audit_first(first, query)
    values = vector[0]; gold = query['answer_ids'][0]; predicted = values.argmax().item()
    other = values.clone(); other[gold] = -torch.inf; alternative = other.argmax().item()
    if (first['logits_sha256'] != tensor_hash(vector) or first['argmax_id'] != predicted or
        first['non_gold_id'] != alternative or first['gold_logit'] != values[gold].item() or
        first['non_gold_logit'] != values[alternative].item() or
        first['argmax_text'] != tokenizer.decode([predicted], skip_special_tokens=False) or
        not 0 <= row['fixed_competitor_id'] < vocab_size or row['fixed_competitor_id'] == gold or
        row['fixed_pair_margin'] != values[gold].item() - values[row['fixed_competitor_id']].item()):
        raise ValueError('Reported first token, fixed-pair margin or hash contradicts archived full logits')


def audit_query(scope, entry, rows, proofs, vectors, native_prefix, tokenizer, vocab_size):
    import torch
    plan, q = scope['plan'], scope['queries'][entry['id']]
    expected = [(c, p) for c in plan['context_order'] for p in plan['patch_order']]
    if (len(rows) != 8 or len(proofs) != 8 or [(r['context'], r['patch']) for r in rows] != expected or
        [(p['context'], p['patch']) for p in proofs] != expected or
        set(vectors) != {r['logits_key'] for r in rows}):
        raise ValueError('Both contexts, all four patches and all raw vectors required for every query')
    base = proofs[0]; base_receipts = base['patch_receipts']
    refs = {r['context']: r for r in plan['prior_references'] if (r['model'], r['id']) == (entry['model'], q['id'])}
    fixed_competitor = refs['unmasked']['first_token']['non_gold_id']
    config = json.loads((Path(plan['model_path']) / 'config.json').read_text())['text_config']
    dimensions = {**{f'gdn{i}': [1, q['prompt_tokens'], config['hidden_size']] for i in (20,21,22)},
        'k23': [1, q['prompt_tokens'], config['num_key_value_heads'], config['head_dim']],
        'v23': [1, q['prompt_tokens'], config['num_key_value_heads'] * config['head_dim']]}
    for index, (row, proof) in enumerate(zip(rows, proofs)):
        context, patch = expected[index]; ctx = proofs[0 if context == 'unmasked' else 4]
        gold_ref = rows[0]; layers = CONTEXTS[context]['selected_layers']
        if (row['id'] != q['id'] or row['model'] != entry['model'] or row['source_regime'] != entry['regime'] or
            row['original_direction'] != entry['direction'] or row['token_policy'] != 'question' or row['prefill'] != 'full' or
            row['arithmetic_profile'] != plan['arithmetic_profile'] or row['registration_sha256'] != data_fingerprint(q) or
            row['prompt_ids_sha256'] != q['prompt_ids_sha256'] or row['fixed_competitor_id'] != fixed_competitor or
            row['logits_key'] != f"{q['id']}--{context}--{patch}" or
            any(row[k] != entry[k] for k in ('mask_registration_sha256','keep_bits_sha256')) or
            proof['id'] != q['id'] or proof['selected_layers'] != layers or
            not proof['complete_prefix_memory_exact'] or not proof['prefix_logits_exact'] or
            proof['prefix_sha256'] != native_prefix or proof['masked_prefix_sha256'] != native_prefix or
            proof['historical_kv_fields_exact'] != 12 or not proof['hooks_restored'] or
            proof['final_row_kv23_original_projection_exact_fields'] != 2 or not proof['gdn20_written_states_unchanged']):
            raise ValueError('Original input, patch, prefix/ledgerKV, current K/V or source-state proof changed')
        audit_vector(vectors[row['logits_key']], row, q, tokenizer, vocab_size)
        audit_operator(proof['calls'], proof['receipts'], proof['batches'], layers, q,
            'earlier_question_tokens', [q['prompt_tokens'], q['ledger_tokens']])
        audit_weights(proof['attention_stats'], layers, q, 'earlier_question_tokens')
        audit_receipts(proof['patch_receipts'], proof['input_calls'], q, patch, base_receipts)
        for name, receipt in proof['patch_receipts'].items():
            if receipt['shape'] != dimensions[name]: raise ValueError('Native tensor target dimensions changed')
            for field in ('target_before_sha256','target_after_sha256','target_reference_sha256',
                          'untargeted_before_sha256','untargeted_after_sha256'): sha(receipt[field])
        for field in ('full_question_cache_sha256','attention23_input_sha256','attention23_last_input_sha256',
                      'attention23_earlier_input_sha256','layer20_last_input_sha256'): sha(proof[field])
        for field in ('question_kv23_sha256','earlier_position_kv_sha256','recurrent_conv_sha256'):
            for value in proof[field].values(): sha(value)
        if (set(proof['question_kv23_sha256']) != {'keys','values'} or
            set(proof['earlier_position_kv_sha256']) != {f'{i}:{k}' for i in (3,7,11,15,19,23) for k in ('keys','values')} or
            len(proof['recurrent_conv_sha256']) != 36 or set(proof['recurrent_conv_sha256']) != set(base['recurrent_conv_sha256']) or
            proof['layer20_last_input_sha256'] != base['layer20_last_input_sha256'] or
            proof['attention23_earlier_input_sha256'] != ctx['attention23_earlier_input_sha256']):
            raise ValueError('Complete cache-region coverage or layer20/current-patch causal boundary changed')
        s20 = lambda data: {k:v for k,v in data.items() if k.startswith('(20,')}
        if len(s20(proof['recurrent_conv_sha256'])) != 2 or s20(proof['recurrent_conv_sha256']) != s20(ctx['recurrent_conv_sha256']):
            raise ValueError('GDN20 already-written states changed under output patch')
        if PATCHES[patch]['gdn_current_outputs'] and proof['attention23_last_input_sha256'] != base['attention23_last_input_sha256']:
            raise ValueError('GDN output patch failed to restore actual final-reader input row')
        if PATCHES[patch]['question_kv'] and proof['question_kv23_sha256'] != base['question_kv23_sha256']:
            raise ValueError('Question K/V after natural RoPE not restored')
        if patch == 'question_kv' and proof['attention23_input_sha256'] != ctx['attention23_input_sha256']:
            raise ValueError('Internal K/V patch changed its input representation')
        if patch == 'gdn_current_outputs' and proof['earlier_position_kv_sha256'] != ctx['earlier_position_kv_sha256']:
            raise ValueError('Current output patch changed earlier-position K/V')
        if patch in ('question_kv','both'):
            source = ctx if patch == 'question_kv' else proofs[1 if context == 'unmasked' else 5]
            if not proof['kv_patch_recurrent_states_unchanged'] or proof['recurrent_conv_sha256'] != source['recurrent_conv_sha256']:
                raise ValueError('K/V patch modified an earlier recurrent/conv state')
        elif proof['kv_patch_recurrent_states_unchanged'] is not None:
            raise ValueError('State proof mislabeled')
        if context == 'unmasked' and (row['first_token'] != gold_ref['first_token'] or
                proof['full_question_cache_sha256'] != base['full_question_cache_sha256'] or
                proof['attention23_input_sha256'] != base['attention23_input_sha256'] or
                not torch.equal(vectors[row['logits_key']], vectors[gold_ref['logits_key']])):
            raise ValueError('Unmasked oracle self-to-self sham is not complete-cache/input/logit identity')
        if patch == 'none' and (row['first_token'] != refs[context]['first_token'] or
                any(proof[k] != refs[context][k] for k in
                    ('prefix_sha256','full_question_cache_sha256','attention23_input_sha256'))):
            raise ValueError('Exact original unpatched reference drifted')
        closure = context == 'earlier_only_layer19' and patch == 'both'
        if closure:
            if not proof['combined_paths_logits_exact'] or not torch.equal(vectors[row['logits_key']], vectors[gold_ref['logits_key']]):
                raise ValueError('Combined oracle paths failed exact first-logit closure')
        elif proof['combined_paths_logits_exact'] is not None:
            raise ValueError('Closure proof mislabeled')
    effects = {}
    for context in plan['context_order']:
        r = {x['patch']: x for x in rows if x['context'] == context}
        v = {k:x['fixed_pair_margin'] for k,x in r.items()}
        effects[context] = {'fixed_pair_margins': v,
            'gdn_output_effect': v['gdn_current_outputs'] - v['none'],
            'question_kv_effect': v['question_kv'] - v['none'],
            'combined_effect': v['both'] - v['none'],
            'patch_margin_difference_in_differences': v['both'] - v['gdn_current_outputs'] - v['question_kv'] + v['none'],
            'first_tokens': {k:x['first_token'] for k,x in r.items()}}
    return {'model':entry['model'],'id':q['id'],'source_regime':entry['regime'],'original_direction':entry['direction'],
        'bindings':entry['bindings'],'gold_first_id':q['answer_ids'][0],'fixed_competitor_id':fixed_competitor,'effects':effects}


def audit_run(path):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    path=Path(path); scope=input_scope(); plan=scope['plan']; manifest=json.loads((path/'manifest.json').read_text())
    after=manifest['weights_after']
    if (manifest['status']!='completed' or not all(manifest[k] for k in ('posthoc','selection_is_posthoc','no_accuracy_gate','save_full_first_logits')) or
        any(manifest[k] for k in ('training','new_adapters','new_data','independent_scientific_confirmation','can_change_primary_gates','full_generation_scored')) or
        manifest['protocol_sha256']!=PROTOCOL_SHA256 or json.loads((path/'plan.json').read_text())!=plan or
        streaming_hash(path/'installed_modeling_source.py')!=plan['installed_modeling_source_sha256'] or
        json.loads((path/'precision_addendum.json').read_text())!=scope['profile'] or
        manifest['original_primary_gates_unchanged']!=scope['primary_gates'] or
        any(manifest[k]!=plan[k] for k in ('model_path','packages','dataset_manifest_sha256','mask_registry_sha256',
            'arithmetic_profile','precision_addendum_sha256','prior_timing_audit_sha256','prior_timing_manifest_sha256','installed_modeling_source_sha256')) or
        manifest['actual_eos_token_id']!=scope['primary_manifest']['actual_eos_token_id'] or
        manifest['generation_config_sha256']!=scope['primary_manifest']['generation_config_sha256'] or
        manifest['generation_config_sha256']!={p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')} or
        manifest['actual_vocab_size']!=json.loads((Path(plan['model_path'])/'config.json').read_text())['text_config']['vocab_size'] or
        manifest['model_order']!=plan['models'] or manifest['completed_models']!=plan['models'] or list(manifest['models'])!=plan['models'] or
        set(manifest['source_sha256'])!=sources() or manifest['weights_before']!=after or after['sha256']!=plan['native_loaded_weights_sha256'] or
        not after['all_parameters_frozen'] or not after['no_parameter_gradients'] or not manifest['hooks_restored'] or
        not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(manifest['started_utc'])<datetime.fromisoformat(manifest['completed_utc'])):
        raise ValueError('Frozen oracle scope, runtime, sources, chronology or native weights changed')
    for name,digest in manifest['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:
            raise ValueError('Actual or archived path-mediation source changed')
    keys=[q['id'] for q in plan['queries']]
    if [r['id'] for r in manifest['native_anchors']]!=keys:raise ValueError('All four native anchors required')
    anchors={}
    for r in manifest['native_anchors']:
        audit_readout(r['readout'],scope['queries'][r['id']])
        if not r['exact_completed_fresh_native_replay'] or r['readout']!=scope['native_references'][r['id']]:raise ValueError('Native anchor drifted')
        anchors[r['id']]=r['readout']
    tokenizer=AutoTokenizer.from_pretrained(plan['model_path'],local_files_only=True); cases=[]; total=detach=0
    for name in plan['models']:
        state=manifest['models'][name]; spec=scope['source_plan']['models'][name]
        selected=[q for q in plan['queries'] if q['model']==name]; ids=[q['id'] for q in selected]
        if ({k:state[k] for k in plan['fixed_checkpoints'][name]}!=plan['fixed_checkpoints'][name] or
            state['adapter_parameters']!=spec['parameters'] or state['adapter_state_sha256_after']!=state['adapter_state_sha256_before'] or
            not state['all_adapter_parameters_frozen'] or not state['arithmetic_hooks_removed'] or not state['hooks_restored'] or
            state['token_policy']!='question' or state['completed_queries']!=2 or state['recorded_rows']!=16 or state['logits_tensors']!=16 or
            'active_query' in state or len(state['proofs'])!=16 or [p['id'] for p in state['prefix_checks']]!=ids):
            raise ValueError('Frozen controller, complete vector coverage or cleanup changed')
        audit_affines(state,plan['arithmetic_profile'])
        result_path=path/f'{name}.jsonl'; logits_path=path/f'{name}.logits.safetensors'
        if streaming_hash(result_path)!=state['results_sha256'] or streaming_hash(logits_path)!=state['logits_file_sha256']:
            raise ValueError('Actual first-token rows or full-vector file changed')
        rows=[json.loads(line) for line in result_path.read_text().splitlines()]; vectors=load_file(str(logits_path))
        if len(rows)!=16 or set(vectors)!={r['logits_key'] for r in rows}:raise ValueError('All16raw vectors per model required')
        for i,entry in enumerate(selected):
            p=state['prefix_checks'][i]; q=scope['queries'][entry['id']]
            if (p['source_key']!=q['source_key'] or p['ledger_tokens']!=q['ledger_tokens'] or
                not p['complete_KV_recurrent_conv_bookkeeping_exact'] or not p['last_prefix_logits_exact']):
                raise ValueError('Actual native ledger-prefix memory/logits changed')
            sha(p['prefix_sha256']); sl=slice(i*8,(i+1)*8); sub=rows[sl]
            cases.append(audit_query(scope,entry,sub,state['proofs'][sl],{r['logits_key']:vectors[r['logits_key']] for r in sub},
                p['prefix_sha256'],tokenizer,manifest['actual_vocab_size']))
        if [r['id'] for r in state['native_restore_checks']]!=keys or any(not r['exact'] or r['readout']!=anchors[r['id']] for r in state['native_restore_checks']):
            raise ValueError('All native anchors must replay after each controller detach')
        total+=len(rows);detach+=len(state['native_restore_checks'])
    if (total,detach,len(cases))!=(32,8,4):raise ValueError('Complete32readout prescribed mediation required')
    return {'status':'completed_and_audited_posthoc_question_path_mediation','operational_controls_passed':True,
        'posthoc':True,'selection_is_posthoc':True,'independent_scientific_confirmation':False,'can_change_primary_gates':False,
        'no_accuracy_gate':True,'full_generation_scored':False,'source_run':str(path),
        'manifest_sha256':streaming_hash(path/'manifest.json'),'protocol_sha256':PROTOCOL_SHA256,
        'prior_timing_audit_sha256':plan['prior_timing_audit_sha256'],'weights_sha256':after['sha256'],
        'arithmetic_profile':plan['arithmetic_profile'],'original_primary_gates_unchanged':scope['primary_gates'],
        'counts':{'physical_first_token_readouts':32,'archived_full_vectors':32,'exact_prior_unpatched_references':8,
            'unmasked_oracle_patch_shams':12,'intervened_oracle_patch_readouts':12,'combined_path_exact_logit_closures':4,
            'actual_row_mask_and_prefix_KV_proofs':32,'actual_tensor_patch_receipts':160,'native_prefix_checks':4,
            'precontroller_native_anchors':4,'detached_native_replays':8},
        'cases':cases,'interpretation':'Oracle patches diagnose GDN branch outputs versus earlier-question Attention23 K/V in four reused cases. GDN outputs include recurrence, convolution, gates, normalization and projection; output patching does not identify recurrent storage loss or repair state writes. Combined first-logit closure is an operator control, not a deployable method, future-state preservation, full-answer result or architecture contribution.',
        'analysis_source_sha256':streaming_hash(Path(__file__))}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous audits')
    report=audit_run(args.run);write_json(args.output,report);source=args.output.with_suffix('.source.py');source.write_bytes(Path(__file__).read_bytes())
    rows=[json.loads(line) for name in ('local_question','gdn_question_only') for line in (args.run/f'{name}.jsonl').read_text().splitlines()]
    table=args.output.with_suffix('.csv')
    with table.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows([{**r,'first_token':json.dumps(r['first_token'],sort_keys=True)} for r in rows])
    write_json(args.output.with_suffix('.provenance.json'),{'source_manifest_sha256':report['manifest_sha256'],'counts':report['counts'],
        'artifacts_sha256':{str(p):streaming_hash(p) for p in (args.output,source,table)},
        'runtime_sources_sha256':json.loads((args.run/'manifest.json').read_text())['source_sha256']})
    print(json.dumps({'status':report['status'],'counts':report['counts']}),flush=True)
