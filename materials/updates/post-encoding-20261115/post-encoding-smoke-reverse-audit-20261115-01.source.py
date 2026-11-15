"""Independent coverage, cache, gate, native-replay and outcome audit."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json
from scripts.post_encoding_data import (read_plan,registered_inputs,sources,selected_contexts,context_queries,
    DESTINATION,PROTOCOL,MODES,READERS)
from scripts.post_encoding_runtime import check_gate_stats,check_cached_attention
from scripts.post_encoding_metrics import summarize
from scripts.copy_readout_runtime import readout
from scripts.analyze_trained_timing_evidence_access import audit_readout
from scripts.analyze_gate_components import audit_kernel_checks
from scripts.canonical_necessity_common import audit_affines
from scripts.gate_component_runtime import validation_addendum,VALIDATION_ADDENDUM_SHA256,GDN_LAYERS


def core_cache(record):return {k:v for k,v in record.items() if k!='source_guard_checks'}


def audit_cache(record,tokens,checks):
    expected=({(i,'recurrent_states',0) for i in GDN_LAYERS}|{(i,'conv_states',0) for i in GDN_LAYERS}|
        {(i,f,0) for i in (3,7,11,15,19,23) for f in ('keys','values')})
    if (set(record['tensors'])!={str(k) for k in expected} or set(record['layouts'])!=set(record['tensors']) or
        record['source_guard_checks']!=checks or len(record['metadata']['layers'])!=24 or
        record['metadata']['original_metadata']['sequence_length']!=tokens):
        raise ValueError('Complete guarded48tensor cache/layout or sequence bookkeeping changed')
    for key,t in record['tensors'].items():
        if len(t['sha256'])!=64 or len(record['logits_sha256'])!=64 or not t['shape'] or not t['dtype']:
            raise ValueError('Cache tensor/logits byte fingerprints required')
        if ("'keys'" in key or "'values'" in key) and t['shape'][-2]!=tokens:raise ValueError('KV physical length changed')


def audit_reader(proof,q,reader):
    check_cached_attention(proof['attention'],reader=='oracle')
    if proof['reader_keep_sha256']!=data_fingerprint(q['history_keep_bits']):raise ValueError('Question-specific evidence reader changed')
    if reader=='oracle':
        if proof['reader_calls']['prefill']!=0 or proof['reader_calls']['cached']<1:raise ValueError('Cached-only reader required')
    elif proof['reader_calls'] is not None:raise ValueError('Full reader unexpectedly restricted')


def audit_run(path,allow_smoke=False):
    path=Path(path);m=json.loads((path/'manifest.json').read_text());plan=read_plan()
    registered,cases,registry,queries,references=registered_inputs();oldrefs=references['predecessor_manifest']['references']
    correction=validation_addendum();order=plan['model_order'][::-1] if m['reverse_model_order'] else plan['model_order']
    weights=m['weights_after'];profile=json.loads(Path(plan['precision_addendum']).read_text())
    if (m['status']!='completed' or (m['smoke'] and not allow_smoke) or m['training'] or m['new_adapters'] or
        (m['reverse_model_order'] and not m['smoke']) or m['model_order']!=order or m['completed_models']!=order or list(m['models'])!=order or
        m['dataset']!=str(DESTINATION) or m['dataset_manifest_sha256']!=streaming_hash(DESTINATION/'manifest.json') or
        m['protocol_sha256']!=streaming_hash(PROTOCOL) or m['source_sha256']!=registered['source_sha256'] or set(m['source_sha256'])!=sources() or
        m['model_path']!=plan['model_path'] or m['packages']!=plan['packages'] or m['generation']!=plan['generation'] or
        m['actual_eos_token_id']!=oldrefs['actual_eos_token_id'] or m['generation_config_sha256']!=oldrefs['generation_config_sha256'] or
        m['generation_config_sha256']!={p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')} or
        m['precision_addendum_sha256']!=streaming_hash(Path(plan['precision_addendum'])) or m['arithmetic_profile']!=profile['profile'] or
        m['validation_addendum_sha256']!=VALIDATION_ADDENDUM_SHA256 or json.loads((path/'plan.json').read_text())!=plan or
        json.loads((path/'precision_addendum.json').read_text())!=profile or json.loads((path/'kernel_validation_addendum.json').read_text())!=correction or
        streaming_hash(path/'installed_modeling_source.py')!=plan['installed_modeling_source_sha256'] or
        streaming_hash(Path(plan['installed_modeling_source']))!=plan['installed_modeling_source_sha256'] or
        weights!=m['weights_before'] or weights['sha256']!=plan['native_loaded_weights_sha256'] or not weights['all_parameters_frozen'] or
        not weights['no_parameter_gradients'] or not m['hooks_restored'] or not m['question_only_native_memory_exact'] or
        not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(registered['created_utc'])<
            datetime.fromisoformat(m['started_utc'])<datetime.fromisoformat(m['completed_utc'])):
        raise ValueError('Frozen runtime,206sources, complete allocation or chronological registration changed')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Measured source bytes changed')
    audit_kernel_checks(m['native_component_kernel_checks'])
    if m['smoke']:
        if m['smoke_equivalence_gate'] is not None:raise ValueError('Unexpected smoke gate')
    else:
        from scripts.run_post_encoding import checked_gate
        gate=m['smoke_equivalence_gate'];eq=json.loads(Path(gate['path']).read_text())
        if (checked_gate(Path(gate['path']),m['dataset_manifest_sha256'])!=gate or eq['runtime_source_sha256']!=m['source_sha256'] or
            any(datetime.fromisoformat(t)>=datetime.fromisoformat(m['started_utc']) for t in eq['smoke_completed_utc'])):
            raise ValueError('Opposite-order exact operational smokes must precede primary inference')
    controls=references['legacy_ids'];anchors={v['id']:v['readout'] for v in m['native_anchors']}
    if [v['id'] for v in m['native_anchors']]!=controls or anchors!=references['legacy_full']['native']:raise ValueError('All12original native anchors required')
    contexts=selected_contexts(registry,m['smoke']);per_source=6 if m['smoke'] else 12;all_rows={};total_memories=0;tail_count=0;noop_count=0
    for name in order:
        state=m['models'][name];spec=plan['models'][name]
        if state['token_policy']!=(spec['token_policy'] if spec else 'none') or not state['hooks_restored']:raise ValueError('Fixed model policy or restoration changed')
        for label in ('native_before_attach_checks','attached_none_checks','native_restore_checks','legacy_full_checks'):
            values=state[label];expected=[] if label=='attached_none_checks' and spec is None else controls
            reference=references['legacy_full'][name] if label=='legacy_full_checks' else anchors
            if [v['id'] for v in values]!=expected or any(v['readout']!=reference[v['id']] for v in values):raise ValueError('Frozen legacy/native replay failed')
        if spec:
            training=json.loads((Path(spec['training_run'])/'manifest.json').read_text())
            if (state['description']!=training['arms'][spec['training_arm']]['description'] or state['adapter_parameters']!=spec['parameters'] or
                state['checkpoint_sha256']!=spec['checkpoint_sha256'] or streaming_hash(Path(spec['checkpoint']))!=spec['checkpoint_sha256'] or
                state['adapter_state_sha256_before']!=oldrefs['training'][name]['checkpoint_state_sha256'] or
                state['adapter_state_sha256_after']!=state['adapter_state_sha256_before'] or not state['all_adapter_parameters_frozen'] or
                not state['arithmetic_hooks_removed']):raise ValueError('Frozen checkpoint or scope changed')
            audit_affines(state,m['arithmetic_profile'])
        rp=path/f'{name}.jsonl'
        if streaming_hash(rp)!=state['results_sha256']:raise ValueError('Raw readout bytes changed')
        rows=[json.loads(line) for line in rp.read_text().splitlines()];expected=[];expected_memories=[]
        for context in contexts:
            for mode in (('none',) if context['episode']=='before' else MODES):
                mid=context['id']+'.'+mode;expected_memories.append(mid);memory=state['memories'][mid];prefix=state['prefixes'][context['prefix_key']]
                if any(memory[k]!=v for k,v in {'context_id':context['id'],'mode':mode,'prefix_key':context['prefix_key'],'source_key':context['source_key']}.items()):
                    raise ValueError('Question-independent source identity changed')
                audit_cache(memory['cache'],context['ledger_tokens'],per_source+1)
                # Every prefix is checked after each branch and every readout, including no-op controls.
                audit_cache(prefix,context['prefix_tokens'],per_source+4*(7+4*per_source)+1)
                if context['episode']=='before':
                    if memory['tail_proof'] is not None or core_cache(memory['cache'])!=core_cache(prefix):raise ValueError('Before cache must be the normally encoded prefix')
                else:
                    proof=memory['tail_proof'];check_gate_stats(proof['gates'],mode,context['tail_tokens'])
                    kv={k:t['sha256'] for k,t in prefix['tensors'].items() if "'keys'" in k or "'values'" in k}
                    recurrent=[str((i,'recurrent_states',0)) for i in GDN_LAYERS] if mode=='both' else []
                    if (proof['tail_tokens']!=context['tail_tokens'] or not proof['isolated_from_prefix'] or
                        proof['write_keep_sha256']!=data_fingerprint('1'*context['prefix_tokens']+'0'*context['tail_tokens']) or
                        proof['historical_kv_sha256']!=kv or proof['both_mode_exact_recurrent_keys']!=recurrent or
                        any(memory['cache']['tensors'][k]!=prefix['tensors'][k] for k in recurrent)):
                        raise ValueError('Historical prefix KV, isolated tail scope or exact18matrix hold failed')
                    tail_count+=1
                for q in context_queries(context,queries,m['smoke']):
                    for reader in READERS:expected.append((q['id'],mode,reader,mid))
        if (list(state['memories'])!=expected_memories or len(state['prefixes'])!=len(contexts)//5 or
            state['completed_readouts']!=len(expected) or [(r['id'],r['mode'],r['reader'],r['memory_id']) for r in rows]!=expected):
            raise ValueError('Complete cached source and raw row coverage/order required')
        indexed={}
        for row in rows:
            q=queries[row['id']];audit_readout(readout(row),{**q,'question_tokens':len(q['input_ids'])-q['ledger_tokens']})
            audit_reader(row['proof'],q,row['reader'])
            if (row['model']!=name or row['prefill']!='cached' or row['registration_sha256']!=data_fingerprint(q) or
                row['prompt_ids_sha256']!=q['prompt_ids_sha256'] or row['source_key']!=q['source_key']):raise ValueError('Original fresh tokens, source or cached partition changed')
            indexed[(row['id'],row['mode'],row['reader'])]=row
        after=[c for c in contexts if c['episode']!='before']
        if [v['context_id'] for v in state['noop_controls']]!=[c['id'] for c in after]:raise ValueError('All unwrapped-vs-none tail controls required')
        for control,context in zip(state['noop_controls'],after):
            audit_cache(control['unwrapped_cache'],context['ledger_tokens'],3)
            actual=state['memories'][context['id']+'.none']['cache']
            if (core_cache(actual)!=core_cache(control['unwrapped_cache']) or not control['complete_cache_and_logits_exact'] or
                control['matched_primary_readouts']!=2):raise ValueError('No-op cache/logit identity failed')
            q=next(q for q in context_queries(context,queries,m['smoke']) if q['role']=='target_current' and q['question_style']=='original')
            for reader in READERS:
                audit_reader(control['reader_proofs'][reader],q,reader)
                if control['readouts'][reader]!=readout(indexed[(q['id'],'none',reader)]):raise ValueError('No-op five-field readout identity failed')
                noop_count+=1
        all_rows[name]=rows;total_memories+=len(state['memories'])
    native,question=[m['models'][n] for n in ('native','shared_question_trained_question')]
    if native['prefixes']!=question['prefixes'] or native['memories']!=question['memories']:raise ValueError('Question-only controller wrote before question arrival')
    return {'status':'audited_post_encoding_smoke' if m['smoke'] else 'completed_and_audited_post_encoding',
        'source_run':str(path),'manifest_sha256':streaming_hash(path/'manifest.json'),'dataset_manifest_sha256':m['dataset_manifest_sha256'],
        'primary_readouts_audited':sum(map(len,all_rows.values())),'source_caches_audited':total_memories,'tail_conditions_audited':tail_count,
        'noop_control_readouts_audited':noop_count,'legacy_operational_readouts':240,'source_files_verified':len(sources()),
        'question_only_native_memory_exact':True,'original_ideal_kernel_failure_preserved':True,'baseline_failures_excluded':0,
        'prior_timing_criteria_unchanged':oldrefs['prior_timing_criteria_unchanged'],'architecture_effectiveness_claim':False,
        'metrics':None if m['smoke'] else summarize(all_rows,queries,plan),'analysis_source_sha256':streaming_hash(Path(__file__))}


def audit_equivalence(forward_path,reverse_path):
    paths=[Path(forward_path),Path(reverse_path)];reports=[];manifests=[]
    for p in paths:
        r=json.loads(p.read_text())
        if (r['status']!='audited_post_encoding_smoke' or r['primary_readouts_audited']!=510 or
            audit_run(Path(r['source_run']),True)!=r or streaming_hash(p.with_suffix('.source.py'))!=r['analysis_source_sha256']):
            raise ValueError('Two independently audited complete510readout smokes required')
        reports.append(r);manifests.append(json.loads((Path(r['source_run'])/'manifest.json').read_text()))
    fm,rm=manifests;plan=read_plan()
    if (fm['model_order']!=plan['model_order'] or rm['model_order']!=plan['model_order'][::-1] or
        any(fm[k]!=rm[k] for k in ('native_anchors','native_component_kernel_checks','source_sha256','dataset_manifest_sha256','weights_before','weights_after'))):
        raise ValueError('Opposite-order operational environment or controls changed')
    for name in plan['model_order']:
        if fm['models'][name]!=rm['models'][name] or (Path(reports[0]['source_run'])/f'{name}.jsonl').read_bytes()!=(Path(reports[1]['source_run'])/f'{name}.jsonl').read_bytes():
            raise ValueError('Opposite-order caches, gates, native restores or raw readouts differ')
    return {'status':'audited_post_encoding_smoke_equivalence','forward_audit':str(paths[0]),'reverse_audit':str(paths[1]),
        'forward_audit_sha256':streaming_hash(paths[0]),'reverse_audit_sha256':streaming_hash(paths[1]),
        'dataset_manifest_sha256':reports[0]['dataset_manifest_sha256'],'runtime_source_sha256':fm['source_sha256'],
        'smoke_completed_utc':[m['completed_utc'] for m in manifests],'exact_readouts':510,'accuracy_gate':False,
        'primary_evaluated':False,'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',nargs='?',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-smoke',action='store_true');p.add_argument('--equivalence',nargs=2,type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite an audit receipt')
    value=audit_equivalence(*a.equivalence) if a.equivalence else audit_run(a.run,a.allow_smoke)
    write_json(a.output,value);a.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in value.items() if k!='metrics'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
