"""Independent complete-grid audit and descriptive paired normal-update outcomes."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.natural_load_data import (read_plan, registered_inputs, sources, selected_contexts,
    context_queries, DESTINATION, PROTOCOL, MODELS, READERS)
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import write_json
from scripts.analyze_post_encoding import audit_cache, audit_reader
from scripts.post_encoding_runtime import check_gate_stats
from scripts.analyze_trained_timing_evidence_access import audit_readout
from scripts.analyze_gate_components import audit_kernel_checks
from scripts.canonical_necessity_common import audit_affines
from scripts.gate_component_runtime import validation_addendum, VALIDATION_ADDENDUM_SHA256


def aggregate(rows):
    return {'questions':len(rows), 'correct':sum(r['greedy_exact'] for r in rows),
        'answer_nll_mean':sum(r['answer_nll'] for r in rows)/len(rows)}


def paired(left, right):
    pairs = list(zip(left, right))
    if not pairs or len(left) != len(right):
        raise ValueError('Complete paired rows required')
    return {'pairs':len(pairs), 'left':aggregate(left), 'right':aggregate(right),
        'gains':sum(b['greedy_exact'] and not a['greedy_exact'] for a,b in pairs),
        'losses':sum(a['greedy_exact'] and not b['greedy_exact'] for a,b in pairs),
        'generated_changed':sum(a['generated'] != b['generated'] for a,b in pairs),
        'answer_nll_delta_right_minus_left':sum(b['answer_nll']-a['answer_nll'] for a,b in pairs)/len(pairs)}


def summarize(all_rows, queries):
    cells = defaultdict(list); strata = defaultdict(list); indexed = {}; contrasts = []
    for model, rows in all_rows.items():
        for row in rows:
            q = queries[row['id']]
            cells[(model,row['reader'],q['episode'])].append(row)
            strata[(model,row['reader'],q['episode'],q['role'],q['question_style'],q['naming'])].append(row)
            indexed[(model,row['reader'],q['seed'],q['naming'],q['kind'],q['tail_records'],q['active_entities'],q['role'],q['question_style'])] = row
    # Every registered row is retained; baseline errors are never filtered out.
    groups = sorted({(k[0],k[1],k[2],k[3],k[7],k[8]) for k in indexed})
    for model,reader,seed,naming,role,style in groups:
        def get(kind,count,load):
            return indexed[(model,reader,seed,naming,kind,count,load,role,style)]
        pairs = []
        for kind in ('unrelated','related'):
            for count in (64,512):
                pairs.append((f'load2_to16.{kind}.{count}', get(kind,count,2), get(kind,count,16)))
            for load in (2,16):
                pairs.append((f'length64_to512.{kind}.load{load}', get(kind,64,load), get(kind,512,load)))
                for count in (64,512):
                    pairs.append((f'before_to_{kind}.{count}.load{load}', get('before',0,0), get(kind,count,load)))
        for label, left, right in pairs:
            contrasts.append({'model':model,'reader':reader,'seed':seed,'naming':naming,'role':role,
                'question_style':style,'contrast':label, 'left_id':left['id'],'right_id':right['id'],
                **paired([left],[right])})
    combined = defaultdict(list)
    for c in contrasts:
        combined[(c['model'],c['reader'],c['contrast'],c['role'])].append(c)
    return {'cells':[{'model':m,'reader':r,'episode':e,**aggregate(v)} for (m,r,e),v in sorted(cells.items())],
        'strata':[{'model':m,'reader':r,'episode':e,'role':role,'question_style':style,'naming':naming,**aggregate(v)}
            for (m,r,e,role,style,naming),v in sorted(strata.items())],
        'per_stream_paired_contrasts':contrasts,
        'paired_contrasts':[{'model':m,'reader':r,'contrast':label,'role':role,'pairs':len(v),
            'gains':sum(c['gains'] for c in v),'losses':sum(c['losses'] for c in v),
            'generated_changed':sum(c['generated_changed'] for c in v),
            'answer_nll_delta_right_minus_left':sum(c['answer_nll_delta_right_minus_left'] for c in v)/len(v)}
            for (m,r,label,role),v in sorted(combined.items())],
        'independent_streams':len({queries[r['id']]['seed'] for rows in all_rows.values() for r in rows}),
        'interpretation':'Descriptive paired pilot. Renamings/styles/questions repeat underlying streams; no significance, causal-memory, architecture or scaleup claim.'}


def audit_run(path, allow_smoke=False):
    path = Path(path); m = json.loads((path/'manifest.json').read_text()); plan = read_plan()
    registered, cases, registry, queries, refs = registered_inputs()
    order = list(MODELS[::-1] if m['reverse_model_order'] else MODELS)
    weights = m.get('weights_after', {})
    if (m['status'] != 'completed' or (m['smoke'] and not allow_smoke) or m['training'] or m['new_adapters'] or
            (m['reverse_model_order'] and not m['smoke']) or m['model_order'] != order or m['completed_models'] != order or
            list(m['models']) != order or m['packet'] != str(DESTINATION) or
            m['packet_manifest_sha256'] != streaming_hash(DESTINATION/'manifest.json') or
            m['protocol_sha256'] != streaming_hash(PROTOCOL) or m['source_sha256'] != registered['source_sha256'] or
            set(m['source_sha256']) != sources() or len(sources()) != 240 or
            m['model_path'] != plan['model_path'] or m['packages'] != plan['packages'] or
            m['actual_eos_token_id'] != refs['runtime']['actual_eos_token_id'] or
            m['generation_config_sha256'] != refs['runtime']['generation_config_sha256'] or
            m['generation_config_sha256'] != {p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')} or
            m['generation'] != plan['generation'] or m['arithmetic_profile'] != plan['arithmetic_profile'] or
            m['precision_addendum_sha256'] != streaming_hash(Path(plan['precision_addendum'])) or
            m['validation_addendum_sha256'] != VALIDATION_ADDENDUM_SHA256 or
            json.loads((path/'plan.json').read_text()) != plan or
            json.loads((path/'precision_addendum.json').read_text()) != json.loads(Path(plan['precision_addendum']).read_text()) or
            json.loads((path/'kernel_validation_addendum.json').read_text()) != validation_addendum() or
            streaming_hash(path/'installed_modeling_source.py') != plan['installed_modeling_source_sha256'] or
            streaming_hash(Path(plan['installed_modeling_source'])) != plan['installed_modeling_source_sha256'] or
            weights != m['weights_before'] or weights['sha256'] != plan['native_loaded_weights_sha256'] or
            not weights['all_parameters_frozen'] or not weights['no_parameter_gradients'] or not m['hooks_restored'] or
            not datetime.fromisoformat(plan['registered_utc']) < datetime.fromisoformat(registered['created_utc']) <
                datetime.fromisoformat(m['started_utc']) < datetime.fromisoformat(m['completed_utc'])):
        raise ValueError('Completed frozen240source runtime and chronological gates required')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name) != digest or streaming_hash(path/'sources'/name) != digest:
            raise ValueError('Measured source bytes changed')
    audit_kernel_checks(m['native_component_kernel_checks'])
    if m['smoke']:
        if m['smoke_equivalence_gate'] is not None:
            raise ValueError('Smoke cannot have a primary gate')
    else:
        from scripts.run_natural_load import checked_gate
        gate = m['smoke_equivalence_gate']
        if checked_gate(Path(gate['path']),m['packet_manifest_sha256']) != gate:
            raise ValueError('Complete opposite-order smoke gate changed')
        eq = json.loads(Path(gate['path']).read_text())
        if eq['source_sha256'] != m['source_sha256'] or any(datetime.fromisoformat(t) >= datetime.fromisoformat(m['started_utc']) for t in eq['smoke_completed_utc']):
            raise ValueError('Exact same-source smoke equivalence must precede primary inference')
    controls = refs['legacy_ids']; anchors = {v['id']:v['readout'] for v in m['native_anchors']}
    if [v['id'] for v in m['native_anchors']] != controls or anchors != refs['legacy_full']['native']:
        raise ValueError('All12unchanged native anchors required')
    contexts = selected_contexts(registry,m['smoke']); nq = 3 if m['smoke'] else 6
    all_rows = {}; prefix_count = 0; after_count = 0
    for name in order:
        state = m['models'][name]; spec = plan['models'][name]
        if state['token_policy'] != ('all' if spec else 'none') or not state['hooks_restored']:
            raise ValueError('Frozen token policy or restored hooks changed')
        for label in ('native_before_attach_checks','attached_none_checks','legacy_full_checks','native_restore_checks'):
            ids = [] if label == 'attached_none_checks' and spec is None else controls
            reference = refs['legacy_full'][name] if label == 'legacy_full_checks' else anchors
            if [v['id'] for v in state[label]] != ids or any(v['readout'] != reference[v['id']] for v in state[label]):
                raise ValueError('All96legacy operational replays must be exact')
        if spec:
            if (state['description'] != refs['model_states'][name]['description'] or state['adapter_parameters'] != spec['parameters'] or
                    state['checkpoint_sha256'] != spec['checkpoint_sha256'] or streaming_hash(Path(spec['checkpoint'])) != spec['checkpoint_sha256'] or
                    state['adapter_state_sha256_before'] != refs['model_states'][name]['adapter_state_sha256_before'] or
                    state['adapter_state_sha256_after'] != state['adapter_state_sha256_before'] or
                    not state['all_adapter_parameters_frozen'] or not state['arithmetic_hooks_removed']):
                raise ValueError('Frozen adapter bytes or arithmetic hooks changed')
            audit_affines(state,plan['arithmetic_profile'])
        before = [c for c in contexts if c['episode'] == 'before']; after = [c for c in contexts if c['episode'] != 'before']
        if list(state['prefixes']) != [c['prefix_key'] for c in before] or list(state['sources']) != [c['id'] for c in after]:
            raise ValueError('Every normal common prefix and tail cache must be retained')
        for c in before:
            audit_cache(state['prefixes'][c['prefix_key']],c['prefix_tokens'],10+18*nq); prefix_count += 1
        for c in after:
            source = state['sources'][c['id']]; parent = state['prefixes'][c['prefix_key']]
            if not source['all_sources_built_before_any_question'] or not source['pairwise_storage_isolated']:
                raise ValueError('Answer-independent isolated normal sources required')
            audit_cache(source['cache'],c['ledger_tokens'],2+2*nq)
            proof = source['tail_proof']; check_gate_stats(proof['gates'],'none',c['tail_tokens'])
            kv = {k:t['sha256'] for k,t in parent['tensors'].items() if "'keys'" in k or "'values'" in k}
            if (proof['tail_tokens'] != c['tail_tokens'] or not proof['isolated_from_prefix'] or
                    proof['historical_kv_sha256'] != kv or proof['both_mode_exact_recurrent_keys'] != [] or
                    proof['write_keep_sha256'] != data_fingerprint('1'*c['prefix_tokens']+'0'*c['tail_tokens'])):
                raise ValueError('Normal unmodified tail and exact historical12KV bytes required')
            after_count += 1
        raw = path/f'{name}.jsonl'
        if streaming_hash(raw) != state['results_sha256']:
            raise ValueError('Raw result bytes changed')
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        expected = [(q['id'],reader) for c in contexts for q in context_queries(c,queries,m['smoke']) for reader in READERS]
        if [(r['id'],r['reader']) for r in rows] != expected or state['completed_primary'] != len(expected):
            raise ValueError('Complete unfiltered ordered query/reader grid required')
        for row in rows:
            q = queries[row['id']]
            audit_readout(row,{**q,'question_tokens':len(q['input_ids'])-q['ledger_tokens']})
            if (row['model'] != name or row['registration_sha256'] != data_fingerprint(q) or
                    row['source_key'] != q['source_key'] or row['prompt_ids_sha256'] != q['prompt_ids_sha256'] or
                    row['prefill'] != 'cached' or row['measurement_origin'] != 'normally_encoded_without_intervention'):
                raise ValueError('Actual registered normal readout identity changed')
            audit_reader(row['proof'],q,row['reader'])
        all_rows[name] = rows
    total = sum(map(len,all_rows.values()))
    if total != (108 if m['smoke'] else 3456) or prefix_count != (2 if m['smoke'] else 32) or after_count != (16 if m['smoke'] else 256):
        raise ValueError('Complete registered readout/cache count required')
    return {'status':'audited_natural_load_smoke' if m['smoke'] else 'completed_and_audited_natural_load',
        'source_run':str(path),'manifest_sha256':streaming_hash(path/'manifest.json'),
        'packet_manifest_sha256':m['packet_manifest_sha256'],'source_files_verified':240,
        'primary_readouts_audited':total,'prefix_caches_audited':prefix_count,'after_caches_audited':after_count,
        'legacy_readouts_audited':96,'smoke':m['smoke'],'prior_failed_criteria_unchanged':refs['prior_failed_criteria'],
        'fresh_confirmation':False,'metrics':summarize(all_rows,queries),
        'analysis_source_sha256':streaming_hash(Path(__file__))}


def audit_equivalence(forward, reverse):
    f = audit_run(forward,allow_smoke=True); r = audit_run(reverse,allow_smoke=True)
    fm = json.loads((forward/'manifest.json').read_text()); rm = json.loads((reverse/'manifest.json').read_text())
    if (not f['smoke'] or not r['smoke'] or fm['model_order'] != list(MODELS) or rm['model_order'] != list(MODELS[::-1]) or
            fm['packet_manifest_sha256'] != rm['packet_manifest_sha256'] or fm['source_sha256'] != rm['source_sha256'] or
            fm['weights_before'] != rm['weights_before'] or fm['native_anchors'] != rm['native_anchors'] or
            fm['native_component_kernel_checks'] != rm['native_component_kernel_checks'] or f['metrics'] != r['metrics']):
        raise ValueError('Opposite-order operational inputs and complete outcomes must match exactly')
    for name in MODELS:
        if fm['models'][name] != rm['models'][name] or (forward/f'{name}.jsonl').read_bytes() != (reverse/f'{name}.jsonl').read_bytes():
            raise ValueError('Every raw readout,48cache fields,logits,tail proof,legacy/adapter state must match exactly')
    return {'status':'audited_natural_load_smoke_equivalence','forward_run':str(forward),'reverse_run':str(reverse),
        'forward_manifest_sha256':f['manifest_sha256'],'reverse_manifest_sha256':r['manifest_sha256'],
        'packet_manifest_sha256':fm['packet_manifest_sha256'],'source_sha256':fm['source_sha256'],
        'smoke_completed_utc':[fm['completed_utc'],rm['completed_utc']],
        'readouts_per_smoke':108,'raw_files_exact':2,'accuracy_gate':False,
        'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path)
    p.add_argument('--allow-smoke',action='store_true'); p.add_argument('--forward',type=Path); p.add_argument('--reverse',type=Path)
    p.add_argument('--output',type=Path,required=True); args = p.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite an independent audit')
    if args.run is not None and args.forward is None and args.reverse is None:
        value = audit_run(args.run,args.allow_smoke)
    elif args.run is None and args.forward is not None and args.reverse is not None:
        value = audit_equivalence(args.forward,args.reverse)
    else:
        raise ValueError('Choose one complete-run audit or opposite-order equivalence')
    write_json(args.output,value); args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:value[k] for k in ('status','primary_readouts_audited','source_files_verified') if k in value}),flush=True)


if __name__ == '__main__':
    main()
