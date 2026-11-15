"""Posthoc, complete paired model/reader/name/style comparisons; no inference."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.natural_load_data import registered_inputs
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def operational_replay(run, manifest):
    """Additional read-only check; never changes the sealed protocol or gates."""
    gate=manifest['smoke_equivalence_gate'];gate_path=Path(gate['path'])
    if streaming_hash(gate_path)!=gate['sha256']:raise ValueError('Operational equivalence receipt changed')
    equivalence=json.loads(gate_path.read_text());forward=Path(equivalence['forward_run'])
    before=json.loads((forward/'manifest.json').read_text())
    if streaming_hash(forward/'manifest.json')!=equivalence['forward_manifest_sha256']:
        raise ValueError('Completed smoke manifest changed')
    if before['source_sha256']!=manifest['source_sha256'] or before['weights_before']!=manifest['weights_before']:
        raise ValueError('Same source and native weights required for repeated subset')
    result={}
    def core(cache):return {k:v for k,v in cache.items() if k!='source_guard_checks'}
    for model in ('native','gdn_all_tokens'):
        smoke_path=forward/f'{model}.jsonl'
        if streaming_hash(smoke_path)!=before['models'][model]['results_sha256']:
            raise ValueError('Completed smoke raw bytes changed')
        smoke=[json.loads(line) for line in smoke_path.read_text().splitlines()]
        ids={r['id'] for r in smoke}
        primary=[json.loads(line) for line in (run/f'{model}.jsonl').read_text().splitlines()]
        overlapping=[r for r in primary if r['id'] in ids]
        if len(smoke)!=54 or overlapping!=smoke:
            raise ValueError('Primary first-stream original-style readouts differ from operational smoke')
        old=before['models'][model];new=manifest['models'][model]
        for key,cache in old['prefixes'].items():
            if core(cache)!=core(new['prefixes'][key]):raise ValueError('Repeated normally encoded prefix differs')
        for key,source in old['sources'].items():
            actual=new['sources'][key]
            if core(source['cache'])!=core(actual['cache']) or source['tail_proof']!=actual['tail_proof']:
                raise ValueError('Repeated normal tail cache or actual gate proof differs')
        result[model]={'raw_readouts_exact':54,'normal_prefix_cache_exact':1,'normal_tail_caches_exact':8,
            'all48cache_fields_logits_and_layouts_exact':True,'tail_gate_and_historical_kv_proofs_exact':True}
    return {'status':'additional_posthoc_read_only_operational_subset_replay_exact',
        'forward_run':str(forward),'primary_run':str(run),'models':result,
        'scope':'All shared first-stream original-style readouts and normal caches. Guard-check counts differ by registered number of questions; all remaining cache fields match. This adds no model inference, tolerance or scientific success criterion.'}


def extract(audit_path):
    audit=json.loads(audit_path.read_text());run=Path(audit['source_run']);manifest=json.loads((run/'manifest.json').read_text())
    registered,_,_,queries,refs=registered_inputs()
    if (audit['status']!='completed_and_audited_natural_load' or audit['primary_readouts_audited']!=3456 or
        audit['source_files_verified']!=240 or audit['manifest_sha256']!=streaming_hash(run/'manifest.json') or
        audit['packet_manifest_sha256']!=manifest['packet_manifest_sha256'] or
        manifest['source_sha256']!=registered['source_sha256'] or manifest['status']!='completed' or
        manifest['weights_before']!=manifest['weights_after'] or not manifest['hooks_restored'] or
        any(audit['prior_failed_criteria_unchanged'].values())):
        raise ValueError('Complete audited immutable primary required before descriptive analysis')
    values={};indexed={}
    for model in ('native','gdn_all_tokens'):
        raw=run/f'{model}.jsonl'
        if streaming_hash(raw)!=manifest['models'][model]['results_sha256']:
            raise ValueError('Complete raw readout bytes changed')
        rows=[json.loads(line) for line in raw.read_text().splitlines()]
        if len(rows)!=1728:raise ValueError('All1728readouts per model required')
        for row in rows:
            q=queries[row['id']];key=(model,row['reader'],row['id'])
            if key in values:raise ValueError('Duplicate physical readout')
            values[key]=row
            other=(model,row['reader'],q['seed'],q['naming'],q['episode'],q['role'],q['question_style'])
            if other in indexed:raise ValueError('Duplicate semantic condition')
            indexed[other]=row
    comparisons=[]
    def add(label,left,right):
        ql,qr=(queries[r['id']] for r in (left,right))
        if ql['answer']!=qr['answer']:raise ValueError('Cross-arm comparisons must preserve true answer')
        comparisons.append({'comparison':label,'seed':ql['seed'],'episode':ql['episode'],'role':ql['role'],
            'left_model':left['model'],'right_model':right['model'],'left_reader':left['reader'],'right_reader':right['reader'],
            'left_id':left['id'],'right_id':right['id'],'left_naming':ql['naming'],'right_naming':qr['naming'],
            'left_style':ql['question_style'],'right_style':qr['question_style'],
            'left_correct':left['greedy_exact'],'right_correct':right['greedy_exact'],
            'gain':bool(right['greedy_exact'] and not left['greedy_exact']),
            'loss':bool(left['greedy_exact'] and not right['greedy_exact']),
            'generated_changed':left['generated']!=right['generated'],
            'nll_delta_right_minus_left':right['answer_nll']-left['answer_nll']})
    for q in queries.values():
        for reader in ('full','oracle'):
            add('native_to_gdn',values[('native',reader,q['id'])],values[('gdn_all_tokens',reader,q['id'])])
        for model in ('native','gdn_all_tokens'):
            add('full_to_privileged_oracle',values[(model,'full',q['id'])],values[(model,'oracle',q['id'])])
            for reader in ('full','oracle'):
                base=(model,reader,q['seed']);end=(q['episode'],q['role'])
                if q['naming']=='original':
                    add('original_to_permuted_name',values[(model,reader,q['id'])],indexed[(*base,'permuted',*end,q['question_style'])])
                if q['question_style']=='original':
                    add('original_to_entity_first_question',values[(model,reader,q['id'])],indexed[(*base,q['naming'],*end,'entity_first')])
    if len(comparisons)!=6912 or any(sum(c['comparison']==label for c in comparisons)!=1728 for label in
        ('native_to_gdn','full_to_privileged_oracle','original_to_permuted_name','original_to_entity_first_question')):
        raise ValueError('All6912paired cross-arm comparisons required without selection')
    summaries=defaultdict(list);per_seed=defaultdict(list)
    fields=('comparison','left_model','right_model','left_reader','right_reader','episode','role')
    for c in comparisons:
        key=tuple(c[k] for k in fields);summaries[key].append(c);per_seed[(*key,c['seed'])].append(c)
    def summarize(groups,names):
        return [{**dict(zip(names,key)),'pairs':len(rows),'left_correct':sum(c['left_correct'] for c in rows),
            'right_correct':sum(c['right_correct'] for c in rows),'gains':sum(c['gain'] for c in rows),
            'losses':sum(c['loss'] for c in rows),'generated_changed':sum(c['generated_changed'] for c in rows),
            'nll_delta_right_minus_left':sum(c['nll_delta_right_minus_left'] for c in rows)/len(rows)}
            for key,rows in sorted(groups.items())]
    return {'status':'posthoc_complete_cross_arm_descriptive_comparisons','audit_sha256':streaming_hash(audit_path),
        'manifest_sha256':audit['manifest_sha256'],'packet_manifest_sha256':audit['packet_manifest_sha256'],
        'physical_readouts':3456,'paired_comparisons':6912,'independent_streams':8,
        'summary':summarize(summaries,fields),'per_seed_summary':summarize(per_seed,(*fields,'seed')),
        'pairs':comparisons,'prior_failed_criteria_unchanged':refs['prior_failed_criteria'],
        'additional_operational_replay':operational_replay(run,manifest),
        'limits':'No selection or success gate. Names/styles repeat8underlyingstreams. Oracle is privileged. Model/reader gains do not by themselves identify recurrent-memory interference or an architecture improvement.',
        'analysis_source_sha256':streaming_hash(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Never overwrite descriptive results')
    result=extract(args.audit);write_json(args.output,result);args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:result[k] for k in ('status','physical_readouts','paired_comparisons')}),flush=True)


if __name__=='__main__':main()
