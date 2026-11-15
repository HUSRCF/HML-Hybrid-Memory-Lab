"""All paired outcomes, including failed pre-tail baselines and update controls."""
from collections import defaultdict
from statistics import mean
from scripts.post_encoding_tasks import ROLES,STYLES


def stats(rows):
    if not rows:raise ValueError('Empty outcome stratum')
    return {'readouts':len(rows),'greedy_exact':sum(r['greedy_exact'] for r in rows),
        'accuracy':mean(r['greedy_exact'] for r in rows),'answer_nll':mean(r['answer_nll'] for r in rows)}


def paired(candidate,reference):
    if len(candidate)!=len(reference) or not candidate:raise ValueError('Nonempty matched pairs required')
    outcomes={k:0 for k in ('both_correct','gain','loss','both_wrong')}
    values=[]
    for a,b in zip(candidate,reference):
        label=('both_correct' if b['greedy_exact'] else 'gain') if a['greedy_exact'] else ('loss' if b['greedy_exact'] else 'both_wrong')
        outcomes[label]+=1;values.append(a['answer_nll']-b['answer_nll'])
    return {'pairs':len(candidate),**outcomes,'candidate_minus_reference_nll':mean(values)}


def summarize(all_rows,queries,plan):
    cells={};contrasts={};before_after={};role_style_seed={};positive={}
    for name in plan['model_order']:
        rows=all_rows[name];index={(r['id'],r['mode'],r['reader']):r for r in rows}
        groups=defaultdict(list)
        for r in rows:groups[(queries[r['id']]['episode'],r['mode'],r['reader'])].append(r)
        cells[name]={};before_after[name]={};role_style_seed[name]={};positive[name]={};contrasts[name]={}
        baseline={(queries[r['id']]['seed'],queries[r['id']]['question_style'],queries[r['id']]['role'],r['reader']):r
            for r in rows if queries[r['id']]['episode']=='before'}
        for (episode,mode,reader),values in groups.items():
            key=f'{episode}.{mode}.{reader}';by_style=defaultdict(list);by_seed=defaultdict(list)
            for r in values:
                q=queries[r['id']];by_style[(q['seed'],q['question_style'])].append(r);by_seed[q['seed']].append(r)
            if len(values)!=48 or any(len(v)!=3 for v in by_style.values()) or any(len(v)!=6 for v in by_seed.values()):
                raise ValueError('Every cell must preserve all8streams,3roles and2styles')
            cells[name][key]={**stats(values),'joint3_role_cases':sum(all(r['greedy_exact'] for r in v) for v in by_style.values()),
                'joint3_total':len(by_style),'joint6_streams':sum(all(r['greedy_exact'] for r in v) for v in by_seed.values()),'joint6_total':len(by_seed)}
            role_style_seed[name][key]={
                'role':{role:stats([r for r in values if queries[r['id']]['role']==role]) for role in ROLES},
                'style':{style:stats([r for r in values if queries[r['id']]['question_style']==style]) for style in STYLES},
                'seed':{str(seed):stats(v) for seed,v in by_seed.items()}}
            if episode=='before':continue
            ref=[baseline[(queries[r['id']]['seed'],queries[r['id']]['question_style'],queries[r['id']]['role'],reader)] for r in values]
            prefix=[i for i,r in enumerate(values) if queries[r['id']]['prefix_evidence']]
            tail=[i for i,r in enumerate(values) if not queries[r['id']]['prefix_evidence']]
            before_after[name][key]={'all':paired(values,ref),
                'fixed_prefix_evidence':paired([values[i] for i in prefix],[ref[i] for i in prefix]),
                'by_role':{role:paired([r for r in values if queries[r['id']]['role']==role],
                    [b for r,b in zip(values,ref) if queries[r['id']]['role']==role]) for role in ROLES}}
            if tail:
                positive[name][key]={'new_tail_evidence':stats([values[i] for i in tail]),
                    'before_question_accuracy':stats([ref[i] for i in tail]),
                    'interpretation':'Answers change after the related SET. This is an update positive control, not fixed-prefix evidence retention.'}
            if mode!='none':contrasts[name][key+'_vs_none']=paired(values,[index[(r['id'],'none',reader)] for r in values])
            if reader=='oracle':contrasts[name][key+'_vs_full']=paired(values,[index[(r['id'],mode,'full')] for r in values])
            if episode.startswith('related-'):
                uq={(q['seed'],q['tail_records'],q['role'],q['question_style']):q['id'] for q in queries.values() if q['kind']=='unrelated'}
                contrasts[name][key+'_vs_unrelated']=paired(values,[index[(uq[(queries[r['id']]['seed'],queries[r['id']]['tail_records'],
                    queries[r['id']]['role'],queries[r['id']]['question_style'])],mode,reader)] for r in values])
        if len(cells[name])!=34:raise ValueError('All34before/after reader/mode cells required per configuration')
    return {'cell_metrics':cells,'strata':role_style_seed,'before_after':before_after,'update_positive_controls':positive,
        'same_input_and_matched_tail_contrasts':contrasts,'cells':sum(map(len,cells.values())),
        'primary_readouts':sum(map(len,all_rows.values())),'baseline_failures_excluded':0,'success_thresholds':None,
        'training_or_scaleup_trigger':False,'architecture_effectiveness_claim':False,'limits':plan['analysis']}
