"""Report every fresh baseline and Query-source cell without correctness filtering."""
from statistics import mean
from scripts.fresh_query_data import MODELS,READERS,CONDITIONS
from scripts.post_encoding_tasks import ROLES,STYLES
from scripts.post_encoding_metrics import stats,paired
from scripts.post_tail_path_metrics import functional_stats,split_paired


def summarize(all_rows,queries,plan):
    before_ids=[k for k,q in queries.items() if q['episode']=='before'];after_ids=[k for k,q in queries.items() if q['episode']!='before']
    if len(before_ids)!=48 or len(after_ids)!=192:raise ValueError('Complete fresh240question grid required')
    cells={};strata={};episodes={};contrasts={};before_after={};interactions={}
    for name in MODELS:
        rows=all_rows[name]
        if len(rows)!=1632:raise ValueError('All1632primary rows required per model')
        index={(r['id'],r['condition'],r['reader']):r for r in rows}
        if len(index)!=len(rows):raise ValueError('Duplicate primary result')
        cells[name]={};strata[name]={};episodes[name]={};contrasts[name]={};before_after[name]={};interactions[name]={}
        grouped={}
        for condition in ('before',*CONDITIONS):
            ids=before_ids if condition=='before' else after_ids
            for reader in READERS:
                values=[index[(k,condition,reader)] for k in ids];grouped[(condition,reader)]=values;cell=condition+'.'+reader
                cells[name][cell]={**functional_stats(values,queries),'origin':'before_baseline' if condition=='before' else
                    'unpatched_after_baseline' if condition[0]==condition[1] else 'cross_source_query'}
                strata[name][cell]={'role':{r:stats([v for v in values if queries[v['id']]['role']==r]) for r in ROLES},
                    'style':{s:stats([v for v in values if queries[v['id']]['question_style']==s]) for s in STYLES},
                    'seed':{str(seed):stats([v for v in values if queries[v['id']]['seed']==seed]) for seed in plan['seeds']},
                    'evidence':{str(flag):stats([v for v in values if queries[v['id']]['prefix_evidence']==flag]) for flag in (True,False)
                        if any(queries[v['id']]['prefix_evidence']==flag for v in values)}}
                episodes[name][cell]={}
                for episode in dict.fromkeys(queries[k]['episode'] for k in ids):
                    sub=[v for v in values if queries[v['id']]['episode']==episode]
                    episodes[name][cell][episode]={'all':functional_stats(sub,queries),'role':{
                        r:stats([v for v in sub if queries[v['id']]['role']==r]) for r in ROLES}}
                if reader=='oracle':contrasts[name][cell+'_vs_full']=split_paired(values,grouped[(condition,'full')],queries)
        for candidate,reference in (('HN','HH'),('NH','NN'),('HH','NN'),('HN','NN'),('NH','HH')):
            for reader in READERS:contrasts[name][candidate+'.'+reader+'_vs_'+reference]=split_paired(grouped[(candidate,reader)],grouped[(reference,reader)],queries)
        for condition in ('NN','HH'):
            for reader in READERS:
                base={(queries[v['id']]['seed'],queries[v['id']]['role'],queries[v['id']]['question_style']):v for v in grouped[('before',reader)]}
                before_after[name][condition+'.'+reader]={}
                for ep in ('unrelated-8','related-8','unrelated-64','related-64'):
                    values=[v for v in grouped[(condition,reader)] if queries[v['id']]['episode']==ep]
                    reference=[base[(queries[v['id']]['seed'],queries[v['id']]['role'],queries[v['id']]['question_style'])] for v in values]
                    flags=[queries[v['id']]['prefix_evidence'] for v in values]
                    before_after[name][condition+'.'+reader][ep]={'all':paired(values,reference),
                        'fixed_prefix_evidence':paired([v for v,f in zip(values,flags) if f],[v for v,f in zip(reference,flags) if f]),
                        'role':{r:paired([v for v in values if queries[v['id']]['role']==r],
                            [v for v,q in zip(reference,values) if queries[q['id']]['role']==r]) for r in ROLES},
                        'scope':'Related-current gold changes. Only fixed-prefix evidence is a same-gold before/after comparison.'}
        for reader in READERS:
            values=[]
            for key in after_ids:
                loss={c:index[(key,c,reader)]['answer_nll'] for c in CONDITIONS}
                values.append({'id':key,'recipient_state':loss['HN']-loss['NN'],'query_source':loss['NH']-loss['NN'],
                    'interaction':loss['HH']-loss['HN']-loss['NH']+loss['NN']})
            terms=('recipient_state','query_source','interaction')
            interactions[name][reader]={'per_question':values,'means':{t:mean(v[t] for v in values) for t in terms},
                'role_means':{r:{t:mean(v[t] for v in values if queries[v['id']]['role']==r) for t in terms} for r in ROLES},
                'episode_means':{ep:{t:mean(v[t] for v in values if queries[v['id']]['episode']==ep) for t in terms}
                    for ep in ('unrelated-8','related-8','unrelated-64','related-64')}}
    return {'cells':cells,'strata':strata,'episodes':episodes,'paired_contrasts':contrasts,'before_after':before_after,
        'nll_interactions':interactions,'primary_readouts':3264,'after_unpatched_baselines':1536,'after_cross_source_queries':1536,
        'before_baselines':192,'cell_count':20,'independent_streams':8,'baseline_failures_excluded':0,'fresh_streams':True,
        'success_thresholds':None,'architecture_effectiveness_claim':False,'previous_gates_can_change':False,'limits':plan['analysis']}
