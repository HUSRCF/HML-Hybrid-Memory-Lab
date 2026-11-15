"""Complete three-component factorial, with new/reference origins kept explicit."""
from collections import defaultdict
from statistics import mean
from scripts.post_encoding_metrics import stats,paired
from scripts.post_tail_path_data import MODELS,CONDITIONS,PRIMARY,READERS
from scripts.post_encoding_tasks import ROLES,STYLES


def functional_stats(rows,queries):
    groups=[defaultdict(list) for _ in range(3)]
    for r in rows:
        q=queries[r['id']]
        for group,key in zip(groups,((q['seed'],q['episode'],q['question_style']),(q['seed'],q['episode']),q['seed'])):
            group[key].append(r)
    if any(len(v)!=3 for v in groups[0].values()) or any(len(v)!=6 for v in groups[1].values()):
        raise ValueError('Complete three-role and two-style functional groups required')
    return {**stats(rows),**{name:{'correct':sum(all(v['greedy_exact'] for v in values) for values in group.values()),'total':len(group)}
        for name,group in zip(('joint3','joint6','all_questions_per_stream'),groups)}}


def split_paired(candidate,reference,queries):
    if [r['id'] for r in candidate]!=[r['id'] for r in reference]:raise ValueError('Fixed-query contrast required')
    result={'all':paired(candidate,reference),'by_role':{}}
    for role in ROLES:
        which=[i for i,r in enumerate(candidate) if queries[r['id']]['role']==role]
        result['by_role'][role]=paired([candidate[i] for i in which],[reference[i] for i in which])
    for name,flag in (('fixed_prefix_evidence',True),('new_tail_evidence',False)):
        which=[i for i,r in enumerate(candidate) if queries[r['id']]['prefix_evidence']==flag]
        if which:result[name]=paired([candidate[i] for i in which],[reference[i] for i in which])
    return result


def factorial_values(loss):
    if set(loss)!=set(CONDITIONS):raise ValueError('Every factorial corner required')
    return {'recurrent':loss['HNN']-loss['NNN'],'convolution':loss['NHN']-loss['NNN'],'kv':loss['NNH']-loss['NNN'],
        'recurrent_convolution':loss['HHN']-loss['HNN']-loss['NHN']+loss['NNN'],
        'recurrent_kv':loss['HNH']-loss['HNN']-loss['NNH']+loss['NNN'],
        'convolution_kv':loss['NHH']-loss['NHN']-loss['NNH']+loss['NNN'],
        'three_way':loss['HHH']-loss['HHN']-loss['HNH']-loss['NHH']+loss['HNN']+loss['NHN']+loss['NNH']-loss['NNN']}


def summarize(new_rows,endpoints,queries,plan):
    cells={};strata={};episodes={};contrasts={};interactions={}
    for name in MODELS:
        indexed={}
        for label in CONDITIONS:
            for reader in READERS:
                values=([r for r in new_rows[name] if r['condition']==label and r['reader']==reader] if label in PRIMARY else
                    [r for r in endpoints[name][label] if r['reader']==reader])
                if len(values)!=192 or len({r['id'] for r in values})!=192:raise ValueError('All192fixed questions required in every cell')
                indexed[(label,reader)]={r['id']:r for r in values}
        ids=list(queries);cells[name]={};strata[name]={};episodes[name]={};contrasts[name]={};interactions[name]={}
        if len(ids)!=192:raise ValueError('Complete unfiltered reused question grid required')
        for (label,reader),values in indexed.items():
            rows=[values[key] for key in ids];cell=label+'.'+reader
            cells[name][cell]={**functional_stats(rows,queries),'measurement_origin':'new_primary' if label in PRIMARY else 'archived_endpoint_reference'}
            strata[name][cell]={'role':{r:stats([v for v in rows if queries[v['id']]['role']==r]) for r in ROLES},
                'style':{s:stats([v for v in rows if queries[v['id']]['question_style']==s]) for s in STYLES},
                'seed':{str(seed):stats([v for v in rows if queries[v['id']]['seed']==seed]) for seed in plan['seeds']},
                'evidence':{str(flag):stats([v for v in rows if queries[v['id']]['prefix_evidence']==flag]) for flag in (True,False)}}
            episodes[name][cell]={}
            for episode in ('unrelated-8','related-8','unrelated-64','related-64'):
                sub=[v for v in rows if queries[v['id']]['episode']==episode]
                episodes[name][cell][episode]={'all':functional_stats(sub,queries),'role':{
                    r:stats([v for v in sub if queries[v['id']]['role']==r]) for r in ROLES}}
            if reader=='oracle':contrasts[name][cell+'_vs_full']=split_paired(rows,[indexed[(label,'full')][key] for key in ids],queries)
        for candidate,reference in (('HNN','NNN'),('NHN','NNN'),('NNH','NNN'),('NHH','HHH'),('HNH','HHH'),('HHN','HHH')):
            for reader in READERS:
                contrasts[name][candidate+'.'+reader+'_vs_'+reference]=split_paired(
                    [indexed[(candidate,reader)][k] for k in ids],[indexed[(reference,reader)][k] for k in ids],queries)
        matches={(q['seed'],q['tail_records'],q['role'],q['question_style']):q['id'] for q in queries.values() if q['kind']=='unrelated'}
        for label in CONDITIONS:
            for reader in READERS:
                for count in (8,64):
                    related=[q for q in queries.values() if q['kind']=='related' and q['tail_records']==count]
                    a=[indexed[(label,reader)][q['id']] for q in related]
                    b=[indexed[(label,reader)][matches[(q['seed'],count,q['role'],q['question_style'])]] for q in related]
                    contrasts[name][label+'.'+reader+f'.related-{count}_vs_unrelated']={
                        'all':paired(a,b),'role':{r:paired([v for v,q in zip(a,related) if q['role']==r],
                            [v for v,q in zip(b,related) if q['role']==r]) for r in ROLES},
                        'scope':'Current answers differ; this is a matched-tail comparison, not a same-gold retention contrast.'}
        for reader in READERS:
            values=[{'id':key,**factorial_values({label:indexed[(label,reader)][key]['answer_nll'] for label in CONDITIONS})} for key in ids]
            terms=[k for k in values[0] if k!='id']
            interactions[name][reader]={'all_query_values':values,'means':{t:mean(v[t] for v in values) for t in terms},
                'role_means':{r:{t:mean(v[t] for v in values if queries[v['id']]['role']==r) for t in terms} for r in ROLES},
                'episode_means':{e:{t:mean(v[t] for v in values if queries[v['id']]['episode']==e) for t in terms}
                    for e in ('unrelated-8','related-8','unrelated-64','related-64')}}
    return {'cells':cells,'strata':strata,'episodes':episodes,'paired_contrasts':contrasts,'nll_factorial_interactions':interactions,
        'measurement_origin':{c:'new_primary' if c in PRIMARY else 'archived_endpoint_reference' for c in CONDITIONS},
        'new_primary_readouts':4608,'archived_endpoint_references':1536,'unified_factorial_readouts':6144,'cell_count':32,
        'independent_streams':8,'fresh_confirmation':False,'baseline_failures_excluded':0,'success_thresholds':None,
        'architecture_effectiveness_claim':False,'old_gates_can_change':False,'limits':plan['analysis']}
