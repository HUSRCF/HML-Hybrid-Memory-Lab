"""All-case evidence-access effects, without converting old failures into success."""
from scripts.canonical_oracle_metrics import model_stats
from scripts.copy_readout_metrics import paired_current
from scripts.copy_transfer_metrics import paired_outcomes
from scripts.unseen_shared_timing_metrics import pointwise
from scripts.trained_timing_evidence_access_data import MODELS,REGIMES,selected_inputs


def summarize(all_rows,cases,registry,plan):
    if list(all_rows)!=list(MODELS) or any(list(all_rows[n])!=list(REGIMES) for n in MODELS):
        raise ValueError('All5fixed models and all3evidence regimes required')
    stats={n:{r:model_stats(all_rows[n][r],cases,registry) for r in REGIMES} for n in MODELS}
    _,bindings=selected_inputs(registry);comparisons={}
    for name in MODELS:
        indexed={r:{v['id']:v for v in all_rows[name][r]} for r in REGIMES};comparisons[name]={}
        for ref in ('full','nonslice'):
            comparisons[name][ref]={'current_target':paired_current(indexed['oracle'],indexed[ref],bindings),
                'joint_cases':paired_outcomes(stats[name]['oracle']['case_outcomes'],stats[name][ref]['case_outcomes']),
                'physical':pointwise(all_rows[name]['oracle'],all_rows[name][ref],cases,registry)}
    return {'models':stats,'oracle_paired_contrasts':comparisons,'old_gates_can_change':False,
        'architecture_effectiveness_claim':False,'prospective_unseen_replication':False,'primary_evaluated':True,
        'success_thresholds':None,'interpretation':plan['interpretation_limits']}


def supplemental_depth2_queries(cases,registry):
    """Fixed short/long, original/entity-first depth2 controls; never inspect scores."""
    seeds=sorted({c['seed'] for c in cases})
    chosen=[]
    for seed,style,variant in [(seeds[0],'original','control'),(seeds[-1],'entity_first','treatment')]:
        case=next(c for c in cases if c['seed']==seed and c['dependency_depth']==2 and c['question_style']==style)
        binding=next(b for b in registry['bindings'] if b['case_id']==case['id'] and b['variant']==variant and b['role']=='target_current')
        chosen.append(binding['physical_id'])
    result=[q for q in registry['physical'] if q['id'] in chosen]
    if len(result)!=2:raise ValueError('Two complete predetermined depth2 controls required')
    return result
