"""Registered pointwise timing sufficiency and complete natural baselines."""

from scripts.canonical_oracle_metrics import model_stats
from scripts.copy_transfer_metrics import paired_outcomes
from scripts.unseen_shared_timing_data import NATURAL,ORDER,selected_inputs


def pointwise(candidate,reference,cases,registry):
    physical,bindings=selected_inputs(registry)
    keys=[q['id'] for q in physical]
    if any([r['id'] for r in rows]!=keys for rows in (candidate,reference)):
        raise ValueError('Pointwise comparisons require the identical complete physical registry')
    pairs={c['id']:{'id':c['id'],'reference_correct':r['greedy_exact'],'candidate_correct':c['greedy_exact'],
        'preserved':r['greedy_exact'] and c['greedy_exact'],
        'regression':r['greedy_exact'] and not c['greedy_exact'],
        'repair':c['greedy_exact'] and not r['greedy_exact'],
        'both_wrong':not c['greedy_exact'] and not r['greedy_exact']} for c,r in zip(candidate,reference)}
    def count(ids):
        selected=set(ids)
        values=[pairs[k] for k in keys if k in selected]
        return {'physical_queries':len(values),**{name:sum(v[name] for v in values) for name in
            ('reference_correct','candidate_correct','preserved','regression','repair','both_wrong')}}
    by_case={c['id']:c for c in cases}
    expanded=[{**b,'seed':by_case[b['case_id']]['seed']} for b in bindings]
    breakdown={}
    for factor in ('seed','dependency_depth','question_style','distractor_updates','role','variant'):
        breakdown[factor]=[{'value':v,**count(b['physical_id'] for b in expanded if b[factor]==v)}
            for v in sorted({b[factor] for b in expanded})]
    result=count(keys)
    return {**result,'no_shared_correct_physical_regression':result['regression']==0,
        'breakdowns':breakdown,'individual_physical_pairs':list(pairs.values()),
        'unit_note':'Each breakdown deduplicates its physical queries; related bindings and overlapping strata are not independent samples.'}


def summarize(all_rows,cases,registry,registered):
    if list(all_rows)!=list(ORDER):raise ValueError('All ten prospectively registered arms required')
    models={name:model_stats(all_rows[name],cases,registry) for name in ORDER}
    comparisons=[('shared_question_timing','shared'),('shared_ledger_timing','shared'),
        ('shared_ledger_timing','shared_question_timing'),('shared_question_timing','native'),
        *[('shared',r) for r in NATURAL if r!='shared']]
    joint=[{'candidate':c,'reference':r,**paired_outcomes(models[c]['case_outcomes'],models[r]['case_outcomes'])}
        for c,r in comparisons]
    preservation=pointwise(all_rows['shared_question_timing'],all_rows['shared'],cases,registry)
    retained=next(c for c in joint if (c['candidate'],c['reference'])==('shared_question_timing','native'))
    return {'models':models,'joint_paired_comparisons':joint,
        'within_checkpoint_question_functional_preservation':preservation,
        'question_only_retained_gain_over_native':retained,
        'registered_criteria':{'question_only_no_shared_correct_regression':preservation['regression']==0,
            'question_only_positive_joint_gain_over_native':retained['candidate_passed']},
        'prior_fresh_gates_unchanged':registered['references']['prior_fresh_gates'],
        'scope':'One unseen task-family cohort, same frozen all-token-trained shared checkpoint. Timing losses are deployment-schedule effects, not proof of semantic write necessity or a new architecture.'}
