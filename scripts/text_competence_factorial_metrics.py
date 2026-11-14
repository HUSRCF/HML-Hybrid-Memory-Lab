"""Complete factorial task counts and paired losses; no architecture success gate."""

from statistics import mean
from scripts.canonical_oracle_metrics import model_stats
from scripts.copy_readout_metrics import paired_current
from scripts.copy_transfer_metrics import paired_outcomes
from scripts.unseen_shared_timing_metrics import pointwise
from scripts.text_competence_factorial_data import MODELS,REGIMES


def summarize(rows,cases,registry,plan,collisions):
    if list(rows)!=list(MODELS) or any(list(rows[n])!=list(REGIMES) for n in MODELS):
        raise ValueError('Complete five configurations and four text conditions required')
    stats={n:{r:model_stats(rows[n][r],cases,registry) for r in REGIMES} for n in MODELS}
    contrasts={};interactions={};unique_counts={}
    for name in MODELS:
        indexed={r:{v['id']:v for v in rows[name][r]} for r in REGIMES};contrasts[name]=[]
        for candidate,reference in plan['analysis']['contrasts']:
            contrasts[name].append({'candidate':candidate,'reference':reference,
                'current_target':paired_current(indexed[candidate],indexed[reference],registry['bindings']),
                'joint_cases':paired_outcomes(stats[name][candidate]['case_outcomes'],stats[name][reference]['case_outcomes']),
                'physical':pointwise(rows[name][candidate],rows[name][reference],cases,registry)})
        physical=[{'id':q['id'],'nll_interaction':indexed['compiled_support'][q['id']]['answer_nll']-
            indexed['support_text'][q['id']]['answer_nll']-indexed['compiled_full'][q['id']]['answer_nll']+
            indexed['full'][q['id']]['answer_nll']} for q in registry['physical']]
        by_id={v['id']:v['nll_interaction'] for v in physical}
        current=[{**b,'nll_interaction':by_id[b['physical_id']]} for b in registry['bindings']
                 if b['role']=='target_current' and b['variant']!='before']
        interactions[name]={'physical_queries':336,'mean_physical_nll_interaction':mean(by_id.values()),
            'all_physical_interactions':physical,'current_bindings':96,'mean_current_nll_interaction':mean(v['nll_interaction'] for v in current),
            'all_current_interactions':current,'current_by_depth':[{'depth':d,'bindings':32,
                'mean_nll_interaction':mean(v['nll_interaction'] for v in current if v['dependency_depth']==d)} for d in (1,2,3)]}
        unique_counts[name]={}
        for r in REGIMES:
            groups={}
            for v in rows[name][r]:
                key=v['actual_prompt_sha256']
                if key in groups and any(groups[key][f]!=v[f] for f in ('answer','generated','greedy_exact','answer_nll','query_tokens')):
                    raise ValueError('Identical actual transformed input produced different readouts')
                groups[key]=v
            if len(groups)!=collisions['by_regime'][r]['unique_prompts']:raise ValueError('Transformed collision coverage changed')
            unique_counts[name][r]={'unique_inputs':len(groups),'correct':sum(v['greedy_exact'] for v in groups.values())}
    return {'models':stats,'paired_contrasts':contrasts,'factorial_nll_interactions':interactions,
        'deduplicated_actual_input_counts':unique_counts,'collision_registry':collisions,
        'old_gates_can_change':False,'architecture_effectiveness_claim':False,'prospective_unseen_replication':False,
        'success_thresholds':None,'interpretation':plan['interpretation_limits']}
