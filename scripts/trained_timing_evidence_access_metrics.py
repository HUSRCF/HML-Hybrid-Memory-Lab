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
