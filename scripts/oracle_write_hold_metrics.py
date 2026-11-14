"""Complete paired outcomes; no architecture effectiveness criterion."""
from statistics import mean
from scripts.canonical_oracle_metrics import model_stats
from scripts.copy_readout_metrics import paired_current
from scripts.copy_transfer_metrics import paired_outcomes
from scripts.unseen_shared_timing_metrics import pointwise
from scripts.oracle_write_hold_data import MODELS,REGIMES,selected_inputs

def summarize(rows,cases,registry,plan):
    stats={n:{r:model_stats(rows[n][r],cases,registry) for r in REGIMES} for n in MODELS}
    _,bindings=selected_inputs(registry);paired={};interaction={}
    for n in MODELS:
        indexed={r:{v['id']:v for v in rows[n][r]} for r in REGIMES};paired[n]={}
        for treatment,reference in [('hold','full'),('hold_oracle','oracle'),('oracle','full'),('hold_oracle','hold')]:
            paired[n][f'{treatment}_vs_{reference}']={'current_target':paired_current(indexed[treatment],indexed[reference],bindings),
                'joint_cases':paired_outcomes(stats[n][treatment]['case_outcomes'],stats[n][reference]['case_outcomes']),
                'physical':pointwise(rows[n][treatment],rows[n][reference],cases,registry)}
        values=[{'id':key,'nll_interaction':indexed['hold_oracle'][key]['answer_nll']-indexed['hold'][key]['answer_nll']-
            indexed['oracle'][key]['answer_nll']+indexed['full'][key]['answer_nll']} for key in indexed['full']]
        interaction[n]={'physical_mean':mean(v['nll_interaction'] for v in values),'all_physical_values':values}
    return {'models':stats,'paired_contrasts':paired,'loss_interaction':interaction,'success_thresholds':None,
        'old_gates_can_change':False,'architecture_effectiveness_claim':False,'interpretation':plan['interpretation_limits']}
