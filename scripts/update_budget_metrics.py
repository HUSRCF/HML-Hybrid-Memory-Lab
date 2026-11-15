"""Keep every matched-policy result and the semantic/original archived references."""
from statistics import mean
from scripts.canonical_oracle_metrics import model_stats
from scripts.copy_readout_metrics import paired_current
from scripts.copy_transfer_metrics import paired_outcomes
from scripts.unseen_shared_timing_metrics import pointwise
from scripts.update_budget_data import MODELS,FAMILIES,REGIMES,UNIFIED,selected_inputs

def summarize(new_rows,archived,cases,registry,plan):
    rows={n:{r:(new_rows[n][r] if r in REGIMES else archived[n][r]) for r in UNIFIED} for n in MODELS}
    stats={n:{r:model_stats(rows[n][r],cases,registry) for r in UNIFIED} for n in MODELS}
    _,bindings=selected_inputs(registry);paired={};interactions={}
    for n in MODELS:
        indexed={r:{v['id']:v for v in cell} for r,cell in rows[n].items()};paired[n]={}
        for candidate,reference in plan['contrasts']:
            paired[n][candidate+'_vs_'+reference]={'current_target':paired_current(indexed[candidate],indexed[reference],bindings),
                'joint_cases':paired_outcomes(stats[n][candidate]['case_outcomes'],stats[n][reference]['case_outcomes']),
                'physical':pointwise(rows[n][candidate],rows[n][reference],cases,registry)}
        values=[]
        for key in indexed['full']:
            loss={r:indexed[r][key]['answer_nll'] for r in UNIFIED}
            values.append({'id':key,**{f:loss[f+'_oracle']-loss[f+'_full']-loss['delta_oracle']+loss['delta_full'] for f in FAMILIES}})
        by_id={v['id']:v for v in values}
        interactions[n]={'all_physical_values':values,'physical_means':{f:mean(v[f] for v in values) for f in FAMILIES},
            'current_binding_means':{f:mean(by_id[b['physical_id']][f] for b in bindings if b['role']=='target_current' and b['variant']!='before') for f in FAMILIES}}
    return {'models':stats,'measurement_origin':{r:'new_primary' if r in REGIMES else 'archived_reference' for r in UNIFIED},
        'new_primary_readouts':13440,'archived_reference_readouts':6720,'unified_logical_cells':20160,
        'paired_contrasts':paired,'loss_interactions':interactions,'success_thresholds':None,'old_gates_can_change':False,
        'architecture_effectiveness_claim':False,'interpretation':plan['interpretation_limits']}
