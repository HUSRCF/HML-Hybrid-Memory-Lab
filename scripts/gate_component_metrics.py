"""Explicit new/archive origin and all single-gate factorial outcomes."""
from statistics import mean
from scripts.canonical_oracle_metrics import model_stats
from scripts.copy_readout_metrics import paired_current
from scripts.copy_transfer_metrics import paired_outcomes
from scripts.unseen_shared_timing_metrics import pointwise
from scripts.gate_component_data import MODELS,REGIMES,ARCHIVED,UNIFIED,selected_inputs

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
            full=loss['hold']-loss['decay_full']-loss['delta_full']+loss['full']
            oracle=loss['hold_oracle']-loss['decay_oracle']-loss['delta_oracle']+loss['oracle']
            values.append({'id':key,'gates_full':full,'gates_oracle':oracle,
                'reader_decay':loss['decay_oracle']-loss['decay_full']-loss['oracle']+loss['full'],
                'reader_delta':loss['delta_oracle']-loss['delta_full']-loss['oracle']+loss['full'],
                'three_way':oracle-full})
        by_id={v['id']:v for v in values};keys=('gates_full','gates_oracle','reader_decay','reader_delta','three_way')
        interactions[n]={'all_physical_values':values,'physical_means':{k:mean(v[k] for v in values) for k in keys},
            'current_binding_means':{k:mean(by_id[b['physical_id']][k] for b in bindings if b['role']=='target_current' and b['variant']!='before') for k in keys}}
    return {'models':stats,'measurement_origin':{r:'new_primary' if r in REGIMES else 'archived_reference' for r in UNIFIED},
        'new_primary_readouts':6720,'archived_reference_readouts':6720,'unified_logical_cells':13440,
        'paired_contrasts':paired,'loss_interactions':interactions,'success_thresholds':None,'old_gates_can_change':False,
        'architecture_effectiveness_claim':False,'interpretation':plan['interpretation_limits']}
