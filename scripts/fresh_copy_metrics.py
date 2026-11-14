"""Fresh natural architecture controls and readout gates, kept as separate claims."""

from scripts.canonical_oracle_metrics import model_stats
from scripts.copy_readout_metrics import paired_current, sufficiency_checks
from scripts.copy_necessity_metrics import necessity_checks
from scripts.copy_transfer_metrics import summarize as full_summary
from scripts.fresh_copy_data import MODELS, PRIMARY, MASKED_REGIMES, selected_inputs


def summarize(all_rows, cases, registry, source_plan, plan):
    if list(all_rows) != list(MODELS) or any(list(all_rows[m]) != plan['regimes_by_model'][m] for m in MODELS):
        raise ValueError('All8natural arms and all3extra masks for both prescribed models required')
    natural = full_summary({m: all_rows[m]['full'] for m in MODELS}, cases, registry, source_plan)
    regimes = ('full', *MASKED_REGIMES)
    readouts = {r: {m: model_stats(all_rows[m][r], cases, registry) for m in PRIMARY} for r in regimes}
    _, bindings = selected_inputs(registry)
    effects, vs_full, sufficient, necessary = {}, {}, {}, {}
    for model in PRIMARY:
        indexed = {r: {v['id']: v for v in all_rows[model][r]} for r in regimes}
        vs_full[model] = {r: paired_current(indexed[r], indexed['full'], bindings) for r in MASKED_REGIMES}
        effects[model] = paired_current(indexed['remove_matched_other'], indexed['remove_support'], bindings)
        sufficient[model] = sufficiency_checks(readouts['full'][model], readouts['oracle'][model], vs_full[model]['oracle'])
        necessary[model] = necessity_checks(readouts['full'][model], readouts['remove_matched_other'][model], effects[model])
    gate = natural['descriptive_shared_candidate_gate']
    return {'natural_full_models': natural['models'], 'natural_full_comparisons': natural['comparisons'],
        'generic_shared_controller': {'passed': gate['passed'], 'comparators': gate['comparators'],
            'scope': 'Fixed fresh8stream natural full-history candidate screen. Generic shared latents do not implement explicit semantic constraints. Does not erase previous failed comparisons or establish an architecture contribution.'},
        'readout_models_by_regime': readouts, 'readout_comparisons_vs_full': vs_full,
        'support_removal_effects': effects,
        'frozen_prefix_sufficiency': {'passed': all(all(c.values()) for c in sufficient.values()), 'model_checks': sufficient},
        'whole_record_necessity': {'passed': all(all(c.values()) for c in necessary.values()), 'model_checks': necessary},
        'arithmetic_profile': plan['arithmetic_profile'], 'cross_profile_rows_in_primary': 0,
        'new_data_confirmation': True, 'prior_shared_history_repeat_primaries_remain_failed': True,
        'prior_canonical_development_sufficiency_and_necessity_remain_passed': True,
        'interpretation': plan['preserve']}
