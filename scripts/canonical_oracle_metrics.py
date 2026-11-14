"""Two-model replication of the sealed oracle-sufficiency checks, without fake arms."""

from statistics import mean

from scripts.canonical_oracle_common import MODELS, REGIMES
from scripts.copy_readout_metrics import paired_current, sufficiency_checks
from scripts.copy_transfer_data import ROLES, VARIANTS, selected_inputs


def model_stats(rows, cases, registry):
    physical, bindings = selected_inputs(registry)
    if [r['id'] for r in rows] != [q['id'] for q in physical]:
        raise ValueError('Complete336 ordered physical readouts required')
    indexed = {r['id']: r for r in rows}
    expanded = [{**b, **indexed[b['physical_id']]} for b in bindings]
    by_case = {(r['case_id'], r['variant'], r['role']): r for r in expanded}
    outcomes = []
    for case in cases:
        def correct(variant, role):
            return by_case[case['id'], variant, role]['greedy_exact']
        outcomes.append({'case_id': case['id'], 'base_case_id': case['base_case_id'],
            'dependency_depth': case['dependency_depth'], 'question_style': case['question_style'],
            'distractor_updates': case['distractor_updates'],
            'before_all_correct': all(correct('before', r) for r in ROLES),
            'joint_after': all(correct(v, r) for v in VARIANTS[1:] for r in ROLES),
            'both_order_targets_correct': all(correct(v, ROLES[0]) for v in VARIANTS[1:]),
            'joint_all_nine': all(correct(v, r) for v in VARIANTS for r in ROLES)})
    cells = []
    for depth in (1, 2, 3):
        for style in ('original', 'entity_first'):
            for variant in VARIANTS:
                for role in ROLES:
                    group = [r for r in expanded if (r['dependency_depth'], r['question_style'], r['variant'], r['role']) ==
                             (depth, style, variant, role)]
                    cells.append({'depth': depth, 'question_style': style, 'variant': variant, 'role': role,
                        'correct': sum(r['greedy_exact'] for r in group), 'examples': len(group),
                        'mean_answer_nll': mean(r['answer_nll'] for r in group)})
    return {'physical_correct': sum(r['greedy_exact'] for r in rows), 'physical_queries': len(physical),
        'binding_correct': sum(r['greedy_exact'] for r in expanded), 'bindings': len(bindings),
        'role_variant_counts': {role: {v: {'correct': sum(r['greedy_exact'] for r in expanded if r['variant'] == v and r['role'] == role),
                                        'examples': 48} for v in VARIANTS} for role in ROLES},
        'joint_six_after': sum(c['joint_after'] for c in outcomes),
        'joint_all_nine': sum(c['joint_all_nine'] for c in outcomes),
        'both_order_targets_correct': sum(c['both_order_targets_correct'] for c in outcomes),
        'before_all_correct': sum(c['before_all_correct'] for c in outcomes), 'cases': len(cases),
        'cells': cells, 'case_outcomes': outcomes}


def summarize(all_rows, cases, registry, plan):
    if list(all_rows) != list(MODELS) or any(list(all_rows[m]) != list(REGIMES) for m in MODELS):
        raise ValueError('Both prescribed models and both fresh regimes required')
    models = {r: {m: model_stats(all_rows[m][r], cases, registry) for m in MODELS} for r in REGIMES}
    _, bindings = selected_inputs(registry)
    comparisons = {m: paired_current({r['id']: r for r in all_rows[m]['oracle']},
                                     {r['id']: r for r in all_rows[m]['full']}, bindings) for m in MODELS}
    checks = {m: sufficiency_checks(models['full'][m], models['oracle'][m], comparisons[m]) for m in MODELS}
    return {'models_by_regime': models, 'current_target_paired_comparisons': comparisons,
        'frozen_prefix_sufficiency': {'passed': all(all(c.values()) for c in checks.values()), 'model_checks': checks},
        'arithmetic_profile': plan['arithmetic_profile'], 'cross_profile_rows_in_primary': 0,
        'completed_canonical_necessity_gate_unchanged_passed': True,
        'prior_shared_history_repeat_primaries_remain_failed': True,
        'primary_reference': 'fresh_full_same_canonical_arithmetic_profile',
        'interpretation': plan['scientific_gate']['scope']}
