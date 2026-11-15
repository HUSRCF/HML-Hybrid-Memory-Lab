"""All single-layer cells, with archived all-layer references clearly labelled."""
from scripts.query_layer_data import MODELS, LAYERS, DIRECTIONS, REFERENCES
from scripts.post_encoding_tasks import ROLES, STYLES
from scripts.post_encoding_metrics import stats
from scripts.post_tail_path_metrics import functional_stats, split_paired


def summarize(all_rows, queries, refs):
    ids = [key for key, q in queries.items() if q['episode'] != 'before']
    if len(ids) != 192:
        raise ValueError('All192reused after questions required')
    cells, strata, episodes, contrasts = {}, {}, {}, {}
    for model in MODELS:
        rows = all_rows[model]
        index = {(r['id'], r['condition'], r['layer']): r for r in rows}
        archived = {(r['id'], r['condition']): r for r in refs['full_readouts'][model]}
        if len(rows) != 2304 or len(index) != 2304 or len(archived) != 768:
            raise ValueError('Complete2304new rows and768archived references per model required')
        groups = {c + '.all6': [archived[(key, c)] for key in ids] for c in REFERENCES}
        for direction in DIRECTIONS:
            for layer in LAYERS:
                groups[f'{direction}.L{layer:02d}'] = [index[(key, direction, layer)] for key in ids]
        cells[model], strata[model], episodes[model], contrasts[model] = {}, {}, {}, {}
        for cell, values in groups.items():
            cells[model][cell] = {**functional_stats(values, queries),
                'measurement_origin': 'archived_reference' if cell.endswith('.all6') else 'new_single_layer_query'}
            strata[model][cell] = {
                'role': {r: stats([v for v in values if queries[v['id']]['role'] == r]) for r in ROLES},
                'style': {s: stats([v for v in values if queries[v['id']]['question_style'] == s]) for s in STYLES},
                'seed': {str(seed): stats([v for v in values if queries[v['id']]['seed'] == seed])
                         for seed in sorted({queries[key]['seed'] for key in ids})},
                'evidence': {str(flag): stats([v for v in values if queries[v['id']]['prefix_evidence'] == flag]) for flag in (True, False)}}
            episodes[model][cell] = {}
            for episode in ('unrelated-8', 'related-8', 'unrelated-64', 'related-64'):
                sub = [v for v in values if queries[v['id']]['episode'] == episode]
                episodes[model][cell][episode] = {'all': functional_stats(sub, queries),
                    'role': {r: stats([v for v in sub if queries[v['id']]['role'] == r]) for r in ROLES}}
        for direction in DIRECTIONS:
            baseline = direction[0] * 2 + '.all6'
            for layer in LAYERS:
                cell = f'{direction}.L{layer:02d}'
                for reference in (baseline, direction + '.all6'):
                    contrasts[model][cell + '_vs_' + reference] = split_paired(groups[cell], groups[reference], queries)
    return {'cells': cells, 'strata': strata, 'episodes': episodes, 'paired_contrasts': contrasts,
            'new_primary_readouts': 4608, 'archived_references': 1536, 'unified_readouts': 6144,
            'new_cell_count': 24, 'archived_cell_count': 8, 'cell_count': 32,
            'independent_streams': 8, 'fresh_confirmation': False, 'baseline_failures_excluded': 0,
            'success_thresholds': None, 'architecture_effectiveness_claim': False,
            'previous_gates_can_change': False,
            'limits': 'Reused cohort; single-layer success does not establish necessity, additivity or a new architecture. Later native Queries can change causally.'}
