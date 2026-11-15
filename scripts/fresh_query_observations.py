"""Descriptive extraction of the entire completed fresh Query factorial."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fresh_query_data import MODELS, READERS, CONDITIONS
from scripts.post_encoding_metrics import paired
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def extract(audit_path):
    audit = json.loads(audit_path.read_text())
    metrics = audit['metrics']
    if (audit['status'] != 'completed_and_audited_fresh_query' or
            audit['primary_readouts_audited'] != 3264 or metrics['cell_count'] != 20):
        raise ValueError('Complete independently audited fresh factorial required')
    run = Path(audit['source_run'])
    manifest = json.loads((run / 'manifest.json').read_text())
    if streaming_hash(run / 'manifest.json') != audit['manifest_sha256']:
        raise ValueError('Audited manifest changed')
    query_file = Path(manifest['packet']) / 'queries.jsonl'
    # Use the registered query list, rather than any outcome-based subset.
    if not query_file.exists():
        query_file = Path(manifest['packet']) / 'queries.json'
    payload = query_file.read_text()
    raw_queries = json.loads(payload) if query_file.suffix == '.json' else [json.loads(v) for v in payload.splitlines()]
    if isinstance(raw_queries, dict):
        raw_queries = raw_queries.get('queries', raw_queries)
    queries = raw_queries if isinstance(raw_queries, dict) else {q['id']: q for q in raw_queries}
    after_ids = [key for key, q in queries.items() if q['episode'] != 'before']
    if len(queries) != 240 or len(after_ids) != 192:
        raise ValueError('All registered questions required')
    grid, patterns, by_seed, query_differences = [], {}, {}, {}
    for model in MODELS:
        for cell, values in metrics['cells'][model].items():
            strata = metrics['strata'][model][cell]
            grid.append({'model': model, 'cell': cell, 'all': values,
                         'fixed_prefix': strata['evidence']['True'],
                         'new_tail': strata['evidence'].get('False'), 'by_role': strata['role']})
        rows_path = run / f'{model}.primary.jsonl'
        donor_path = run / f'{model}.donors.jsonl'
        for kind, path in (('primary', rows_path), ('donors', donor_path)):
            if streaming_hash(path) != manifest['models'][model]['results_sha256'][kind]:
                raise ValueError('Audited raw rows changed')
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        index = {(r['id'], r['condition'], r['reader']): r for r in rows}
        donors = [json.loads(line) for line in donor_path.read_text().splitlines()]
        banks = {(r['id'], r['origin'], r['reader']): r['bank'] for r in donors}
        if len(rows) != 1632 or len(index) != 1632 or len(donors) != 768 or len(banks) != 768:
            raise ValueError('Complete primary and donor rows required')
        patterns[model], by_seed[model], query_differences[model] = {}, {}, {}
        for reader in READERS:
            counts = Counter(''.join(str(int(index[(key, c, reader)]['greedy_exact'])) for c in CONDITIONS)
                             for key in after_ids)
            patterns[model][reader] = {
                'condition_order': list(CONDITIONS), 'readouts': 192,
                'all16_correctness_patterns': {f'{i:04b}': counts[f'{i:04b}'] for i in range(16)}}
            by_seed[model][reader] = {}
            for candidate, reference in (('HN', 'HH'), ('NH', 'NN'), ('HH', 'NN'), ('HN', 'NN'), ('NH', 'HH')):
                by_seed[model][reader][candidate + '_vs_' + reference] = {
                    str(seed): paired([index[(key, candidate, reader)] for key in after_ids if queries[key]['seed'] == seed],
                                      [index[(key, reference, reader)] for key in after_ids if queries[key]['seed'] == seed])
                    for seed in sorted({queries[key]['seed'] for key in after_ids})}
            query_differences[model][reader] = {}
            layers = list(banks[(after_ids[0], 'N', reader)]['tensors'])
            if len(layers) != 6:
                raise ValueError('All six captured Attention Queries required')
            for layer in layers:
                query_differences[model][reader][layer] = {
                    episode: {'pairs': len(keys), 'different': sum(
                        banks[(key, 'N', reader)]['tensors'][layer]['sha256'] !=
                        banks[(key, 'H', reader)]['tensors'][layer]['sha256'] for key in keys)}
                    for episode, keys in [('all', after_ids)] + [
                        (ep, [key for key in after_ids if queries[key]['episode'] == ep])
                        for ep in ('unrelated-8', 'related-8', 'unrelated-64', 'related-64')]}
    return {'status': 'posthoc_descriptive_extraction_of_complete_fresh_query_factorial',
            'audit_sha256': streaming_hash(audit_path), 'source_sha256': streaming_hash(Path(__file__)),
            'all20_cells_with_fixed_and_new_evidence': grid,
            'all_paired_contrasts': metrics['paired_contrasts'], 'all_seed_paired_contrasts': by_seed,
            'correctness_patterns': patterns, 'actual_query_source_difference_counts': query_differences,
            'nll_interaction_means': {model: {reader: v['means'] for reader, v in groups.items()}
                                      for model, groups in metrics['nll_interactions'].items()},
            'independent_fresh_streams': 8, 'baseline_failures_excluded': 0,
            'new_scientific_gate': False, 'architecture_effectiveness_claim': False,
            'previous_failed_criteria_unchanged': audit['prior_failed_criteria_unchanged'],
            'interpretation_limits': [
                'Only the question-suffix forward is patched; answer continuation remains native.',
                'Full and privileged Oracle donors are reader-matched and never exchanged.',
                'Donor Queries include upstream Attention and recurrent computations.',
                'A crossed Query can be off distribution; a rescue is not a unique semantic-address explanation.',
                'The eight streams, rather than repeated questions or donor tensors, define the independent cohort.',
                'This phase does not compare or establish a new shared architecture.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite an extraction')
    result = extract(args.audit)
    write_json(args.output, result)
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: result[k] for k in ('status', 'audit_sha256', 'independent_fresh_streams')}), flush=True)


if __name__ == '__main__':
    main()
