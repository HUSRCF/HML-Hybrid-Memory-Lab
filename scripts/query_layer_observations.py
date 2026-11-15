"""Descriptive extraction of all completed single-layer interventions; no gates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.query_layer_data import MODELS, LAYERS, DIRECTIONS
from scripts.post_encoding_metrics import paired
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def extract(audit_path):
    audit = json.loads(audit_path.read_text())
    if (audit['status'] != 'completed_and_audited_query_layer' or
            audit['new_primary_readouts_audited'] != 4608 or audit['total_source_files_verified'] != 234):
        raise ValueError('Complete independently audited primary required')
    run = Path(audit['source_run'])
    manifest = json.loads((run / 'manifest.json').read_text())
    if streaming_hash(run / 'manifest.json') != audit['manifest_sha256']:
        raise ValueError('Audited manifest changed')
    packet = Path(manifest['packet'])
    if streaming_hash(packet / 'manifest.json') != audit['packet_manifest_sha256']:
        raise ValueError('Audited input packet changed')
    registration = json.loads((packet / 'manifest.json').read_text())
    for name in ('queries.json', 'references.json'):
        if streaming_hash(packet / name) != registration['payload_sha256'][name]:
            raise ValueError('Frozen query or reference payload changed')
    queries = json.loads((packet / 'queries.json').read_text())
    refs = json.loads((packet / 'references.json').read_text())
    ids = [key for key, q in queries.items() if q['episode'] != 'before']
    if len(ids) != 192: raise ValueError('Complete reused question grid required')
    metrics = audit['metrics']
    grid, seeds, patterns, downstream = [], {}, {}, {}
    for model in MODELS:
        raw = {}
        for kind in ('primary', 'donors'):
            path = run / f'{model}.{kind}.jsonl'
            if streaming_hash(path) != manifest['models'][model]['results_sha256'][kind]:
                raise ValueError('Audited raw bytes changed')
            raw[kind] = [json.loads(line) for line in path.read_text().splitlines()]
        index = {(r['id'], r['condition'], r['layer']): r for r in raw['primary']}
        old = {(r['id'], r['condition']): r for r in refs['full_readouts'][model]}
        donors = {(r['id'], r['origin']): r['bank'] for r in raw['donors']}
        if len(index) != 2304 or len(old) != 768 or len(donors) != 384:
            raise ValueError('Complete new and archived coverage required')
        for cell, value in metrics['cells'][model].items():
            evidence = metrics['strata'][model][cell]['evidence']
            grid.append({'model': model, 'cell': cell, 'all': value,
                         'new_tail': evidence['False'], 'fixed_prefix': evidence['True'],
                         'by_role': metrics['strata'][model][cell]['role']})
        seeds[model], patterns[model], downstream[model] = {}, {}, {}
        for direction in DIRECTIONS:
            recipient = direction[0]
            patterns[model][direction] = {}
            for evidence, flag in (('all', None), ('fixed_prefix', True), ('new_tail', False)):
                keys = [key for key in ids if flag is None or queries[key]['prefix_evidence'] == flag]
                counts = Counter(''.join(str(int(v['greedy_exact'])) for v in
                    [old[(key, recipient * 2)], *[index[(key, direction, layer)] for layer in LAYERS], old[(key, direction)]]) for key in keys)
                patterns[model][direction][evidence] = {'readouts': len(keys),
                    'column_order': ['recipient_baseline', *[f'L{layer:02d}' for layer in LAYERS], 'all6_cross'],
                    'all256_correctness_patterns': {f'{i:08b}': counts[f'{i:08b}'] for i in range(256)}}
            seeds[model][direction], downstream[model][direction] = {}, {}
            for layer in LAYERS:
                seeds[model][direction][str(layer)] = {
                    reference: {str(seed): paired(
                        [index[(key, direction, layer)] for key in ids if queries[key]['seed'] == seed],
                        [old[(key, reference)] for key in ids if queries[key]['seed'] == seed])
                        for seed in sorted({queries[key]['seed'] for key in ids})}
                    for reference in (recipient * 2, direction)}
                downstream[model][direction][str(layer)] = {}
                for later in LAYERS:
                    if later <= layer: continue
                    values = [index[(key, direction, layer)]['proof']['query_exchange']['layers'][str(later)] for key in ids]
                    if not all(v['original_query_object_preserved'] and not v['actual_replacement'] for v in values):
                        raise ValueError('Every unselected downstream Query must retain actual native identity')
                    downstream[model][direction][str(layer)][str(later)] = {'readouts': 192,
                        'native_query_changed_from_pure_recipient': sum(
                            value['native_query']['sha256'] != donors[(key, recipient)]['tensors'][str(later)]['sha256']
                            for key, value in zip(ids, values))}
    return {'status': 'posthoc_descriptive_extraction_of_complete_query_layer_grid',
            'audit_sha256': streaming_hash(audit_path), 'source_sha256': streaming_hash(Path(__file__)),
            'all32_cells': grid, 'all_seed_paired_contrasts': seeds, 'all_correctness_patterns': patterns,
            'unclamped_downstream_query_changes': downstream,
            'independent_reused_streams': 8, 'fresh_confirmation': False, 'baseline_failures_excluded': 0,
            'new_scientific_gate': False, 'architecture_effectiveness_claim': False,
            'previous_failed_criteria_unchanged': audit['prior_failed_criteria_unchanged'],
            'limits': ['Correctness patterns are descriptive, not a deployable per-question layer selector.',
                'A pattern involving several layers does not establish synergy, necessity or additivity.',
                'Later native question Queries and question KV can change causally; historical source KV stay guarded.',
                'Crossed activations can be off distribution, and donors include upstream Attention computation.',
                'Current inputs use211prefix and112/896tail tokens; they do not establish100K-context behavior.',
                'This experiment does not demonstrate a new shared architecture or change any previous failed criterion.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Never overwrite an extraction')
    result = extract(args.audit)
    write_json(args.output, result)
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: result[k] for k in ('status', 'audit_sha256', 'independent_reused_streams')}), flush=True)


if __name__ == '__main__':
    main()
