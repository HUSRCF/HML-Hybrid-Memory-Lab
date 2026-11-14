"""Posthoc four-point margins: separate a decision threshold from pair-logit interaction."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_fresh_token_timing import audit_run
from scripts.run_controller_trial import write_json


def analyze(path):
    report = json.loads(path.read_text())
    if audit_run(Path(report['source_run'])) != report:
        raise ValueError('Actual timing observations or independent audit changed')
    cells = []
    for case in report['cases']:
        baseline = case['full_first_token']
        for item in case['conditions']:
            values = [baseline, item['last_question_token'], item['earlier_question_tokens'], item['complete_question']]
            competitor_ids = [v['non_gold_id'] for v in values]
            stable = len(set(competitor_ids)) == 1
            diff = values[3]['gold_margin'] - values[1]['gold_margin'] - values[2]['gold_margin'] + values[0]['gold_margin']
            joint_only = (values[3]['argmax_id'] != values[0]['argmax_id'] and
                values[1]['argmax_id'] == values[0]['argmax_id'] and values[2]['argmax_id'] == values[0]['argmax_id'])
            cells.append({'model': case['model'], 'id': case['id'], 'source_regime': case['source_regime'],
                'condition': item['condition'], 'order': ['unmasked', 'last_only', 'earlier_only', 'complete_question'],
                'first_tokens': values, 'strongest_non_gold_ids': competitor_ids,
                'same_non_gold_competitor_in_all4': stable,
                'fixed_competitor_margin_difference_in_differences': diff if stable else None,
                'combined_first_argmax_flip_without_either_alone': joint_only,
                'zero_fixed_pair_interaction_for_joint_threshold_crossing': joint_only and stable and diff == 0})
    return {'status': 'posthoc_descriptive_factorial_margins', 'source_audit': str(path),
        'source_audit_sha256': streaming_hash(path), 'source_manifest_sha256': report['manifest_sha256'],
        'no_new_inference': True, 'no_accuracy_gate': True, 'can_change_primary_gates': False,
        'independent_scientific_confirmation': False, 'all32case_layer_cells': cells,
        'counts': {'case_layer_cells': len(cells),
            'stable_competitor_cells': sum(c['same_non_gold_competitor_in_all4'] for c in cells),
            'combined_first_flip_without_either_alone': sum(c['combined_first_argmax_flip_without_either_alone'] for c in cells),
            'joint_threshold_crossings_with_exact_zero_fixed_pair_interaction': sum(c['zero_fixed_pair_interaction_for_joint_threshold_crossing'] for c in cells)},
        'interpretation': 'For a fixed gold and fixed non-gold token, D=(both-last-earlier+unmasked) is a four-point logit-margin interaction, not a semantic-memory metric. Only compute it when the strongest non-gold token is unchanged in all four observations. An exact zero with a combined-only argmax flip shows that threshold crossing needs no positive pair-margin interaction on these four points. Nonzero values at BF16 logit granularity alone do not establish semantic complementarity or a novel architecture. No tolerance, subset exclusion or primary gate relabeling is introduced.',
        'analysis_source_sha256': streaming_hash(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve prior descriptive analyses')
    report = analyze(args.audit); write_json(args.output, report)
    source = args.output.with_suffix('.source.py'); source.write_bytes(Path(__file__).read_bytes())
    write_json(args.output.with_suffix('.provenance.json'), {'source_audit_sha256': report['source_audit_sha256'],
        'source_manifest_sha256': report['source_manifest_sha256'],
        'artifacts_sha256': {str(p): streaming_hash(p) for p in (args.output, source)}})
    print(json.dumps(report['counts']), flush=True)
