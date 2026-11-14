"""Export same-profile oracle replication and clearly sourced necessity context."""

import argparse
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_canonical_oracle_replication import audit_run
from scripts.canonical_oracle_common import MODELS, REGIMES, read_plan, registered_inputs
from scripts.canonical_oracle_metrics import summarize
from scripts.copy_necessity_metrics import after_count
from scripts.copy_readout_metrics import paired_current
from scripts.copy_readout_runtime import readout
from scripts.copy_transfer_data import ROLES, selected_inputs
from scripts.run_controller_trial import write_json

COMBINED = ('full', 'oracle', 'remove_support', 'remove_matched_other')
COLORS = ('#64748b', '#16a34a', '#e11d48', '#0284c7')
LABELS = ('Fresh full history', 'Oracle whole-event evidence',
          'Remove evidence SET (prior canonical)', 'Remove nearby other SET (prior canonical)')
DISPLAY = ('Local question', 'GDN question')


def complete_inputs(audit_path):
    report = json.loads(audit_path.read_text())
    run = Path(report['source_run'])
    if (report['status'] != 'completed_and_audited_canonical_oracle' or report['new_physical_readouts_audited'] != 1344 or
        report['fresh_full_reference_readouts_audited'] != 672 or report['new_related_bindings_audited'] != 1728 or
        report['exact_completed_canonical_full_replays'] != 672 or
        streaming_hash(audit_path.with_suffix('.source.py')) != report['analysis_source_sha256'] or audit_run(run) != report):
        raise ValueError('Complete independently reproducible1344-readout oracle audit required')
    plan = read_plan()
    manifest = json.loads((run / 'manifest.json').read_text())
    _, source_plan, cases, registry, masks, references = registered_inputs(Path(manifest['dataset']))
    rows = {m: {r: [json.loads(line) for line in (run / f'{m}.{r}.jsonl').read_text().splitlines()]
                for r in REGIMES} for m in MODELS}
    metrics = summarize(rows, cases, registry, plan)
    if metrics != report['metrics']:
        raise ValueError('Fresh same-profile metrics do not reproduce the independent audit')
    prior = json.loads(Path(plan['completed_necessity_audit']).read_text())
    prior_run = Path(prior['source_run'])
    combined = {r: {} for r in COMBINED}
    for m in MODELS:
        for r in COMBINED:
            combined[r][m] = rows[m][r] if r in REGIMES else [json.loads(line) for line in (prior_run / f'{m}.{r}.jsonl').read_text().splitlines()]
    _, bindings = selected_inputs(registry)
    effects = {m: paired_current({r['id']: r for r in combined['remove_matched_other'][m]},
                                 {r['id']: r for r in combined['remove_support'][m]}, bindings) for m in MODELS}
    if any(effects[m] != prior['metrics']['support_removal_effects'][m] for m in MODELS):
        raise ValueError('Sealed same-profile necessity contrasts changed in the joined export')
    return report, manifest, registry, combined, metrics, prior, effects


def make_plot(stats, passed, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=160)
    current, joint, depth, preserve = axes.flat
    y = np.arange(2)
    for shift, regime, color, label in zip((-.3, -.1, .1, .3), COMBINED, COLORS, LABELS):
        for ax, values in ((current, [after_count(stats[regime][m], ROLES[0]) for m in MODELS]),
                           (joint, [stats[regime][m]['joint_six_after'] for m in MODELS])):
            ax.barh(y + shift, values, height=.19, color=color, label=label)
            for pos, value in zip(y + shift, values):
                ax.text(value + .5, pos, str(value), va='center', fontsize=9)
    for ax, limit, title, xlabel in (
        (current, 96, 'Current COPY target after updates', 'Correct answers /96 related bindings'),
        (joint, 48, 'All roles in both update orders', 'All six correct /48 related cases')):
        ax.set_yticks(y, DISPLAY); ax.invert_yaxis(); ax.set_xlim(0, limit * 1.1)
        ax.set_xticks(np.linspace(0, limit, 7)); ax.set_title(title, fontsize=12); ax.set_xlabel(xlabel)
    positions = np.arange(6)
    for shift, regime, color in zip((-.3, -.1, .1, .3), COMBINED, COLORS):
        values = [sum(c['correct'] for c in stats[regime][m]['cells'] if c['depth'] == d and
                      c['role'] == ROLES[0] and c['variant'] != 'before') for m in MODELS for d in (1, 2, 3)]
        depth.bar(positions + shift, values, width=.19, color=color)
        for pos, value in zip(positions + shift, values):
            depth.text(pos, value + .3, str(value), ha='center', fontsize=7)
    depth.set_xticks(positions, [f'Local Q\nD{d}' for d in (1, 2, 3)] + [f'GDN Q\nD{d}' for d in (1, 2, 3)])
    depth.set_ylim(0, 36); depth.set_yticks(range(0, 33, 8))
    depth.set_ylabel('Correct current answers /32 related bindings')
    depth.set_title('Every registered dependency depth', fontsize=12)
    counts = np.array([[after_count(stats[r][m], role) for r in COMBINED]
                       for m in MODELS for role in (ROLES[2], ROLES[1])])
    preserve.imshow(counts / 96, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    preserve.set_yticks(range(4), ['Local Q: history', 'Local Q: unrelated', 'GDN Q: history', 'GDN Q: unrelated'])
    preserve.set_xticks(range(4), ['Full', 'Oracle', 'Drop evidence', 'Drop other'])
    preserve.set_title('Historical versions and unrelated facts after updates', fontsize=12)
    for i in range(4):
        for j in range(4):
            preserve.text(j, i, f'{counts[i,j]}/96', ha='center', va='center', fontsize=10,
                          color='white' if counts[i,j] >= 58 else '#111827')
    handles, labels = current.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .922), ncol=2, fontsize=9)
    outcome = 'passed' if passed else 'not passed'
    fig.suptitle(f'Same-profile oracle sufficiency replication\nRegistered sufficiency screen: {outcome}', fontsize=16, y=.985)
    fig.text(.5, .014, '1,344 NEW full/oracle readouts; all672full observations exactly replay the completed canonical necessity phase.\n'
             'The two removal conditions reuse1,344audited canonical observations; 48related cases share8base streams.\n'
             'All weights and original prefix memories preserved. Program-oracle addresses are provided; this is not a learned router or architecture result.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .095, 1, .875), h_pad=3.5, w_pad=3)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    if any(legend.get_window_extent(renderer).overlaps(ax.get_tightbbox(renderer)) for ax in axes.flat):
        raise ValueError('Figure-wide legend overlaps a plot or its text')
    width, height = fig.canvas.get_width_height()
    outside = []
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text().strip(): continue
        box = item.get_window_extent(renderer)
        if box.width > 0 and box.height > 0 and (box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1):
            outside.append({'text': item.get_text(), 'bbox': list(box.bounds)})
    if outside: raise ValueError(f'Figure text outside image: {outside}')
    fig.savefig(path, dpi=160); fig.savefig(path.with_suffix('.pdf'))
    write_json(path.with_suffix('.layout.json'), {'status': 'all_visible_text_within_figure_and_legend_disjoint',
        'width': width, 'height': height, 'outside_text': outside, 'source_sha256': streaming_hash(Path(__file__)),
        'png_sha256': streaming_hash(path), 'pdf_sha256': streaming_hash(path.with_suffix('.pdf'))})
    plt.close(fig)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def export(audit_path, output, plot):
    report, manifest, registry, rows, metrics, prior, effects = complete_inputs(audit_path)
    output.parent.mkdir(parents=True, exist_ok=True); plot.parent.mkdir(parents=True, exist_ok=True)
    stats = {r: metrics['models_by_regime'][r] if r in REGIMES else
             {m: prior['metrics']['models_by_regime'][r][m] for m in MODELS} for r in COMBINED}
    source_sha = streaming_hash(Path(__file__))
    output_format = {r: {m: {'physical_queries': len(rows[r][m]),
        'greedy_correct': sum(v['greedy_exact'] for v in rows[r][m]),
        'integer_output': sum(bool(re.fullmatch(r'\s*-?\d+\s*', v['generated'])) for v in rows[r][m])}
        for m in MODELS} for r in COMBINED}
    result = {'status': 'complete_canonical_oracle_replication_export', 'source_audit': str(audit_path),
        'source_audit_sha256': streaming_hash(audit_path), 'source_manifest_sha256': report['manifest_sha256'],
        'completed_necessity_audit': read_plan()['completed_necessity_audit'],
        'completed_necessity_audit_sha256': read_plan()['completed_necessity_audit_sha256'],
        'completed_necessity_manifest_sha256': prior['manifest_sha256'],
        'dataset_manifest_sha256': report['dataset_manifest_sha256'], 'mask_registry_sha256': report['mask_registry_sha256'],
        'arithmetic_profile': report['arithmetic_profile'], 'weights_sha256': report['weights_sha256'],
        'analysis_source_sha256': source_sha, 'references': report['references'],
        'counts': {'new_physical': 1344, 'new_full': 672, 'new_oracle': 672, 'new_related_bindings': 1728,
            'exact_prior_canonical_full_replays': 672, 'reused_canonical_masked_physical': 1344,
            'combined_physical_export_rows': 2688, 'combined_related_bindings': 3456,
            'new_oracle_full_current_pairs': 192, 'combined_current_pairs': 768, 'cross_profile_rows': 0},
        'metrics': metrics, 'combined_models_by_regime': stats,
        'recorded_canonical_necessity_effects': effects,
        'same_profile_sufficiency_and_necessity_screens': {
            'sufficiency_passed': metrics['frozen_prefix_sufficiency']['passed'],
            'necessity_passed': prior['metrics']['whole_record_necessity']['passed'],
            'all672_full_reference_readouts_exact': True,
            'interpretation': 'Functional fixed-readout access dependence and oracle recoverability on the same8reused streams. This is not evidence that a GDN state lost or retained every fact, nor a learned addressing mechanism, shared-control advantage, new architecture or independent-data replication.'},
        'descriptive_output_format_not_gate': output_format,
        'prior_shared_history_repeat_primaries_remain_failed': True}
    write_json(output, result); output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    combined_rows = [{'model': m, 'regime': r, 'fresh_observation': r in REGIMES,
        'source_audit_sha256': result['source_audit_sha256'] if r in REGIMES else result['completed_necessity_audit_sha256'],
        'arithmetic_profile': report['arithmetic_profile'], 'id': row['id'], **readout(row)}
        for r in COMBINED for m in MODELS for row in rows[r][m]]
    primary = [r for r in combined_rows if r['fresh_observation']]
    write_csv(output.with_suffix('.csv'), primary)
    combined_path = output.with_name(output.stem + '-combined.csv'); write_csv(combined_path, combined_rows)
    _, bindings = selected_inputs(registry)
    combined_bindings = []
    for r in COMBINED:
        for m in MODELS:
            indexed = {row['id']: row for row in combined_rows if row['model'] == m and row['regime'] == r}
            combined_bindings.extend({**b, **indexed[b['physical_id']]} for b in bindings)
    primary_bindings = [b for b in combined_bindings if b['fresh_observation']]
    binding_path = output.with_name(output.stem + '-bindings.csv'); write_csv(binding_path, primary_bindings)
    combined_binding_path = output.with_name(output.stem + '-combined-bindings.csv'); write_csv(combined_binding_path, combined_bindings)
    pairs = [{'model': m, 'candidate_regime': 'oracle', 'reference_regime': 'full', 'new_oracle_contrast': True, **p}
             for m in MODELS for p in metrics['current_target_paired_comparisons'][m]['individual_pairs']]
    pair_path = output.with_name(output.stem + '-pairs.csv'); write_csv(pair_path, pairs)
    all_pairs = pairs + [{'model': m, 'candidate_regime': r, 'reference_regime': 'full', 'new_oracle_contrast': False, **p}
        for m in MODELS for r in ('remove_support', 'remove_matched_other')
        for p in prior['metrics']['current_comparisons_vs_full'][m][r]['individual_pairs']]
    all_pairs += [{'model': m, 'candidate_regime': 'remove_matched_other', 'reference_regime': 'remove_support',
        'new_oracle_contrast': False, **p} for m in MODELS for p in effects[m]['individual_pairs']]
    combined_pair_path = output.with_name(output.stem + '-combined-pairs.csv'); write_csv(combined_pair_path, all_pairs)
    if (len(primary), len(primary_bindings), len(combined_rows), len(combined_bindings), len(pairs), len(all_pairs)) != (1344, 1728, 2688, 3456, 192, 768):
        raise ValueError('Complete primary and source-labelled combined export counts changed')
    make_plot(stats, metrics['frozen_prefix_sufficiency']['passed'], plot)
    artifacts = (output, output.with_suffix('.source.py'), output.with_suffix('.csv'), combined_path, binding_path,
                 combined_binding_path, pair_path, combined_pair_path, plot, plot.with_suffix('.pdf'), plot.with_suffix('.layout.json'))
    write_json(output.with_suffix('.provenance.json'), {'status': 'complete_export_source_and_artifact_closure',
        'source_audit_sha256': result['source_audit_sha256'], 'source_manifest_sha256': report['manifest_sha256'],
        'completed_necessity_audit_sha256': result['completed_necessity_audit_sha256'],
        'exporter_sha256': source_sha, 'artifacts_sha256': {str(p): streaming_hash(p) for p in artifacts},
        'run_sources_sha256': manifest['source_sha256'], 'counts': result['counts'],
        'visual_inspection': 'pending separate inspection'})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path); parser.add_argument('--output', type=Path, required=True); parser.add_argument('--plot', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.audit, args.output, args.plot)
    print(json.dumps({'status': result['status'], 'counts': result['counts'],
                      'gate': result['metrics']['frozen_prefix_sufficiency']}), flush=True)
