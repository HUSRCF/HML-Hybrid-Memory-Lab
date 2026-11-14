"""Export every fresh natural control and readout intervention with separate claims."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_fresh_copy_confirmation import audit_run
from scripts.copy_necessity_metrics import after_count
from scripts.copy_readout_runtime import readout
from scripts.fresh_copy_data import MODELS, PRIMARY, MASKED_REGIMES, ROLES, read_plan, registered_inputs, selected_inputs
from scripts.fresh_copy_metrics import summarize
from scripts.run_controller_trial import write_json

REGIMES = ('full', *MASKED_REGIMES)
COLORS = ('#64748b', '#16a34a', '#e11d48', '#0284c7')
LABELS = ('Full history', 'Oracle whole-event evidence', 'Remove evidence SET', 'Remove nearby other SET')
DISPLAY = {'native': 'Native', 'shared': 'Shared', 'independent': 'Independent', 'local': 'Local all tokens',
    'local_question': 'Local question', 'attention_only': 'Attention only',
    'gdn_all_tokens': 'GDN all tokens', 'gdn_question_only': 'GDN question'}


def complete_inputs(audit_path):
    report = json.loads(audit_path.read_text())
    run = Path(report['source_run'])
    if (report['status'] != 'completed_and_audited_fresh_copy' or
        report['new_physical_readouts_audited'] != 4704 or
        report['fresh_full_reference_readouts_audited'] != 2688 or
        report['new_masked_readouts_audited'] != 2016 or
        report['new_related_bindings_audited'] != 6048 or
        streaming_hash(audit_path.with_suffix('.source.py')) != report['analysis_source_sha256'] or
        audit_run(run) != report):
        raise ValueError('Complete independently reproducible4704-readout fresh audit required')
    manifest = json.loads((run / 'manifest.json').read_text())
    _, source_plan, cases, registry, _, = registered_inputs(Path(manifest['dataset']))
    plan = read_plan()
    rows = {m: {r: [json.loads(line) for line in (run / f'{m}.{r}.jsonl').read_text().splitlines()]
                for r in plan['regimes_by_model'][m]} for m in MODELS}
    metrics = summarize(rows, cases, registry, source_plan, plan)
    if metrics != report['metrics']:
        raise ValueError('Fresh metrics do not reproduce the independent audit')
    return report, manifest, registry, rows, metrics


def make_plot(metrics, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(17, 12.5), dpi=160)
    natural, current, comparison, preserve = axes.flat
    full = metrics['natural_full_models']
    values = [full[m]['joint_six_after'] for m in MODELS]
    positions = np.arange(len(MODELS))
    natural.barh(positions, values, color=['#7c3aed' if m == 'shared' else '#64748b' for m in MODELS])
    for pos, value in zip(positions, values):
        natural.text(value + .35, pos, str(value), va='center', fontsize=10)
    natural.set_yticks(positions, [DISPLAY[m] for m in MODELS]); natural.invert_yaxis()
    natural.set_xlim(0, 53); natural.set_xticks(range(0, 49, 8))
    natural.set_xlabel('All six after-update answers correct /48 related cases')
    natural.set_title('All eight fixed models: natural full-history evaluation', fontsize=12)

    stats = metrics['readout_models_by_regime']
    y = np.arange(2)
    for shift, regime, color, label in zip((-.3, -.1, .1, .3), REGIMES, COLORS, LABELS):
        counts = [after_count(stats[regime][m], ROLES[0]) for m in PRIMARY]
        current.barh(y + shift, counts, height=.19, color=color, label=label)
        for pos, value in zip(y + shift, counts):
            current.text(value + .5, pos, str(value), va='center', fontsize=9)
    current.set_yticks(y, [DISPLAY[m] for m in PRIMARY]); current.invert_yaxis()
    current.set_xlim(0, 106); current.set_xticks(range(0, 97, 16))
    current.set_xlabel('Correct current answers /96 related bindings')
    current.set_title('Fresh evidence-access interventions: both prescribed models', fontsize=12)

    refs = ('independent', 'local_question', 'gdn_question_only')
    selected = [next(c for c in metrics['natural_full_comparisons']
                     if c['candidate'] == 'shared' and c['reference'] == ref) for ref in refs]
    y = np.arange(3)
    comparison.barh(y - .12, [c['gains'] for c in selected], height=.23, color='#16a34a', label='Cases gained')
    comparison.barh(y + .12, [-c['losses'] for c in selected], height=.23, color='#e11d48', label='Cases lost')
    for pos, c in zip(y, selected):
        comparison.text(c['gains'] + .25, pos - .12, str(c['gains']), va='center', fontsize=10)
        comparison.text(-c['losses'] - .25, pos + .12, str(c['losses']), ha='right', va='center', fontsize=10)
    limit = max(3, max(max(c['gains'], c['losses']) for c in selected) + 3)
    comparison.set_xlim(-limit, limit); comparison.axvline(0, color='#475569', linewidth=.8)
    comparison.set_yticks(y, [DISPLAY[m] for m in refs]); comparison.invert_yaxis()
    comparison.set_xlabel('Paired joint-case change, shared minus reference')
    comparison.set_title('Shared controller versus each strong control', fontsize=12)
    comparison.legend(loc='lower right', fontsize=9)

    counts = np.array([[after_count(stats[r][m], role) for r in REGIMES]
                       for m in PRIMARY for role in (ROLES[2], ROLES[1])])
    preserve.imshow(counts / 96, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    preserve.set_yticks(range(4), ['Local Q: history', 'Local Q: unrelated', 'GDN Q: history', 'GDN Q: unrelated'])
    preserve.set_xticks(range(4), ['Full', 'Oracle', 'Drop evidence', 'Drop other'])
    preserve.set_title('Historical and unrelated facts after updates', fontsize=12)
    for i in range(4):
        for j in range(4):
            preserve.text(j, i, f'{counts[i,j]}/96', ha='center', va='center', fontsize=10,
                          color='white' if counts[i,j] >= 58 else '#111827')
    handles, labels = current.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .919), ncol=2, fontsize=10)
    gates = [metrics['generic_shared_controller']['passed'], metrics['frozen_prefix_sufficiency']['passed'],
             metrics['whole_record_necessity']['passed']]
    outcome = ', '.join(f'{name}: {"pass" if value else "not passed"}'
                        for name, value in zip(('shared', 'oracle recovery', 'record dependence'), gates))
    fig.suptitle('Fresh eight-stream confirmation\nSeparate registered screens: ' + outcome, fontsize=16, y=.985)
    fig.text(.5, .017, '4,704 new readouts; 48 related cases from8new base streams, same generation distribution; frozen weights.\n'
             'Oracle addresses are supplied. Shared generic latents do not implement explicit dependency constraints.\n'
             'Functional evidence access is distinct from recurrent-write damage or architecture benefit; all earlier failed comparisons remain.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .095, 1, .875), h_pad=4, w_pad=4)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    if any(legend.get_window_extent(renderer).overlaps(ax.get_tightbbox(renderer)) for ax in axes.flat):
        raise ValueError('Figure-wide legend overlaps plot content')
    width, height = fig.canvas.get_width_height()
    outside = []
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        box = item.get_window_extent(renderer)
        if box.width > 0 and box.height > 0 and (box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1):
            outside.append({'text': item.get_text(), 'bbox': list(box.bounds)})
    if outside:
        raise ValueError(f'Figure text outside image: {outside}')
    fig.savefig(path, dpi=160); fig.savefig(path.with_suffix('.pdf'))
    write_json(path.with_suffix('.layout.json'), {'status': 'all_visible_text_within_figure_and_legend_disjoint',
        'width': width, 'height': height, 'outside_text': outside, 'source_sha256': streaming_hash(Path(__file__)),
        'png_sha256': streaming_hash(path), 'pdf_sha256': streaming_hash(path.with_suffix('.pdf'))})
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty export table')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def export(audit_path, output, plot):
    report, manifest, registry, rows, metrics = complete_inputs(audit_path)
    output.parent.mkdir(parents=True, exist_ok=True); plot.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or plot.exists() or output.with_suffix('.provenance.json').exists():
        raise ValueError('Preserve previous exports; choose a new output prefix')
    source_sha = streaming_hash(Path(__file__))
    plan = read_plan()
    audit_sha = streaming_hash(audit_path)
    physical = [{'model': m, 'regime': r, 'fresh_observation': True,
        'source_audit_sha256': audit_sha, 'arithmetic_profile': report['arithmetic_profile'],
        'id': row['id'], **readout(row)} for m in MODELS for r in plan['regimes_by_model'][m]
        for row in rows[m][r]]
    _, bindings = selected_inputs(registry)
    expanded = []
    for m in MODELS:
        for r in plan['regimes_by_model'][m]:
            indexed = {v['id']: v for v in physical if v['model'] == m and v['regime'] == r}
            expanded.extend({**b, **indexed[b['physical_id']]} for b in bindings)
    pairs = [{'model': m, 'candidate_regime': r, 'reference_regime': 'full', **p}
        for m in PRIMARY for r in MASKED_REGIMES
        for p in metrics['readout_comparisons_vs_full'][m][r]['individual_pairs']]
    pairs += [{'model': m, 'candidate_regime': 'remove_matched_other', 'reference_regime': 'remove_support', **p}
        for m in PRIMARY for p in metrics['support_removal_effects'][m]['individual_pairs']]
    natural_pairs = [{'candidate': comparison['candidate'], 'reference': comparison['reference'], **p}
        for comparison in metrics['natural_full_comparisons'] for p in comparison['individual_pairs']]
    cells = [{'model': m, 'regime': r, **cell}
        for m in MODELS for r in plan['regimes_by_model'][m]
        for cell in (metrics['natural_full_models'][m]['cells'] if r == 'full'
                     else metrics['readout_models_by_regime'][r][m]['cells'])]
    if ((len(physical), len(expanded), len(pairs), len(cells)) != (4704, 6048, 768, 756) or
        len(natural_pairs) != 48 * len(metrics['natural_full_comparisons']) or
        any(not row['fresh_observation'] for row in physical)):
        raise ValueError('Complete fresh observation/binding/contrast/cell counts changed')
    counts = {'new_physical': len(physical), 'new_full': 2688, 'new_masked': 2016,
        'new_related_bindings': len(expanded), 'readout_current_pairs': len(pairs),
        'natural_joint_case_pairs': len(natural_pairs), 'complete_cells': len(cells),
        'base_streams': 8, 'related_cases': 48, 'models_full': 8,
        'models_with_three_extra_interventions': 2, 'reused_observations': 0, 'cross_profile_rows': 0}
    result = {'status': 'complete_fresh_copy_confirmation_export', 'source_audit': str(audit_path),
        'source_audit_sha256': audit_sha, 'source_manifest_sha256': report['manifest_sha256'],
        'dataset_manifest_sha256': report['dataset_manifest_sha256'],
        'mask_registry_sha256': report['mask_registry_sha256'], 'references': report['references'],
        'arithmetic_profile': report['arithmetic_profile'], 'weights_sha256': report['weights_sha256'],
        'analysis_source_sha256': source_sha, 'counts': counts,
        'independent_novelty_recheck': manifest['independent_novelty_recheck'], 'metrics': metrics,
        'scope': 'New same-distribution histories, not new linguistic templates or training initializations. All8natural controls and every fresh intervention retained. Oracle evidence access, whole-record dependence and shared generic control are separate claims; none establishes irreversible GDN damage or explicit semantic co-design. Previous failed shared/history/repeat primaries remain failed.'}
    write_json(output, result); output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    paths = [output.with_suffix('.csv'), output.with_name(output.stem + '-bindings.csv'),
        output.with_name(output.stem + '-readout-pairs.csv'), output.with_name(output.stem + '-natural-pairs.csv'),
        output.with_name(output.stem + '-cells.csv')]
    for path, table in zip(paths, (physical, expanded, pairs, natural_pairs, cells)):
        write_csv(path, table)
    make_plot(metrics, plot)
    artifacts = [output, output.with_suffix('.source.py'), *paths, plot, plot.with_suffix('.pdf'), plot.with_suffix('.layout.json')]
    write_json(output.with_suffix('.provenance.json'), {'status': 'complete_export_source_and_artifact_closure',
        'source_audit_sha256': audit_sha, 'source_manifest_sha256': report['manifest_sha256'],
        'exporter_sha256': source_sha, 'artifacts_sha256': {str(p): streaming_hash(p) for p in artifacts},
        'run_sources_sha256': manifest['source_sha256'], 'counts': counts,
        'visual_inspection': 'pending separate inspection'})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.audit, args.output, args.plot)
    print(json.dumps({'status': result['status'], 'counts': result['counts'],
        'shared': result['metrics']['generic_shared_controller']['passed'],
        'sufficiency': result['metrics']['frozen_prefix_sufficiency']['passed'],
        'necessity': result['metrics']['whole_record_necessity']['passed']}), flush=True)
