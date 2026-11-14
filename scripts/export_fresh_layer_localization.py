"""Export all56layer-localization cells and their fixed first-token margins."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_fresh_layer_localization import audit_run
from scripts.fresh_layer_controls import input_scope
from scripts.run_controller_trial import write_json


def export(audit_path, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    report = json.loads(audit_path.read_text()); run = Path(report['source_run'])
    if audit_run(run) != report:
        raise ValueError('Independent layer audit or actual source rows changed')
    plan = input_scope()['plan']; order = plan['condition_order']
    rows = [json.loads(line) for name in plan['models'] for line in (run / f'{name}.jsonl').read_text().splitlines()]
    indexed = {(r['model'], r['id'], r['condition']): r for r in rows}
    matrix, labels, cells = [], [], []
    for index, entry in enumerate(plan['queries']):
        values = []
        model = 'Local-Q' if entry['model'] == 'local_question' else 'GDN-Q'
        labels.append(f"{index + 1}. {model} | {entry['direction']} | gold {entry['expected_full']['answer']}")
        for condition in order:
            row = indexed[(entry['model'], entry['id'], condition)]
            values.append(row['first_token']['gold_margin'])
            cells.append({'case_index': index + 1, **row})
        matrix.append(values)
    matrix = np.asarray(matrix)
    extent = max(float(np.abs(matrix).max()), 1e-9)
    fig, ax = plt.subplots(figsize=(16, 5.4))
    fig.subplots_adjust(left=.22, right=.90, bottom=.29, top=.82)
    im = ax.imshow(matrix, cmap='RdBu',
        norm=TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent), aspect='auto')
    ax.set_xticks(range(14), ['Full', 'All', *[f'Only {i}' for i in plan['attention_layers']],
        *[f'Except {i}' for i in plan['attention_layers']]], rotation=45, ha='right')
    ax.set_yticks(range(4), labels)
    ax.set_title('All registered layer interventions; cell text = greedy answer', pad=22)
    ax.set_xlabel('Restricted Attention layers (zero-based); original question and history positions')
    for i in range(4):
        for j in range(14):
            row = cells[i * 14 + j]
            label = row['generated'].strip().replace('\n', ' / ')
            if len(label) > 13:
                label = label[:10] + '...'
            color = 'white' if abs(matrix[i, j]) > .55 * extent else 'black'
            ax.text(j, i, label + (' *' if row['greedy_exact'] else ''),
                ha='center', va='center', fontsize=9, color=color)
    for edge in (1.5, 7.5):
        ax.axvline(edge, color='black', linewidth=1)
    cbax = fig.add_axes([.92, .30, .012, .48])
    colorbar = fig.colorbar(im, cax=cbax)
    colorbar.set_ticks(np.linspace(-extent, extent, 5))
    cbax.set_ylabel('Gold first-token logit minus strongest alternative')
    fig.suptitle('Question-time readout localization on four reused control flips', fontsize=14, y=.96)
    fig.text(.22, .055,
        '* Exact gold answer. All56conditions retained; no accuracy gate. Posthoc cases are not independent confirmation.',
        fontsize=9)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); bounds = fig.bbox
    checked = 0
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            if not (bounds.x0 <= box.x0 and bounds.y0 <= box.y0 and box.x1 <= bounds.x1 and box.y1 <= bounds.y1):
                raise ValueError(f'Figure text falls outside canvas: {item.get_text()}')
            checked += 1
    paths = [Path(str(prefix) + ext) for ext in ('.png', '.pdf', '.json', '.layout.json', '.source.py')]
    provenance = Path(str(prefix) + '.provenance.json')
    if any(p.exists() for p in [*paths, provenance]):
        raise ValueError('Preserve all previous figure exports')
    fig.savefig(paths[0], dpi=180); fig.savefig(paths[1]); plt.close(fig)
    write_json(paths[2], {'posthoc': True, 'independent_scientific_confirmation': False,
        'source_audit': str(audit_path), 'source_audit_sha256': streaming_hash(audit_path),
        'conditions': order, 'all_cells': cells, 'gold_first_token_margin_range': [-extent, extent]})
    write_json(paths[3], {'all_text_inside_canvas': True, 'checked_text_objects': checked,
        'physical_cells': len(cells), 'rows': 4, 'columns': 14,
        'source_manifest_sha256': report['manifest_sha256']})
    paths[4].write_bytes(Path(__file__).read_bytes())
    write_json(provenance, {'status': 'complete_posthoc_layer_figure',
        'source_audit_sha256': streaming_hash(audit_path),
        'source_manifest_sha256': report['manifest_sha256'], 'all56cells_retained': True,
        'artifacts_sha256': {str(p): streaming_hash(p) for p in paths}})
    print(json.dumps({'exported_artifacts': len(paths), 'cells': len(cells),
        'all_text_inside_canvas': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path); parser.add_argument('--prefix', type=Path, required=True)
    args = parser.parse_args(); export(args.audit, args.prefix)
