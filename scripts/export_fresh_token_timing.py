"""Render every timing/layer first-token observation from its audited source run."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_fresh_token_timing import audit_run
from scripts.fresh_timing_controls import input_scope
from scripts.run_controller_trial import write_json


def export(audit_path, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    report = json.loads(audit_path.read_text()); run = Path(report['source_run'])
    if audit_run(run) != report:
        raise ValueError('Independent timing audit or actual source observations changed')
    plan = input_scope()['plan']; conditions = plan['condition_order']; timings = plan['timing_order']
    rows = [json.loads(line) for name in plan['models'] for line in (run / f'{name}.jsonl').read_text().splitlines()]
    indexed = {(r['model'], r['id'], r['timing'], r['condition']): r for r in rows}
    cells, matrices = [], []
    for timing in timings:
        matrix = []
        for index, entry in enumerate(plan['queries']):
            values = []
            for condition in conditions:
                row = indexed[(entry['model'], entry['id'], timing, condition)]
                values.append(row['first_token']['gold_margin'])
                cells.append({'case_index': index + 1, **row})
            matrix.append(values)
        matrices.append(np.asarray(matrix))
    extent = max(max(float(np.abs(m).max()) for m in matrices), 1e-9)
    labels = [f"{i + 1}. {'Local-Q' if e['model'] == 'local_question' else 'GDN-Q'} | {e['direction']} | gold {e['expected_full']['answer'][0]}"
        for i, e in enumerate(plan['queries'])]
    fig, axes = plt.subplots(3, 1, figsize=(13.8, 10.6))
    fig.subplots_adjust(left=.24, right=.88, top=.90, bottom=.13, hspace=.56)
    titles = ('Restrict all question rows', 'Restrict final question token only', 'Restrict earlier question tokens only')
    norm = TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent)
    for panel, (ax, matrix, title) in enumerate(zip(axes, matrices, titles)):
        im = ax.imshow(matrix, cmap='RdBu', norm=norm, aspect='auto')
        ax.set_xticks(range(8), ['Full', 'All', *[f'Only {i}' for i in plan['attention_layers']]])
        ax.set_yticks(range(4), labels); ax.set_title(title, pad=12, fontsize=11)
        for i in range(4):
            for j in range(8):
                row = cells[panel * 32 + i * 8 + j]; value = row['first_token']
                text = value['argmax_text'].replace('\n', '\\n')
                if len(text) > 10: text = text[:7] + '...'
                star = ' *' if value['argmax_id'] == value['gold_id'] else ''
                ax.text(j, i, f"{text}{star}\n{value['gold_margin']:+.3f}",
                    ha='center', va='center', fontsize=9,
                    color='white' if abs(matrix[i, j]) > .55 * extent else 'black')
        ax.axvline(1.5, color='black', linewidth=1)
    axes[-1].set_xlabel('Restricted Attention layers (zero-based); original token IDs and positions')
    cax = fig.add_axes([.91, .25, .017, .53]); colorbar = fig.colorbar(im, cax=cax)
    colorbar.set_ticks(np.linspace(-extent, extent, 5))
    cax.set_ylabel('Gold first-token logit minus strongest alternative')
    fig.suptitle('Question-token timing on four reused control cases', fontsize=14, y=.97)
    fig.text(.24, .065, 'Each cell: first-token argmax text and signed gold margin. * Gold first token; these cells do not score full answers.', fontsize=9)
    fig.text(.24, .038, 'All 96 observations retained, including ties. Posthoc diagnostic; no accuracy gate or architecture validation.', fontsize=9)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); bounds = fig.bbox; checked = 0
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            if not (bounds.x0 <= box.x0 and bounds.y0 <= box.y0 and box.x1 <= bounds.x1 and box.y1 <= bounds.y1):
                raise ValueError(f'Figure text outside canvas: {item.get_text()}')
            checked += 1
    paths = [Path(str(prefix) + ext) for ext in ('.png', '.pdf', '.json', '.layout.json', '.source.py')]
    provenance = Path(str(prefix) + '.provenance.json')
    if any(p.exists() for p in [*paths, provenance]): raise ValueError('Preserve prior exports')
    fig.savefig(paths[0], dpi=180); fig.savefig(paths[1]); plt.close(fig)
    write_json(paths[2], {'posthoc': True, 'independent_scientific_confirmation': False,
        'full_generation_scored': False, 'source_audit': str(audit_path),
        'source_audit_sha256': streaming_hash(audit_path), 'condition_order': conditions,
        'timing_order': timings, 'all96cells': cells, 'gold_first_token_margin_range': [-extent, extent]})
    write_json(paths[3], {'all_text_inside_canvas': True, 'checked_text_objects': checked,
        'physical_cells': 96, 'panels': 3, 'rows_per_panel': 4, 'columns': 8,
        'source_manifest_sha256': report['manifest_sha256']})
    paths[4].write_bytes(Path(__file__).read_bytes())
    write_json(provenance, {'status': 'complete_posthoc_timing_figure',
        'source_audit_sha256': streaming_hash(audit_path), 'source_manifest_sha256': report['manifest_sha256'],
        'all96cells_retained': True, 'artifacts_sha256': {str(p): streaming_hash(p) for p in paths}})
    print(json.dumps({'exported_artifacts': len(paths), 'cells': 96, 'all_text_inside_canvas': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path); parser.add_argument('--prefix', type=Path, required=True)
    args = parser.parse_args(); export(args.audit, args.prefix)
