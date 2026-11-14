"""Posthoc descriptive control changes; no new inference or primary-gate revision."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.fresh_copy_data import PRIMARY, registered_inputs, selected_inputs
from scripts.copy_readout_runtime import readout
from scripts.run_controller_trial import write_json

AUDIT = Path('/media/990Pro/Code/SAS/artifacts/memory_update/fresh-copy-confirmation-audit-20261114-01.json')
AUDIT_SHA256 = 'd28f7568ed6d1d6ddf8314e694c12f11b87b3a7544adeec27f82ce706268d2e9'


def analyze(output):
    if output.exists() or streaming_hash(AUDIT) != AUDIT_SHA256:
        raise ValueError('Preserve reports and require the completed fresh audit')
    audit = json.loads(AUDIT.read_text())
    if audit['status'] != 'completed_and_audited_fresh_copy':
        raise ValueError('Completed fresh confirmation required')
    run = Path(audit['source_run'])
    manifest = json.loads((run / 'manifest.json').read_text())
    if streaming_hash(run / 'manifest.json') != audit['manifest_sha256']:
        raise ValueError('Completed run manifest changed')
    _, _, cases, registry, masks = registered_inputs(Path(manifest['dataset']))
    _, bindings = selected_inputs(registry)
    changes = []
    for model in PRIMARY:
        indexed = {}
        for regime in ('full', 'oracle', 'remove_matched_other'):
            path = run / f'{model}.{regime}.jsonl'
            if streaming_hash(path) != manifest['models'][model]['results_sha256'][regime]:
                raise ValueError('Audited observations changed')
            indexed[regime] = {r['id']: r for r in map(json.loads, path.read_text().splitlines())}
        for regime in ('oracle', 'remove_matched_other'):
            for key, row in indexed[regime].items():
                full = indexed['full'][key]
                if full['greedy_exact'] == row['greedy_exact']:
                    continue
                linked = [b for b in bindings if b['physical_id'] == key]
                # Oracle's intended current-target repair is a separate primary
                # statistic. This report covers non-target preservation and all
                # changes under the purported irrelevant-record control.
                if regime == 'oracle' and all(b['role'] == 'target_current' for b in linked):
                    continue
                changes.append({'model': model, 'regime': regime, 'id': key,
                    'direction': 'harm' if full['greedy_exact'] else 'repair', 'bindings': linked,
                    'full': readout(full), 'intervened': readout(row), 'mask_registration': masks[key]})
    counts = {direction: sum(r['direction'] == direction for r in changes) for direction in ('harm', 'repair')}
    if counts != {'harm': 3, 'repair': 1}:
        raise ValueError('Preserve all observed control flips, including the matched-control repair')
    result = {'status': 'posthoc_descriptive_control_changes_no_new_inference',
        'audit': str(AUDIT), 'audit_sha256': AUDIT_SHA256, 'manifest_sha256': audit['manifest_sha256'],
        'dataset_manifest_sha256': audit['dataset_manifest_sha256'],
        'analysis_source_sha256': streaming_hash(Path(__file__)), 'counts': counts, 'changes': changes,
        'selection_rule': 'Every exact-answer correctness flip for oracle non-target roles or matched-other removal in both preselected models. All96current-target oracle repairs remain in the primary export; none is relabeled here.',
        'interpretation': 'Posthoc descriptive localization, not fresh scientific confirmation. Aggregate non-decrease can conceal paired harms and repairs. Original primary gates remain failed; unchanged prefix memory does not establish that stored information was absent or intact.'}
    write_json(output, result)
    output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status': result['status'], 'counts': counts,
        'output': str(output), 'sha256': streaming_hash(output)}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    analyze(parser.parse_args().output)
