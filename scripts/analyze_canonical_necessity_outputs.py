"""Posthoc output-format description; never changes the registered necessity gate."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.canonical_necessity_common import REGIMES
from scripts.copy_necessity_data import MODELS
from scripts.run_controller_trial import write_json

BASE = Path('/media/990Pro/Code/SAS/artifacts/memory_update/canonical-copy-necessity-comparison-20261114-01.json')
OUTPUT = BASE.with_name('canonical-copy-necessity-output-format-20261114-01.json')


def main():
    if OUTPUT.exists():
        raise ValueError('Preserve existing descriptive reports')
    export = json.loads(BASE.read_text())
    provenance = json.loads(BASE.with_suffix('.provenance.json').read_text())
    for name, digest in provenance['artifacts_sha256'].items():
        if streaming_hash(Path(name)) != digest:
            raise ValueError('Audited export artifact changed before descriptive analysis')
    with BASE.with_suffix('.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 8064 or export['counts']['reused_full_physical_in_primary'] != 0:
        raise ValueError('Complete same-profile8064new observations required')
    stats = {}
    for regime in REGIMES:
        stats[regime] = {}
        for model in MODELS:
            group = [r for r in rows if r['model'] == model and r['regime'] == regime]
            if len(group) != 336 or len({r['id'] for r in group}) != 336:
                raise ValueError('Complete physical model/regime coverage required')
            stats[regime][model] = {'physical_observations': len(group),
                'greedy_correct': sum(r['greedy_exact'] == 'True' for r in group),
                'integer_output': sum(bool(re.fullmatch(r'\s*-?\d+\s*', r['generated'])) for r in group),
                'unique_generated_strings': len({r['generated'] for r in group})}
    result = {'status': 'posthoc_descriptive_output_format_not_gate',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_export': str(BASE), 'source_export_sha256': streaming_hash(BASE),
        'source_physical_csv_sha256': streaming_hash(BASE.with_suffix('.csv')),
        'source_audit_sha256': export['source_audit_sha256'],
        'analysis_source_sha256': streaming_hash(Path(__file__)), 'stats': stats,
        'interpretation': 'Integer output is a format count, not answer correctness or information absence. These descriptive counts do not change the registered scientific decision.'}
    write_json(OUTPUT, result)
    OUTPUT.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
