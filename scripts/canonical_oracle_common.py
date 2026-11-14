"""Registered same-arithmetic oracle replication; sealed previous phases stay intact."""

from datetime import datetime
import json
from pathlib import Path

from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.canonical_necessity_common import sources as previous_sources
from scripts.copy_readout_data import registered_inputs as oracle_inputs
from scripts.copy_readout_runtime import readout

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/canonical_oracle_sufficiency_replication_plan.json'
PROTOCOL_SHA256 = '2444974c7a9e55e50cb60e132abd12b971c9c1e4903c5e827558bf6c3fd92c66'
MODELS = ('local_question', 'gdn_question_only')
REGIMES = ('full', 'oracle')
MASKED_REGIMES = ('oracle',)
DESTINATION = Path('/media/990Pro/Code/SAS/datasets/memory_update/copy-readout-sufficiency-v1')
RUNTIME_SOURCES = tuple(sorted(previous_sources() | {
    'configs/canonical_oracle_sufficiency_replication_plan.json',
    'scripts/canonical_oracle_common.py', 'scripts/canonical_oracle_metrics.py',
    'scripts/run_canonical_oracle_replication.py', 'scripts/analyze_canonical_oracle_replication.py',
    'scripts/compare_canonical_oracle_smokes.py',
}))


def sources():
    return set(RUNTIME_SOURCES)


def model_order(reverse=False):
    return list(reversed(MODELS)) if reverse else list(MODELS)


def read_plan():
    plan = json.loads(PROTOCOL.read_text())
    if (streaming_hash(PROTOCOL) != PROTOCOL_SHA256 or plan['models'] != list(MODELS) or
        plan['regimes'] != list(REGIMES) or not plan['local_files_only'] or
        any(plan[k] for k in ('training', 'new_adapters', 'new_models', 'new_task_cases', 'primary_inference_started'))):
        raise ValueError('Registered canonical oracle replication changed')
    return plan


def registered_inputs(path=DESTINATION):
    """Rebuild masks, validate sealed source/raw closures, and load same-profile refs."""
    from scripts.analyze_canonical_copy_necessity import audit_full_rows
    plan = read_plan()
    path = Path(path)
    registered, source_plan, cases, registry, masks, _ = oracle_inputs(path)
    audit_path = Path(plan['completed_necessity_audit'])
    audit = json.loads(audit_path.read_text())
    run = Path(plan['completed_necessity_source_run'])
    manifest = json.loads((run / 'manifest.json').read_text())
    profile = json.loads(Path(plan['precision_addendum']).read_text())
    equivalence = Path(plan['prior_exact_order_smoke_equivalence'])
    if (path != Path(plan['existing_oracle_dataset']) or
        streaming_hash(audit_path) != plan['completed_necessity_audit_sha256'] or
        streaming_hash(audit_path.with_suffix('.source.py')) != audit['analysis_source_sha256'] or
        audit['status'] != 'completed_and_audited_canonical' or audit['new_physical_readouts_audited'] != 8064 or
        not audit['metrics']['whole_record_necessity']['passed'] or
        not audit['metrics']['prior_shared_history_repeat_primaries_remain_failed'] or
        audit['arithmetic_profile'] != plan['arithmetic_profile'] or
        audit['cross_profile_rows_in_primary'] != 0 or
        audit['weights_sha256'] != plan['native_loaded_weights_sha256'] or
        manifest['packages'] != plan['packages'] or manifest['generation'] != plan['generation'] or
        streaming_hash(run / 'manifest.json') != plan['completed_necessity_manifest_sha256'] or
        audit['manifest_sha256'] != plan['completed_necessity_manifest_sha256'] or
        streaming_hash(Path(plan['existing_oracle_protocol'])) != plan['existing_oracle_protocol_sha256'] or
        registered['mask_registry_sha256'] != plan['existing_oracle_mask_registry_sha256'] or
        data_fingerprint(masks) != plan['existing_oracle_mask_registry_sha256'] or
        streaming_hash(Path(plan['precision_addendum'])) != plan['precision_addendum_sha256'] or
        profile['profile'] != plan['arithmetic_profile'] or
        streaming_hash(ROOT / plan['profile_source']) != plan['profile_source_sha256'] or
        streaming_hash(equivalence) != plan['prior_exact_order_smoke_equivalence_sha256'] or
        not datetime.fromisoformat(manifest['completed_utc']) < datetime.fromisoformat(plan['registered_utc'])):
        raise ValueError('Pinned completed canonical evidence, numerical profile or original oracle masks changed')
    for name, digest in manifest['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / 'sources' / name) != digest:
            raise ValueError('Completed canonical live/archived runtime source changed')
    rows = {}
    for model, state in manifest['models'].items():
        for regime, digest in state['results_sha256'].items():
            file = run / f'{model}.{regime}.jsonl'
            if streaming_hash(file) != digest:
                raise ValueError('Completed canonical raw readouts changed')
            if regime == 'full':
                rows[model] = [json.loads(line) for line in file.read_text().splitlines()]
                spec = source_plan['models'][model]
                audit_full_rows(rows[model], registry['physical'], model, spec['token_policy'] if spec else 'none')
                if any(r['arithmetic_profile'] != plan['arithmetic_profile'] or
                       r['reference_policy'] != 'fresh_full_same_canonical_arithmetic_profile' for r in rows[model]):
                    raise ValueError('Canonical reference includes cross-profile observations')
    registered = {**registered, 'references': {**registered['references'],
        'completed_canonical_necessity_audit': str(audit_path),
        'completed_canonical_necessity_audit_sha256': streaming_hash(audit_path),
        'completed_canonical_necessity_manifest_sha256': streaming_hash(run / 'manifest.json'),
        'completed_canonical_necessity_passed': True}}
    return registered, source_plan, cases, registry, masks, rows


def require_exact_full(actual, expected, model, query_id):
    differences = {k: {'expected': expected[k], 'actual': value}
                   for k, value in readout(actual).items() if expected[k] != value}
    if differences:
        raise ValueError(f'Canonical full reference drift: {model},{query_id},{differences}')


def checked_smoke_equivalence(path, dataset_digest):
    from scripts.compare_canonical_oracle_smokes import audit_equivalence
    path = Path(path)
    report = json.loads(path.read_text())
    if (report != audit_equivalence(Path(report['forward_audit']), Path(report['reverse_audit'])) or
        report['dataset_manifest_sha256'] != dataset_digest or
        streaming_hash(path.with_suffix('.source.py')) != report['analysis_source_sha256']):
        raise ValueError('Both independent72-readout oracle smokes and exact order equivalence required')
    return {'path': str(path), 'sha256': streaming_hash(path)}
