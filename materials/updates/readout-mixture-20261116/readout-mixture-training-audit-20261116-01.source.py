"""Independent complete exposure, checkpoint, native-safety and observed-answer audit."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.readout_mixture_data import registered_inputs, verify_seal, build_inputs, REGIMES
from scripts.readout_adapter_data import read_plan as base_plan
from scripts.analyze_readout_adapter_training import checkpoint, expected_parameters, verify_scope
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.run_controller_trial import write_json
from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash


def grade(row, example):
    match = re.fullmatch(r'\s*(-?\d+)\s*', row['generated'])
    correct = bool(match and match[1] == example['answer'])
    if (row['id'] != example['id'] or row['answer'] != example['answer']
        or row['greedy_exact'] != correct
        or row['cached_teacher_tokens'] != len(example['teacher_ids'])
        or not math.isfinite(row['teacher_nll_per_token']) or not math.isfinite(row['answer_nll'])):
        raise ValueError('Complete unfiltered raw answer/teacher grading differs')
    return correct


def audit_run(run, allow_smoke=False):
    run = Path(run)
    p, packet, data = registered_inputs()
    seal = verify_seal()
    # Reconstruct actual observed token inputs, truth programs and both schedules.
    if build_inputs(tokenizer_runtime(p['model_path'])) != data:
        raise ValueError('Complete observed registry no longer independently reconstructs')
    m = json.loads((run / 'manifest.json').read_text())
    if (m['status'] != 'completed' or m['smoke'] and not allow_smoke
        or m['source_sha256'] != seal['source_sha256'] or len(m['source_sha256']) != 296
        or m['protocol_sha256'] != packet['protocol_sha256']
        or m['packet_manifest_sha256'] != streaming_hash(Path(p['dataset']) / 'manifest.json')
        or json.loads((run / 'plan.json').read_text()) != p
        or m['regime_order'] != list(REGIMES) or list(m['arms']) != list(REGIMES)
        or m['weights_before'] != m['weights_after']
        or m['weights_before']['sha256'] != p['native_loaded_weights_sha256']
        or not m['weights_before']['all_parameters_frozen'] or not m['weights_before']['no_parameter_gradients']
        or m['packages'] != p['packages'] or not m['hooks_restored']
        or any(m[k] for k in ('oracle', 'donor', 'future_confirmation_generated'))
        or m['training'] == m['smoke']
        or not datetime.fromisoformat(seal['created_utc']) < datetime.fromisoformat(m['started_utc']) < datetime.fromisoformat(m['completed_utc'])):
        raise ValueError('Complete sealed296source paired training run required')
    for name, digest in m['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / 'sources' / name) != digest:
            raise ValueError('Measured mixture source changed')
    bp = base_plan()
    legacy = json.loads((Path(bp['source_packet']) / 'references.json').read_text())
    for field, model in (('native_legacy_before', 'native'), ('native_legacy_after', 'native'),
                         ('fixed_legacy_before', 'gdn_all_tokens'), ('fixed_legacy_after', 'gdn_all_tokens')):
        if ([r['id'] for r in m[field]] != legacy['legacy_ids']
            or any(r['readout'] != legacy['legacy_full'][model][r['id']] for r in m[field])):
            raise ValueError('Every exact existing native/fixed-GDN anchor required')
    reference_path = run / 'normal-history-references.json'
    references = json.loads(reference_path.read_text())
    if streaming_hash(reference_path) != m['normal_reference_sha256'] or set(references) != set(data['contexts']):
        raise ValueError('Complete256history fixed-GDN reference bank required')
    if any(len(v['tensors']) != 48 or len(v['logits_sha256']) != 64 or not v['metadata'] or not v['layouts'] for v in references.values()):
        raise ValueError('Every reference requires all48tensor/layout/metadata and logit fields')
    output = {}
    shapes = expected_parameters('gate', 32)
    for regime in REGIMES:
        rec = m['arms'][regime]
        if (rec['initial_state_sha256'] != p['initial_state_sha256']
            or rec['description']['parameters'] != 131072 or rec['description']['mode'] != 'gate'
            or rec['description']['layer_index'] != 15 or not rec['hooks_restored']
            or rec['fixed_parameters']['state_sha256'] != p['fixed_gdn_state_sha256']
            or not rec['fixed_parameters']['all_frozen_and_gradient_free']):
            raise ValueError('Paired original Gate initialization or frozen state differs')
        log_path = run / regime / 'training.jsonl'
        log_bytes = log_path.read_bytes()
        log = [json.loads(line) for line in log_bytes.splitlines()]
        if m['smoke']:
            s = rec['safety']
            expected_ids = data['set_chunks'][0]['query_ids'] + data['copy_chunks'][0]['query_ids']
            if (s['status'] != 'passed_real_paired_mixture_safety' or s['scratch_optimizer_updates'] != 1
                or s['teacher_lengths_checked'] != [27, 29, 30] or s['second_backward_teacher_tokens'] != 30
                or [r['id'] for r in s['zero_readouts']] != expected_ids
                or any(r['readout'] != r['disabled_readout'] for r in s['zero_readouts'])
                or not s['source_caches_unchanged'] or len(log) != 1
                or s['scratch_state_sha256'] == rec['initial_state_sha256']):
                raise ValueError('Actual paired zero replay and variable native backwards required')
            for table, part in (('first_gradient_norms', 'b'), ('second_gradient_norms', 'a')):
                v = s[table][f'branches.gate.{part}.weight']
                if v is None or not math.isfinite(v) or v <= 0:
                    raise ValueError('Actual native active-branch gradient required')
            for phase in ('before_update', 'after_update'):
                values = s['cache_commit_equivalence'][phase]
                if len(values) != 3 or len({x['id'] for x in values}) != 3:
                    raise ValueError('SET27 and COPY29/30 equivalence required')
                for v in values:
                    if not all(v[k] for k in ('all_logits_bit_exact', 'loss_bit_exact',
                            'all48final_cache_tensors_bit_exact', 'native_no_grad_corrected_with_grad', 'corrected_loss_requires_grad')):
                        raise ValueError('Teacher cache correction changed native forward values')
            checks = s['prefix_after_nonzero_update']
            if {r['context_id'] for r in checks} != {data['set_chunks'][0]['context_id'], data['copy_chunks'][0]['context_id']}:
                raise ValueError('Both task histories must be rebuilt after the nonzero update')
            for r in checks:
                if r['sha256'] != references[r['context_id']]['sha256'] or not r['all48fields_exact']:
                    raise ValueError('Nonzero adapter changed a fact history')
            verify_scope(s['scope'])
            ck = checkpoint(run / regime / 'step-000001', 1, log_bytes, shapes)
            if ck['state_sha256'] != s['scratch_state_sha256'] or ck['controller_sha256'] != s['checkpoint_sha256']:
                raise ValueError('Scratch checkpoint differs')
            output[regime] = {'initial_state_sha256': rec['initial_state_sha256'], 'checkpoint': ck, 'real_native_safety': True}
        else:
            schedule = data['schedules'][regime]
            expected_contexts = {e['context_id'] for e in schedule}
            if ([r['step'] for r in log] != list(range(1, 433))
                or any(len(r['examples']) != 4 or not r['native_and_fixed_gdn_gradient_free']
                       or not math.isfinite(r['gradient_norm']) for r in log)):
                raise ValueError('Complete432actual optimizer steps required')
            details = [e for row in log for e in row['examples']]
            for row, exposure in zip(details, schedule):
                e = data['examples'][exposure['query_id']]
                if (row['exposure_id'] != exposure['exposure_id'] or row['id'] != exposure['query_id']
                    or row['context_id'] != exposure['context_id']
                    or row['cached_teacher_tokens'] != len(e['teacher_ids'])
                    or not math.isfinite(row['teacher_nll_per_token'])):
                    raise ValueError('Missing, altered, reordered or wrong-length exposure')
            if len(details) != 1728 or sum(r['cached_teacher_tokens'] for r in details) != p['training']['teacher_input_tokens'][regime]:
                raise ValueError('Exact teacher-token exposure budget required')
            for stage in ('training', 'final_fit'):
                checks = rec[stage]['history_checks']
                if set(checks) != expected_contexts:
                    raise ValueError('Every training and final normal fact history must be checked')
                for cid, v in checks.items():
                    if v != {'context_id': cid, 'sha256': references[cid]['sha256'], 'all48fields_exact': True}:
                        raise ValueError('A learned readout changed a normally encoded fact history')
            if (set(rec['training']['source_guard_checks']) != expected_contexts
                or any(v < 6 for v in rec['training']['source_guard_checks'].values())
                or rec['training']['ordered_exposures_sha256'] != data_fingerprint([e['exposure_id'] for e in schedule])
                or rec['training']['training_log_sha256'] != streaming_hash(log_path)):
                raise ValueError('All source guards and complete schedule digest required')
            checkpoints = [checkpoint(run / regime / f'step-{s:06d}', s, log_bytes, shapes) for s in (144, 288, 432)]
            if rec['final_state_sha256'] != checkpoints[-1]['state_sha256']:
                raise ValueError('Only final432weights may be evaluated')
            fit_path = run / regime / 'final-fit.jsonl'
            rows = [json.loads(line) for line in fit_path.read_text().splitlines()]
            wanted = {e['query_id'] for e in schedule}
            if (len(rows) != len(wanted) or {r['id'] for r in rows} != wanted
                or streaming_hash(fit_path) != rec['final_fit']['results_sha256']):
                raise ValueError('Every unique physical fit input must be retained exactly once')
            correct = sum(grade(row, data['examples'][row['id']]) for row in rows)
            verify_scope(rec['final_fit']['scope'])
            output[regime] = {'initial_state_sha256': rec['initial_state_sha256'], 'checkpoints': checkpoints,
                              'optimizer_updates': 432, 'exposures': 1728, 'fit_questions': len(rows), 'fit_correct': correct}
    if not m['smoke']:
        gate = m['smoke_gate']
        smoke = json.loads(Path(gate['path']).read_text())
        if streaming_hash(Path(gate['path'])) != gate['sha256'] or smoke != audit_run(Path(smoke['source_run']), True):
            raise ValueError('Prior complete independently audited real smoke required')
    return {'status': 'audited_paired_mixture_smoke' if m['smoke'] else 'completed_and_audited_paired_mixture_training',
            'source_run': str(run), 'manifest_sha256': streaming_hash(run / 'manifest.json'),
            'source_files_verified': 296, 'arms': output,
            'optimizer_updates_audited': 2 if m['smoke'] else 864,
            'physical_observed_readouts_audited': 48 if m['smoke'] else 2400,
            'zero_adapter_replays_audited': 24 if m['smoke'] else 0,
            'disabled_reference_readouts_audited': 24 if m['smoke'] else 0,
            'future_confirmation_generated': False, 'prior_failed_criteria_unchanged': p['prior_failed_criteria_unchanged'],
            'analysis_source_sha256': streaming_hash(Path(__file__)), 'completed_utc': m['completed_utc']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite an independent audit')
    result = audit_run(args.run, args.allow_smoke)
    write_json(args.output, result)
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
