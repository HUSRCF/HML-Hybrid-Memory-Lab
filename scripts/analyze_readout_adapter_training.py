"""Independent literal-order, checkpoint, source-cache and fit audit."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import load_file
from scripts.readout_adapter_data import ARMS, registered_inputs, verify_seal
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash
from sas_memory.training_data import data_fingerprint


def expected_parameters(arm, rank):
    names = ('query', 'gate') if arm == 'query_gate' else (arm,)
    return {f'branches.{name}.{part}.weight': shape for name in names
            for part, shape in (('a', [rank, 2048]), ('b', [2048, rank]))}


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def checkpoint(path, step, log_bytes, shapes):
    record = json.loads((path / 'checkpoint.json').read_text())
    weights = path / 'controller.safetensors'
    opt_path = path / 'optimizer.pt'
    prefix = b''.join(log_bytes.splitlines(keepends=True)[:step])
    if (record['step'] != step or record['controller_sha256'] != streaming_hash(weights)
        or record['optimizer_sha256'] != streaming_hash(opt_path)
        or record['training_log_prefix_bytes'] != len(prefix)
        or record['training_log_prefix_sha256'] != hashlib.sha256(prefix).hexdigest()):
        raise ValueError('Checkpoint/log bytes differ')
    tensors = load_file(str(weights), device='cpu')
    if (set(tensors) != set(shapes) or any(list(tensors[n].shape) != shape or tensors[n].dtype != torch.float32
        or not torch.isfinite(tensors[n]).all() for n, shape in shapes.items())
        or sum(t.numel() for t in tensors.values()) != 131072):
        raise ValueError('Declared parameter budget or storage changed')
    saved = torch.load(opt_path, map_location='cpu', weights_only=True)
    state = saved['optimizer']['state']
    groups = saved['optimizer']['param_groups']
    if (len(groups) != 1 or groups[0]['lr'] != .001 or groups[0]['weight_decay'] != 0
        or len(groups[0]['params']) != len(shapes)
        or set(groups[0]['params']) != set(state)):
        raise ValueError('Registered optimizer configuration changed')
    for index, shape in zip(groups[0]['params'], shapes.values()):
        if any(list(state[index][k].shape) != shape or state[index][k].dtype != torch.float32
               for k in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('Optimizer moments do not match the declared parameters')
    if (len(state) != len(shapes) or not saved['rng']['cuda'] or saved['rng']['cpu'].dtype != torch.uint8
        or any(int(s['step'].item()) != step or not torch.isfinite(s['exp_avg']).all()
               or not torch.isfinite(s['exp_avg_sq']).all() for s in state.values())):
        raise ValueError('Complete actual optimizer/RNG state required')
    return {'step': step, 'controller_sha256': record['controller_sha256'],
            'optimizer_sha256': record['optimizer_sha256'], 'state_sha256': state_hash(tensors)}


def verify_scope(scope):
    if (scope['prefix_calls'] < 1 or scope['active_calls'] < 1 or scope['active_rows'] < 27
        or not scope['inactive_halves_bit_preserved'] or not scope['prefix_rows_bit_preserved']
        or any(not math.isfinite(scope[k]) for k in ('max_query_delta', 'max_gate_delta'))):
        raise ValueError('Actual prefix/active projection scope missing')


def audit_run(path, allow_smoke=False):
    path = Path(path)
    p, packet, old, cases, registry, queries, examples, refs = registered_inputs()
    m = json.loads((path / 'manifest.json').read_text())
    seal = verify_seal(packet)
    if (m['status'] != 'completed' or m['smoke'] and not allow_smoke
        or m['source_sha256'] != packet['source_sha256'] or len(m['source_sha256']) != 277
        or m['registration_seal'] != seal or m['protocol_sha256'] != packet['protocol_sha256']
        or m['packet_manifest_sha256'] != streaming_hash(Path(p['registration_destination']) / 'manifest.json')
        or json.loads((path / 'plan.json').read_text()) != p or list(m['arms']) != list(ARMS)
        or not m['hooks_restored'] or m['weights_before'] != m['weights_after']
        or m['weights_before']['sha256'] != p['native_loaded_weights_sha256']
        or not m['weights_before']['all_parameters_frozen'] or not m['weights_before']['no_parameter_gradients']
        or m['packages'] != p['packages'] or m['python'] != old['python']
        or not datetime.fromisoformat(seal['created_utc']) < datetime.fromisoformat(m['started_utc']) < datetime.fromisoformat(m['completed_utc'])):
        raise ValueError('Complete sealed native learned-readout run required')
    for name, digest in m['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(path / 'sources' / name) != digest:
            raise ValueError('Measured training source changed')
    legacy = json.loads((Path(p['source_packet']) / 'references.json').read_text())
    for field, model in (('native_legacy_before', 'native'), ('native_legacy_after', 'native'),
                         ('fixed_legacy_before', 'gdn_all_tokens'), ('fixed_legacy_after', 'gdn_all_tokens')):
        if ([r['id'] for r in m[field]] != legacy['legacy_ids']
            or any(r['readout'] != legacy['legacy_full'][model][r['id']] for r in m[field])):
            raise ValueError('All native/fixed-GDN anchor replays required')
    states = old['models']['gdn_all_tokens']
    contexts = {c['id']: c for c in registry['contexts']}
    source_records = {}
    for cid, c in contexts.items():
        expected = states['prefixes'][c['prefix_key']] if c['episode'] == 'before' else states['sources'][cid]['cache']
        source_records[cid] = {k: expected[k] for k in ('sha256', 'logits_sha256')}
    base_rows = {r['id']: r for r in refs}
    output = {}
    for arm in ARMS:
        rec = m['arms'][arm]
        shapes = expected_parameters(arm, p['arms'][arm]['rank'])
        if (rec['description']['parameters'] != 131072 or rec['description']['mode'] != arm
            or rec['description']['layer_index'] != 15 or not rec['hooks_restored']
            or rec['fixed_parameters']['state_sha256'] != p['fixed_gdn_state_sha256']
            or not rec['fixed_parameters']['all_frozen_and_gradient_free']):
            raise ValueError('Matched scope and frozen-GDN state required')
        if m['smoke']:
            safety = rec['safety']
            if (safety['status'] != 'passed_real_one_update_readout_safety'
                or safety['scratch_optimizer_updates'] != 1 or not safety['second_backward_without_update']
                or [r['id'] for r in safety['zero_normal_readouts']] != contexts[examples[0]['context_id']]['query_ids']
                or any(r['readout'] != {k: base_rows[r['id']][k] for k in r['readout']} for r in safety['zero_normal_readouts'])
                or safety['prefix_after_nonzero_update'] != source_records[examples[0]['context_id']]
                or safety['scratch_state_sha256'] == rec['initial_state_sha256']):
                raise ValueError('Real throwaway update, exact references and prefix identity required')
            for name in p['arms'][arm]['branches']:
                for table, part in (('first_gradient_norms', 'b'), ('second_gradient_norms', 'a')):
                    value = safety[table][f'branches.{name}.{part}.weight']
                    if value is None or not math.isfinite(value) or value <= 0:
                        raise ValueError('Actual active branch gradient missing')
            verify_scope(safety['scope'])
            log = (path / arm / 'training.jsonl').read_bytes()
            ck = checkpoint(path / arm / 'step-000001', 1, log, shapes)
            if ck['state_sha256'] != safety['scratch_state_sha256'] or ck['controller_sha256'] != safety['checkpoint_sha256']:
                raise ValueError('Scratch checkpoint differs')
            output[arm] = {'initial_state_sha256': rec['initial_state_sha256'], 'checkpoint': ck,
                           'new_parameters': 131072, 'native_prefix_identity': True}
        else:
            record = rec['training']
            log = (path / arm / 'training.jsonl').read_bytes()
            steps = [json.loads(line) for line in log.splitlines()]
            seen = [e['id'] for row in steps for e in row['examples']]
            if (len(steps) != 432 or [r['step'] for r in steps] != list(range(1, 433))
                or any(len(r['examples']) != 4 or not r['native_and_fixed_gdn_gradient_free']
                    or not math.isfinite(r['gradient_norm']) or any(v is None or not math.isfinite(v)
                    for v in r['parameter_gradient_norms'].values()) for r in steps)
                or seen != list(queries) or record['ordered_ids_sha256'] != data_fingerprint(seen)
                or record['normal_cache_records'] != source_records
                or record['training_log_sha256'] != hashlib.sha256(log).hexdigest()):
                raise ValueError('One complete fixed-order pass and all normal source caches required')
            checkpoints = [checkpoint(path / arm / f'step-{s:06d}', s, log, shapes) for s in (144, 288, 432)]
            fit_path = path / arm / 'final-fit.jsonl'
            fit = [json.loads(line) for line in fit_path.read_text().splitlines()]
            result = rec['final_fit']
            if (len(fit) != 1728 or [r['id'] for r in fit] != list(queries)
                or result['file_sha256'] != streaming_hash(fit_path)
                or result['normal_cache_records'] != source_records or result['examples'] != 1728
                or rec['final_state_sha256'] != checkpoints[-1]['state_sha256']):
                raise ValueError('All final432fit outputs and unchanged normal histories required')
            for r in fit:
                q = queries[r['id']]
                import re
                match = re.fullmatch(r'\s*(-?\d+)\s*', r['generated'])
                if r['answer'] != q['answer'] or r['greedy_exact'] != bool(match and match[1] == q['answer']) or not math.isfinite(r['teacher_nll_per_token']):
                    raise ValueError('Independent fit grading differs')
            verify_scope(result['scope'])
            output[arm] = {'initial_state_sha256': rec['initial_state_sha256'], 'checkpoints': checkpoints,
                           'new_parameters': 131072, 'fit_correct': sum(r['greedy_exact'] for r in fit),
                           'fit_questions': 1728, 'teacher_nll_mean': sum(r['teacher_nll_per_token'] for r in fit) / 1728}
    if m['arms']['query']['initial_branch_affines']['query'] != m['arms']['gate']['initial_branch_affines']['gate']:
        raise ValueError('Query-only and gate-only matrices must have paired initialization')
    if not m['smoke']:
        gate = m['smoke_gate']
        g = json.loads(Path(gate['path']).read_text())
        if streaming_hash(Path(gate['path'])) != gate['sha256'] or g != audit_run(Path(g['source_run']), True):
            raise ValueError('Original real smoke audit changed')
        baseline = path / 'query/disabled_teacher_baseline.jsonl'
        values = [json.loads(line) for line in baseline.read_text().splitlines()]
        if (len(values) != 1728 or [r['id'] for r in values] != list(queries)
            or streaming_hash(baseline) != m['arms']['query']['training']['disabled_baseline_sha256']):
            raise ValueError('Complete disabled-readout teacher baseline required')
    return {'status': 'audited_readout_adapter_smoke' if m['smoke'] else 'completed_and_audited_readout_adapter_training',
            'source_run': str(path), 'manifest_sha256': streaming_hash(path / 'manifest.json'),
            'packet_manifest_sha256': m['packet_manifest_sha256'], 'source_files_verified': 277,
            'optimizer_updates_audited': 3 if m['smoke'] else 1296, 'checkpoints_audited': 3 if m['smoke'] else 9,
            'arms': output, 'prior_failed_criteria_unchanged': p['prior_failed_criteria_unchanged'],
            'all_data_observed_fit_not_confirmation': True,
            'analysis_source_sha256': streaming_hash(Path(__file__)), 'completed_utc': m['completed_utc']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--allow-smoke', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Never overwrite an audit')
    result = audit_run(a.run, a.allow_smoke)
    write_json(a.output, result)
    a.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
