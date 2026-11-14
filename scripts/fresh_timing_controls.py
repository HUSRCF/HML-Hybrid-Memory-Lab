"""Frozen timing inputs and actual Attention row-restriction observations."""

from datetime import datetime
import json
import math
from pathlib import Path

from sas_memory.kv_readout import question_kv_mask
from sas_memory.write_locality_data import streaming_hash
from scripts.fresh_layer_controls import (LAYERS, LayerReadout, input_scope as layer_inputs,
    sources as layer_sources)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/fresh_control_token_timing_plan.json'
PROTOCOL_SHA256 = '0e6bc1bd3cd6b619af3ee565b757d6a233827c4a6cd1936dd9bd36c63e7c112b'
TIMINGS = ('complete_question', 'last_question_token', 'earlier_question_tokens')


def sources():
    return layer_sources() | {'configs/fresh_control_token_timing_plan.json',
        'scripts/fresh_timing_controls.py', 'scripts/run_fresh_token_timing.py',
        'scripts/analyze_fresh_token_timing.py'}


def input_scope():
    scope = layer_inputs(); old = scope['plan']; plan = json.loads(PROTOCOL.read_text())
    labels = ['full', 'all_layers', *[f'only_layer_{i}' for i in LAYERS]]
    if (streaming_hash(PROTOCOL) != PROTOCOL_SHA256 or plan['status'] != 'registered_before_token_timing_inference' or
        plan['condition_order'] != labels or plan['conditions'] != {k: old['conditions'][k] for k in labels} or
        plan['timing_order'] != list(TIMINGS) or not plan['posthoc'] or not plan['no_accuracy_gate'] or
        any(plan[k] for k in ('independent_scientific_confirmation', 'can_change_prior_gates',
            'new_queries', 'new_training', 'new_models', 'new_environment', 'full_generation_scored')) or
        (plan['physical_first_token_readouts'], plan['exact_prior_first_token_reference_readouts'],
            plan['new_timing_counterfactual_first_token_readouts'], plan['preserved_runtime_source_count']) != (96, 32, 64, 117) or
        any(plan[k] != old[k] for k in ('model_path', 'packages', 'native_loaded_weights_sha256',
            'arithmetic_profile', 'precision_addendum_sha256', 'dataset', 'dataset_manifest_sha256',
            'mask_registry_sha256', 'models', 'fixed_checkpoints', 'queries', 'attention_layers')) or
        streaming_hash(Path(plan['prior_localization_plan'])) != plan['prior_localization_plan_sha256']):
        raise ValueError('Registered four-case, all-layer, three-timing scope changed')
    run, audit_path = Path(plan['prior_localization_run']), Path(plan['prior_localization_audit'])
    manifest = json.loads((run / 'manifest.json').read_text()); audit = json.loads(audit_path.read_text())
    if (manifest['status'] != 'completed' or audit['status'] != 'completed_and_audited_posthoc_layer_localization' or
        not audit['operational_controls_passed'] or audit['counts']['recorded_readouts'] != 56 or
        audit['counts']['actual_mask_and_complete_prefix_proofs'] != 56 or audit['source_run'] != str(run) or
        streaming_hash(run / 'manifest.json') != plan['prior_localization_manifest_sha256'] or
        audit['manifest_sha256'] != plan['prior_localization_manifest_sha256'] or
        streaming_hash(audit_path) != plan['prior_localization_audit_sha256'] or
        streaming_hash(audit_path.with_suffix('.source.py')) != audit['analysis_source_sha256'] or
        audit['original_primary_gates_unchanged'] != scope['primary_gates'] or
        set(manifest['source_sha256']) != layer_sources() or
        not datetime.fromisoformat(manifest['completed_utc']) < datetime.fromisoformat(plan['registered_utc'])):
        raise ValueError('Completed layer audit, original scientific gates or registration chronology changed')
    for name, digest in manifest['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / 'sources' / name) != digest:
            raise ValueError('Preserve all117sealed localization sources')
    rows = []
    for name in plan['models']:
        path = run / f'{name}.jsonl'
        if streaming_hash(path) != manifest['models'][name]['results_sha256']:
            raise ValueError('Completed actual layer observations changed')
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    refs = []
    for entry in plan['queries']:
        for label in labels:
            row = next(r for r in rows if (r['model'], r['id'], r['condition']) == (entry['model'], entry['id'], label))
            proof = next(p for p in manifest['models'][entry['model']]['proofs']
                if (p['id'], p['condition']) == (entry['id'], label))
            refs.append({'model': entry['model'], 'id': entry['id'], 'condition': label,
                'first_token': row['first_token'], **{k: proof[k] for k in
                    ('prefix_sha256', 'full_question_cache_sha256', 'attention23_input_sha256')}})
    if refs != plan['prior_references']:
        raise ValueError('All32prior first-token, full-question memory and final-input references must match actual rows')
    return {**scope, 'plan': plan, 'localization_manifest': manifest, 'localization_audit': audit}


def active_positions(boundary, question_tokens, timing):
    if type(question_tokens) is not int or question_tokens < 2 or timing not in TIMINGS:
        raise ValueError('At least two original question tokens and a registered timing required')
    stop = boundary + question_tokens
    if timing == 'last_question_token':
        return [stop - 1]
    return list(range(boundary, stop - (timing == 'earlier_question_tokens')))


class TimingReadout(LayerReadout):
    def __init__(self, model, keep, selected, question_tokens, timing):
        self.positions = active_positions(len(keep), question_tokens, timing)
        self.question_tokens, self.timing, self.batches = question_tokens, timing, []
        super().__init__(model, keep, selected)
        for stat in self.receipts.values():
            stat.update(native_causal_mask_exact=True, inactive_rows_exact=True,
                nonhistorical_columns_exact=True)

    def prepare(self, module, args, kwargs):
        import torch
        super().prepare(module, args, kwargs)
        ids = kwargs['input_ids']
        self.natural = question_kv_mask([True] * len(self.keep), self.past, ids.shape[1], ids.shape[0],
            self.mask.dtype, ids.device)
        positions = torch.arange(self.past, self.past + ids.shape[1], device=ids.device)
        self.active = torch.isin(positions, torch.tensor(self.positions, device=ids.device))
        self.mask = torch.where(self.active[None, None, :, None], self.mask, self.natural)
        self.batches.append({'past_length': self.past, 'query_length': ids.shape[1],
            'active_positions': [p for p in self.positions if self.past <= p < self.past + ids.shape[1]]})

    def apply(self, index):
        parent = super().apply(index)

        def before(module, args, kwargs):
            import torch
            original = kwargs.get('attention_mask')
            if (not isinstance(original, torch.Tensor) or original.dtype != self.natural.dtype or
                original.shape != self.natural.shape or not torch.equal(original, self.natural)):
                raise ValueError('Actual native causal Attention mask changed')
            args, kwargs = parent(module, args, kwargs)
            actual, stat = kwargs['attention_mask'], self.receipts[index]
            stat['native_causal_mask_exact'] &= torch.equal(original, self.natural)
            stat['inactive_rows_exact'] &= torch.equal(actual[..., ~self.active, :], original[..., ~self.active, :])
            stat['nonhistorical_columns_exact'] &= torch.equal(actual[..., len(self.keep):], original[..., len(self.keep):])
            if not all(stat[k] for k in ('native_causal_mask_exact', 'inactive_rows_exact', 'nonhistorical_columns_exact')):
                raise ValueError('Timing restriction changed inactive rows or nonhistorical columns')
            return args, kwargs
        return before


class TimingWeights:
    """Observe selected versus other question rows, using actual eager weights."""

    def __init__(self, model, keep, prelude, question_tokens, timing):
        self.keep, self.boundary, self.prelude = list(keep), len(keep), prelude
        self.positions = active_positions(len(keep), question_tokens, timing)
        self.question_tokens = question_tokens
        if (not 0 < prelude < len(keep) or not all(keep[:prelude]) or
            all(keep) or not any(keep[prelude:])):
            raise ValueError('Actual mixed whole-record historical mask required')
        self.stats = {i: {'calls': 0, 'selected_query_rows': 0, 'other_query_rows': 0,
            'selected_max_suppressed_probability': 0., 'selected_max_suppressed_mass': 0.,
            'other_max_suppressed_probability': 0., 'other_max_suppressed_mass': 0.,
            'max_kept_record_mass': 0.} for i in LAYERS}
        self.handles = []
        for i in LAYERS:
            self.handles.append(model.model.layers[i].self_attn.register_forward_hook(self.after(i)))

    def after(self, index):
        def observe(module, args, result):
            import torch
            weights = result[1]; total = self.boundary + self.question_tokens
            if not isinstance(weights, torch.Tensor) or weights.ndim != 4 or weights.shape[-2:] != (total, total):
                raise ValueError('First-token observer requires one complete original-prompt eager forward')
            stat = self.stats[index]; stat['calls'] += 1
            blocked = torch.tensor([j for j, bit in enumerate(self.keep) if not bit], device=weights.device)
            retained = torch.tensor([j for j in range(self.prelude, self.boundary) if self.keep[j]], device=weights.device)
            for name, rows in (('selected', self.positions), ('other',
                    [j for j in range(self.boundary, total) if j not in self.positions])):
                stat['selected_query_rows' if name == 'selected' else 'other_query_rows'] += len(rows)
                if not rows:
                    continue
                values = weights.index_select(-2, torch.tensor(rows, device=weights.device))[..., :self.boundary]
                suppressed = values.index_select(-1, blocked)
                stat[f'{name}_max_suppressed_probability'] = max(stat[f'{name}_max_suppressed_probability'], suppressed.abs().max().item())
                stat[f'{name}_max_suppressed_mass'] = max(stat[f'{name}_max_suppressed_mass'], suppressed.float().sum(-1).abs().max().item())
                stat['max_kept_record_mass'] = max(stat['max_kept_record_mass'], values.index_select(-1, retained).float().sum(-1).abs().max().item())
        return observe

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def audit_operator(calls, receipts, batches, layers, query, timing, lengths):
    h, q = query['ledger_tokens'], query['question_tokens']
    if timing == 'complete_question':
        positions = list(range(h, h + q))
    elif timing == 'last_question_token':
        positions = [h + q - 1]
    elif timing == 'earlier_question_tokens':
        positions = list(range(h, h + q - 1))
    else:
        raise ValueError('Unknown timing')
    expected = [{'past_length': 0, 'query_length': n, 'active_positions': [p for p in positions if p < n]} for n in lengths]
    if calls != {'prefill': len(lengths), 'cached': 0} or batches != expected or set(map(int, receipts)) != set(LAYERS):
        raise ValueError('Exact first-token forward lengths and absolute active rows required')
    for i, stat in receipts.items():
        if (stat['prefill_calls'] != len(lengths) or stat['cached_calls'] != 0 or
            stat['overridden_calls'] != (len(lengths) if int(i) in layers else 0) or
            stat['untouched_calls'] != (0 if int(i) in layers else len(lengths)) or
            not all(stat[k] for k in ('mask_identity_exact', 'native_causal_mask_exact',
                'inactive_rows_exact', 'nonhistorical_columns_exact'))):
            raise ValueError('Actual row-mask receipt, natural inactive rows or unselected mask identity failed')


def audit_weights(stats, layers, query, timing):
    q = query['question_tokens']
    expected = q if timing == 'complete_question' else 1 if timing == 'last_question_token' else q - 1
    if set(map(int, stats)) != set(LAYERS):
        raise ValueError('All six actual Attention probability observations required')
    fields = ('selected_max_suppressed_probability', 'selected_max_suppressed_mass',
        'other_max_suppressed_probability', 'other_max_suppressed_mass', 'max_kept_record_mass')
    for i, stat in stats.items():
        if (stat['calls'] != 1 or stat['selected_query_rows'] != expected or stat['other_query_rows'] != q - expected or
            any(not math.isfinite(stat[k]) or stat[k] < 0 for k in fields) or
            (int(i) in layers and (stat['selected_max_suppressed_probability'] != 0 or stat['selected_max_suppressed_mass'] != 0))):
            raise ValueError('Selected-row actual suppression or full probability-row coverage failed')
