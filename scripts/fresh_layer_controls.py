"""Isolated layer-specific question-time masks and frozen localization inputs."""

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from sas_memory.kv_readout import question_kv_mask
from sas_memory.write_locality_data import streaming_hash
from scripts.run_fresh_control_replay import input_scope as replay_inputs, sources as replay_sources

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/fresh_control_layer_localization_plan.json'
PROTOCOL_SHA256 = '9cdde60299233fe414332f69d71c45fba8fa13a1a70717dc8bbf5f3333dceb1f'
LAYERS = (3, 7, 11, 15, 19, 23)


def sources():
    return replay_sources() | {'configs/fresh_control_layer_localization_plan.json',
        'scripts/fresh_layer_controls.py', 'scripts/run_fresh_layer_localization.py',
        'scripts/analyze_fresh_layer_localization.py'}


def input_scope():
    scope = replay_inputs(); old = scope['plan']
    plan = json.loads(PROTOCOL.read_text())
    conditions = {'full': [], 'all_layers': list(LAYERS),
        **{f'only_layer_{i}': [i] for i in LAYERS},
        **{f'all_except_layer_{i}': [j for j in LAYERS if j != i] for i in LAYERS}}
    if (streaming_hash(PROTOCOL) != PROTOCOL_SHA256 or plan['status'] != 'registered_before_layer_localization' or
        plan['attention_layers'] != list(LAYERS) or plan['conditions'] != conditions or
        plan['condition_order'] != list(conditions) or plan['physical_scored_readouts'] != 56 or
        not plan['posthoc'] or not plan['no_accuracy_gate'] or any(plan[k] for k in
            ('independent_scientific_confirmation', 'can_change_prior_gates', 'new_training', 'new_models', 'new_environment', 'new_queries')) or
        any(plan[k] != old[k] for k in ('model_path', 'packages', 'native_loaded_weights_sha256',
            'arithmetic_profile', 'precision_addendum_sha256', 'dataset', 'dataset_manifest_sha256',
            'mask_registry_sha256', 'models', 'fixed_checkpoints', 'queries')) or
        streaming_hash(Path(plan['prior_replay_plan'])) != plan['prior_replay_plan_sha256']):
        raise ValueError('Frozen four-query,14condition localization changed')
    audit_path, run = Path(plan['prior_replay_audit']), Path(plan['prior_replay_run'])
    audit = json.loads(audit_path.read_text()); manifest = json.loads((run / 'manifest.json').read_text())
    if (streaming_hash(audit_path) != plan['prior_replay_audit_sha256'] or
        streaming_hash(run / 'manifest.json') != plan['prior_replay_manifest_sha256'] or
        audit['status'] != 'completed_and_audited_posthoc_control_replay' or not audit['operational_replay_passed'] or
        audit['manifest_sha256'] != plan['prior_replay_manifest_sha256'] or audit['source_run'] != str(run) or
        audit['counts']['recorded_readouts'] != 24 or audit['counts']['measure_calls_including_controls'] != 60 or
        audit['original_primary_gates_unchanged'] != scope['primary_gates'] or
        streaming_hash(audit_path.with_suffix('.source.py')) != audit['analysis_source_sha256'] or
        manifest['status'] != 'completed' or set(manifest['source_sha256']) != replay_sources() or
        not datetime.fromisoformat(manifest['completed_utc']) < datetime.fromisoformat(plan['registered_utc'])):
        raise ValueError('Completed exact replay, source closure or chronology changed')
    for name, digest in manifest['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / 'sources' / name) != digest:
            raise ValueError('Preserve all113sealed replay sources')
    for name, state in manifest['models'].items():
        if streaming_hash(run / f'{name}.jsonl') != state['results_sha256']:
            raise ValueError('Completed actual replay rows changed')
    return {**scope, 'plan': plan, 'prior_replay_manifest': manifest, 'prior_replay_audit': audit}


def tensor_hash(tensor):
    import torch
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


class Attention23Input:
    def __init__(self, model, prompt_tokens):
        self.prompt_tokens, self.hidden, self.calls = prompt_tokens, None, 0
        self.handle = model.model.layers[23].self_attn.register_forward_pre_hook(self.before, with_kwargs=True)

    def before(self, module, args, kwargs):
        if args or 'hidden_states' not in kwargs:
            raise ValueError('Native Attention23 keyword hidden states required')
        hidden = kwargs['hidden_states']
        if hidden.shape[-2] == self.prompt_tokens:
            self.hidden = hidden.detach().clone(); self.calls += 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.handle.remove()


class LayerReadout:
    """Leave global model inputs and unselected layer masks untouched."""

    def __init__(self, model, history_keep, selected):
        actual = tuple(i for i, kind in enumerate(model.config.layer_types) if kind == 'full_attention')
        if (model.config._attn_implementation != 'eager' or actual != LAYERS or
            len(set(selected)) != len(selected) or not set(selected) <= set(LAYERS) or
            not history_keep or any(type(v) is not bool for v in history_keep)):
            raise ValueError('Native eager six-layer layout and boolean historical mask required')
        self.model, self.keep, self.selected = model, list(history_keep), tuple(selected)
        self.calls = {'prefill': 0, 'cached': 0}
        self.receipts = {i: {'prefill_calls': 0, 'cached_calls': 0, 'overridden_calls': 0,
            'untouched_calls': 0, 'mask_identity_exact': True} for i in LAYERS}
        self.handles = [model.register_forward_pre_hook(self.prepare, with_kwargs=True)]
        for i in LAYERS:
            self.handles.append(model.model.layers[i].self_attn.register_forward_pre_hook(self.apply(i), with_kwargs=True))

    def prepare(self, module, args, kwargs):
        if args or kwargs.get('attention_mask') is not None or kwargs.get('position_ids') is not None:
            raise ValueError('Unpadded keyword model inputs and natural positions required')
        ids = kwargs.get('input_ids')
        if ids is None:
            raise ValueError('input_ids required')
        cache = kwargs.get('past_key_values')
        self.past = cache.get_seq_length() if cache is not None else 0
        self.mask = question_kv_mask(self.keep, self.past, ids.shape[1], ids.shape[0],
            module.get_input_embeddings().weight.dtype, ids.device)
        self.calls['cached' if self.past else 'prefill'] += 1

    def apply(self, index):
        def before(module, args, kwargs):
            if args or 'attention_mask' not in kwargs:
                raise ValueError('Native Attention keyword mask required')
            original = kwargs['attention_mask']; stat = self.receipts[index]
            stat['cached_calls' if self.past else 'prefill_calls'] += 1
            if index in self.selected:
                kwargs['attention_mask'] = self.mask
                stat['overridden_calls'] += 1
                stat['mask_identity_exact'] &= kwargs['attention_mask'] is self.mask
            else:
                stat['untouched_calls'] += 1
                stat['mask_identity_exact'] &= kwargs['attention_mask'] is original
            return args, kwargs
        return before

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def audit_receipts(calls, receipts, selected, prefill):
    if (calls['prefill'] != prefill or calls['cached'] < (1 if prefill == 1 else 0) or
        set(map(int, receipts)) != set(LAYERS)):
        raise ValueError('Complete actual layer-mask receipt coverage required')
    total = calls['prefill'] + calls['cached']
    for i, stat in receipts.items():
        if (stat['prefill_calls'] != calls['prefill'] or stat['cached_calls'] != calls['cached'] or
            not stat['mask_identity_exact'] or
            stat['overridden_calls'] != (total if int(i) in selected else 0) or
            stat['untouched_calls'] != (0 if int(i) in selected else total)):
            raise ValueError('Selected mask receipt or unselected mask identity failed')


def audit_probabilities(stats, selected, calls):
    if set(map(int, stats)) != set(LAYERS):
        raise ValueError('Actual probabilities in all six layers required')
    for index, stat in stats.items():
        if (stat['prefill_calls'] != 1 or stat['cached_calls'] != calls['cached'] or stat['query_rows'] < 1 or
            any(not math.isfinite(stat[k]) or stat[k] < 0 for k in
                ('max_suppressed_probability', 'max_suppressed_mass', 'max_kept_record_mass')) or
            (int(index) in selected and (stat['max_suppressed_probability'] != 0 or stat['max_suppressed_mass'] != 0))):
            raise ValueError('Actual selected-layer suppression or finite probability observations failed')


def first_token(runtime, logits, query):
    import torch
    scores = logits[0].detach().float()
    gold = query['answer_ids'][0]
    predicted = scores.argmax().item()
    other = scores.clone(); other[gold] = -torch.inf
    alternative = other.argmax().item()
    return {'argmax_id': predicted, 'argmax_text': runtime.tokenizer.decode([predicted], skip_special_tokens=False),
        'gold_id': gold, 'gold_logit': scores[gold].item(), 'non_gold_id': alternative,
        'non_gold_logit': scores[alternative].item(), 'gold_margin': scores[gold].item() - scores[alternative].item(),
        'logits_sha256': tensor_hash(logits)}
