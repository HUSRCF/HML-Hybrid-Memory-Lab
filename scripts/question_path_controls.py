"""Frozen path-mediation inputs and row-exact oracle activation patches."""

from datetime import datetime
import json
from pathlib import Path

from sas_memory.write_locality_data import streaming_hash
from scripts.fresh_layer_controls import tensor_hash
from scripts.fresh_timing_controls import input_scope as timing_inputs, sources as timing_sources

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/fresh_question_path_mediation_plan.json'
PROTOCOL_SHA256 = 'f2eadecbeb209e95b2dc58f546853b548e6bb8a43e87063438508b8ef3f08146'
PATCHES = {'none': {'gdn_current_outputs': [], 'question_kv': False},
    'gdn_current_outputs': {'gdn_current_outputs': [20, 21, 22], 'question_kv': False},
    'question_kv': {'gdn_current_outputs': [], 'question_kv': True},
    'both': {'gdn_current_outputs': [20, 21, 22], 'question_kv': True}}
CONTEXTS = {'unmasked': {'timing': 'earlier_question_tokens', 'selected_layers': []},
    'earlier_only_layer19': {'timing': 'earlier_question_tokens', 'selected_layers': [19]}}
TARGETS = ('gdn20', 'gdn21', 'gdn22', 'k23', 'v23')


def sources():
    return timing_sources() | {'configs/fresh_question_path_mediation_plan.json',
        'scripts/question_path_controls.py', 'scripts/run_question_path_mediation.py',
        'scripts/analyze_question_path_mediation.py'}


def input_scope():
    scope = timing_inputs(); old = scope['plan']; plan = json.loads(PROTOCOL.read_text())
    if (streaming_hash(PROTOCOL) != PROTOCOL_SHA256 or plan['status'] != 'registered_before_question_path_mediation_inference' or
        plan['patches'] != PATCHES or plan['patch_order'] != list(PATCHES) or
        plan['contexts'] != CONTEXTS or plan['context_order'] != list(CONTEXTS) or
        not all(plan[k] for k in ('posthoc', 'selection_is_posthoc', 'no_accuracy_gate', 'save_full_first_logits')) or
        any(plan[k] for k in ('independent_scientific_confirmation', 'can_change_prior_gates',
            'new_queries', 'new_training', 'new_models', 'new_environment', 'full_generation_scored')) or
        (plan['physical_first_token_readouts'], plan['exact_prior_unpatched_references'],
         plan['unmasked_nonidentity_patch_shams'], plan['intervened_nonidentity_patch_readouts'],
         plan['preserved_runtime_source_count']) != (32, 8, 12, 12, 121) or
        any(plan[k] != old[k] for k in ('model_path', 'packages', 'native_loaded_weights_sha256',
            'arithmetic_profile', 'precision_addendum_sha256', 'dataset', 'dataset_manifest_sha256',
            'mask_registry_sha256', 'models', 'fixed_checkpoints', 'queries', 'attention_layers')) or
        streaming_hash(Path(plan['prior_timing_plan'])) != plan['prior_timing_plan_sha256'] or
        streaming_hash(Path(plan['installed_modeling_source'])) != plan['installed_modeling_source_sha256']):
        raise ValueError('Registered four-case, two-context, four-patch mediation scope changed')
    run, audit_path = Path(plan['prior_timing_run']), Path(plan['prior_timing_audit'])
    manifest = json.loads((run / 'manifest.json').read_text()); audit = json.loads(audit_path.read_text())
    if (manifest['status'] != 'completed' or audit['status'] != 'completed_and_audited_posthoc_token_timing' or
        not audit['operational_controls_passed'] or audit['counts']['physical_first_token_readouts'] != 96 or
        audit['counts']['preceding_question_complete_memory_logits_and_KV_checks'] != 32 or
        audit['source_run'] != str(run) or streaming_hash(run / 'manifest.json') != plan['prior_timing_manifest_sha256'] or
        audit['manifest_sha256'] != plan['prior_timing_manifest_sha256'] or
        streaming_hash(audit_path) != plan['prior_timing_audit_sha256'] or
        streaming_hash(audit_path.with_suffix('.source.py')) != audit['analysis_source_sha256'] or
        audit['original_primary_gates_unchanged'] != scope['primary_gates'] or
        set(manifest['source_sha256']) != timing_sources() or
        not datetime.fromisoformat(manifest['completed_utc']) < datetime.fromisoformat(plan['registered_utc'])):
        raise ValueError('Completed timing audit, source closure or chronology changed')
    rows = []
    for name, digest in manifest['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest or streaming_hash(run / 'sources' / name) != digest:
            raise ValueError('Preserve all121sealed timing sources')
    for name in plan['models']:
        path = run / f'{name}.jsonl'
        if streaming_hash(path) != manifest['models'][name]['results_sha256']:
            raise ValueError('Actual completed timing observations changed')
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    refs = []
    for entry in plan['queries']:
        for context, timing, label in (('unmasked', 'complete_question', 'full'),
                ('earlier_only_layer19', 'earlier_question_tokens', 'only_layer_19')):
            row = next(r for r in rows if (r['model'], r['id'], r['timing'], r['condition']) ==
                (entry['model'], entry['id'], timing, label))
            proof = next(p for p in manifest['models'][entry['model']]['proofs']
                if (p['id'], p['timing'], p['condition']) == (entry['id'], timing, label))
            refs.append({'model': entry['model'], 'id': entry['id'], 'context': context,
                'timing': timing, 'condition': label, 'first_token': row['first_token'],
                **{k: proof[k] for k in ('prefix_sha256', 'full_question_cache_sha256', 'attention23_input_sha256')}})
    if refs != plan['prior_references']:
        raise ValueError('All8registered unpatched references must match completed actual timing rows')
    return {**scope, 'plan': plan, 'timing_manifest': manifest, 'timing_audit': audit}


class PathTrace:
    """Capture original tensors; patch only the registered absolute token rows."""

    def __init__(self, model, query, patch='none', baseline=None):
        if patch not in PATCHES or (patch != 'none' and baseline is None):
            raise ValueError('Registered patch and actual unmasked baseline required')
        self.n, self.h = query['prompt_tokens'], query['ledger_tokens']
        if self.n != self.h + query['question_tokens'] or self.h < 1 or self.n - self.h < 2:
            raise ValueError('Original full prompt and question boundaries required')
        self.patch, self.baseline, self.outputs, self.receipts = patch, baseline, {}, {}
        self.input20, self.input23, self.cos_last, self.sin_last = None, None, None, None
        self.input_calls = {'gdn20_full': 0, 'gdn20_prefix': 0, 'attention23_full': 0, 'attention23_prefix': 0}
        modules = {f'gdn{i}': model.model.layers[i].linear_attn for i in (20, 21, 22)}
        modules.update(k23=model.model.layers[23].self_attn.k_norm, v23=model.model.layers[23].self_attn.v_proj)
        self.handles = []
        self.handles.append(modules['gdn20'].register_forward_pre_hook(self.before20, with_kwargs=True))
        self.handles.append(model.model.layers[23].self_attn.register_forward_pre_hook(self.before23, with_kwargs=True))
        for name, module in modules.items():
            self.receipts[name] = {'full_calls': 0, 'prefix_calls': 0, 'prefix_output_identity_exact': True}
            self.handles.append(module.register_forward_hook(self.after(name)))

    def hidden(self, args, kwargs):
        import torch
        value = kwargs.get('hidden_states')
        if args or not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1 or value.shape[1] not in (self.h, self.n):
            raise ValueError('Original unpadded keyword hidden tensor required')
        return value

    def before20(self, module, args, kwargs):
        value = self.hidden(args, kwargs); full = value.shape[1] == self.n
        self.input_calls['gdn20_full' if full else 'gdn20_prefix'] += 1
        if full: self.input20 = value[:, -1:].detach().clone()

    def before23(self, module, args, kwargs):
        value = self.hidden(args, kwargs); full = value.shape[1] == self.n
        self.input_calls['attention23_full' if full else 'attention23_prefix'] += 1
        if full:
            self.input23 = value.detach().clone()
            cos, sin = kwargs['position_embeddings']
            self.cos_last, self.sin_last = cos[..., -1:, :].detach().clone(), sin[..., -1:, :].detach().clone()

    def after(self, name):
        def output(module, args, value):
            import torch
            if (not isinstance(value, torch.Tensor) or value.ndim != (4 if name == 'k23' else 3) or
                value.shape[0] != 1 or value.shape[1] not in (self.h, self.n)):
                raise ValueError('Actual native GDN/Knorm/V projection tensor layout changed')
            stat = self.receipts[name]
            if value.shape[1] == self.h:
                stat['prefix_calls'] += 1
                return None
            stat['full_calls'] += 1
            start, stop = (self.n - 1, self.n) if name.startswith('gdn') else (self.h, self.n - 1)
            enabled = (int(name[3:]) in PATCHES[self.patch]['gdn_current_outputs']) if name.startswith('gdn') else PATCHES[self.patch]['question_kv']
            reference = self.baseline.outputs[name] if self.baseline is not None else value
            if reference.shape != value.shape or reference.dtype != value.dtype or reference.device != value.device:
                raise ValueError('Actual oracle activation shape/dtype/device differs')
            result = value.clone() if enabled else value
            if enabled: result[:, start:stop] = reference[:, start:stop]
            rows = torch.ones(self.n, dtype=torch.bool, device=value.device); rows[start:stop] = False
            stat.update(patch_enabled=enabled, selected_rows=list(range(start, stop)), shape=list(value.shape), dtype=str(value.dtype),
                target_before_sha256=tensor_hash(value[:, start:stop]), target_after_sha256=tensor_hash(result[:, start:stop]),
                target_reference_sha256=tensor_hash(reference[:, start:stop]),
                target_equals_reference=bool(torch.equal(result[:, start:stop], reference[:, start:stop])),
                untargeted_before_sha256=tensor_hash(value[:, rows]), untargeted_after_sha256=tensor_hash(result[:, rows]),
                untargeted_rows_exact=bool(torch.equal(value[:, rows], result[:, rows])),
                output_object_identity_exact=result is value)
            if not stat['untargeted_rows_exact'] or (enabled and not stat['target_equals_reference']):
                raise ValueError('Oracle patch changed untargeted rows or missed target values')
            self.outputs[name] = result.detach().clone()
            if name == 'k23': self.key_before_last = value[:, -1:].detach().clone()
            if name == 'v23': self.value_before_last = value[:, -1:].detach().clone()
            return result
        return output

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for handle in self.handles: handle.remove()
        self.handles.clear()


def recurrent_hashes(cache):
    from sas_memory.runtime import cache_tensors
    return {str(key): tensor_hash(value) for key, value in cache_tensors(cache)
        if key[1] in ('recurrent_states', 'conv_states')}


def audit_receipts(receipts, calls, query, patch, baseline_receipts, dtype='torch.bfloat16'):
    if set(receipts) != set(TARGETS) or calls != {'gdn20_full': 1, 'gdn20_prefix': 1, 'attention23_full': 1, 'attention23_prefix': 1}:
        raise ValueError('Every target and original-prefix hook must execute exactly once')
    n, h = query['prompt_tokens'], query['ledger_tokens']
    for name, stat in receipts.items():
        selected = [n - 1] if name.startswith('gdn') else list(range(h, n - 1))
        enabled = (int(name[3:]) in PATCHES[patch]['gdn_current_outputs']) if name.startswith('gdn') else PATCHES[patch]['question_kv']
        if (stat['full_calls'] != 1 or stat['prefix_calls'] != 1 or not stat['prefix_output_identity_exact'] or
            stat['patch_enabled'] != enabled or stat['selected_rows'] != selected or stat['shape'][:2] != [1, n] or
            len(stat['shape']) != (4 if name == 'k23' else 3) or stat['dtype'] != dtype or
            not stat['untargeted_rows_exact'] or stat['untargeted_before_sha256'] != stat['untargeted_after_sha256'] or
            stat['target_reference_sha256'] != baseline_receipts[name]['target_after_sha256'] or
            stat['target_equals_reference'] != (stat['target_after_sha256'] == stat['target_reference_sha256']) or
            stat['output_object_identity_exact'] != (not enabled) or
            (enabled and not stat['target_equals_reference']) or
            (not enabled and stat['target_before_sha256'] != stat['target_after_sha256'])):
            raise ValueError('Actual target-row reference, untargeted rows or prefix patch fidelity failed')
