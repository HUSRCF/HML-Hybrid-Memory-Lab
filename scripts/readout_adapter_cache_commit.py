"""Registered copy-on-write commits for disposable teacher caches only."""
from datetime import datetime
import json
from pathlib import Path
import re
from types import MethodType

import torch
from torch.nn import functional as F

from scripts.readout_adapter_runtime import regular_cache_clone, teacher as original_teacher
from scripts.readout_adapter_modules import bits_equal
from sas_memory.runtime import cache_tensors, assert_cache_equal
from sas_memory.write_locality_data import streaming_hash

ROOT = Path(__file__).resolve().parents[1]
ADDENDUM = 'configs/readout_adapter_cache_commit_addendum.json'
ADDENDUM_SHA256 = 'a1ff022ad0d2ed33a6978490aa08a47960398a50758336c3657bd6138dea6d7a'


def copy_on_write_commit(layer, recurrent_states, state_idx=0, **kwargs):
    if kwargs or not layer.is_recurrent_states_initialized[state_idx]:
        raise ValueError('Only initialized native teacher states are supported')
    previous = layer.recurrent_states[state_idx]
    if (torch.is_inference(previous) or torch.is_inference(recurrent_states)
        or previous.shape != recurrent_states.shape or previous.dtype != recurrent_states.dtype
        or previous.device != recurrent_states.device):
        raise ValueError('Teacher commit cannot change state layout, dtype or device')
    # No detach: preserve gradients to the incoming final state. Never overwrite the
    # initial tensor saved by the native chunk autograd function.
    value = recurrent_states.clone(memory_format=torch.preserve_format)
    layer.recurrent_states[state_idx] = value
    return value


def install_teacher_commits(cache):
    recurrent = [layer for layer in cache.layers if hasattr(layer, 'recurrent_states')]
    if len(recurrent) != 18:
        raise ValueError('Exactly18native recurrent teacher layers required')
    for layer in recurrent:
        if 'update_recurrent_state' in vars(layer):
            raise ValueError('Never override an existing instance policy')
        if (set(layer.recurrent_states) != {0} or not layer.is_recurrent_states_initialized[0]
            or torch.is_inference(layer.recurrent_states[0])
            or layer.recurrent_states[0].requires_grad):
            raise ValueError('Install only on fresh regular detached teacher clones')
    for layer in recurrent:
        layer.update_recurrent_state = MethodType(copy_on_write_commit, layer)
    return cache


def corrected_teacher(runtime, adapter, source, example, train):
    if not train:
        return original_teacher(runtime, adapter, source, example, False)
    if len(example['teacher_ids']) != 27 or len(example['answer_ids']) != 4:
        raise ValueError('One native multi-token teacher chunk required')
    if source.get_seq_length() != example['ledger_tokens']:
        raise ValueError('Wrong question-independent history')
    with torch.inference_mode(False), torch.enable_grad():
        working = install_teacher_commits(regular_cache_clone(source))
        if adapter is not None:
            adapter.set_policy(True, example['ledger_tokens'])
        output = runtime.model(input_ids=runtime.tensor([example['teacher_ids']]),
                               past_key_values=working, use_cache=True, logits_to_keep=4)
        labels = runtime.tensor([example['answer_ids']])
        scores = output.logits.float()
        loss = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), labels.reshape(-1))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite teacher loss')
        details = {'id': example['id'], 'teacher_nll_per_token': float(loss.detach().item()),
                   'teacher_all_tokens_correct': bool(torch.equal(scores.detach().argmax(-1), labels)),
                   'teacher_numeric_tokens_correct': int((scores.detach().argmax(-1)[:, :3] == labels[:, :3]).sum().item()),
                   'teacher_newline_correct': bool((scores.detach().argmax(-1)[:, 3] == labels[:, 3]).all().item()),
                   'cached_teacher_tokens': 27, 'cache_fields_cloned': 48}
        return loss, details


def cache_commit_equivalence(runtime, adapter, source, example):
    """Real logits, loss and48final tensors: native no-grad versus corrected grad."""
    from scripts.fresh_layer_controls import tensor_hash
    adapter.set_policy(True, example['ledger_tokens'])
    with torch.inference_mode(False):
        native = regular_cache_clone(source)
        safe = install_teacher_commits(regular_cache_clone(source))
        labels = runtime.tensor([example['answer_ids']])
        ids = runtime.tensor([example['teacher_ids']])
        with torch.no_grad():
            old = runtime.model(input_ids=ids, past_key_values=native, use_cache=True, logits_to_keep=4)
            old_loss = F.cross_entropy(old.logits.float().reshape(-1, old.logits.shape[-1]), labels.reshape(-1))
        with torch.enable_grad():
            new = runtime.model(input_ids=ids, past_key_values=safe, use_cache=True, logits_to_keep=4)
            new_loss = F.cross_entropy(new.logits.float().reshape(-1, new.logits.shape[-1]), labels.reshape(-1))
        if (not new_loss.requires_grad or not bits_equal(old.logits, new.logits)
            or not bits_equal(old_loss.reshape(1), new_loss.reshape(1))):
            raise ValueError('Cache-only correction changed actual native teacher logits/loss')
        assert_cache_equal(native, safe)
        old_fields = dict(cache_tensors(native))
        new_fields = dict(cache_tensors(safe))
        if len(old_fields) != 48 or any(not bits_equal(v, new_fields[k]) for k, v in old_fields.items()):
            raise ValueError('Cache-only correction changed a final cache tensor byte')
        return {'id': example['id'], 'logits_sha256': tensor_hash(old.logits),
                'loss_sha256': tensor_hash(old_loss.reshape(1)), 'all_logits_bit_exact': True,
                'loss_bit_exact': True, 'all48final_cache_tensors_bit_exact': True,
                'native_no_grad_corrected_with_grad': True, 'corrected_loss_requires_grad': True}


def verify_correction_seal():
    path = ROOT / ADDENDUM
    if streaming_hash(path) != ADDENDUM_SHA256:
        raise ValueError('Registered cache correction changed')
    plan = json.loads(path.read_text())
    seal_path = Path(plan['source_seal'])
    seal = json.loads(seal_path.read_text())
    parent_path = Path(plan['parent_seal'])
    parent = json.loads(parent_path.read_text())
    expected = set(parent['source_sha256']) | set(plan['new_sources'])
    if (streaming_hash(parent_path) != plan['parent_seal_sha256']
        or seal['status'] != 'sealed_cache_commit_correction_before_native_smoke'
        or set(seal['source_sha256']) != expected or len(expected) != 282
        or seal['full_tests_passed'] != 364 or seal['addendum_sha256'] != ADDENDUM_SHA256
        or not datetime.fromisoformat(plan['created_utc']) < datetime.fromisoformat(seal['created_utc'])):
        raise ValueError('Complete pre-native282source correction seal required')
    for name, digest in seal['source_sha256'].items():
        if streaming_hash(ROOT / name) != digest:
            raise ValueError('Measured corrected source changed: ' + name)
    for f, key in ((plan['failed_smoke'] + '/manifest.json', 'failed_manifest_sha256'),
                   (plan['failed_log'], 'failed_log_sha256')):
        if streaming_hash(Path(f)) != plan[key]:
            raise ValueError('Original failed native smoke changed')
    if streaming_hash(Path(seal['full_tests_log'])) != seal['full_tests_log_sha256']:
        raise ValueError('Corrected full regression proof changed')
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$', Path(seal['full_tests_log']).read_text())
    failed = json.loads((Path(plan['failed_smoke']) / 'manifest.json').read_text())
    if (not match or int(match[1]) != 364 or failed['status'] != 'failed'
        or 'modified by an inplace operation' not in failed['error']
        or list(Path(plan['failed_smoke']).glob('*/step-*'))):
        raise ValueError('Preserved zero-update failure and complete364test log required')
    return {'seal': str(seal_path), 'seal_sha256': streaming_hash(seal_path),
            'addendum_sha256': ADDENDUM_SHA256, 'source_files': 282,
            'source_sha256': seal['source_sha256'], 'created_utc': seal['created_utc']}
