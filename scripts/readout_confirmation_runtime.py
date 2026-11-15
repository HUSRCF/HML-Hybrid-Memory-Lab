"""Normal query-independent history construction and frozen final readout models."""
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from scripts.readout_adapter_modules import ReadoutAdapter
from scripts.readout_adapter_runtime import (base, FixedGDN, normal_readout, normal_groups,
                                             legacy_controls)
from scripts.post_encoding_runtime import assert_isolated
from scripts.run_controller_trial import state_fingerprint
from sas_memory.write_locality_data import streaming_hash


def history_record(guard):
    value = guard.record()
    if len(value['tensors']) != 48:
        raise ValueError('Every hybrid history field must be checked')
    return {k: value[k] for k in ('sha256', 'metadata', 'tensors', 'layouts', 'logits_sha256')}


def compare_history(actual, expected):
    if actual != expected or len(actual['tensors']) != 48:
        raise ValueError('Readout adapter altered a normal frozen-GDN fact history')


@contextmanager
def frozen_model(runtime, name, train_plan, lock):
    fixed = adapter = None
    descriptor = {'name': name, 'new_readout_parameters': 0}
    try:
        if name != 'native':
            fixed = FixedGDN(runtime, train_plan)
            descriptor['fixed_gdn'] = fixed.verify()
        if name in ('query', 'gate', 'query_gate'):
            path = Path(lock['training']['training_run']) / name / 'step-000432/controller.safetensors'
            if streaming_hash(path) != lock['training']['files_sha256'][str(path)]:
                raise ValueError('Use only the audited final432checkpoint')
            adapter = ReadoutAdapter(runtime.model, name, train_plan['arms'][name]['rank'], 15)
            adapter.load_state_dict(load_file(str(path), device=str(runtime.model.device)), strict=True)
            adapter.eval().requires_grad_(False)
            descriptor.update(new_readout_parameters=sum(p.numel() for p in adapter.parameters()),
                              checkpoint_sha256=streaming_hash(path), state_sha256=state_fingerprint(adapter),
                              description=adapter.description())
            if (descriptor['new_readout_parameters'] != 131072
                or descriptor['state_sha256'] != lock['training']['final_states'][name]):
                raise ValueError('Actual frozen readout weight state differs')
        elif name not in ('native', 'fixed_gdn'):
            raise ValueError('Unknown preregistered model')
        yield fixed, adapter, descriptor
        if fixed is not None:
            descriptor['fixed_gdn_after'] = fixed.verify()
        if adapter is not None:
            if (any(p.requires_grad or p.grad is not None for p in adapter.parameters())
                or state_fingerprint(adapter) != descriptor['state_sha256']):
                raise ValueError('Frozen readout weight or gradient changed')
            descriptor['scope'] = dict(adapter.scope)
            descriptor['state_sha256_after'] = state_fingerprint(adapter)
    finally:
        if adapter is not None:
            adapter.close()
        if fixed is not None:
            fixed.close()


def normal_contexts(runtime, adapter, inputs):
    """All nine SET histories exist before questions; COPY uses unique full ledgers."""
    contexts = inputs['set_registry']['contexts']
    for case_id in dict.fromkeys(c['case_id'] for c in contexts):
        group = [c for c in contexts if c['case_id'] == case_id]
        before = group[0]
        if len(group) != 9 or before['episode'] != 'before':
            raise ValueError('Complete normal SET context group required')
        with torch.inference_mode():
            if adapter is not None:
                adapter.set_policy(True, before['prefix_tokens'])
            prefix, logits, count = runtime.advance(before['prefix_text'])
            if count != before['prefix_tokens']:
                raise ValueError('Normal prefix boundary changed')
            parent = base.CacheGuard(prefix, logits)
            children, guards = {}, {}
            for c in group[1:]:
                if adapter is not None:
                    adapter.set_policy(True, c['ledger_tokens'])
                cache, last, _ = base.advance_tail(runtime, prefix, c['prefix_text'], c['tail_text'],
                                                  c['prefix_tokens'], 'none')
                parent.check()
                children[c['id']], guards[c['id']] = cache, base.CacheGuard(cache, last)
            all_caches = [prefix, *children.values()]
            for index, left in enumerate(all_caches):
                for right in all_caches[index + 1:]:
                    assert_isolated(left, right)
        for c in group:
            source, guard = (prefix, parent) if c['episode'] == 'before' else (children[c['id']], guards[c['id']])
            yield 'set', c, source, guard
            guard.check()
            parent.check()
    for c in inputs['copy_contexts']:
        with torch.inference_mode():
            if adapter is not None:
                adapter.set_policy(True, c['ledger_tokens'])
            source, logits, count = runtime.advance(c['ledger'])
            if count != c['ledger_tokens']:
                raise ValueError('COPY full ledger boundary changed')
            guard = base.CacheGuard(source, logits)
        yield 'copy', c, source, guard
        guard.check()


def observed_fit_smoke(runtime, name, adapter, train_plan, training_run, old, old_registry, old_queries):
    """Only30observed inputs across models; never a prospective inference."""
    model_key = 'native' if name == 'native' else 'gdn_all_tokens'
    source = Path(train_plan['source_run']) / f'{model_key}.jsonl'
    if name in ('query', 'gate', 'query_gate'):
        source = Path(training_run) / name / 'final-fit.jsonl'
    reference = {r['id']: r for r in (json.loads(line) for line in source.read_text().splitlines())}
    iterator = normal_groups(runtime, adapter, old_registry, old['models'][model_key], smoke=True)
    try:
        c, cache, guard = next(iterator)
        record = history_record(guard)
        rows = []
        for key in c['query_ids']:
            value = normal_readout(runtime, adapter, c, cache, old_queries[key])
            if value != {k: reference[key][k] for k in value}:
                raise ValueError('Final frozen observed-fit replay differs')
            guard.check()
            rows.append({'id': key, 'model': name, 'task': 'observed_fit', **value})
        if len(rows) != 6:
            raise ValueError('All first-context questions required')
        return rows, {c['id']: record}, {c['id']: guard.checks}
    finally:
        iterator.close()


def measure_context(runtime, adapter, context, source, query):
    return normal_readout(runtime, adapter, context, source, query)
