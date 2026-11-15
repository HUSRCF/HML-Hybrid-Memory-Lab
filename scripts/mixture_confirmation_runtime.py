"""Seven frozen checkpoint contexts and append-only full-history evidence records."""
from contextlib import contextmanager
import json
from pathlib import Path
import re

import torch
from safetensors.torch import load_file
from scripts.mixture_confirmation_data import model_spec, OLD_MODELS, REGIMES
from scripts.readout_confirmation_runtime import (normal_contexts, history_record, compare_history,
    measure_context, frozen_model as old_frozen_model)
from scripts.readout_adapter_runtime import base, FixedGDN, normal_groups, normal_readout, legacy_controls
from scripts.readout_adapter_modules import ReadoutAdapter
from scripts.run_controller_trial import state_fingerprint
from sas_memory.write_locality_data import streaming_hash


@contextmanager
def frozen_model(runtime, name, base_plan, lock):
    if name in OLD_MODELS:
        with old_frozen_model(runtime, name, base_plan, {'training': lock['original_training']}) as value:
            yield value
        return
    spec = model_spec(name, lock)
    fixed = adapter = None
    try:
        fixed = FixedGDN(runtime, base_plan)
        if streaming_hash(Path(spec['checkpoint'])) != spec['checkpoint_sha256']:
            raise ValueError('Only locked final432mixture weights allowed')
        adapter = ReadoutAdapter(runtime.model, 'gate', 32, 15)
        adapter.load_state_dict(load_file(spec['checkpoint'], device=str(runtime.model.device)), strict=True)
        adapter.eval().requires_grad_(False)
        descriptor = {'name': name, 'new_readout_parameters': sum(p.numel() for p in adapter.parameters()),
                      'checkpoint_sha256': spec['checkpoint_sha256'], 'state_sha256': state_fingerprint(adapter),
                      'description': adapter.description(), 'fixed_gdn': fixed.verify()}
        if descriptor['new_readout_parameters'] != 131072 or descriptor['state_sha256'] != spec['state_sha256']:
            raise ValueError('Actual frozen Gate checkpoint differs from training lock')
        yield fixed, adapter, descriptor
        descriptor['fixed_gdn_after'] = fixed.verify()
        descriptor['state_sha256_after'] = state_fingerprint(adapter)
        descriptor['scope'] = dict(adapter.scope)
        if (descriptor['state_sha256_after'] != spec['state_sha256']
            or any(p.requires_grad or p.grad is not None for p in adapter.parameters())):
            raise ValueError('Frozen confirmation weights or gradients changed')
    finally:
        if adapter is not None:
            adapter.close()
        if fixed is not None:
            fixed.close()


def observed_fit_replay(runtime, name, adapter, p, base_plan, old, registry, queries):
    key = 'native' if name == 'native' else 'gdn_all_tokens'
    source = Path(base_plan['source_run']) / (key + '.jsonl')
    if name in ('query', 'gate', 'query_gate'):
        source = Path(p['original_training_run']) / name / 'final-fit.jsonl'
    elif name in REGIMES:
        source = Path(p['mixture_training_run']) / REGIMES[name] / 'final-fit.jsonl'
    reference = {r['id']: r for r in (json.loads(line) for line in source.read_text().splitlines())}
    stream = normal_groups(runtime, adapter, registry, old['models'][key], smoke=True)
    try:
        c, cache, guard = next(stream)
        record = history_record(guard)
        rows = []
        for qid in c['query_ids']:
            value = normal_readout(runtime, adapter, c, cache, queries[qid])
            if value != {field: reference[qid][field] for field in value}:
                raise ValueError('Final frozen model does not reproduce its recorded fit answer')
            guard.check()
            rows.append({'id': qid, 'model': name, 'task': 'observed_fit', **value})
        return rows, c['id'], record, guard.checks
    finally:
        stream.close()


def archive_exact(source, destination, expected_sha256):
    source, destination = Path(source), Path(destination)
    if destination.exists() or streaming_hash(source) != expected_sha256:
        raise ValueError('Never overwrite or alter archived baseline bytes')
    destination.write_bytes(source.read_bytes())
    if streaming_hash(destination) != expected_sha256:
        raise ValueError('Archived raw baseline copy differs')


def valid_history(record):
    return (isinstance(record, dict)
        and set(record) == {'sha256', 'metadata', 'tensors', 'layouts', 'logits_sha256'}
        and all(isinstance(record[k], str) and re.fullmatch(r'[0-9a-f]{64}', record[k])
                for k in ('sha256', 'logits_sha256'))
        and all(isinstance(record[k], dict) for k in ('metadata', 'tensors', 'layouts'))
        and len(record['tensors']) == 48)


def write_history(handle, context_id, record):
    if not valid_history(record):
        raise ValueError('Complete48field normal history record required')
    handle.write(json.dumps({'context_id': context_id, 'record': record}) + '\n')


def read_histories(path):
    result = {}
    for row in (json.loads(line) for line in Path(path).read_text().splitlines()):
        cid, record = row['context_id'], row['record']
        if cid in result or not valid_history(record):
            raise ValueError('Missing fields or duplicated normal history')
        result[cid] = record
    return result
