"""Frozen normal caches and gradient-safe multi-token readout training."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from sas_memory.controllers import GroupController
from sas_memory.readout_adapters import ConstantInputFrozenAffines
from sas_memory.runtime import fork_cache, cache_tensors, assert_cache_equal
from sas_memory.write_locality_data import streaming_hash
from scripts import run_natural_load as base
from scripts.post_encoding_runtime import assert_isolated, assert_complete_tensor_inventory
from scripts.run_controller_trial import state_fingerprint
from scripts.copy_readout_runtime import readout


def regular_cache_clone(source):
    """Turn every one of the48source tensors into an isolated regular tensor."""
    with torch.inference_mode(False):
        result = fork_cache(source)
        for (index, field, slot), value in list(cache_tensors(result)):
            cloned = value.detach().clone(memory_format=torch.preserve_format)
            if field in ('keys', 'values'):
                setattr(result.layers[index], field, cloned)
            else:
                getattr(result.layers[index], field)[slot] = cloned
        if len(list(cache_tensors(result))) != 48 or any(torch.is_inference(t) or t.requires_grad for _, t in cache_tensors(result)):
            raise ValueError('A teacher cache contains inference tensors or a prior graph')
        assert_complete_tensor_inventory(result)
        assert_cache_equal(source, result)
        assert_isolated(source, result)
        return result


class FixedGDN:
    def __init__(self, runtime, plan):
        self.runtime, self.plan = runtime, plan
        spec = plan['fixed_gdn']
        if streaming_hash(Path(spec['checkpoint'])) != spec['checkpoint_sha256']:
            raise ValueError('Fixed GDN checkpoint changed')
        self.controller = GroupController(runtime.model, spec['group_index'], spec['rank'],
                                           spec['mode'], adapt_gdn_keys=spec['adapt_gdn_keys'])
        self.controller.load_state_dict(load_file(spec['checkpoint'], device=str(runtime.model.device)), strict=True)
        self.controller.eval().requires_grad_(False)
        self.controller.set_token_policy('all')
        self.before = state_fingerprint(self.controller)
        if self.before != plan['fixed_gdn_state_sha256']:
            self.controller.close()
            raise ValueError('Fixed GDN state differs from the strong reference')
        self.arithmetic = ConstantInputFrozenAffines(self.controller)

    def verify(self):
        if (state_fingerprint(self.controller) != self.before
            or any(p.requires_grad or p.grad is not None for p in self.controller.parameters())
            or any(p.requires_grad or p.grad is not None for p in self.runtime.model.parameters())):
            raise ValueError('A frozen parameter or gradient changed')
        return {'state_sha256': self.before, 'parameters': sum(p.numel() for p in self.controller.parameters()),
                'all_frozen_and_gradient_free': True, 'arithmetic_calls': dict(self.arithmetic.calls)}

    def close(self):
        self.arithmetic.close()
        self.controller.close()


def check_normal_source(guard, expected):
    actual = guard.record()
    for key in ('sha256', 'metadata', 'tensors', 'layouts', 'logits_sha256'):
        if actual[key] != expected[key]:
            raise ValueError('Normally encoded strong-GDN history changed: ' + key)
    if len(actual['tensors']) != 48:
        raise ValueError('Complete hybrid cache inventory required')
    return {k: actual[k] for k in ('sha256', 'logits_sha256')}


def normal_groups(runtime, adapter, registry, old_state, smoke=False):
    """Build all nine normal histories before any question of each case."""
    contexts = registry['contexts']
    case_ids = list(dict.fromkeys(c['case_id'] for c in contexts))
    if smoke:
        case_ids = case_ids[:1]
    for case_id in case_ids:
        group = [c for c in contexts if c['case_id'] == case_id]
        before = group[0]
        if len(group) != 9 or before['episode'] != 'before':
            raise ValueError('All nine original context conditions required')
        with torch.inference_mode():
            if adapter is not None:
                adapter.set_policy(True, before['prefix_tokens'])
            prefix, logits, count = runtime.advance(before['prefix_text'])
            if count != before['prefix_tokens']:
                raise ValueError('Prefix token boundary changed')
            parent = base.CacheGuard(prefix, logits)
            parent.normal_source_record = check_normal_source(parent, old_state['prefixes'][before['prefix_key']])
            children, guards = {}, {}
            for c in group[1:]:
                if adapter is not None:
                    adapter.set_policy(True, c['ledger_tokens'])
                cache, last, _ = base.advance_tail(runtime, prefix, c['prefix_text'], c['tail_text'],
                                                     c['prefix_tokens'], 'none')
                parent.check()
                guard = base.CacheGuard(cache, last)
                guard.normal_source_record = check_normal_source(guard, old_state['sources'][c['id']]['cache'])
                children[c['id']], guards[c['id']] = cache, guard
            actual = [prefix, *children.values()]
            for index, left in enumerate(actual):
                for right in actual[index + 1:]:
                    assert_isolated(left, right)
        for c in group:
            cache, guard = (prefix, parent) if c['episode'] == 'before' else (children[c['id']], guards[c['id']])
            yield c, cache, guard
            parent.check()
            guard.check()


def teacher(runtime, adapter, source, example, train):
    if len(example['teacher_ids']) != 27 or len(example['answer_ids']) != 4:
        raise ValueError('One native multi-token teacher chunk required')
    if source.get_seq_length() != example['ledger_tokens']:
        raise ValueError('Wrong question-independent history')
    with torch.inference_mode(False):
        working = regular_cache_clone(source)
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
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


def normal_readout(runtime, adapter, context, cache, query):
    if adapter is not None:
        adapter.set_policy(True, query['ledger_tokens'])
    with torch.inference_mode():
        return readout(runtime.probe(context['ledger'], cache, query['question'], query['answer'],
                                     [query['answer']], generate=True, prefill='cached', score_batch=False))


def legacy_controls(runtime, plan, model):
    refs = json.loads((Path(plan['source_packet']) / 'references.json').read_text())
    results = []
    with torch.inference_mode():
        for key in refs['legacy_ids']:
            q = refs['legacy_queries'][key]
            value = base.measure(runtime, base.actual_prompt(runtime, q), q)
            if value != refs['legacy_full'][model][key]:
                raise ValueError('Existing legacy anchor changed')
            results.append({'id': key, 'readout': value})
    return results
