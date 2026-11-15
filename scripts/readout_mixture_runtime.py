"""Variable-length native teacher and normal fact histories for observed mixture inputs."""
import torch
from torch.nn import functional as F
from scripts.readout_adapter_runtime import regular_cache_clone, normal_groups, check_normal_source, base
from scripts.readout_adapter_cache_commit import install_teacher_commits
from scripts.readout_confirmation_runtime import history_record


def teacher(runtime, adapter, source, example, train):
    known, labels = example['known_ids'], example['answer_ids']
    if (len(known) not in (24, 26, 27) or len(labels) != 4
        or example['teacher_ids'] != known + labels[:-1]
        or source.get_seq_length() != example['ledger_tokens']):
        raise ValueError('Exact known question, four answer targets and cache boundary required')
    with torch.inference_mode(False), torch.enable_grad() if train else torch.no_grad():
        working = regular_cache_clone(source)
        if train:
            install_teacher_commits(working)
        if adapter is not None:
            adapter.set_policy(True, example['ledger_tokens'])
        output = runtime.model(input_ids=runtime.tensor([example['teacher_ids']]),
                               past_key_values=working, use_cache=True, logits_to_keep=4)
        target = runtime.tensor([labels])
        scores = output.logits.float()
        if scores.shape[:2] != (1, 4):
            raise ValueError('Only four aligned answer logits are supervised')
        loss = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite variable teacher loss')
        predicted = scores.detach().argmax(-1)
        return loss, {'id': example['id'], 'task': example['task'],
                      'teacher_nll_per_token': float(loss.detach().item()),
                      'teacher_all_tokens_correct': bool(torch.equal(predicted, target)),
                      'teacher_numeric_tokens_correct': int((predicted[:, :3] == target[:, :3]).sum().item()),
                      'teacher_newline_correct': bool((predicted[:, 3] == target[:, 3]).all().item()),
                      'cached_teacher_tokens': len(example['teacher_ids']), 'cache_fields_cloned': 48}


def copy_history(runtime, adapter, context):
    if adapter is not None:
        adapter.set_policy(True, context['ledger_tokens'])
    with torch.inference_mode():
        cache, logits, count = runtime.advance(context['ledger'])
    if count != context['ledger_tokens']:
        raise ValueError('Literal normal COPY history length changed')
    return cache, base.CacheGuard(cache, logits)


def context_stream(runtime, adapter, data, old_state):
    """SET uses unchanged continuation policy; each distinct COPY ledger one fullprefill."""
    for c, cache, guard in normal_groups(runtime, adapter, data['set_registry'], old_state):
        yield c, cache, guard
    for c in data['contexts'].values():
        if c['task'] == 'copy':
            cache, guard = copy_history(runtime, adapter, c)
            yield c, cache, guard
            guard.check()


def normal_reference(runtime, data, old_state):
    records = {}
    for c, _, guard in context_stream(runtime, None, data, old_state):
        records[c['id']] = history_record(guard)
        guard.check()
    if len(records) != 256:
        raise ValueError('All256normal fixed-GDN histories required')
    return records


def check_reference(context, guard, records):
    actual = history_record(guard)
    if actual != records[context['id']]:
        raise ValueError('Learned readout changed fact-prefix values,layout,metadata orlogits')
    return {'context_id': context['id'], 'sha256': actual['sha256'], 'all48fields_exact': True}
