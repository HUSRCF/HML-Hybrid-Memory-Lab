"""Question-only historical KV interventions, without rebuilding the prefix.

This is an oracle diagnostic using dense eager attention, not an efficient sparse
kernel or a learned router. All GDN updates and all prefix-token rows run normally.
"""

from __future__ import annotations

import torch


def evidence_token_keep(tokenizer, prompt, ledger, prelude, event_texts, selected_indices):
    """Keep prelude plus selected *original* event spans at original positions."""
    if ledger != prelude + "".join(event_texts) or not prompt.startswith(ledger):
        raise ValueError("Event spans do not reconstruct the original ledger")
    selected = set(selected_indices)
    if not selected or not selected <= set(range(len(event_texts))):
        raise ValueError("Invalid selected event indices")
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    prefix_ids = tokenizer.encode(ledger, add_special_tokens=False)
    if encoded["input_ids"][:len(prefix_ids)] != prefix_ids:
        raise ValueError("Tokenization crosses ledger/question boundary")
    spans, start = [], len(prelude)
    for index, text in enumerate(event_texts):
        spans.append((start, start + len(text), index))
        start += len(text)
    keep = []
    for left, right in encoded["offset_mapping"][:len(prefix_ids)]:
        overlap = {index for begin, end, index in spans if left < end and right > begin}
        if overlap & selected and overlap - selected:
            raise ValueError("A token crosses selected/unselected event boundaries")
        # Special chat tokens can have zero-width offsets; these are prelude.
        keep.append(left < len(prelude) or left == right or bool(overlap & selected))
    if not any(keep) or all(keep):
        raise ValueError("Expected both retained and suppressed historical tokens")
    return keep


def question_kv_mask(history_keep, past_length, query_length, batch_size, dtype, device):
    """Causal prefix rows; restrict historical columns only for question rows."""
    if past_length < 0 or query_length < 1 or batch_size < 1:
        raise ValueError("Invalid attention dimensions")
    history = torch.as_tensor(history_keep, dtype=torch.bool, device=device)
    if history.ndim != 1 or not len(history):
        raise ValueError("Nonempty one-dimensional history keep mask required")
    boundary = len(history)
    total = past_length + query_length
    queries = torch.arange(past_length, total, device=device)
    keys = torch.arange(total, device=device)
    key_keep = torch.ones(total, dtype=torch.bool, device=device)
    key_keep[:min(total, boundary)] = history[:min(total, boundary)]
    allowed = (keys[None, :] <= queries[:, None]) & (
        (queries[:, None] < boundary) | key_keep[None, :]
    )
    mask = torch.zeros((query_length, total), dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None].expand(batch_size, 1, -1, -1)


class QuestionKVReadout:
    """Install a mask on every model call, including cached answer decoding."""

    def __init__(self, model, history_keep):
        if model.config._attn_implementation != "eager":
            raise ValueError("This diagnostic is validated only for eager attention")
        self.model, self.history_keep = model, list(history_keep)
        if not self.history_keep:
            raise ValueError("Empty ledger boundary")
        self.calls = {"prefill": 0, "cached": 0}
        self.handle = model.register_forward_pre_hook(self._before, with_kwargs=True)

    def _before(self, module, args, kwargs):
        if args or kwargs.get("attention_mask") is not None or kwargs.get("position_ids") is not None:
            raise ValueError("Diagnostic requires unpadded keyword inputs and natural positions")
        ids = kwargs.get("input_ids")
        if ids is None:
            raise ValueError("input_ids are required")
        cache = kwargs.get("past_key_values")
        past = cache.get_seq_length() if cache is not None else 0
        mask = question_kv_mask(self.history_keep, past, ids.shape[1], ids.shape[0],
                                module.get_input_embeddings().weight.dtype, ids.device)
        kwargs["attention_mask"] = {"full_attention": mask, "linear_attention": None}
        self.calls["cached" if past else "prefill"] += 1
        return args, kwargs

    def close(self):
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
