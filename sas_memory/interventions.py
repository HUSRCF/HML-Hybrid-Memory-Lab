"""Matched-position cache transplants, used only as counterfactual diagnostics."""

from __future__ import annotations

import copy

import torch

from sas_memory.runtime import fork_cache, state_items


def transplant(recipient, donor, layer_types: list[str], groups: list[int], path: str):
    """Replace selected groups' complete recurrent path, KV path, or both.

    This intentionally creates an unnatural mixed cache. It is not a model
    improvement, a proof of semantic storage, or a natural-generation baseline.
    Require equal KV positions and shapes, and include convolution state whenever
    replacing a recurrent state. No token is deleted, shifted, or recomputed.
    """
    if path not in ("recurrent", "kv", "both"):
        raise ValueError(path)
    if len(layer_types) % 4 or any(layer_types[i:i + 4] != ["linear_attention"] * 3 + ["full_attention"]
                                   for i in range(0, len(layer_types), 4)):
        raise ValueError("This experiment requires explicit 3:1 groups")
    if recipient.get_seq_length() != donor.get_seq_length():
        raise ValueError("Transplants require matched sequence lengths")
    if len(set(groups)) != len(groups) or any(g < 0 or g >= len(layer_types) // 4 for g in groups):
        raise ValueError("Invalid or repeated group")
    result = fork_cache(recipient)
    for group in groups:
        for index in range(4 * group, 4 * group + 4):
            target, source = result.layers[index], donor.layers[index]
            is_recurrent = layer_types[index] == "linear_attention"
            if is_recurrent and path in ("recurrent", "both"):
                for field in ("conv_states", "recurrent_states"):
                    dst, src = getattr(target, field), getattr(source, field)
                    if dict(state_items(dst)).keys() != dict(state_items(src)).keys():
                        raise ValueError("Incompatible recurrent state structure")
                    for key, tensor in state_items(src):
                        if tensor is None or dst[key] is None or tensor.shape != dst[key].shape:
                            raise ValueError("Incompatible recurrent state shape")
                        dst[key] = tensor.clone()
                for field in ("is_conv_states_initialized", "is_recurrent_states_initialized", "has_previous_state", "conv_kernel_size"):
                    setattr(target, field, copy.deepcopy(getattr(source, field)))
            elif not is_recurrent and path in ("kv", "both"):
                if target.keys.shape != source.keys.shape or target.values.shape != source.values.shape:
                    raise ValueError("Incompatible KV shape")
                target.keys, target.values = source.keys.clone(), source.values.clone()
    return result
