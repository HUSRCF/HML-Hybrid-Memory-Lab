"""Question-stage GDN query capacity control; native write paths are untouched.

This is an opt-in diagnostic, separate from the archived GroupController. A
changed query output can affect deeper question-time states; it is not a probe
of an isolated matrix with all subsequent computation held fixed.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class GDNQueryController(nn.Module):
    def __init__(self, model, group_index: int, rank: int):
        super().__init__()
        if rank <= 0 or group_index < 0:
            raise ValueError((group_index, rank))
        if getattr(model, "is_gradient_checkpointing", False):
            raise ValueError("Cached-position hooks require checkpointing to be disabled")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Freeze the backbone before attaching query controls")
        start = 4 * group_index
        if list(model.config.layer_types[start:start + 4]) != ["linear_attention"] * 3 + ["full_attention"]:
            raise ValueError("Selected group is not GDN-GDN-GDN-Attention")
        self.group_index, self.rank, self.mode = group_index, rank, "gdn_query"
        self.encoders, self.outputs = nn.ModuleDict(), nn.ModuleDict()
        self._specs, self._handles = [], []
        self._token_policy, self._token_boundary, self._position_start = "none", 0, 0
        for offset in range(3):
            gdn = model.model.layers[start + offset].linear_attn
            projection = gdn.in_proj_qkv
            if (projection.in_features != model.config.hidden_size or
                projection.out_features != 2 * gdn.key_dim + gdn.value_dim):
                raise ValueError("Unexpected packed GDN query/key/value dimensions")
            name = f"gdn_{offset}_query"
            self.encoders[name] = nn.Linear(projection.in_features, rank, bias=False)
            self.outputs[name] = nn.Linear(rank, gdn.key_dim, bias=False)
            nn.init.zeros_(self.outputs[name].weight)
            self._specs.append((projection, name, gdn.key_dim))
        self._enabled_outputs = set(self.outputs)
        self.to(device=next(model.parameters()).device, dtype=torch.float32)
        self._handles.append(model.register_forward_pre_hook(self._positions, with_kwargs=True))
        for projection, name, query_dim in self._specs:
            self._handles.append(projection.register_forward_hook(self._make_hook(name, query_dim)))

    def _positions(self, module, args, kwargs):
        cache = kwargs.get("past_key_values", kwargs.get("cache"))
        self._position_start = cache.get_seq_length() if cache is not None else 0

    def _make_hook(self, name, query_dim):
        def hook(module, args, output):
            if self._token_policy == "none" or name not in self._enabled_outputs:
                return output
            length = output.shape[-2]
            if self._position_start + length <= self._token_boundary:
                return output
            control = self.encoders[name](args[0].float()) / math.sqrt(self.rank)
            delta = self.outputs[name](control).to(output.dtype)
            positions = torch.arange(self._position_start, self._position_start + length, device=output.device)
            delta = delta * (positions >= self._token_boundary).to(delta.dtype)[None, :, None]
            # The first key_dim channels are queries in the installed Qwen3.5
            # implementation. K/V channels and all other native projections stay
            # byte-for-byte the original projection output at this layer.
            return torch.cat((output[..., :query_dim] + delta, output[..., query_dim:]), dim=-1)
        return hook

    def set_token_policy(self, policy: str, boundary: int = 0):
        if policy not in ("none", "question") or boundary < 0:
            raise ValueError((policy, boundary))
        self._token_policy, self._token_boundary = policy, boundary

    def set_enabled_outputs(self, names):
        names = set(names)
        if not names.issubset(self.outputs):
            raise ValueError(f"Unknown query controls: {names - set(self.outputs)}")
        self._enabled_outputs = names

    def description(self):
        return {"mode": self.mode, "rank": self.rank, "group_index": self.group_index,
                "layers": list(range(4 * self.group_index, 4 * self.group_index + 3)),
                "trainable_parameters": sum(p.numel() for p in self.parameters()),
                "encoder_shapes": {name: list(module.weight.shape) for name, module in self.encoders.items()},
                "output_shapes": {name: list(module.weight.shape) for name, module in self.outputs.items()},
                "packed_projection_shapes": {name: list(projection.weight.shape) for projection, name, _ in self._specs},
                "input": "local projection hidden state", "target": "query channels before convolution only",
                "cache_expansion": False}

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
