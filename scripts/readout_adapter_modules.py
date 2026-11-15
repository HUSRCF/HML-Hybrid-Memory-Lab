"""Causal local Query/pre-sigmoid-gate controls on a frozen attention layer."""
from __future__ import annotations

import math
import weakref

import torch
from torch import nn
from torch.nn import functional as F


class StableTrainableAffine(nn.Linear):
    """The declared FP64 accumulation / FP32 parameter and output profile."""

    def forward(self, x):
        if x.dtype != torch.float32 or self.weight.dtype != torch.float32 or self.bias is not None:
            raise ValueError("Bias-free FP32 storage and inputs required")
        return F.linear(x.double(), self.weight.double()).float()


def bits_equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.detach().contiguous().view(torch.uint8),
                            right.detach().contiguous().view(torch.uint8)))


class ReadoutAdapter(nn.Module):
    """Only position boundaries enter policy; no answers or program labels."""

    MODES = {"query": ("query",), "gate": ("gate",), "query_gate": ("query", "gate")}
    PROFILE = "trainable_readout_affine_fp64_accumulate_round_fp32_v1"

    def __init__(self, model, mode, rank, layer_index=15):
        super().__init__()
        if mode not in self.MODES or rank <= 0:
            raise ValueError("Known control and positive rank required")
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("Freeze the native model first")
        if getattr(model, "is_gradient_checkpointing", False):
            raise ValueError("Checkpoint replay is outside this hook contract")
        attention = model.model.layers[layer_index].self_attn
        projection = attention.q_proj
        self.head_dim, self.hidden_size = attention.head_dim, projection.in_features
        self.query_width = projection.out_features // 2
        if projection.out_features % (2 * self.head_dim):
            raise ValueError("Expected interleaved query/gate heads")
        self.mode, self.rank, self.layer_index = mode, rank, layer_index
        self.branches = nn.ModuleDict()
        for name in self.MODES[mode]:
            self.branches[name] = nn.ModuleDict({
                "a": StableTrainableAffine(self.hidden_size, rank, bias=False),
                "b": StableTrainableAffine(rank, self.query_width, bias=False),
            })
            nn.init.zeros_(self.branches[name]["b"].weight)
        self.to(device=projection.weight.device, dtype=torch.float32)
        self._boundary, self._enabled, self._start, self._length = 0, False, None, None
        self._model_ref = weakref.ref(model)
        self.handles = []
        self.reset_scope()
        self.handles.append(model.register_forward_pre_hook(self._before, with_kwargs=True))
        self.handles.append(projection.register_forward_hook(self._after))

    def reset_scope(self):
        self.scope = {"prefix_calls": 0, "disabled_calls": 0, "active_calls": 0,
                      "active_rows": 0, "prefix_rows_preserved": 0,
                      "inactive_halves_bit_preserved": True, "prefix_rows_bit_preserved": True,
                      "max_query_delta": 0., "max_gate_delta": 0.}

    def set_policy(self, enabled, boundary):
        if type(enabled) is not bool or type(boundary) is not int or boundary < 0:
            raise ValueError("Explicit enabled flag and known nonnegative boundary required")
        self._enabled, self._boundary = enabled, boundary

    def _before(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Single-example known token input required")
        cache = kwargs.get("past_key_values")
        self._start = cache.get_seq_length() if cache is not None else 0
        self._length = ids.shape[1]

    def _after(self, module, args, output):
        if output.shape[:-1] != (1, self._length) or output.shape[-1] != self.query_width * 2:
            raise ValueError("Projection must match this forward's known positions")
        if not self._enabled:
            self.scope["disabled_calls"] += 1
            return output
        positions = torch.arange(self._start, self._start + self._length, device=output.device)
        active = positions >= self._boundary
        count = int(active.sum().item())
        if count == 0:
            self.scope["prefix_calls"] += 1
            self.scope["prefix_rows_preserved"] += self._length
            return output
        if args[0].shape != (1, self._length, self.hidden_size) or args[0].requires_grad:
            raise ValueError("One local adapter layer requires frozen upstream input")
        original = output.reshape(1, self._length, -1, 2 * self.head_dim)
        query, gate = original.split(self.head_dim, dim=-1)
        parts = {"query": query, "gate": gate}
        mask = active[None, :, None, None]
        for name, branch in self.branches.items():
            update = branch["b"](branch["a"](args[0].float()) / math.sqrt(self.rank))
            delta = update.to(output.dtype).reshape_as(parts[name])
            self.scope["max_" + name + "_delta"] = max(self.scope["max_" + name + "_delta"],
                                                        float(delta.detach().abs().max().item()))
            parts[name] = torch.where(mask, parts[name] + delta, parts[name])
        changed = torch.cat((parts["query"], parts["gate"]), dim=-1)
        for name, value in (("query", query), ("gate", gate)):
            if name not in self.branches and not bits_equal(parts[name], value):
                raise AssertionError("Inactive head half changed")
        if (~active).any() and not bits_equal(changed[:, ~active], original[:, ~active]):
            raise AssertionError("Fact-prefix rows changed")
        self.scope["active_calls"] += 1
        self.scope["active_rows"] += count
        self.scope["prefix_rows_preserved"] += self._length - count
        return changed.reshape_as(output)

    def description(self):
        return {"mode": self.mode, "rank": self.rank, "layer_index": self.layer_index,
                "branches": list(self.branches), "parameters": sum(p.numel() for p in self.parameters()),
                "arithmetic": self.PROFILE, "input": "detached local normalized attention input",
                "query_width": self.query_width, "head_dim": self.head_dim,
                "prefix_projection_returned_unchanged": True, "cache_expansion": False}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._start = self._length = None


class ConstantInputFrozenAffines:
    """Same old frozen affine math, also valid before the sole trainable layer.

    Global autograd may be enabled, but frozen adapter inputs and parameters
    must not require gradients. No gradient-carrying path may pass this hook.
    """

    def __init__(self, controller):
        if any(p.requires_grad or p.grad is not None for p in controller.parameters()):
            raise ValueError("Only the fixed GDN controller is supported")
        self.handles, self.calls = [], {}
        for name, module in controller.named_modules():
            if isinstance(module, nn.Linear):
                if module.bias is not None or module.weight.dtype != torch.float32:
                    self.close()
                    raise ValueError("Frozen FP32 bias-free storage required")
                self.calls[name] = 0
                self.handles.append(module.register_forward_hook(self._after(name)))
        if not self.handles:
            raise ValueError("No frozen affines found")

    def _after(self, name):
        def hook(module, args, output):
            if (args[0].requires_grad or module.weight.requires_grad or module.weight.grad is not None
                or args[0].dtype != torch.float32 or output.dtype != torch.float32):
                raise ValueError("A gradient path entered the frozen constant-input controller")
            with torch.no_grad():
                result = F.linear(args[0].double(), module.weight.double()).float()
            self.calls[name] += 1
            return result
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
