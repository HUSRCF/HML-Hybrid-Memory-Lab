"""Small causal group-entry adapters, without modifying the backbone or caches.

These test the value of shared control. They do not by themselves enforce or
establish a semantic dependency interface. Checkpointing is intentionally not
supported: the group-entry hook keeps the current forward's detached input.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def query_residual(output: torch.Tensor, delta: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Qwen interleaves each head's query and output-gate projection."""
    if output.shape[-1] != 2 * delta.shape[-1] or delta.shape[-1] % head_dim:
        raise ValueError("Incompatible gated query projection")
    original = output.reshape(*output.shape[:-1], -1, 2 * head_dim)
    update = delta.reshape(*delta.shape[:-1], -1, head_dim)
    return torch.cat((original[..., :head_dim] + update, original[..., head_dim:]), dim=-1).reshape_as(output)


class GroupController(nn.Module):
    MODES = ("shared", "independent", "attention_only", "gdn_only", "local")

    def __init__(self, model, group_index: int, rank: int, mode: str, adapt_gdn_keys: bool = False):
        super().__init__()
        if mode not in self.MODES or rank <= 0:
            raise ValueError((mode, rank))
        if getattr(model, "is_gradient_checkpointing", False):
            raise ValueError("Group-entry hooks require checkpointing to be disabled")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Freeze the backbone before attaching a controller")
        layers = model.model.layers
        start = 4 * group_index
        types = model.config.layer_types[start:start + 4]
        if list(types) != ["linear_attention"] * 3 + ["full_attention"]:
            raise ValueError("Selected group is not GDN-GDN-GDN-Attention")
        self.group_index, self.rank, self.mode = group_index, rank, mode
        self.adapt_gdn_keys = adapt_gdn_keys
        self.hidden_size = model.config.hidden_size
        attention = layers[start + 3].self_attn
        self.head_dim = attention.head_dim
        self.encoders = nn.ModuleDict()
        self.outputs = nn.ModuleDict()
        self._context = None
        self._handles = []
        self._specs = []
        self._key_slices = {}
        self._token_policy, self._token_boundary, self._position_start = "all", 0, 0
        use_gdn, use_attention = mode != "attention_only", mode != "gdn_only"
        if mode == "local":
            branches = ()
        elif mode == "shared":
            branches = ("shared",)
        else:
            branches = tuple(branch for branch, enabled in (("gdn", use_gdn), ("attention", use_attention)) if enabled)
        for branch in branches:
            self.encoders[branch] = nn.Linear(self.hidden_size, rank, bias=False)
        if use_gdn:
            for offset in range(3):
                gdn = layers[start + offset].linear_attn
                for field in ("a", "b"):
                    projection = getattr(gdn, f"in_proj_{field}")
                    name = f"gdn_{offset}_{field}"
                    self.outputs[name] = nn.Linear(rank, projection.out_features, bias=False)
                    self._specs.append((projection, name, "gdn", False))
                if adapt_gdn_keys:
                    name = f"gdn_{offset}_key"
                    self.outputs[name] = nn.Linear(rank, gdn.key_dim, bias=False)
                    self._key_slices[name] = (gdn.key_dim, 2 * gdn.key_dim)
                    self._specs.append((gdn.in_proj_qkv, name, "gdn", False))
        if use_attention:
            for field, output_dim, is_query in (
                ("q_proj", attention.q_proj.out_features // 2, True),
                ("k_proj", attention.k_proj.out_features, False),
            ):
                name = f"attention_{field}"
                self.outputs[name] = nn.Linear(rank, output_dim, bias=False)
                self._specs.append((getattr(attention, field), name, "attention", is_query))
        for output in self.outputs.values():
            nn.init.zeros_(output.weight)
        if mode == "local":
            for projection, name, _, _ in self._specs:
                self.encoders[name] = nn.Linear(projection.in_features, rank, bias=False)
        self._enabled_outputs = set(self.outputs)
        device = next(model.parameters()).device
        self.to(device=device, dtype=torch.float32)
        self._handles.append(model.register_forward_pre_hook(self._positions, with_kwargs=True))
        self._handles.append(layers[start].register_forward_pre_hook(self._capture, with_kwargs=True))
        for projection, name, branch, is_query in self._specs:
            self._handles.append(projection.register_forward_hook(self._make_hook(name, branch, is_query)))

    def _capture(self, module, args, kwargs):
        hidden = kwargs.get("hidden_states")
        if hidden is None:
            hidden = args[0]
        # The prefix is frozen. Detaching avoids retaining its entire graph.
        self._context = hidden.detach()

    def _positions(self, module, args, kwargs):
        if self._token_policy != "all":
            cache = kwargs.get("past_key_values", kwargs.get("cache"))
            self._position_start = cache.get_seq_length() if cache is not None else 0

    def _make_hook(self, name, branch, is_query):
        def hook(module, args, output):
            if name not in self._enabled_outputs or self._token_policy == "none":
                return output
            if self._context is None or self._context.shape[:-1] != output.shape[:-1]:
                raise RuntimeError("Missing or mismatched causal group-entry context")
            if self.mode == "local":
                # Ordinary linear low-rank residual, using this projection's own
                # input. It has no cross-layer group-entry bypass or shared code.
                control = self.encoders[name](args[0].float()) / math.sqrt(self.rank)
            else:
                encoder = self.encoders["shared" if self.mode == "shared" else branch]
                # Recompute rather than retaining a small autograd graph across
                # calls. Shared/independent branches see exactly the same h0.
                control = torch.nn.functional.silu(encoder(self._context.float())) / math.sqrt(self.rank)
            delta = self.outputs[name](control).to(output.dtype)
            if self._token_policy in ("ledger", "question"):
                positions = torch.arange(self._position_start, self._position_start + output.shape[-2], device=output.device)
                mask = positions < self._token_boundary if self._token_policy == "ledger" else positions >= self._token_boundary
                delta = delta * mask.to(delta.dtype)[None, :, None]
            if is_query:
                return query_residual(output, delta, self.head_dim)
            if name in self._key_slices:
                start, stop = self._key_slices[name]
                return torch.cat((output[..., :start], output[..., start:stop] + delta, output[..., stop:]), dim=-1)
            return output + delta
        return hook

    def set_enabled_outputs(self, names):
        names = set(names)
        if not names.issubset(self.outputs):
            raise ValueError(f"Unknown controller outputs: {names - set(self.outputs)}")
        self._enabled_outputs = names

    def set_token_policy(self, policy: str, boundary: int = 0):
        if policy not in ("all", "none", "ledger", "question") or boundary < 0:
            raise ValueError((policy, boundary))
        self._token_policy, self._token_boundary = policy, boundary

    def description(self):
        return {"mode": self.mode, "rank": self.rank, "group_index": self.group_index,
                "adapt_gdn_keys": self.adapt_gdn_keys,
                "layers": list(range(4 * self.group_index, 4 * self.group_index + 4)),
                "trainable_parameters": sum(p.numel() for p in self.parameters()),
                "encoder_shapes": {name: list(module.weight.shape) for name, module in self.encoders.items()},
                "output_shapes": {name: list(module.weight.shape) for name, module in self.outputs.items()},
                "input": "local projection hidden state" if self.mode == "local" else "detached causal group-entry hidden state",
                "cache_expansion": False}

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._context = None

