"""Explicit diagnostic arithmetic for frozen adapter Linear modules only.

This changes floating-point execution and requires fresh full-history references.
It must never be silently mixed with the sealed FP32-adapter observations.
"""

import torch
from torch import nn
from torch.nn import functional as F


class StableAdapterLinears:
    """Recompute frozen adapter affines in FP64, then round once to FP32.

    Forward hooks leave every parameter's storage, dtype and bytes unchanged.
    The original FP32 result is discarded. This is for frozen diagnostic runs,
    not a training implementation or an inference-speed claim.
    """

    PROFILE = "frozen_adapter_affine_fp64_accumulate_round_fp32_v1"

    def __init__(self, controller):
        if any(p.requires_grad or p.grad is not None for p in controller.parameters()):
            raise ValueError("Canonical adapter arithmetic requires frozen parameters without gradients")
        self.handles, self.calls = [], {}
        for name, module in controller.named_modules():
            if isinstance(module, nn.Linear):
                if module.bias is not None or module.weight.dtype != torch.float32:
                    self.close()
                    raise ValueError("Expected existing bias-free FP32 adapter linears")
                self.calls[name] = 0
                self.handles.append(module.register_forward_hook(self.after(name)))
        if not self.handles:
            raise ValueError("No adapter Linear modules found")

    def after(self, name):
        def recompute(module, args, output):
            if torch.is_grad_enabled() or args[0].dtype != torch.float32 or output.dtype != torch.float32:
                raise ValueError("Canonical affine profile is frozen inference with FP32 inputs/outputs only")
            self.calls[name] += 1
            return F.linear(args[0].double(), module.weight.double()).float()
        return recompute

    def description(self):
        return {"profile": self.PROFILE, "linear_modules": list(self.calls),
                "parameter_storage_and_dtype_changed": False, "backbone_arithmetic_changed": False}

    def close(self):
        for handle in self.handles:
            handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
