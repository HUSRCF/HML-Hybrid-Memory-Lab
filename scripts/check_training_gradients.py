"""Compare FLA GDN gradients with the native Torch recurrence, including gates."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
    torch.set_num_threads(4)
    torch.manual_seed(2027)
    shape = (1, 128, 2, 128)
    tensors = {
        "query": torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
        "key": torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
        "value": torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
        "g": -0.01 - torch.rand(shape[:3], device="cuda", dtype=torch.float32) * 0.09,
        "beta": 0.2 + torch.rand(shape[:3], device="cuda", dtype=torch.bfloat16) * 0.6,
    }
    upstream = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gradients, losses = {}, {}
    for name, implementation in (("torch", torch_recurrent_gated_delta_rule), ("fla", chunk_gated_delta_rule)):
        values = {key: value.detach().clone().requires_grad_() for key, value in tensors.items()}
        output, state = implementation(values["query"], values["key"], values["value"], g=values["g"], beta=values["beta"],
                                       output_final_state=True, initial_state=None,
                                       use_qk_l2norm_in_kernel=True)
        loss = (output.float() * upstream.float()).mean() + 0.01 * state.float().square().mean()
        loss.backward()
        losses[name] = loss.item()
        gradients[name] = {key: value.grad.detach().float() for key, value in values.items()}
    comparisons = {}
    for key in tensors:
        actual, reference = gradients["fla"][key], gradients["torch"][key]
        error = (actual - reference).norm() / reference.norm().clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(), reference.flatten(), dim=0)
        comparisons[key] = {"relative_l2_error": error.item(), "cosine": cosine.item(),
                            "max_abs_error": (actual - reference).abs().max().item(),
                            "finite": bool(torch.isfinite(actual).all())}
    passed = all(row["finite"] and row["relative_l2_error"] < 0.03 and row["cosine"] > 0.995
                 for row in comparisons.values())
    report = {"status": "passed" if passed else "failed", "shape": list(shape), "losses": losses,
              "comparisons": comparisons, "tolerance": {"relative_l2_error": 0.03, "cosine_min": 0.995},
              "scope": "Two chunks, normalized q/k, gradients for q/k/v/decay/write gates and final recurrent state."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
