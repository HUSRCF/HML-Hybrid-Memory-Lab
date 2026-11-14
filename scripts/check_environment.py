"""Validate the local research runtime; this does not run research experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import struct
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def checkpoint_manifest(path: Path) -> dict:
    """Check shard lengths and index coverage without loading tensor data."""
    index = json.loads((path / "model.safetensors.index.json").read_text())
    shards = []
    for name in sorted(set(index["weight_map"].values())):
        shard = path / name
        with shard.open("rb") as file:
            header_bytes = struct.unpack("<Q", file.read(8))[0]
            header = json.loads(file.read(header_bytes))
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        expected_bytes = 8 + header_bytes + max(v["data_offsets"][1] for v in tensors.values())
        expected_keys = {k for k, v in index["weight_map"].items() if v == name}
        if shard.stat().st_size != expected_bytes or not expected_keys.issubset(tensors):
            raise ValueError(f"Incomplete or inconsistent checkpoint shard: {shard}")
        shards.append({"file": name, "bytes": expected_bytes, "tensors": len(tensors)})
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config", config)
    return {
        "path": str(path),
        "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
        "shards": shards,
        "layers": text["num_hidden_layers"],
        "layer_types": text["layer_types"],
        "verification": "safetensors header length and index coverage; not a full-file checksum",
    }


def check_kernels() -> dict:
    """Check FLA prefill, recurrent continuation, and backward against Torch."""
    import torch
    import torch.nn.functional as F
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

    torch.manual_seed(42)
    shape = (1, 64, 2, 128)
    q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(3)]
    g = -F.softplus(torch.randn(shape[:3], device="cuda")) * 0.1
    beta = torch.rand(shape[:3], device="cuda", dtype=torch.bfloat16)
    kwargs = dict(g=g, beta=beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
    reference, reference_state = torch_recurrent_gated_delta_rule(q, k, v, **kwargs)
    output, state = chunk_gated_delta_rule(q, k, v, **kwargs)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), reference.float(), atol=3e-3, rtol=3e-2)
    torch.testing.assert_close(state.float(), reference_state.float(), atol=3e-3, rtol=3e-2)
    continuation = dict(g=g[:, :1], beta=beta[:, :1], output_final_state=True, use_qk_l2norm_in_kernel=True)
    reference_next, reference_next_state = torch_recurrent_gated_delta_rule(
        q[:, :1], k[:, :1], v[:, :1], initial_state=state.clone(), **continuation
    )
    recurrent, recurrent_state = fused_recurrent_gated_delta_rule(
        q[:, :1], k[:, :1], v[:, :1], initial_state=state.clone(), **continuation
    )
    torch.testing.assert_close(recurrent.float(), reference_next.float(), atol=3e-3, rtol=3e-2)
    torch.testing.assert_close(recurrent_state.float(), reference_next_state.float(), atol=3e-3, rtol=3e-2)
    q, k, v = [x.detach().requires_grad_() for x in (q, k, v)]
    differentiable, _ = chunk_gated_delta_rule(q, k, v, **kwargs)
    differentiable.float().square().mean().backward()
    if not all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0 for x in (q, k, v)):
        raise RuntimeError("FLA backward did not produce finite, nonzero Q/K/V gradients")
    return {
        "status": "passed",
        "dtype": "bfloat16",
        "shape": list(shape),
        "chunk_max_abs_error": (output.float() - reference.float()).abs().max().item(),
        "state_max_abs_error": (state.float() - reference_state.float()).abs().max().item(),
        "recurrent_max_abs_error": (recurrent.float() - reference_next.float()).abs().max().item(),
        "backward_finite_nonzero": True,
        "tolerance": {"atol": 0.003, "rtol": 0.03},
    }


def check_model(name: str, path: Path, mode: str) -> dict:
    import torch
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    config = AutoConfig.from_pretrained(path, local_files_only=True).text_config
    started = time.monotonic()
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        path,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": "cuda:0"},
        local_files_only=True,
        output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError(f"Text checkpoint loading failed: {loading}")
    model.eval().requires_grad_(False)
    if mode == "reference":
        for layer in model.model.layers:
            if hasattr(layer, "linear_attn"):
                gdn = layer.linear_attn
                gdn.chunk_gated_delta_rule = implementation.torch_chunk_gated_delta_rule
                gdn.recurrent_gated_delta_rule = implementation.torch_recurrent_gated_delta_rule
                gdn.causal_conv1d_fn = None
                gdn.causal_conv1d_update = implementation.torch_causal_conv1d_update
                norm = implementation.Qwen3_5RMSNormGated(gdn.head_v_dim, eps=gdn.layer_norm_epsilon)
                norm.to(device=model.device, dtype=model.dtype)
                norm.weight.data.copy_(gdn.norm.weight.data)
                gdn.norm = norm
    if name == "it":
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    else:
        prompt = "The capital of France is"
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        prefill = model(**tokens, use_cache=True, logits_to_keep=1)
        if not torch.isfinite(prefill.logits).all():
            raise RuntimeError("Non-finite prefill logits")
        first = prefill.logits[:, -1].argmax(-1, keepdim=True)
        decoded = model(input_ids=first, past_key_values=prefill.past_key_values, use_cache=True, logits_to_keep=1)
        if not torch.isfinite(decoded.logits).all():
            raise RuntimeError("Non-finite cached decode logits")
        continuation = model.generate(**tokens, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    gdn = next(layer.linear_attn for layer in model.model.layers if hasattr(layer, "linear_attn"))
    text = tokenizer.decode(continuation[0, tokens.input_ids.shape[1]:], skip_special_tokens=True)
    result = {
        "status": "passed", "mode": mode, "path": str(path),
        "class": type(model).__name__,
        "text_parameters": sum(p.numel() for p in model.parameters()),
        "vision_tower_loaded": any("visual" in key for key in model.state_dict()),
        "prefill_and_cached_decode_finite": True,
        "gdn_chunk_backend": f"{gdn.chunk_gated_delta_rule.__module__}.{gdn.chunk_gated_delta_rule.__name__}",
        "convolution_backend": "causal_conv1d" if gdn.causal_conv1d_fn else "torch",
        "generated_text": text,
        "elapsed_including_load_and_compile_seconds": round(time.monotonic() - started, 3),
        "peak_allocated_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "missing_text_keys": sorted(loading.get("missing_keys", [])),
    }
    del model, prefill, decoded, continuation, tokens, gdn
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "configs/models.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Load each text model and check prefill/decode/generation")
    parser.add_argument("--kernels", action="store_true", help="Also check FLA forward, continuation, and backward")
    parser.add_argument("--mode", choices=("reference", "auto"), default="reference")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(8)
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch_runtime": torch.__version__, "hip": torch.version.hip,
        "packages": {}, "checkpoints": {}, "smoke": {}, "errors": {},
    }
    for package in ("torch", "triton", "transformers", "accelerate", "huggingface-hub", "safetensors", "fla-core", "flash-linear-attention", "datasets", "pyarrow", "pandas", "numpy"):
        try:
            report["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["packages"][package] = None
    try:
        from datasets import Dataset
        sample = Dataset.from_dict({"text": ["runtime check"]})
        if sample[0]["text"] != "runtime check":
            raise RuntimeError("Dataset round trip failed")
        report["data_runtime"] = {"status": "passed", "arrow_round_trip_rows": sample.num_rows}
    except Exception as error:
        report["errors"]["data_runtime"] = f"{type(error).__name__}: {error}"
    if not torch.cuda.is_available() or not torch.version.hip:
        report["errors"]["gpu"] = "Expected an available ROCm GPU"
    else:
        properties = torch.cuda.get_device_properties(0)
        report["gpu"] = {"name": properties.name, "memory_GiB": round(properties.total_memory / 2**30, 3), "architecture": properties.gcnArchName}
    models = json.loads(args.models.read_text())
    for name, value in models.items():
        try:
            report["checkpoints"][name] = checkpoint_manifest(Path(value))
        except Exception as error:
            report["errors"][f"checkpoint:{name}"] = f"{type(error).__name__}: {error}"
    if args.kernels and "gpu" not in report["errors"]:
        print("Checking FLA forward, recurrent continuation, and backward...", flush=True)
        try:
            report["kernels"] = check_kernels()
        except Exception as error:
            report["errors"]["kernels"] = f"{type(error).__name__}: {error}"
    if args.smoke and "gpu" not in report["errors"]:
        for name, value in models.items():
            if f"checkpoint:{name}" in report["errors"]:
                continue
            print(f"Checking {name} ({args.mode})...", flush=True)
            try:
                report["smoke"][name] = check_model(name, Path(value), args.mode)
            except Exception as error:
                report["errors"][f"smoke:{name}"] = f"{type(error).__name__}: {error}"
                import traceback
                traceback.print_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
