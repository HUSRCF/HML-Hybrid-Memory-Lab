"""Small CPU/GPU numeric reproduction, with explicit BLAS and unchanged inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import load_file
from scripts.run_controller_trial import write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def differences(actual, reference):
    delta = actual.double() - reference.double()
    return {"exact_equal": torch.equal(actual, reference), "different_elements": int((actual != reference).sum()),
            "maximum_absolute_error": delta.abs().max().item(), "rms_error": delta.square().mean().sqrt().item(),
            "different_after_bfloat16_rounding": int((actual.bfloat16() != reference.bfloat16()).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cublas", "cublaslt"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve numeric replay report")
    torch.set_num_threads(4)
    root = Path("/media/990Pro/Code/SAS/artifacts/memory_update")
    cold_path = root / "attention-projection-fixture-20261114-01.json"
    warm_path = root / "attention-projection-warm-fixture-20261114-01.json"
    cold = json.loads(cold_path.read_text())
    warm = json.loads(warm_path.read_text()) if warm_path.exists() else None
    fixtures = [(cold_path, cold)] + ([(warm_path, warm)] if warm else [])
    for path, meta in fixtures:
        if digest(path.with_suffix(".safetensors")) != meta["fixture_sha256"]:
            raise ValueError("Numeric fixture changed")
    if ((warm and cold["adapter_state_sha256"] != warm["adapter_state_sha256"]) or
        digest(Path(cold["checkpoint"])) != cold["checkpoint_sha256"]):
        raise ValueError("Actual adapter weights changed")
    a = load_file(cold_path.with_suffix(".safetensors"), device="cpu")
    b = load_file(warm_path.with_suffix(".safetensors"), device="cpu") if warm else None
    trace_path = root / "copy-arm-replay-diagnostic-20261114-01.json"
    warm_trace = next(x for x in json.loads(trace_path.read_text())["records"] if x["stage"] == "warm_trace")["activation_trace"]
    observed_warm_hashes = {k: next(x["sha256"] for x in warm_trace if x["name"] == f"adapter.outputs.attention_{k}_proj") for k in ("q", "k")}
    checkpoint = load_file(cold["checkpoint"], device="cpu")
    device = "cpu" if args.backend == "cpu" else "cuda"
    if device == "cuda":
        torch.backends.cuda.preferred_blas_library(args.backend)
    report = {"status": "numeric_diagnostic_only", "backend": args.backend,
        "actual_blas": str(torch.backends.cuda.preferred_blas_library()),
        "cold_meta_sha256": digest(cold_path), "warm_meta_sha256": digest(warm_path) if warm else None,
        "warm_trace_sha256": digest(trace_path), "warm_full_fixture_captured": bool(warm),
        "checkpoint_sha256": cold["checkpoint_sha256"], "source_sha256": digest(Path(__file__)), "projections": {}}
    with torch.inference_mode():
        for kind in ("q", "k"):
            control, weight = a[kind + "_input"], checkpoint[f"outputs.attention_{kind}_proj.weight"]
            truth = torch.nn.functional.linear(control.double(), weight.double())
            trials = []
            weight = weight.to(device)
            for number, offset in enumerate((0, 1, 2, 4, 8, 16, 32, 64, 0, 0)):
                allocation = torch.empty(control.numel() + offset, device=device, dtype=control.dtype)
                actual_input = allocation[offset:].view_as(control)
                actual_input.copy_(control)
                if number >= 8:
                    for rank in (22, 54, 64, 84):
                        torch.nn.functional.linear(torch.ones((1, 421, rank), device=device), torch.ones((2048, rank), device=device))
                actual = torch.nn.functional.linear(actual_input, weight).cpu()
                trials.append({"trial": number, "storage_offset": offset, "sha256": tensor_sha(actual),
                    "matches_actual_warm_projection_sha256": tensor_sha(actual) == observed_warm_hashes[kind],
                    "versus_cold": differences(actual, a[kind]), "versus_warm": differences(actual, b[kind]) if b else None,
                    "versus_fp64_truth": differences(actual, truth)})
            report["projections"][kind] = {"same_input_cold_warm": torch.equal(control, b[kind + "_input"]) if b else None,
                "input_difference": differences(control, b[kind + "_input"]) if b else None,
                "output_cold_versus_warm": differences(a[kind], b[kind]) if b else None,
                "cold_vs_fp64_truth": differences(a[kind], truth),
                "warm_vs_own_fp64_truth": differences(b[kind], torch.nn.functional.linear(b[kind + "_input"].double(), weight.cpu().double())) if b else None,
                "distinct_output_hashes": len({t["sha256"] for t in trials}), "trials": trials}
    write_json(args.output, report)
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"backend": args.backend, "projections": {k: {f: v[f] for f in (
        "same_input_cold_warm", "input_difference", "output_cold_versus_warm", "distinct_output_hashes")} for k, v in report["projections"].items()}}), flush=True)


if __name__ == "__main__":
    main()
