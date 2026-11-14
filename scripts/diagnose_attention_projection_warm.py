"""Capture the failed arm-sequence projection; diagnostic-only, no gate eligibility."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import save_file
from scripts.diagnose_copy_arm_replay import controller_for
from scripts.run_controller_trial import state_fingerprint, write_json
from scripts.run_oracle_kv_readout import measure
import scripts.run_copy_evidence_necessity as runner

OUTPUT = Path("/media/990Pro/Code/SAS/artifacts/memory_update/attention-projection-warm-fixture-20261114-01")
ORIGINAL = runner.prove_and_measure


def capture_failure(runtime, ledger, query, masks, expected):
    try:
        return ORIGINAL(runtime, ledger, query, masks, expected)
    except ValueError as error:
        if "Actual unmasked reference readout changed" not in str(error):
            raise
        controller = controller_for(runtime)
        tensors, layouts, handles = {}, {}, []
        def observe(kind):
            def after(module, args, result):
                if kind not in tensors:
                    layouts[kind] = {"input_stride": list(args[0].stride()), "output_stride": list(result.stride()),
                                     "input_storage_offset": args[0].storage_offset(), "output_storage_offset": result.storage_offset()}
                    tensors[kind + "_input"] = args[0].detach().cpu().contiguous()
                    tensors[kind] = result.detach().cpu().contiguous()
            return after
        for kind in ("q", "k"):
            handles.append(controller.outputs[f"attention_{kind}_proj"].register_forward_hook(observe(kind)))
        try:
            actual = measure(runtime, ledger, query)
            if tensors["q"].shape != (1, 421, 2048):
                raise ValueError("Unexpected failed projection shape")
            save_file(tensors, OUTPUT.with_suffix(".safetensors"))
            write_json(OUTPUT.with_suffix(".json"), {"status": "diagnostic_only_failed_sequence_fixture", "error": str(error),
                "query_id": query["id"], "actual": actual, "adapter_state_sha256": state_fingerprint(controller),
                "blas_backend": str(torch.backends.cuda.preferred_blas_library()), "layouts": layouts,
                "fixture_sha256": hashlib.sha256(OUTPUT.with_suffix(".safetensors").read_bytes()).hexdigest(),
                "tensors": {k: {"shape": list(v.shape), "sha256": hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()} for k, v in tensors.items()},
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
            print(json.dumps({"event": "warm_fixture_captured", "actual": actual, "layouts": layouts}), flush=True)
        finally:
            for handle in handles:
                handle.remove()
        raise


def main():
    if OUTPUT.with_suffix(".json").exists() or OUTPUT.with_suffix(".safetensors").exists():
        raise ValueError("Preserve failed-sequence numeric fixture")
    runner.prove_and_measure = capture_failure
    runner.RUNTIME_SOURCES = tuple(sorted(set(runner.RUNTIME_SOURCES) | {
        str(Path(__file__).relative_to(ROOT)), "scripts/diagnose_copy_arm_replay.py"}))
    try:
        runner.main()
    finally:
        OUTPUT.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
