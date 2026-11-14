"""Deterministic cycling and atomic adapter/optimizer checkpoints for calibration."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import torch


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def microbatch_indices(step, examples, accumulation):
    """One-based optimizer step, cycling the unchanged ordered examples."""
    if step < 1 or examples < 1 or accumulation < 1:
        raise ValueError("Positive step, example count and accumulation required")
    return [((step - 1) * accumulation + micro) % examples for micro in range(accumulation)]


def rng_state(module):
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if next(module.parameters()).is_cuda else []}


def assert_rng_equal(left, right):
    if not torch.equal(left["cpu"], right["cpu"]) or len(left["cuda"]) != len(right["cuda"]):
        raise ValueError("Evaluation changed the training RNG")
    if any(not torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"])):
        raise ValueError("Evaluation changed the GPU RNG")


def save_checkpoint(parent, step, module, optimizer, log_path, metadata):
    from safetensors.torch import save_file
    parent = Path(parent)
    destination = parent / f"step-{step:06d}"
    if destination.exists():
        raise ValueError("A committed checkpoint must not be overwritten")
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=parent) as temporary:
        temporary = Path(temporary)
        save_file({key: value.detach().cpu().contiguous() for key, value in module.state_dict().items()},
                  str(temporary / "controller.safetensors"))
        torch.save({"optimizer": optimizer.state_dict(), "rng": rng_state(module)}, temporary / "optimizer.pt")
        log_bytes = Path(log_path).read_bytes()
        if len(log_bytes.splitlines()) != step:
            raise ValueError("Checkpoint and training log have different step counts")
        record = {**metadata, "step": step, "controller_sha256": file_hash(temporary / "controller.safetensors"),
                  "optimizer_sha256": file_hash(temporary / "optimizer.pt"),
                  "training_log_prefix_bytes": len(log_bytes),
                  "training_log_prefix_sha256": hashlib.sha256(log_bytes).hexdigest()}
        (temporary / "checkpoint.json").write_text(json.dumps(record, indent=2) + "\n")
        temporary.rename(destination)
        # TemporaryDirectory tolerates its path having been atomically renamed.
    return destination


def load_checkpoint(path, module, optimizer):
    from safetensors.torch import load_file
    path = Path(path)
    record = json.loads((path / "checkpoint.json").read_text())
    if (file_hash(path / "controller.safetensors") != record["controller_sha256"] or
        file_hash(path / "optimizer.pt") != record["optimizer_sha256"]):
        raise ValueError("Checkpoint weights or optimizer changed")
    device = str(next(module.parameters()).device)
    module.load_state_dict(load_file(str(path / "controller.safetensors"), device=device), strict=True)
    saved = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"]["cpu"])
    if saved["rng"]["cuda"]:
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
    return record


def restore_log(path, checkpoint):
    """Verify the saved prefix, preserving any interrupted unsaved tail separately."""
    path = Path(path)
    content = path.read_bytes()
    boundary = checkpoint["training_log_prefix_bytes"]
    prefix, tail = content[:boundary], content[boundary:]
    if hashlib.sha256(prefix).hexdigest() != checkpoint["training_log_prefix_sha256"]:
        raise ValueError("Saved training-log prefix changed")
    if len(prefix.splitlines()) != checkpoint["step"]:
        raise ValueError("Saved training-log prefix has wrong step count")
    retained = None
    if tail:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        retained = path.with_name(f"interrupted-tail-{stamp}.jsonl")
        retained.write_bytes(tail)
        path.write_bytes(prefix)
    return str(retained) if retained else None
