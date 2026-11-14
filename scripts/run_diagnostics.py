"""Run reproducible frozen-model memory-update diagnostics on local checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sas_memory.tasks import INSTRUCTIONS, make_case, render
from sas_memory.metrics import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/diagnostic_pilot.json")
    parser.add_argument("--model", choices=("it", "base"), default="it")
    parser.add_argument("--backend", choices=("auto", "reference"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--prefill", choices=("full", "cached"), default="full")
    parser.add_argument("--score-batch", action="store_true", help="Optional speed tradeoff; default uses serial scoring")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("protocol_version", 1) == 2:
        from sas_memory.stress_tasks import make_stress_case
        if config["families"] != ["overwrite"]:
            raise ValueError("Protocol 2 currently tests overwrite only")
        cases = [make_stress_case(seed, length, entities) for length in config["distractor_updates"]
                 for entities in config["distractor_entities"] for seed in config["seeds"]]
    else:
        cases = [make_case(family, seed, length) for length in config["distractor_updates"]
                 for seed in config["seeds"] for family in config["families"]]
    if args.limit is not None:
        cases = cases[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    data = "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases)
    (args.output / "cases.jsonl").write_text(data)
    import torch
    from sas_memory.runtime import TextRuntime, assert_cache_equal, fork_cache
    torch.set_num_threads(4)
    torch.manual_seed(0)
    path = json.loads((ROOT / "configs/models.json").read_text())[args.model]
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "model": args.model, "model_path": path,
        "backend": args.backend, "config": config, "cases_sha256": hashlib.sha256(data.encode()).hexdigest(),
        "model_config_sha256": hashlib.sha256((Path(path) / "config.json").read_bytes()).hexdigest(),
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "fla-core")},
        "gpu": torch.cuda.get_device_name(0), "python": sys.executable, "training": False,
        "generation": not args.no_generate, "status": "running", "case_count": len(cases),
        "prefill": args.prefill, "score_batch": args.score_batch,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__)]},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    print(json.dumps({"event": "loading", "model": args.model, "cases": len(cases)}), flush=True)
    try:
        runtime = TextRuntime(path, args.model, args.backend)
        torch.cuda.reset_peak_memory_stats()
        rows = []
        with (args.output / "results.jsonl").open("w", buffering=1) as result_file:
            for case_index, case in enumerate(cases):
                ledger = runtime.prompt_start + INSTRUCTIONS + render(case["prefix"])
                root_cache, _, root_tokens = runtime.advance(ledger)
                root_saved = fork_cache(root_cache)
                if case_index == 0:
                    verification = runtime.verify(ledger, root_cache, case["variants"]["before"]["queries"][0]["question"], case["candidates"])
                    (args.output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
                    print(json.dumps({"event": "runtime_verified", **verification}), flush=True)
                    if args.prefill == "cached" and not verification["cached_prefill_within_tolerance"]:
                        raise AssertionError("Requested cached prefill exceeds the recorded numerical tolerance; use --prefill full")
                    if args.score_batch and not verification["batch_scoring_within_tolerance"]:
                        raise AssertionError("Requested batched scoring exceeds the recorded numerical tolerance; use serial scoring")
                for variant_name, variant in case["variants"].items():
                    tail = render(variant["tail"])
                    if tail:
                        cache, _, tail_tokens = runtime.advance(tail, root_cache, ledger)
                    else:
                        cache, tail_tokens = root_cache, 0
                    variant_ledger = ledger + tail
                    for query in variant["queries"]:
                        result = runtime.probe(variant_ledger, cache, query["question"], query["answer"], case["candidates"],
                                               not args.no_generate, args.prefill, args.score_batch)
                        row = {"case_id": case["id"], "family": case["family"], "seed": case["seed"],
                               "distractor_updates": case["distractor_updates"], "variant": variant_name,
                               "distractor_entities": case.get("distractor_entities", 4),
                               "observed_distractor_entities": case.get("observed_distractor_entities", 4),
                               "role": query["role"], "question": query["question"],
                               "ledger_tokens": root_tokens + tail_tokens, **result}
                        rows.append(row)
                        result_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                assert_cache_equal(root_cache, root_saved)
                summary = summarize(rows)
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(json.dumps({"event": "case_done", "index": case_index + 1, "total": len(cases),
                                  "case": case["id"], "rows": len(rows),
                                  "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
        torch.cuda.synchronize()
        manifest.update(status="completed", elapsed_seconds=time.monotonic() - started,
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "output": str(args.output), "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
