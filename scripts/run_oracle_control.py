"""Test whether a model can execute the oracle-selected relevant event chain."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model", choices=("it", "base"), default="it")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.cases.read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from sas_memory.oracle import evidence_slice
    from sas_memory.tasks import INSTRUCTIONS, Operation, render
    from sas_memory.runtime import TextRuntime
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model_path = json.loads((ROOT / "configs/models.json").read_text())[args.model]
    manifest = {"status": "running", "model": args.model, "model_path": model_path,
                "started_utc": datetime.now(timezone.utc).isoformat(), "cases": len(cases),
                "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT / "sas_memory").glob("*.py")) + [Path(__file__).resolve()]},
                "interpretation": "Oracle backward slice; tests readout/reasoning with sufficient evidence. Changes positions and computation; not a deployable router."}
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    try:
        runtime = TextRuntime(model_path, args.model)
        count = 0
        with (args.output / "results.jsonl").open("w", buffering=1) as output:
            for index, case in enumerate(cases):
                for variant_name in ("control", "treatment"):
                    variant = case["variants"][variant_name]
                    events = [Operation(**event) for event in case["prefix"] + variant["tail"]]
                    query = next(q for q in variant["queries"] if q["role"] == "target_current")
                    selected = evidence_slice(events, query["entity"], query["version"])
                    selected_events = [asdict(event) for event in selected]
                    ledger = runtime.prompt_start + INSTRUCTIONS + render(selected_events)
                    result = runtime.probe(ledger, None, query["question"], query["answer"], case["candidates"],
                                           generate=True, prefill="full")
                    row = {"case_id": case["id"], "family": case["family"], "seed": case["seed"],
                           "distractor_updates": case["distractor_updates"], "variant": variant_name,
                           "role": query["role"], "oracle_events": selected_events,
                           "original_event_count": len(events), "selected_event_count": len(selected), **result}
                    output.write(json.dumps(row) + "\n")
                    count += 1
                print(json.dumps({"event": "case_done", "index": index + 1, "total": len(cases),
                                  "case": case["id"], "rows": count}), flush=True)
        manifest.update(status="completed", rows=count, elapsed_seconds=time.monotonic() - started,
                        completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
