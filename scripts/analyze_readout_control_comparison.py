"""Audit and pair original-prefix oracle/extra-version/extra-unrelated masks."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_controller_confirmation import scalar_metrics, paired_interval, unary_cluster_scores
from scripts.run_oracle_kv_readout import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--competing", type=Path, required=True)
    parser.add_argument("--unrelated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"oracle": args.oracle, "competing_versions": args.competing, "matched_unrelated": args.unrelated}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_dir = args.output.with_suffix("").with_name(args.output.stem + "-audits")
    audit_dir.mkdir(exist_ok=True)
    manifests, audits = {}, {}
    for kind, path in paths.items():
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["config"].get("selection_kind", "oracle") != kind:
            raise ValueError("Wrong selection protocol")
        target = audit_dir / f"{kind}.json"
        subprocess.run([sys.executable, str(ROOT / "scripts/analyze_oracle_kv_readout.py"), str(path), "--output", str(target)], check=True)
        manifests[kind], audits[kind] = manifest, json.loads(target.read_text())
    common = manifests["oracle"]
    for kind, manifest in manifests.items():
        for field in ("cases_sha256", "checkpoint_sha256", "model_path", "reference_manifest_sha256", "completed_modes", "packages"):
            if manifest[field] != common[field]:
                raise ValueError(f"Unmatched model/cohort/runtime: {kind}/{field}")
        if kind != "oracle":
            validation = Path(manifest["config"]["control_validation"])
            if digest(validation) != manifest["config"]["control_validation_sha256"]:
                raise ValueError("Token-budget registration changed")
    cases = json.loads((args.oracle / "cases.json").read_text())
    lookup = {case["id"]: case for case in cases}
    clusters = {case["id"]: case["base_case_id"] for case in cases}
    validation = json.loads(Path(manifests["competing_versions"]["config"]["control_validation"]).read_text())
    registered = {(row["case_id"], row["variant"]): row for row in validation["rows"]}
    if len(registered) != 2 * len(cases) or not validation["all_token_budgets_equal"]:
        raise ValueError("Incomplete token-budget prevalidation")
    report = {"status": "completed", "base_cases": len(set(clusters.values())), "cases": len(cases),
              "training_initializations": 1, "runs": {kind: str(path) for kind, path in paths.items()},
              "manifest_sha256": {kind: digest(path / "manifest.json") for kind, path in paths.items()}, "models": {},
              "interpretation": "All necessary evidence retained; both added-record conditions have exactly equal historical token budgets on the same original prompt. Extra SET records differ in numeric contents and original positions, so this does not alone isolate semantic version interference from positional effects. Oracle identities, one initialization, eight base facts; no architecture or learned-selector claim."}
    conditions = sorted({(case["dependency_depth"], case["question_style"]) for case in cases})
    for model in common["completed_modes"]:
        rows = {kind: [json.loads(line) for line in (path / f"{model}.jsonl").read_text().splitlines()] for kind, path in paths.items()}
        indexed = {kind: {(row["case_id"], row["variant"]): row for row in members} for kind, members in rows.items()}
        for key, original in indexed["oracle"].items():
            current = [indexed[kind][key] for kind in paths]
            if len({row["prompt_ids_sha256"] for row in current}) != 1 or len({row["ledger_tokens"] for row in current}) != 1:
                raise ValueError("The controls changed original input tokens")
            competing, unrelated = indexed["competing_versions"][key], indexed["matched_unrelated"][key]
            if competing["kept_history_tokens"] != unrelated["kept_history_tokens"]:
                raise ValueError("Retained token budgets differ")
            for kind, row in zip(paths, current):
                if row["selected_event_indices"] != registered[key]["event_indices"][kind]:
                    raise ValueError("Registered event selection changed")
                if kind != "oracle":
                    if not set(original["selected_event_indices"]) <= set(row["selected_event_indices"]) or len(row["selected_event_indices"]) != len(original["selected_event_indices"]) + 2:
                        raise ValueError("Control removed evidence or has the wrong number of extra records")
            if competing["kept_history_tokens"] != registered[key]["kept_history_tokens"]:
                raise ValueError("Registered retained-token count changed")
        scores = {kind: unary_cluster_scores(members, clusters) for kind, members in rows.items()}
        interval = paired_interval(scores["competing_versions"], scores["matched_unrelated"])
        interval["unit"] = "base_case_id, averaging six depth/wording conditions and both orders"
        paired = {"unrelated_correct": sum(row["greedy_exact"] for row in rows["matched_unrelated"]),
                  "competing_regressions": sum(indexed["matched_unrelated"][key]["greedy_exact"] and not row["greedy_exact"] for key, row in indexed["competing_versions"].items()),
                  "competing_recoveries": sum(not indexed["matched_unrelated"][key]["greedy_exact"] and row["greedy_exact"] for key, row in indexed["competing_versions"].items())}
        report["models"][model] = {"regimes": {kind: scalar_metrics(members) for kind, members in rows.items()},
                                   "paired": paired, "competing_minus_unrelated": interval,
                                   "by_condition": [{"dependency_depth": depth, "question_style": style,
                                                     "regimes": {kind: scalar_metrics([row for row in members if lookup[row["case_id"]]["dependency_depth"] == depth and lookup[row["case_id"]]["question_style"] == style]) for kind, members in rows.items()}}
                                                    for depth, style in conditions]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for model, result in report["models"].items():
        print(model, {kind: f"{metrics['correct']}/{metrics['count']}" for kind, metrics in result["regimes"].items()}, result["paired"])


if __name__ == "__main__":
    main()
