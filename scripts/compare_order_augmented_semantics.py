"""Compare paired training-order interventions from a complete independent audit."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from sas_memory.training_data import data_fingerprint
from scripts.analyze_controller_trial import load_rows
from scripts.compare_paired_semantic_controllers import functional_units, paired_counts
from scripts.run_controller_trial import write_json


ARMS = ["shared", "independent", "local", "local_question"]
CONDITIONS = ["canonical", "permuted"]


def checked_rows(directory, filename, digest, expected):
    source = directory / filename
    if file_hash(source) != digest:
        raise ValueError("Audited raw output changed before augmentation comparison")
    return load_rows(source, expected)


def compare(audit):
    run = Path(audit["source_run"])
    if audit["status"] != "completed_and_audited" or set(audit["arms"]) != set(ARMS) or file_hash(run / "manifest.json") != audit["manifest_sha256"]:
        raise ValueError("Current complete four-arm independent audit required")
    plan = json.loads((run / "plan.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    data = json.loads((Path(plan["data_source"]) / "data.json").read_text())
    if data_fingerprint(data) != plan["data_sha256"]:
        raise ValueError("Audited augmented data changed")
    expected = {(example["case"]["id"], example["variant"], example["query"]["role"]): example["query"]["answer"] for example in data["validation"]}
    references = json.loads((run / "reference_audits.json").read_text())
    old_training = Path(plan["training_reference_run"])
    old_order = Path(plan["order_reference_run"])
    if (file_hash(old_training / "manifest.json") != references["training"]["manifest_sha256"] or
        file_hash(old_order / "manifest.json") != references["order"]["manifest_sha256"]):
        raise ValueError("Original audited reference manifest changed")
    old_order_manifest = json.loads((old_order / "manifest.json").read_text())
    rows, units = {}, {}
    for arm in ARMS:
        rows[arm], units[arm] = {}, {}
        original = old_training / arm / "step-000432/evaluation"
        current = run / arm / "step-000432/evaluation"
        permuted = run / arm / "step-000432/initial_order_evaluation"
        for directory, digest in ((original, references["training"]["arms"][arm]["evaluation_summary_sha256"]),
                                  (current, audit["arms"][arm]["canonical"]["evaluation_summary_sha256"]),
                                  (permuted, audit["arms"][arm]["permuted"]["evaluation_summary_sha256"])):
            if file_hash(directory / "summary.json") != digest:
                raise ValueError("Audited evaluation summary changed")
        old_summary = json.loads((original / "summary.json").read_text())
        current_summary = json.loads((current / "summary.json").read_text())
        permuted_summary = json.loads((permuted / "summary.json").read_text())
        rows[arm]["canonical"] = {
            "original_training": checked_rows(original, "validation.jsonl", old_summary["file_sha256"]["validation.jsonl"], expected),
            "augmented_training": checked_rows(current, "validation.jsonl", current_summary["file_sha256"]["validation.jsonl"], expected)}
        rows[arm]["permuted"] = {
            "original_training": checked_rows(old_order, f"{arm}.jsonl", old_order_manifest["models"][arm]["results_sha256"], expected),
            "augmented_training": checked_rows(permuted, "validation.jsonl", permuted_summary["results_sha256"], expected)}
        for condition in CONDITIONS:
            units[arm][condition] = {training: functional_units(members) for training, members in rows[arm][condition].items()}
    changes, counts, family_updates, changed_cases = {}, {}, {}, []
    for arm in ARMS:
        changes[arm], counts[arm], family_updates[arm] = {}, {}, {}
        for condition in CONDITIONS:
            before, after = units[arm][condition]["original_training"], units[arm][condition]["augmented_training"]
            changes[arm][condition] = {metric: paired_counts(before[metric], after[metric]) for metric in before}
            counts[arm][condition] = {training: {metric: {"correct":sum(values.values()), "units":len(values)} for metric,values in model.items()} for training,model in units[arm][condition].items()}
            family_updates[arm][condition] = {training: {family: {"correct": sum(row["greedy_exact"] for row in members if row["family"] == family and row["role"] == "target_current" and row["variant"] != "before"), "examples":sum(row["family"] == family and row["role"] == "target_current" and row["variant"] != "before" for row in members)} for family in sorted({row["family"] for row in members})} for training,members in rows[arm][condition].items()}
            originals = {(row["case_id"],row["variant"],row["role"]):row for row in rows[arm][condition]["original_training"]}
            for row in rows[arm][condition]["augmented_training"]:
                key = (row["case_id"],row["variant"],row["role"])
                old = originals[key]
                if old["greedy_exact"] != row["greedy_exact"]:
                    changed_cases.append({"arm":arm,"condition":condition,"key":list(key),"family":row["family"],"gold":row["answer"],
                                          "original_training_generated":old["generated"],"augmented_training_generated":row["generated"],"rescued":row["greedy_exact"]})
    shared = {condition: {arm: {metric:paired_counts(units[arm][condition]["augmented_training"][metric], units["shared"][condition]["augmented_training"][metric]) for metric in units[arm][condition]["augmented_training"]} for arm in ARMS if arm != "shared"} for condition in CONDITIONS}
    within = {arm: {metric:paired_counts(units[arm]["canonical"]["augmented_training"][metric], units[arm]["permuted"]["augmented_training"][metric]) for metric in units[arm]["canonical"]["augmented_training"]} for arm in ARMS}
    return {"status":"completed_paired_augmentation_comparison", "source_run":str(run), "manifest_sha256":audit["manifest_sha256"],
            "counts":counts,"training_order_change":changes,"shared_vs_augmented_controls":shared,"augmented_canonical_to_permuted":within,
            "target_updates_by_family":family_updates,"correctness_changes":changed_cases,
            "interpretation":plan["interpretation"] + " Equal averages retain paired rescues/regressions. Correlated value/role/variant readouts are not independent experiments. A training-order effect under these matched weights/budgets is not an isolated recurrent-write mechanism."}


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = ["Shared", "Independent", "Local", "Question local"]
    specifications = [("individual", "Answers correct (432)"), ("three_roles", "All 3 roles correct (144)"), ("stable_unchanged_case_role", "Unchanged fact across 3 variants (96)")]
    figure, axes = plt.subplots(2,3,figsize=(14,9))
    positions=np.arange(len(ARMS))
    for r,condition in enumerate(CONDITIONS):
        for c,(metric,title) in enumerate(specifications):
            axis=axes[r,c]
            for offset,(training,label,color) in enumerate((("original_training","Canonical-order training","#94a3b8"),("augmented_training","50/50 order-augmented training","#2563eb"))):
                cells=[report["counts"][arm][condition][training][metric] for arm in ARMS]
                bars=axis.bar(positions+(offset-.5)*.36,[100*cell["correct"]/cell["units"] for cell in cells],.34,label=label,color=color)
                for bar,cell in zip(bars,cells):
                    axis.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1+(4 if offset else 0),str(cell["correct"]),ha="center",va="bottom",fontsize=8)
            axis.set_ylim(0,110)
            axis.set_xticks(positions,labels,rotation=20,ha="right")
            axis.set_title(("Canonical evaluation: " if r==0 else "Permuted evaluation: ")+title,fontsize=10)
            axis.spines[["top","right"]].set_visible(False)
            if c==0:axis.set_ylabel("Required answers correct (%)")
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    figure.legend(handles,legend_labels,loc="lower center",bbox_to_anchor=(.5,.045),ncol=2,frameon=False)
    figure.suptitle("Ordinary order augmentation: frozen Qwen3.5-2B IT, matched initialization and training budgets",fontsize=13)
    figure.text(.5,.015,"Observed development orders; 12 base event streams, one paired initialization. No shared-interface or isolated write-damage claim.",ha="center",fontsize=9,color="#475569")
    figure.tight_layout(rect=(0,.10,1,.95))
    output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(output,dpi=200,bbox_inches="tight")
    figure.savefig(output.with_suffix('.pdf'),bbox_inches="tight")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--plot",type=Path)
    args=parser.parse_args()
    report=compare(json.loads(args.audit.read_text()))
    report.update(audit_report_sha256=file_hash(args.audit),analysis_source_sha256=file_hash(Path(__file__)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    if args.plot:plot(report,args.plot)
    print(json.dumps({arm:{condition:report['counts'][arm][condition]['augmented_training']['individual'] for condition in CONDITIONS} for arm in ARMS}))


if __name__ == '__main__':
    main()
