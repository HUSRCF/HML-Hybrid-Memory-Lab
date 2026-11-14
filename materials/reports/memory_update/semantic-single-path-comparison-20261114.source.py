"""Compare component-only adapters with the four audited augmented controls."""

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


MODELS = ["gdn_only", "attention_only", "shared", "independent", "local", "local_question"]
CONDITIONS = ["canonical", "permuted"]


def evaluated_rows(run, arm, condition, model_audit, expected):
    evaluation = run / arm / "step-000432" / ("evaluation" if condition == "canonical" else "initial_order_evaluation")
    if file_hash(evaluation / "summary.json") != model_audit[condition]["evaluation_summary_sha256"]:
        raise ValueError("Audited component/reference evaluation summary changed")
    summary = json.loads((evaluation / "summary.json").read_text())
    digest = summary["file_sha256"]["validation.jsonl"] if condition == "canonical" else summary["results_sha256"]
    if file_hash(evaluation / "validation.jsonl") != digest:
        raise ValueError("Audited free generations changed")
    return load_rows(evaluation / "validation.jsonl", expected)


def compare(audit):
    if audit["status"] != "completed_and_audited" or set(audit["arms"]) != {"gdn_only", "attention_only"}:
        raise ValueError("Complete independently audited GDN-only and Attention-only controls required")
    run = Path(audit["source_run"])
    if file_hash(run / "manifest.json") != audit["manifest_sha256"]:
        raise ValueError("Component-control manifest changed after audit")
    plan = json.loads((run / "plan.json").read_text())
    reference = Path(plan["augmentation_reference_run"])
    baseline = audit["augmentation_reference_audit"]
    if (baseline["status"] != "completed_and_audited" or file_hash(reference / "manifest.json") != baseline["manifest_sha256"] or
        set(baseline["arms"]) != {"shared", "independent", "local", "local_question"}):
        raise ValueError("Matched four-arm augmentation reference changed")
    data = json.loads((Path(plan["data_source"]) / "data.json").read_text())
    if data_fingerprint(data) != plan["data_sha256"]:
        raise ValueError("Audited component-control data changed")
    expected = {(example["case"]["id"], example["variant"], example["query"]["role"]): example["query"]["answer"] for example in data["validation"]}
    sources = {name: (run, audit["arms"][name]) if name in audit["arms"] else (reference, baseline["arms"][name]) for name in MODELS}
    rows = {condition: {name: evaluated_rows(path, name, condition, model, expected) for name,(path,model) in sources.items()} for condition in CONDITIONS}
    units = {condition: {name: functional_units(members) for name,members in models.items()} for condition,models in rows.items()}
    counts = {condition: {name: {metric:{"correct":sum(values.values()), "units":len(values)} for metric,values in model.items()} for name,model in models.items()} for condition,models in units.items()}
    pairs = [(scope, joint) for scope in ("gdn_only", "attention_only") for joint in ("shared", "independent", "local", "local_question")]
    comparisons = {condition:[{"reference":left, "candidate":right,
                              "metrics":{metric:paired_counts(units[condition][left][metric], units[condition][right][metric]) for metric in units[condition][left]}}
                             for left,right in pairs] for condition in CONDITIONS}
    changes = {name:{metric:paired_counts(units["canonical"][name][metric], units["permuted"][name][metric]) for metric in units["canonical"][name]} for name in MODELS}
    updates = {condition:{name:{family:{"correct":sum(row["greedy_exact"] for row in members if row["family"]==family and row["role"]=="target_current" and row["variant"]!="before"),
                                                  "examples":sum(row["family"]==family and row["role"]=="target_current" and row["variant"]!="before" for row in members)}
                                     for family in sorted({row["family"] for row in members})} for name,members in models.items()} for condition,models in rows.items()}
    return {"status":"completed_component_scope_comparison", "source_run":str(run), "manifest_sha256":audit["manifest_sha256"],
            "augmentation_reference_run":str(reference), "augmentation_manifest_sha256":baseline["manifest_sha256"],
            "counts":counts, "comparisons":comparisons, "canonical_to_permuted":changes, "target_updates_by_family":updates,
            "training_fit":{name:model["teacher_train_fit"] for name,(_,model) in sources.items()},
            "parameters":{name:model["parameters"] for name,(_,model) in sources.items()},
            "interpretation":plan["interpretation"] + " Six models share the same 12 underlying validation event streams. Paired counts are descriptive, not six independent datasets. Good component-only answers weaken a joint-adaptation necessity claim only for the tested group, training budget and tasks; poor answers can reflect fit or capacity."}


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels=["GDN only", "Attention only", "Shared", "Independent", "Local", "Question local"]
    colors=["#059669", "#ea580c", "#2563eb", "#7c3aed", "#64748b", "#0891b2"]
    metrics=[("individual", "Individual answers (432)"), ("stable_unchanged_case_role", "Unchanged fact across 3 variants (96)"),
             ("updated_target_except_add", "Updated target excluding ADD (80)")]
    figure,axes=plt.subplots(2,3,figsize=(14,9))
    for row,condition in enumerate(CONDITIONS):
        for column,(metric,title) in enumerate(metrics):
            axis=axes[row,column]
            cells=[report["counts"][condition][name][metric] for name in MODELS]
            bars=axis.bar(np.arange(len(MODELS)),[100*cell["correct"]/cell["units"] for cell in cells],.7,color=colors)
            for bar,cell in zip(bars,cells):
                axis.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,str(cell["correct"]),ha="center",va="bottom",fontsize=8)
            axis.set_ylim(0,110)
            axis.set_xticks(range(len(MODELS)),labels,rotation=30,ha="right",fontsize=8)
            axis.set_title(("Canonical: " if row==0 else "Permuted: ")+title,fontsize=10)
            axis.spines[["top","right"]].set_visible(False)
            if column==0:axis.set_ylabel("Required answers correct (%)")
    figure.suptitle("Component-specific adaptation on frozen Qwen3.5-2B IT: matched data and training budget",fontsize=13)
    figure.text(.5,.015,"One initialization, 12 base development streams; near-matched parameters. Parameter scope does not isolate a storage pathway.",ha="center",fontsize=9,color="#475569")
    figure.tight_layout(rect=(0,.065,1,.95))
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
    print(json.dumps({condition:{name:model['individual'] for name,model in models.items()} for condition,models in report['counts'].items()}))


if __name__ == '__main__':
    main()
