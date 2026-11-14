"""Export audited native cache-origin effects, with full/cached calibration."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sas_memory.calibration import file_hash
from scripts.analyze_controller_trial import load_rows
from scripts.run_controller_trial import write_json
from scripts.semantic_order_path_common import CONDITIONS, functional_report, nll_contrasts


def export(audit):
    if audit["status"] != "completed_and_audited" or audit["records_audited"] != 1728:
        raise ValueError("Require the independent audit of all four full native cohorts")
    run = Path(audit["source_run"])
    if file_hash(run / "manifest.json") != audit["manifest_sha256"]:
        raise ValueError("Native path manifest changed since audit")
    manifest = json.loads((run / "manifest.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    data = json.loads((Path(plan["canonical_data_source"]) / "data.json").read_text())
    expected = {(e["case"]["id"],e["variant"],e["query"]["role"]):e["query"]["answer"] for e in data["validation"]}
    rows = {}
    for name in CONDITIONS:
        path = run / f"{name}.jsonl"
        if file_hash(path) != manifest["results_sha256"][name]:
            raise ValueError("Audited native path free generations changed")
        rows[name] = load_rows(path,expected)
    functional, contrasts = functional_report(rows), nll_contrasts(rows)
    if functional != audit["functional"] or contrasts != audit["nll_source_contrasts"]:
        raise ValueError("Audited summaries differ from raw native path outcomes")
    for calibration in audit["full_prefill_calibration"].values():
        if file_hash(Path(calibration["full_prefill_source"])) != calibration["full_prefill_sha256"]:
            raise ValueError("Audited full-prefill reference generations changed")
    indexed = {(r["case_id"],r["variant"],r["role"]):r for r in rows["canonical_natural"]}
    by_base, by_cell = defaultdict(list), defaultdict(list)
    for effect in contrasts["paired_examples"]:
        row = indexed[tuple(effect["key"])]
        by_base[row["base_case_id"]].append(effect)
        by_cell[row["family"],row["role"]].append(effect)
    fields = list(contrasts["means"])
    clusters = [{"base_case_id":base,"readouts":len(effects),
                 "means":{field:mean(row[field] for row in effects) for field in fields}}
                for base,effects in sorted(by_base.items())]
    if len(clusters) != 12 or any(cluster["readouts"] != 36 for cluster in clusters):
        raise ValueError("Expected twelve complete underlying development streams")
    if any(abs(mean(cluster["means"][field] for cluster in clusters)-contrasts["means"][field]) > 1e-12 for field in fields):
        raise ValueError("Cluster averages no longer match the paired whole-cohort contrasts")
    cells = [{"family":family,"role":role,"readouts":len(effects),
              "gold_nll":{name:mean(row["gold_nll"][name] for row in effects) for name in CONDITIONS},
              "contrasts":{field:mean(row[field] for row in effects) for field in fields}}
             for (family,role),effects in sorted(by_cell.items())]
    role_counts = {name:{role:{"correct":sum(row["greedy_exact"] for row in members if row["role"] == role),
                              "examples":sum(row["role"] == role for row in members)}
                        for role in ("target_current","protected_current","target_initial")}
                   for name,members in rows.items()}
    return {"status":"completed_native_path_diagnostics_export","source_run":str(run),"manifest_sha256":audit["manifest_sha256"],
        "reader_policy":"full","model_path":manifest["model_path"],"packages":manifest["packages"],
        "conditions":CONDITIONS,"functional":functional,"role_counts":role_counts,
        "full_prefill_calibration":audit["full_prefill_calibration"],"mean_nll_contrasts":contrasts["means"],
        "base_stream_contrasts":clusters,"family_role_contrasts":cells,
        "interpretation":plan["interpretation"] + " This export is descriptive analysis of the complete registered cohort, including all families and roles. Gray plot points summarize 36 related readouts per underlying stream; they are not confidence intervals or 432 independent observations. Cache-source interactions are not semantic-complementarity or architecture-improvement estimates."}


def plot(report,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    figure,axes = plt.subplots(2,2,figsize=(15,10))
    calibration = report["full_prefill_calibration"]
    axis = axes[0,0]
    for offset,(field,label,color) in enumerate((("reference_correct","Old full prefill","#64748b"),("candidate_correct","New cached query","#2563eb"))):
        cells = [calibration[origin]["full_to_cached"]["individual"] for origin in ("canonical","permuted")]
        bars = axis.bar(np.arange(2)+(offset-.5)*.3,[100*cell[field]/cell["units"] for cell in cells],.3,label=label,color=color)
        for bar,cell in zip(bars,cells):
            axis.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,str(cell[field]),ha="center",fontsize=9)
    axis.set_xticks(range(2),["Canonical order","Permuted order"])
    axis.set_ylim(0,110)
    axis.set_ylabel("Answers correct (%)")
    axis.set_title("Natural baseline calibration (432 answers each)",fontsize=11)
    axis.legend(frameon=False,fontsize=9,loc="upper center",ncol=2)

    axis = axes[0,1]
    metrics = [("individual","Answers /432","#2563eb"),("stable_unchanged_case_role","Stable facts /96","#059669"),
               ("updated_target_except_add","Non-ADD updates /80","#d97706")]
    for offset,(metric,label,color) in enumerate(metrics):
        cells = [report["functional"]["counts"][name][metric] for name in CONDITIONS]
        bars = axis.bar(np.arange(4)+(offset-1)*.23,[100*cell["correct"]/cell["units"] for cell in cells],.23,label=label,color=color)
        for bar,cell in zip(bars,cells):
            axis.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,str(cell["correct"]),ha="center",fontsize=8)
    axis.axvspan(1.5,3.5,color="#64748b",alpha=.08,zorder=-1)
    axis.set_xticks(range(4),["C state / C KV\nNatural","P state / P KV\nNatural","C state / P KV\nMixed","P state / C KV\nMixed"],fontsize=9)
    axis.set_ylim(0,110)
    axis.set_title("Matched cached readouts; shaded groups are counterfactual",fontsize=11)
    axis.legend(frameon=False,fontsize=8,loc="upper center",ncol=3)

    axis = axes[1,0]
    families = sorted({row["family"] for row in report["family_role_contrasts"]})
    roles = ["target_current","protected_current","target_initial"]
    indexed = {(row["family"],row["role"]):row for row in report["family_role_contrasts"]}
    values = np.array([[indexed[family,role]["contrasts"]["interaction_difference_in_differences"] for role in roles] for family in families])
    bound = max(float(np.abs(values).max()),1e-8)
    drawing = axis.imshow(values,cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect="auto")
    axis.set_xticks(range(3),["Current target","Protected fact","Initial target"],fontsize=9)
    axis.set_yticks(range(len(families)),[family.replace('_',' ') for family in families],fontsize=9)
    axis.set_title("Cache-origin interaction by family / role (gold NLL)",fontsize=11)
    for row in range(len(families)):
        for column in range(3):
            axis.text(column,row,f"{values[row,column]:.3f}",ha="center",va="center",fontsize=9,
                      color="white" if abs(values[row,column]) > .6*bound else "#0f172a")
    figure.colorbar(drawing,ax=axis,fraction=.035,pad=.03)

    axis = axes[1,1]
    fields = list(report["mean_nll_contrasts"])
    labels = ["Recurrent source\nat C KV","Recurrent source\nat P KV","KV source\nat C state","KV source\nat P state","Interaction\nDiD"]
    for number,field in enumerate(fields):
        values = [row["means"][field] for row in report["base_stream_contrasts"]]
        axis.scatter(number+np.linspace(-.12,.12,len(values)),values,s=22,color="#64748b",alpha=.7)
        axis.plot([number-.17,number+.17],[report["mean_nll_contrasts"][field]]*2,color="#dc2626",linewidth=2.5)
    axis.axhline(0,color="#334155",linewidth=.8,linestyle="--")
    axis.set_xticks(range(5),labels,fontsize=8)
    axis.set_ylabel("Change in gold-answer NLL")
    axis.set_title("Source contrasts across twelve base streams",fontsize=11)
    for axis in axes.flat:
        axis.spines[["top","right"]].set_visible(False)
    figure.suptitle("Frozen Qwen3.5-2B IT: native recurrent/conv and KV source diagnostic",fontsize=14)
    figure.text(.5,.037,"C = canonical; P = permuted. Source effects = P - C; DiD = (PP - CP) - (PC - CC). Gray = stream means; red = cohort means.",ha="center",fontsize=9,color="#475569")
    figure.text(.5,.016,"Twelve reused development streams; mixed caches can be out of distribution. Effects do not establish write damage or semantic complementarity.",ha="center",fontsize=9,color="#475569")
    figure.tight_layout(rect=(0,.075,1,.95))
    output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(output,dpi=200,bbox_inches="tight")
    figure.savefig(output.with_suffix('.pdf'),bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--plot",type=Path)
    args = parser.parse_args()
    report = export(json.loads(args.audit.read_text()))
    report.update(audit_report_sha256=file_hash(args.audit),analysis_source_sha256=file_hash(Path(__file__)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    if args.plot:
        plot(report,args.plot)
    print(json.dumps({name:metrics['individual'] for name,metrics in report['functional']['counts'].items()}))


if __name__ == '__main__':
    main()
