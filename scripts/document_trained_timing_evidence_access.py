"""Publish complete audited evidence-access tables into the repository materials."""

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.trained_timing_evidence_access_data import MODELS,REGIMES


def document(audit,prefix):
    report=json.loads(audit.read_text());metrics=report['metrics']
    provenance=json.loads(prefix.with_suffix('.provenance.json').read_text())
    if (report['status']!='completed_and_audited_trained_timing_evidence_access' or
        report['new_physical_readouts_audited']!=5040 or provenance['audit_sha256']!=streaming_hash(audit) or
        any(report['prior_timing_criteria_unchanged'].values())):raise ValueError('Complete primary report and exact export required')
    for name,digest in provenance['output_sha256'].items():
        if streaming_hash(prefix.parent/name)!=digest:raise ValueError('Export bytes changed')
    folder=ROOT/'materials/updates/trained-timing-evidence-access-20261115'
    notes=folder/'pre-inference-notes.md'
    if notes.exists():raise ValueError('Preserve previous documentation generation')
    shutil.copyfile(folder/'README.md',notes)
    for p in prefix.parent.glob(prefix.name+'.*'):shutil.copyfile(p,folder/p.name)
    for p in (audit,audit.with_suffix('.source.py')):shutil.copyfile(p,folder/p.name)
    run=Path(report['source_run']);shutil.copyfile(run/'manifest.json',folder/'primary-manifest.json')
    labels=dict(zip(MODELS,['Native','Shared：训练all / 部署all','Shared：训练question / 部署all',
                           'Shared：训练question / 部署question','GDN：训练all / 部署all']))
    lines=['# 冻结历史状态的证据访问：完整诊断结果','',
        '**5040次正式读出及独立审计完成。** 固定5个既有配置、3种访问条件，保留全部48个关联案例、336个物理问题、432个关联绑定；没有训练、检查点选择或新环境。',
        '',
        '这次复用63000–63007的8个来源事件流，是解释性诊断。48个深度/表述变体及432个绑定相互关联，不能当作独立重复；不是新的未见队列确认。全部旧研究门槛仍失败，本轮没有架构成功阈值。',
        '',
        'full使用完整历史；oracle只允许问题/回答Attention访问程序真值推导的支持事件；nonslice使用与支持列不重叠、历史Token数完全相同的控制列。所有条件保留原始Token和位置、正常历史GDN更新、完整递归历史以及问题阶段新列。',
        '', '## 全部配置与条件','',
        '联合指标要求两种更新顺序下的当前值、历史值、无关值共6个回答全部正确。角色列为更新后关联绑定，物理问题列已去重。',
        '', '| 配置 | 条件 | 联合 /48 | 当前 /96 | 历史 /96 | 无关 /96 | 物理问题 /336 |',
        '|---|---|---:|---:|---:|---:|---:|']
    totals={}
    for name in MODELS:
        totals[name]={}
        for regime in REGIMES:
            s=metrics['models'][name][regime]
            counts=[sum(s['role_variant_counts'][role][v]['correct'] for v in ('control','treatment'))
                    for role in ('target_current','target_initial','protected_current')]
            lines.append(f"| {labels[name]} | {regime} | {s['joint_six_after']} | {counts[0]} | {counts[1]} | {counts[2]} | {s['physical_correct']} |")
            totals[name][regime]={'joint':s['joint_six_after'],'roles':counts,'physical':s['physical_correct']}
    lines+=['','## 完整深度分解','','| 配置 | 条件 | 深度1 /16 | 深度2 /16 | 深度3 /16 |','|---|---|---:|---:|---:|']
    for name in MODELS:
        for regime in REGIMES:
            s=metrics['models'][name][regime]
            values=[sum(c['joint_after'] for c in s['case_outcomes'] if c['dependency_depth']==d) for d in (1,2,3)]
            lines.append(f'| {labels[name]} | {regime} | '+ ' | '.join(map(str,values))+' |')
    lines+=['','## Oracle配对收益与退化','',
        '每一行都比较同模型、同问题；“修复/退化”按oracle相对参照条件计算。物理计数覆盖全部336题。NLL差为oracle减参照，负值表示金标损失更低。',
        '', '| 配置 | 参照 | 当前改善 /退化（96） | 联合改善 /退化（48） | 物理修复 /退化（336） | 当前平均NLL差 |',
        '|---|---|---:|---:|---:|---:|']
    for name in MODELS:
        for ref in ('full','nonslice'):
            c=metrics['oracle_paired_contrasts'][name][ref];current=c['current_target'];joint=c['joint_cases'];physical=c['physical']
            lines.append(f"| {labels[name]} | {ref} | {current['gains']} / {current['losses']} | {joint['gains']} / {joint['losses']} | {physical['repair']} / {physical['regression']} | {current['mean_nll_candidate_minus_reference']:.9g} |")
    lines+=['','## 解释边界与下一步','',
        'Oracle改善可以支持问题阶段的历史访问或干扰抑制对结果有作用；不能唯一归因于GDN状态受损、语义作用域或某种检索机制。正确证据掩码无效时，KV表示、问题阶段计算、注意力分配和干预分布偏移仍可能造成失败，不能直接断言模型不会组合关系。',
        '',
        'nonslice匹配Token预算，不保证完整事件语法。这里使用dense eager Attention加掩码，不是高效稀疏Kernel，也没有延迟收益或新Router有效性的结论。后续归因若涉及关系组合能力，必须单独登记文本证据与已执行状态等能力控制；本轮没有执行该控制。',
        '',
        '所有角色、深度、表述、更新量及来源流的逐问题修复/退化和NLL差保存在完整JSON/CSV。通用配对工具中的candidate_passed是继承的描述字段，不是本诊断的新成功门槛，不能改变旧判定。',
        '', '## 审计与复现材料','',
        f"两组相反顺序的270次安全运行严格一致，并在正式推理前完成独立审计。正式运行的1680次完整历史重放逐项匹配既有记录的五项字段。封存并核验166份运行源码；50组主要控制加10组深度2控制共核验120次实际掩码Attention及完整历史保留。question部署配置对全部56个历史前缀核验native状态、KV、卷积缓存、簿记及logits的一致性。主干和适配器权重未变，钩子恢复。", '',
        f'- [完整独立审计]({audit.name})',
        f'- [全部5040条原始答案与损失 CSV]({prefix.name}.csv)',
        f'- [全部960条当前目标配对NLL CSV]({prefix.name}.pairs.csv)',
        f'- [全部指标、分层与逐问题比较]({prefix.name}.data.json)',
        f'- [独立计数与CSV逐条核对]({prefix.name}.crosscheck.json)',
        f'- [PDF图表]({prefix.name}.pdf)',
        '- [推理前登记与覆盖修正说明](pre-inference-notes.md)',
        '- [正反顺序等价性审计](trained-timing-evidence-access-smoke-equivalence-20261115-01.json)',
        '- [原训练检查点快照](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases/tag/matched-shared-training-timing-2026-11-15)',
        '', f'![全部十五个配置与条件]({prefix.name}.png)',
        '', '本页由已审计的全部原始结果生成；完整归档与远端发布核验回执在完成后另行加入。']
    (folder/'README.md').write_text('\n'.join(lines)+'\n')
    p=ROOT/'configs/experiment_progress.json';progress=json.loads(p.read_text());progress['updated_utc']=datetime.now(timezone.utc).isoformat()
    progress['trained_timing_evidence_access'].update(status='completed_audited_exported_not_yet_remotely_archived',
        primary_audit_sha256=streaming_hash(audit),readouts_audited=5040,related_bindings_audited=6480,
        exact_full_replays=1680,all15_model_regime_counts=totals,prior_timing_criteria_unchanged=report['prior_timing_criteria_unchanged'],
        source_files_verified=166,export_prefix=str(prefix),composition_failure_attribution_allowed=False)
    p.write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
    p=ROOT/'README.md';p.write_text(p.read_text()+'\n后续完成：[冻结历史状态的证据访问诊断](materials/updates/trained-timing-evidence-access-20261115/README.md)。5040次正式读出、独立审计及全量导出完成；旧失败门槛保持原判，研究继续。\n')
    return {'status':'all15_model_regimes_documented','folder':str(folder),'counts':totals}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('audit',type=Path);parser.add_argument('--prefix',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(document(args.audit,args.prefix),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
