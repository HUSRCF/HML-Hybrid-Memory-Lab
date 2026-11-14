"""Write complete factorial evidence tables from a verified primary export."""

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.text_competence_factorial_data import MODELS,REGIMES


def document(audit,prefix):
    report=json.loads(audit.read_text());metrics=report['metrics'];provenance=json.loads(prefix.with_suffix('.provenance.json').read_text())
    if (report['status']!='completed_and_audited_text_competence_factorial' or report['new_physical_readouts_audited']!=6720 or
        provenance['audit_sha256']!=streaming_hash(audit) or any(report['prior_timing_criteria_unchanged'].values())):
        raise ValueError('Complete independently audited result/export and unchanged failed criteria required')
    for name,digest in provenance['output_sha256'].items():
        if streaming_hash(prefix.parent/name)!=digest:raise ValueError('Successful export bytes changed')
    folder=ROOT/'materials/updates/text-competence-factorial-20261115';notes=folder/'pre-inference-notes.md'
    if notes.exists():raise ValueError('Preserve earlier documentation generation')
    shutil.copyfile(folder/'README.md',notes)
    for p in prefix.parent.glob(prefix.name+'.*'):shutil.copyfile(p,folder/p.name)
    for p in (audit,audit.with_suffix('.source.py')):shutil.copyfile(p,folder/p.name)
    run=Path(report['source_run']);shutil.copyfile(run/'manifest.json',folder/'primary-manifest.json')
    labels=dict(zip(MODELS,['Native','Shared训练all/部署all','Shared训练question/部署all',
        'Shared训练question/部署question','GDN训练all/部署all']))
    lines=['# 文本能力交叉对照：全部6720次正式结果','',
        '**6720次正式读出、独立审计与全量导出完成。** 固定五个既有配置、全部336个原物理问题和432个关联绑定，交叉比较历史范围与事件执行形式。',
        '',
        '复用63000–63007的8个来源流、48个相关案例；没有新训练、新环境或模型选择。各条件改变文本、Token数、位置和历史状态计算，属于特权能力控制，不是冻结历史的因果定位或新架构验证。旧失败门槛保持原判。',
        '',
        '命名范围：五个配置都使用相同的混合主干，六个Attention层始终保留。“GDN-only”指适配器只作用于GDN路径，不是移除Attention后的纯递归模型；该配置的成绩不能证明GDN状态单独包含或输出了答案。',
        '',
        '- full：完整原事件。',
        '- support_text：程序按当前问题反向切片后的完整支持事件。',
        '- compiled_full：完整历史中逐步将COPY改写成该时刻实际执行值的SET。',
        '- compiled_support：支持事件中的COPY逐步改写为SET。',
        '',
        '编译后的每个中间步骤，所有实体的当前值与初始值均与相应原程序相同；不把问题金标附加到提示中。指令、问题表述、分词器、原始无限制答案生成与停止条件固定。',
        '', '## 全部条件与输入去重','',
        '联合指标要求两种更新顺序下的当前、历史与无关值共6个答案全部正确。原任务ID仍用于配对；文本变换后重复输入不构成独立证据。full与compiled_full各336种实际输入，support_text与compiled_support各144种。',
        '', '| 配置 | 条件 | 联合 /48 | 当前 /96 | 历史 /96 | 无关 /96 | 原任务 /336 | 去重输入正确 /总数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    counts={}
    for name in MODELS:
        counts[name]={}
        for r in REGIMES:
            s=metrics['models'][name][r];dedup=metrics['deduplicated_actual_input_counts'][name][r]
            roles=[sum(s['role_variant_counts'][role][v]['correct'] for v in ('control','treatment')) for role in
                   ('target_current','target_initial','protected_current')]
            lines.append(f"| {labels[name]} | {r} | {s['joint_six_after']} | {roles[0]} | {roles[1]} | {roles[2]} | {s['physical_correct']} | {dedup['correct']} / {dedup['unique_inputs']} |")
            counts[name][r]={'joint':s['joint_six_after'],'roles':roles,'original_tasks':s['physical_correct'],'deduplicated':dedup}
    lines+=['','## 全部深度','', '| 配置 | 条件 | 深度1 /16 | 深度2 /16 | 深度3 /16 |','|---|---|---:|---:|---:|']
    for name in MODELS:
        for r in REGIMES:
            values=[sum(c['joint_after'] for c in metrics['models'][name][r]['case_outcomes'] if c['dependency_depth']==d) for d in (1,2,3)]
            lines.append(f'| {labels[name]} | {r} | '+' | '.join(map(str,values))+' |')
    lines+=['','## 全部配对变换效应','',
        '候选相对参照的改善与退化均保留。当前NLL差为候选减参照，负值表示金标损失降低。',
        '', '| 配置 | 候选 | 参照 | 当前改善 /退化 | 联合改善 /退化 | 原任务修复 /退化 | 当前平均NLL差 |',
        '|---|---|---|---:|---:|---:|---:|']
    for name in MODELS:
        for c in metrics['paired_contrasts'][name]:
            a=c['current_target'];b=c['joint_cases'];p=c['physical']
            lines.append(f"| {labels[name]} | {c['candidate']} | {c['reference']} | {a['gains']} / {a['losses']} | {b['gains']} / {b['losses']} | {p['repair']} / {p['regression']} | {a['mean_nll_candidate_minus_reference']:.9g} |")
    lines+=['','## 描述性的交叉NLL项','',
        '`NLL(compiled_support) - NLL(support_text) - NLL(compiled_full) + NLL(full)`逐问题计算。它描述两种文本变换在当前损失尺度上的非加性，不唯一识别检索、组合、GDN损伤或记忆互补机制。',
        '', '| 配置 | 全部336原任务平均值 | 当前目标96绑定平均值 |','|---|---:|---:|']
    for name in MODELS:
        v=metrics['factorial_nll_interactions'][name]
        lines.append(f"| {labels[name]} | {v['mean_physical_nll_interaction']:.9g} | {v['mean_current_nll_interaction']:.9g} |")
    lines+=['','## 审计与解释限制','',
        f"两组相反顺序的360次安全运行、独立审计及精确等价性在正式推理前完成。全部1680次full重放与前轮五字段精确一致；核验172份封存源码、48个独立native锚点、240次启用前native复测、192次attached-none、240次移除后复测和240组六层Attention观察。question配置核验{report['question_only_native_prefix_checks']}个条件内的完整历史前缀，原始full前缀也与既有记录匹配。权重未变，钩子已恢复。",'',
        '支持事件由查询特定的程序Oracle选择，编译由解释器直接执行；成功不说明LLM能自己学习这两个过程。即使compiled条件改善，也可能来自更短位置距离、显式数值、更简单文本或输入分布改变，不能唯一归因于关系组合。变换后失败也不能证明组合能力缺失或递归记忆损伤。',
        '',
        '所有条件、角色、来源流、深度、表述、更新量及逐问题退化均在JSON/CSV中；不把相关变体或重复输入当成独立样本。通用配对工具中的candidate_passed是继承的描述字段，不是本阶段成功门槛，不改变旧失败判定。',
        '', '## 完整材料','',
        f'- [独立审计]({audit.name})',f'- [全部6720条原始任务读出]({prefix.name}.csv)',
        f'- [全部1920条当前目标配对NLL]({prefix.name}.pairs.csv)',f'- [全部1680条逐原任务交叉NLL]({prefix.name}.factorial.csv)',
        f'- [全部指标、分层、重复输入与逐问题比较]({prefix.name}.data.json)',
        f'- [独立计数与逐条CSV核验]({prefix.name}.crosscheck.json)',f'- [PDF图表]({prefix.name}.pdf)',
        '- [输入碰撞登记](input-collisions.json)','- [推理前登记与验证记录](pre-inference-notes.md)',
        '- [独立程序解释器对已保存文本及全部别名绑定的核验](text-competence-factorial-truth-crosscheck-20261115-01.json)',
        '',f'![全部二十个配置与条件]({prefix.name}.png)','',
        '完整归档与远端发布回执将在完成后另行加入。']
    (folder/'README.md').write_text('\n'.join(lines)+'\n')
    p=ROOT/'configs/experiment_progress.json';d=json.loads(p.read_text());d['updated_utc']=datetime.now(timezone.utc).isoformat()
    d['text_competence_factorial'].update(status='complete_audited_exported_pending_snapshot',primary_audit_sha256=streaming_hash(audit),
        readouts_audited=6720,related_bindings_audited=8640,all20_counts=counts,export_prefix=str(prefix),
        prior_timing_criteria_unchanged=report['prior_timing_criteria_unchanged'],architecture_effectiveness_claim=False)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    return {'status':'all_twenty_cells_documented','folder':str(folder),'counts':counts}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path);p.add_argument('--prefix',type=Path,required=True)
    a=p.parse_args();print(json.dumps(document(a.audit,a.prefix),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
