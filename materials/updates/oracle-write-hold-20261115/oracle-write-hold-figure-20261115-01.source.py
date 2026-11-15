"""Publish all20cells and independently crosscheck every exported numeric row."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.oracle_write_hold_data import MODELS,REGIMES,DESTINATION,registered_inputs
from scripts.analyze_oracle_write_hold import audit_run
from scripts.run_controller_trial import write_json


def export(audit,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    report=json.loads(audit.read_text());run=Path(report['source_run']);manifest=json.loads((run/'manifest.json').read_text())
    if report['status']!='completed_and_audited_oracle_write_hold' or audit_run(run)!=report:
        raise ValueError('Complete unchanged full-cohort audit required')
    _,cases,registry,queries,policies,_=registered_inputs()
    rows={n:{r:[json.loads(s) for s in (run/f'{n}.{r}.jsonl').read_text().splitlines()] for r in REGIMES} for n in MODELS}
    indexed={n:{r:{v['id']:v for v in cell} for r,cell in cells.items()} for n,cells in rows.items()}
    metrics=report['metrics'];counts={};interactions={}
    for n in MODELS:
        counts[n]={}
        for r in REGIMES:
            bound={(b['case_id'],b['variant'],b['role']):indexed[n][r][b['physical_id']] for b in registry['bindings']}
            joint={c['id']:all(bound[c['id'],v,role]['greedy_exact'] for v in ('control','treatment')
                for role in ('target_current','target_initial','protected_current')) for c in cases}
            roles={role:sum(bound[c['id'],v,role]['greedy_exact'] for c in cases for v in ('control','treatment'))
                for role in ('target_current','target_initial','protected_current')}
            counts[n][r]={'joint':sum(joint.values()),'roles':roles,'physical':sum(v['greedy_exact'] for v in rows[n][r]),
                'joint_by_depth':{str(d):sum(joint[c['id']] for c in cases if c['dependency_depth']==d) for d in (1,2,3)}}
            s=metrics['models'][n][r]
            if (counts[n][r]['joint']!=s['joint_six_after'] or counts[n][r]['physical']!=s['physical_correct'] or
                any(roles[role]!=sum(s['role_variant_counts'][role][v]['correct'] for v in ('control','treatment')) for role in roles) or
                any(joint[c['case_id']]!=c['joint_after'] for c in s['case_outcomes'])):raise ValueError('Independent raw role/joint/task counts disagree')
        delta=lambda key:indexed[n]['hold_oracle'][key]['answer_nll']-indexed[n]['hold'][key]['answer_nll']-indexed[n]['oracle'][key]['answer_nll']+indexed[n]['full'][key]['answer_nll']
        interactions[n]={'physical_mean':mean(delta(q['id']) for q in registry['physical']),
            'current_binding_mean':mean(delta(b['physical_id']) for b in registry['bindings'] if b['role']=='target_current' and b['variant']!='before')}
        if interactions[n]['physical_mean']!=metrics['loss_interaction'][n]['physical_mean']:raise ValueError('Raw loss interaction disagrees')
    suffixes=('.csv','.pairs.csv','.factorial.csv','.png','.pdf','.data.json','.source.py','.provenance.json')
    if any(prefix.with_suffix(s).exists() for s in suffixes):raise ValueError('Preserve prior exports')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    main_fields=['model','regime','id','token_policy','reader_policy','prompt_ids_sha256','held_token_count','answer','generated','greedy_exact','answer_nll','query_tokens']
    main=[]
    for n in MODELS:
        for r in REGIMES:
            for row in rows[n][r]:main.append({**{k:row[k] for k in main_fields if k!='held_token_count'},'held_token_count':policies[row['id']]['held_token_count'] if r in ('hold','hold_oracle') else 0})
    contrasts=[('hold','full'),('hold_oracle','oracle'),('oracle','full'),('hold_oracle','hold')]
    pairs=[];factorial=[]
    for n in MODELS:
        for candidate,reference in contrasts:
            for b in registry['bindings']:
                if b['variant']=='before' or b['role']!='target_current':continue
                c=indexed[n][candidate][b['physical_id']];r=indexed[n][reference][b['physical_id']]
                pairs.append({'model':n,'candidate':candidate,'reference':reference,**b,
                    'candidate_correct':c['greedy_exact'],'reference_correct':r['greedy_exact'],
                    'gain':c['greedy_exact'] and not r['greedy_exact'],'loss':r['greedy_exact'] and not c['greedy_exact'],
                    'nll_candidate_minus_reference':c['answer_nll']-r['answer_nll']})
        for q in registry['physical']:
            key=q['id'];values={r:indexed[n][r][key]['answer_nll'] for r in REGIMES}
            factorial.append({'model':n,'id':key,**{f'nll_{r}':v for r,v in values.items()},
                'nll_interaction':values['hold_oracle']-values['hold']-values['oracle']+values['full']})
        if [{'id':v['id'],'nll_interaction':v['nll_interaction']} for v in factorial if v['model']==n]!=metrics['loss_interaction'][n]['all_physical_values']:
            raise ValueError('Every independent raw factorial contrast must equal its audited value')
    for suffix,data in [('.csv',main),('.pairs.csv',pairs),('.factorial.csv',factorial)]:
        fields=list(data[0])
        with prefix.with_suffix(suffix).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(data)
        with prefix.with_suffix(suffix).open(newline='') as f:reread=list(csv.DictReader(f))
        for expected,actual in zip(data,reread,strict=True):
            for key,v in expected.items():
                if actual[key]!=str(v):raise ValueError('CSV field/string roundtrip changed')
                if isinstance(v,float) and float(actual[key])!=v:raise ValueError('CSV numeric roundtrip changed')
    if [len(main),len(pairs),len(factorial)]!=[6720,1920,1680]:raise ValueError('Complete export coverage changed')
    # Independently check each pair against the sealed audit, retaining every regression.
    for n in MODELS:
        for candidate,reference in contrasts:
            actual=[p for p in pairs if (p['model'],p['candidate'],p['reference'])==(n,candidate,reference)]
            expected=metrics['paired_contrasts'][n][candidate+'_vs_'+reference]['current_target']
            if (sum(p['gain'] for p in actual)!=expected['gains'] or sum(p['loss'] for p in actual)!=expected['losses'] or
                [{k:p[k] for k in expected['individual_pairs'][0]} for p in actual]!=expected['individual_pairs']):
                raise ValueError('Independent all-pair raw comparison disagrees')
    names=['Native','Shared all / all','Shared question / all','Shared question / question','GDN adapters all / all']
    cells=[(n,r) for n in MODELS for r in REGIMES];labels=[f'{name} | {r}' for name in names for r in REGIMES]
    colors=dict(zip(REGIMES,['#778da0','#d39337','#2b8a73','#8266ad']))
    fig,axes=plt.subplots(1,3,figsize=(16,11))
    for ax,field,denom,title in zip(axes,['joint','current','physical'],[48,96,336],
        ['All six post-update answers','Current target after update','All original questions']):
        values=[counts[n][r]['roles']['target_current'] if field=='current' else counts[n][r][field] for n,r in cells]
        ax.barh(np.arange(20),values,color=[colors[r] for n,r in cells]);ax.invert_yaxis()
        ax.set(yticks=np.arange(20),yticklabels=labels if ax is axes[0] else [],xlim=(0,denom*1.12),title=title,xlabel=f'Correct / {denom}')
        for i,v in enumerate(values):ax.text(v+denom*.015,i,str(v),va='center',fontsize=8)
        for p in (3.5,7.5,11.5,15.5):ax.axhline(p,color='#ddd',lw=.7)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Fixed-input recurrent-hold controls — all five configurations and four conditions',fontsize=14)
    fig.text(.01,.052,'All 6720 readouts; reused 8 source streams. Token IDs / positions fixed. GDN labels describe adapter scope; all six Attention layers remain.',fontsize=9)
    fig.text(.01,.024,'Query-dependent support labels are privileged. Convolution / GDN reads continue; downstream KV can change. No learned-method or architecture claim.',fontsize=9)
    fig.tight_layout(rect=[0,.085,1,.94]);fig.savefig(prefix.with_suffix('.png'),dpi=180);fig.savefig(prefix.with_suffix('.pdf'));plt.close(fig)
    write_json(prefix.with_suffix('.data.json'),{'counts':counts,'loss_interactions':interactions,'csv_rows_verified':{'main':6720,'paired':1920,'factorial':1680}})
    prefix.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    write_json(prefix.with_suffix('.provenance.json'),{'source_run':str(run),'audit_sha256':streaming_hash(audit),
        'manifest_sha256':streaming_hash(run/'manifest.json'),'source_sha256':streaming_hash(Path(__file__)),
        'all_outputs_sha256':{prefix.with_suffix(s).name:streaming_hash(prefix.with_suffix(s)) for s in suffixes if s!='.provenance.json'}})
    dest=ROOT/'materials/updates/oracle-write-hold-20261115';original=dest/'pre-inference-README.md'
    if not original.exists():original.write_bytes((dest/'README.md').read_bytes())
    lines=['# 固定原始输入的递归更新阻断诊断','',
        '完整6720次读出已完成并独立审计。保持原始Token、位置和问题边界，交叉无关记录递归更新阻断与查询阶段Oracle读取；全部五个配置、三种深度和逐题退化均保留。',
        '', '| 配置 | 条件 | 联合 /48 | 当前 /96 | 历史 /96 | 无关 /96 | 全题 /336 | 深度1/2/3（各16） |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for n,r in cells:
        s=counts[n][r];role=s['roles'];depth='/'.join(str(s['joint_by_depth'][str(d)]) for d in (1,2,3))
        lines.append(f"| {n} | {r} | {s['joint']} | {role['target_current']} | {role['target_initial']} | {role['protected_current']} | {s['physical']} | {depth} |")
    lines+=['','`full`：原始更新/完整读取；`hold`：阻断非支持记录更新/完整读取；`oracle`：原始更新/支持读取；`hold_oracle`：阻断更新/支持读取。',
        '', '## 配对的当前答案修复与退化','', '| 配置 | 对照 | 修复 | 退化 |', '| --- | --- | ---: | ---: |']
    for n in MODELS:
        for c,r in contrasts:
            s=metrics['paired_contrasts'][n][c+'_vs_'+r]['current_target'];lines.append(f"| {n} | {c} − {r} | {s['gains']} | {s['losses']} |")
    lines+=['','## 解释边界','',
        '阻断18层GDN在原始非支持记录位置的衰减与写入；短卷积、GDN读出、MLP、残差及全部六层Attention继续执行。GDN-only仅指适配器路径，不代表移除Attention。',
        '', '支持标签来自后续问题的真实依赖，具有后见信息；结果不是可部署的在线更新方法。改变递归更新也会影响下游隐藏表示和KV，不能唯一解释成自然状态损伤或提出新架构的充分理由。',
        '', '来源只有8条事件流；48个变体、432个绑定存在相关性。全部旧架构/时序失败门槛保持失败；本轮无新成功阈值、不训练、不扩模。两项干预的损失交互仅作描述。',
        '', '全部源码、原始读出、预登记、反向顺序smoke、状态/门控/Attention/缓存证明及导出表随本轮快照备份。预登记内容见[原始说明](pre-inference-README.md)。',
        '', f'![全部配置与条件]({prefix.name}.png)','']
    (dest/'README.md').write_text('\n'.join(lines))
    print(json.dumps({'status':'all20cells_exported','counts':counts,'csv_rows_verified':[6720,1920,1680]},ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--prefix',type=Path,required=True);a=p.parse_args();export(a.audit,a.prefix)

if __name__=='__main__':main()
