"""Render all audited budget cells and contrasts, without selecting favorable results."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(lines, headers, rows):
    lines.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|'])
    lines.extend('| '+' | '.join(str(x) for x in row)+' |' for row in rows)
    lines.append('')


def report(audit, exported, output):
    suffixes=('.md','.strata.csv','.source.py','.provenance.json')
    if any(output.with_suffix(s).exists() for s in suffixes):
        raise ValueError('Preserve prior report outputs')
    a=json.loads(audit.read_text());data=json.loads(exported.read_text())
    run=Path(a['source_run']);manifest=json.loads((run/'manifest.json').read_text());plan=json.loads((run/'plan.json').read_text())
    if (a['status']!='completed_and_audited_update_budget' or manifest['status']!='completed' or
        a['manifest_sha256']!=sha(run/'manifest.json') or a['new_physical_readouts_audited']!=13440 or
        data['new_primary_readouts']!=13440 or data['archived_reference_readouts']!=6720 or
        data['csv_rows_verified']!={'main':20160,'paired':9600,'factorial':1680} or
        any(a['prior_timing_criteria_unchanged'].values())):
        raise ValueError('Complete audited primary and independently verified raw export required')
    m=a['metrics'];names=plan['model_order'];regimes=list(m['measurement_origin'])
    if len(names)!=5 or len(regimes)!=12:raise ValueError('All60cells required')
    lines=['# 完整记录写入预算控制：全部审计结果','',
        '本报告由完整审计和逐字段核验的导出数据生成。共13440次新物理读出，加6720项历史参考；统一表20160项，复用8条事件流、48个相关案例。',
        '', '全部六个Attention层保留。控制只改变一条SET的身份；COPY路径已由Oracle类型预算决定。预算仍有后见信息，不是一个对所有问题共用的在线状态；旧科学失败判定不变。','',
        '六项联合为两种顺序×当前、历史、无关三个问题全部正确，分母48；三个角色分别为96项相关绑定，物理读出分母336。联合答案可能来自不同的后见前缀状态。','']
    strata=[]
    for n in names:
        lines.extend(['## '+n,''])
        rows=[]
        for r in regimes:
            s=m['models'][n][r];c=data['counts'][n][r]
            roles={k:sum(s['role_variant_counts'][k][v]['correct'] for v in ('control','treatment'))
                for k in ('target_current','target_initial','protected_current')}
            if (c['joint']!=s['joint_six_after'] or c['physical']!=s['physical_correct'] or c['roles']!=roles):
                raise ValueError('Displayed counts differ from independent raw export')
            rows.append([r,'新测量' if m['measurement_origin'][r]=='new_primary' else '历史参考',s['joint_six_after'],
                roles['target_current'],roles['target_initial'],roles['protected_current'],s['physical_correct'],
                '/'.join(str(c['joint_by_depth'][str(d)]) for d in (1,2,3))])
            cases=s['case_outcomes']
            for factor in ('base_case_id','dependency_depth','question_style','distractor_updates'):
                groups=[]
                for value in sorted({x[factor] for x in cases}):
                    g=[x for x in cases if x[factor]==value]
                    row={'model':n,'regime':r,'measurement_origin':m['measurement_origin'][r],
                        'factor':factor,'value':value,'cases':len(g),'joint_correct':sum(x['joint_after'] for x in g),
                        'both_order_targets_correct':sum(x['both_order_targets_correct'] for x in g),
                        'before_all_correct':sum(x['before_all_correct'] for x in g)}
                    strata.append(row);groups.append(row)
                if sum(x['cases'] for x in groups)!=48 or sum(x['joint_correct'] for x in groups)!=s['joint_six_after']:
                    raise ValueError('Every stratification must retain all48related cases')
        table(lines,['条件','来源','联合/48','当前/96','历史/96','无关/96','物理/336','深度1/2/3各/16'],rows)
    lines.extend(['## 全部预定配对比较','',
        '增益／退化为同一题或同一相关案例上的正确性变化，不能将相关绑定或分层当成独立样本。当前列分母96，联合48，物理336；损失差为候选减参考。',''])
    for n in names:
        rows=[]
        for candidate,reference in plan['contrasts']:
            p=m['paired_contrasts'][n][candidate+'_vs_'+reference]
            current,joint,physical=p['current_target'],p['joint_cases'],p['physical']
            rows.append([candidate+' vs '+reference,f"{current['gains']}/{current['losses']}",
                f"{joint['gains']}/{joint['losses']}",f"{physical['repair']}/{physical['regression']}",
                format(current['mean_nll_candidate_minus_reference'],'.9g')])
        lines.extend(['### '+n,''])
        table(lines,['候选 vs 参考','当前增益/退化','联合增益/退化','物理增益/退化','当前平均NLL差'],rows)
    lines.extend(['## 全部描述性损失交互','',
        '`L(f_oracle)-L(f_full)-L(delta_oracle)+L(delta_full)`；平均值仅作描述，没有新增显著性检验、科学成功阈值或架构有效声明。全部1680个物理问题向量及原始NLL另存于factorial.csv。',''])
    rows=[]
    for n in names:
        for family in ('random0','random1','early','recent'):
            d=m['loss_interactions'][n]
            rows.append([n,family,format(d['physical_means'][family],'.9g'),format(d['current_binding_means'][family],'.9g')])
    table(lines,['配置','写入族','336物理问题均值','96当前绑定均值'],rows)
    lines.extend(['## 完整性与边界','',
        '241项测试、两轮相反模型顺序的各720次smoke、240项完整前缀缓存检查通过。197份正式运行源码、实际18层GDN门控、六层Attention观测、原生权重和固定step432适配检查点均由原始审计核验。原理想FP32内核校验失败及其nativeBF16参考修正继续保留，未更改容差或原生内核。',
        '', '随机策略与原支持写入偶然完全重合的题数为6和8，最早128、最近48，均保留；交叉两种读取和五配置时，必须精确重放1900项旧五字段读出。',
        '', '类型和Token数相同不代表写入能量、全部位置或状态轨迹相同。Oracle读取固定为原支持记录；不同问题可使用不同预算，COPY选择已由预算揭示。本轮不能证明语义控制器学会依赖链、GDN自然损坏，或新混合架构有效。',
        '', '全部角色/顺序/深度/文体的逐项计数见完整审计；事件流、深度、文体和干扰负载的联合分层见strata.csv。所有100组比较、对应逐题退化和分层均保留，没有从结果选择模型或控制族。',''])
    output.parent.mkdir(parents=True,exist_ok=True);output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    p=output.with_suffix('.strata.csv')
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(strata[0]));w.writeheader();w.writerows(strata)
    with p.open(newline='') as f:reloaded=list(csv.DictReader(f))
    if any(any(b[k]!=str(v) for k,v in a.items()) for a,b in zip(strata,reloaded,strict=True)):
        raise ValueError('Every stratification CSV field must roundtrip exactly')
    output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    receipt={'status':'all60cells_100contrasts_reported','created_utc':datetime.now(timezone.utc).isoformat(),
        'primary_audit_sha256':sha(audit),'raw_export_data_sha256':sha(exported),'source_sha256':sha(Path(__file__)),
        'cells':60,'paired_comparisons':100,'case_strata_csv_rows':len(strata),'all_strata_cover48related_cases':True,
        'outputs_sha256':{output.with_suffix(s).name:sha(output.with_suffix(s)) for s in suffixes if s!='.provenance.json'}}
    output.with_suffix('.provenance.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--exported',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(report(a.audit,a.exported,a.output)),flush=True)


if __name__=='__main__':main()
