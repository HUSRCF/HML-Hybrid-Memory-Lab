"""Export every audited factorial cell with explicit new/reference/replay origins."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_post_tail_path import audit_run
from scripts.post_tail_path_data import registered_inputs,read_plan,MODELS,CONDITIONS,PRIMARY,READERS
from scripts.post_encoding_tasks import ROLES
from scripts.run_controller_trial import write_json


def csv_file(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with path.open(newline='') as handle:
        if len(list(csv.DictReader(handle)))!=len(rows):raise ValueError('Exported CSV row count changed')


def flatten(value):
    return {k:v for k,v in value.items() if not isinstance(v,dict)}|{
        k+'_'+field:val for k,v in value.items() if isinstance(v,dict) for field,val in v.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite exports')
    audit=json.loads(a.audit.read_text());run=Path(audit['source_run']);plan=read_plan()
    if audit['status']!='completed_and_audited_post_tail_path' or audit_run(run)!=audit:raise ValueError('Complete independent primary audit required')
    registered,cases,registry,queries,endpoints,refs=registered_inputs();metrics=audit['metrics'];a.output.mkdir(parents=True,exist_ok=False)
    raw=[];replays=[];cells=[];strata=[];episodes=[];contrasts=[];interactions=[]
    def readout_row(row,model,label,origin):
        q=queries[row['id']]
        return {'model':model,'condition':label,'measurement_origin':origin,**{k:row[k] for k in
            ('id','reader','answer','generated','greedy_exact','answer_nll','query_tokens')},
            'memory_identity':row.get('memory_identity',''),**{k:q[k] for k in
            ('seed','episode','role','question_style','prefix_evidence','support_event_index','question','source_key')}}
    for model in MODELS:
        for line in (run/f'{model}.primary.jsonl').read_text().splitlines():
            row=json.loads(line);raw.append(readout_row(row,model,row['condition'],'new_primary'))
        for label,rows in endpoints[model].items():
            raw.extend(readout_row(row,model,label,'archived_endpoint_reference') for row in rows)
        for line in (run/f'{model}.endpoints.jsonl').read_text().splitlines():
            row=json.loads(line);replays.append(readout_row(row,model,row['condition'],'endpoint_operational_replay'))
        for cell,values in metrics['cells'][model].items():
            label,reader=cell.split('.');role=metrics['strata'][model][cell]['role']
            cells.append({'model':model,'condition':label,'reader':reader,**flatten(values),
                **{r+'_correct':role[r]['greedy_exact'] for r in ROLES}})
            for kind,groups in metrics['strata'][model][cell].items():
                for group,v in groups.items():strata.append({'model':model,'cell':cell,'stratum':kind,'group':group,**v})
            for episode,groups in metrics['episodes'][model][cell].items():
                episodes.append({'model':model,'cell':cell,'episode':episode,**flatten(groups['all']),
                    **{r+'_correct':groups['role'][r]['greedy_exact'] for r in ROLES}})
        for name,groups in metrics['paired_contrasts'][model].items():
            for group,v in groups.items():
                if group in ('by_role','role'):
                    for role,values in v.items():contrasts.append({'model':model,'contrast':name,'stratum':role,**values})
                elif isinstance(v,dict):contrasts.append({'model':model,'contrast':name,'stratum':group,**v})
        for reader,values in metrics['nll_factorial_interactions'][model].items():
            for row in values['all_query_values']:
                q=queries[row['id']];interactions.append({'model':model,'reader':reader,**row,
                    **{k:q[k] for k in ('seed','episode','role','question_style','prefix_evidence')}})
    if (len(raw),len(replays),len(cells),len(episodes),len(strata),len(interactions))!=(6144,1536,32,128,480,768):
        raise ValueError('Complete unfiltered factorial exports required')
    tables={'readouts':raw,'endpoint-replays':replays,'cells':cells,'episodes':episodes,'strata':strata,'contrasts':contrasts,'nll-interactions':interactions}
    for name,rows in tables.items():csv_file(a.output/(name+'.csv'),rows)
    write_json(a.output/'metrics.json',metrics)
    lines=['# 尾部后递归状态、卷积与KV交叉：完整结果','',
        '复用8条已测事件流、32个尾部上下文、192个问题，保留所有基线失败。固定Qwen3.5-2B IT与已有GDN适配器，全部6层Attention仍在；不是Base实测或新训练。',
        '每个尾部结束后构造全部8种缓存，再供全部问题读取。来源顺序为18个递归矩阵、18个卷积状态、6层Attention KV；N为正常尾部，H为尾部双门关闭。12份前缀KV切片固定。',
        '正式新测4608项；1536条历史NNN/HHH端点参考组成6144项完整因子表。另1536项新端点重放全部与历史五字段答案精确一致，作为运行控制单列，不计为新主结果。',
        f'正式运行manifest SHA256：`{audit["manifest_sha256"]}`。登记包SHA256：`{audit["packet_manifest_sha256"]}`。',
        '214份测量源码不变，全部128份N/H源缓存和512份因子缓存通过审计。混合缓存没有冒用源logits，问题后缀重新计算实际输出；源缓存与所有组合独立存储。',
        '', '## 全部32条件','',
        '每行192个读出。Joint3为64个流×尾部×文体，Joint6为32个流×尾部；独立事件流仅8条。Oracle使用真实支持标注，只是能力对照。',
        '', '| Model | R/C/KV | Reader | Origin | Current /64 | Initial /64 | Protected /64 | Exact /192 | Joint3 /64 | Joint6 /32 | All /8 | NLL |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in cells:
        lines.append('| '+' | '.join(map(str,(r['model'],r['condition'],r['reader'],r['measurement_origin'],r['target_current_correct'],r['target_initial_correct'],
            r['protected_current_correct'],r['greedy_exact'],r['joint3_correct'],r['joint6_correct'],r['all_questions_per_stream_correct'],f'{r["answer_nll"]:.6f}')))+' |')
    lines.extend(['','## 全部128个尾部条件','',
        '| Model | Condition.Reader | Episode | Current /16 | Initial /16 | Protected /16 | Exact /48 | Joint6 /8 | NLL |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|'])
    for r in episodes:
        lines.append('| '+' | '.join(map(str,(r['model'],r['cell'],r['episode'],r['target_current_correct'],r['target_initial_correct'],r['protected_current_correct'],
            r['greedy_exact'],r['joint6_correct'],f'{r["answer_nll"]:.6f}')))+' |')
    lines.extend(['','## 解读边界','',
        '`readouts.csv`明确区分新主测量和历史参考；`endpoint-replays.csv`单列精确端点重放。全部角色、文体、流和证据来源见`strata.csv`，全部单分量替换/修复、Oracle及相关/无关对照见`contrasts.csv`。',
        '`nll-interactions.csv`给出每个固定问题的7项有限差分，含二阶与三阶项；它们是冻结缓存干预的NLL交互，不是学到的集合效用或显著性证据。',
        '相关尾部的当前答案来自新写入，和固定前缀证据分开报告。相关/无关匹配时该答案改变，不能把这个比较叫同一事实损伤。',
        '交叉缓存可能偏离自然分布。分量替换会改变后续Query和答案计算；回答恢复不证明事实存储或删除，也不证明共享控制的新架构必要。',
        '保留完整逐张量指纹、布局、实际属性与运行检查，没有保存所有缓存张量的全量副本。精确重跑需要相同本地权重及后端。',
        '本轮复用已测队列，是机制定位，不是新数据确认；没有训练、显著性声明、架构成功或扩模门槛，旧失败门槛保持原判。'])
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharey=True)
    index={(r['model'],r['reader'],r['condition']):r for r in cells}
    for i,model in enumerate(MODELS):
        for j,reader in enumerate(READERS):
            ax=axes[i,j];values=[index[(model,reader,c)]['greedy_exact'] for c in CONDITIONS]
            ax.bar(range(8),values,color=['#388496' if c in PRIMARY else '#8b9299' for c in CONDITIONS])
            ax.set_xticks(range(8),CONDITIONS);ax.set_ylim(0,212);ax.set_title(model+' / '+reader);ax.set_ylabel('Correct readouts /192');ax.grid(axis='y',alpha=.2)
            for x,y in enumerate(values):ax.text(x,y+2,str(y),ha='center',fontsize=9)
    fig.suptitle('Post-tail RNN / convolution / KV origin crossover: all32cells')
    fig.text(.5,.018,'N: normal tail; H: both gates held. Blue: new; gray: archived endpoints. 8 reused streams; no architecture or significance claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.96))
    for suffix in ('png','svg'):fig.savefig(a.output/('all32cells.'+suffix),dpi=180)
    plt.close(fig);(a.output/'export-source.py').write_bytes(Path(__file__).read_bytes())
    receipt={'status':'complete_post_tail_path_exports','rows':{k:len(v) for k,v in tables.items()},'audit_sha256':streaming_hash(a.audit),
        'export_source_sha256':streaming_hash(Path(__file__)),'files_sha256':{p.name:streaming_hash(p) for p in a.output.iterdir()}}
    write_json(a.output/'export-receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
