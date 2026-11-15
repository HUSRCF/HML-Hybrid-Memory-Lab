"""Export every audited cached-prefix condition, raw readout and paired outcome."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_post_encoding import audit_run
from scripts.post_encoding_data import registered_inputs,read_plan,MODES,READERS
from scripts.post_encoding_tasks import ROLES
from scripts.run_controller_trial import write_json


def csv_file(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with path.open(newline='') as handle:
        if len(list(csv.DictReader(handle)))!=len(rows):raise ValueError('Exported CSV row count changed')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite exports')
    audit=json.loads(a.audit.read_text());run=Path(audit['source_run']);plan=read_plan()
    if audit['status']!='completed_and_audited_post_encoding' or audit_run(run)!=audit:raise ValueError('Complete independent primary audit required')
    registered,cases,registry,queries,_=registered_inputs();metrics=audit['metrics'];a.output.mkdir(parents=True,exist_ok=False)
    raw=[];cells=[];strata=[];transitions=[];contrasts=[]
    for model in plan['model_order']:
        for line in (run/f'{model}.jsonl').read_text().splitlines():
            row=json.loads(line);q=queries[row['id']]
            raw.append({k:row[k] for k in ('id','model','mode','reader','memory_id','answer','generated','greedy_exact','answer_nll','query_tokens')}|
                {k:q[k] for k in ('seed','episode','role','question_style','prefix_evidence','support_event_index','question','source_key')})
        for cell,value in metrics['cell_metrics'][model].items():
            episode,mode,reader=cell.split('.');r=metrics['strata'][model][cell]['role']
            cells.append({'model':model,'episode':episode,'mode':mode,'reader':reader,**value,
                **{role+'_correct':r[role]['greedy_exact'] for role in ROLES}})
            for kind,groups in metrics['strata'][model][cell].items():
                for group,values in groups.items():strata.append({'model':model,'cell':cell,'stratum':kind,'group':group,**values})
        for cell,value in metrics['before_after'][model].items():
            for kind in ('all','fixed_prefix_evidence'):
                transitions.append({'model':model,'cell':cell,'stratum':kind,**value[kind]})
            for role,v in value['by_role'].items():transitions.append({'model':model,'cell':cell,'stratum':role,**v})
        for label,value in metrics['same_input_and_matched_tail_contrasts'][model].items():contrasts.append({'model':model,'contrast':label,**value})
    if len(raw)!=8160 or len(cells)!=170 or len(strata)!=2210 or len(transitions)!=800:raise ValueError('Complete primary exports required')
    for filename,rows in (('readouts',raw),('cells',cells),('strata',strata),('before-after',transitions),('contrasts',contrasts)):
        csv_file(a.output/(filename+'.csv'),rows)
    write_json(a.output/'metrics.json',metrics)
    lines=['# 正常编码后再更新：完整诊断结果','',
        '本轮使用固定Qwen3.5-2B IT及四种已有适配器部署，不是Base实测或新架构训练。8条新事件流、40个上下文、240个唯一问题，正式读出8160项。',
        '前缀正常编码一次；同一源缓存服务全部问题。干预只覆盖后续更新，Oracle只在问题阶段限制历史读取，仍使用后见支持标签。',
        '',f'数据manifest SHA256：`{audit["dataset_manifest_sha256"]}`。正式运行manifest SHA256：`{audit["manifest_sha256"]}`。',
        '259项测试、两轮相反模型顺序的各510项smoke、206份源码、全部680份源缓存、640种尾部条件、160项no-op缓存/输出检查与320项no-op读出通过审计。',
        '所有前缀KV切片逐位保留；问题阶段部署与原生模型在问题到来前的缓存完全一致。两门均关闭只保证18个递归矩阵不变，卷积仍运行。',
        '', '## 全部170条件','',
        '每行48个问题读出（8流×3角色×2文体）。Joint3以16个流×文体为分母，Joint6以8个流为分母；这些读出不是48个独立事件流。',
        '', '| Model | Episode | Gates | Reader | Current /16 | Initial /16 | Protected /16 | Exact /48 | Joint3 /16 | Joint6 /8 | NLL |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in cells:
        lines.append('| '+ ' | '.join(str(v) for v in (r['model'],r['episode'],r['mode'],r['reader'],r['target_current_correct'],
            r['target_initial_correct'],r['protected_current_correct'],r['greedy_exact'],r['joint3_role_cases'],r['joint6_streams'],f'{r["answer_nll"]:.6f}'))+' |')
    lines.extend(['','## 更新与保持的边界','',
        '全部编码后基线失败均保留。`before-after.csv`逐条件报告both_correct、gain、loss与both_wrong，并单列固定前缀证据和三个角色；`strata.csv`保留角色、文体及逐流统计。',
        '相关尾部的当前值需要编码新的最后一条SET，每条件16次，单独作为正确更新对照；这些答案的证据不在固定前缀。其余问题对应已编码的前缀证据。',
        '`contrasts.csv`保留同一尾部/问题下各门控对none、Oracle对full，以及同长度相关对无关尾部的比较。相关与无关尾部的当前答案不同，不能把其比较解释为同一目标答案的损伤。',
        '', '## 解释限制','',
        '门控仍会改变后续Query、隐藏表示、尾部KV与卷积状态。旧KV相同不唯一定位到递归矩阵存储损坏，也不分离Delta擦除与插入。接尾部前后还存在距离/位置变化；主要干预对照固定实际尾部Token和位置。',
        '本轮是冻结模型的机制诊断，没有新架构成功门槛、显著性声明、训练或扩模触发条件。此前失败的科学门槛全部保持原判。'])
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'none':'#61717e','decay':'#daa449','delta':'#3b8c9b','both':'#ac617a'}
    fig,axes=plt.subplots(5,2,figsize=(18,22),sharey=True)
    indexed={(r['model'],r['reader'],r['episode'],r['mode']):r for r in cells}
    order=[('before','none')]+[(e,m) for e in ('unrelated-8','related-8','unrelated-64','related-64') for m in MODES]
    for i,model in enumerate(plan['model_order']):
        for j,reader in enumerate(READERS):
            ax=axes[i,j];values=[indexed[(model,reader,e,m)]['greedy_exact'] for e,m in order]
            ax.bar(range(17),values,color=[colors[m] for e,m in order]);ax.set_ylim(0,53)
            ax.set_xticks(range(17),[e+'\n'+m for e,m in order],rotation=65,ha='right',fontsize=7)
            ax.set_title(model+' / '+reader);ax.set_ylabel('Correct readouts /48');ax.grid(axis='y',alpha=.2)
            for x,y in enumerate(values):ax.text(x,y+.6,str(y),ha='center',fontsize=7)
    fig.suptitle('Normally encoded prefix, subsequent-update interventions: all170cells',fontsize=16)
    fig.text(.5,.008,'8 fresh streams; 8160 paired readouts, not8160independent samples. Frozen IT diagnostic; oracle support is privileged. No architecture or significance claim.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.025,1,.98))
    for suffix in ('png','svg'):fig.savefig(a.output/('all170cells.'+suffix),dpi=180)
    plt.close(fig)
    receipt={'status':'complete_post_encoding_exports','primary_readouts':len(raw),'cells':len(cells),'strata':len(strata),
        'paired_before_after_strata':len(transitions),'contrasts':len(contrasts),'audit_sha256':streaming_hash(a.audit),
        'export_source_sha256':streaming_hash(Path(__file__)),'files_sha256':{p.name:streaming_hash(p) for p in a.output.iterdir()}}
    write_json(a.output/'export-receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
