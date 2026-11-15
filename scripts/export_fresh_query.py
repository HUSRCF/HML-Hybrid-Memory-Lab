"""Export every audited fresh baseline, Query swap, replay and donor descriptor."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fresh_query_data import registered_inputs,read_plan,MODELS,READERS,CONDITIONS
from scripts.analyze_fresh_query import audit_run
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def csv_file(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with path.open(newline='') as handle:
        if len(list(csv.DictReader(handle)))!=len(rows):raise ValueError('CSV roundtrip row count changed')


def flat(value):
    return {k:v for k,v in value.items() if not isinstance(v,dict)}|{
        key+'_'+field:item for key,v in value.items() if isinstance(v,dict) for field,item in v.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Never overwrite exports')
    audit=json.loads(args.audit.read_text());run=Path(audit['source_run'])
    if audit['status']!='completed_and_audited_fresh_query' or audit_run(run)!=audit:raise ValueError('Complete independent primary audit required')
    registered,cases,registry,queries,refs=registered_inputs();plan=read_plan();metrics=audit['metrics'];manifest=json.loads((run/'manifest.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    tables={k:[] for k in ('primary','same-source-replays','donor-queries','replacement-calls','cells','episodes','strata','contrasts','before-after','nll-interactions','cache-fields')}
    def readout_row(row):
        q=queries[row['id']]
        return {k:row[k] for k in ('id','model','condition','reader','measurement_origin','answer','generated','greedy_exact','answer_nll','query_tokens')}|{
            k:q[k] for k in ('seed','episode','role','question_style','prefix_evidence','support_event_index','question','source_key')}
    for model in MODELS:
        state=manifest['models'][model]
        for kind in ('primary','same-source'):
            for line in (run/f'{model}.{kind}.jsonl').read_text().splitlines():
                row=json.loads(line);tables['primary' if kind=='primary' else 'same-source-replays'].append(readout_row(row))
                exchange=row['proof']['query_exchange']
                if exchange:
                    for layer,value in exchange['layers'].items():
                        tables['replacement-calls'].append({'model':model,'id':row['id'],'condition':row['condition'],'reader':row['reader'],
                            'measurement_origin':row['measurement_origin'],'layer':layer,'native_query_sha256':value['native_query']['sha256'],
                            'used_query_sha256':value['used_query']['sha256'],'actual_replacement':value['actual_replacement'],
                            **{k:value[k] for k in ('eager_calls','question_calls','question_rows','passthrough_calls','non_query_tensor_checks','native_query_checks')}})
        for line in (run/f'{model}.donors.jsonl').read_text().splitlines():
            row=json.loads(line);q=queries[row['id']]
            for layer,value in row['bank']['tensors'].items():
                tables['donor-queries'].append({'model':model,'id':row['id'],'origin':row['origin'],'reader':row['reader'],
                    'seed':q['seed'],'episode':q['episode'],'role':q['role'],'question_style':q['question_style'],'layer':layer,
                    'source_cache_identity':row['source_cache_identity'],'guard_checks':row['bank']['guard_checks'],
                    **{k:(json.dumps(v) if isinstance(v,list) else v) for k,v in value.items()}})
        cache_records=[(key,'before',record) for key,record in state['prefixes'].items()]+[
            (key,origin,record) for key,source in state['sources'].items() for origin,record in source['caches'].items()]
        for key,origin,record in cache_records:
            for field,value in record['tensors'].items():
                tables['cache-fields'].append({'model':model,'context_or_prefix':key,'origin':origin,'field':field,
                    'sha256':value['sha256'],'shape':json.dumps(value['shape']),'dtype':value['dtype'],
                    'layout':json.dumps(record['layouts'][field]),'guard_checks':record['source_guard_checks'],
                    'source_logits_sha256':record['logits_sha256']})
        for cell,value in metrics['cells'][model].items():
            condition,reader=cell.split('.');role=metrics['strata'][model][cell]['role']
            tables['cells'].append({'model':model,'condition':condition,'reader':reader,**flat(value),**{r+'_correct':v['greedy_exact'] for r,v in role.items()}})
            for kind,groups in metrics['strata'][model][cell].items():
                for group,v in groups.items():tables['strata'].append({'model':model,'cell':cell,'stratum':kind,'group':group,**v})
            for episode,groups in metrics['episodes'][model][cell].items():
                tables['episodes'].append({'model':model,'cell':cell,'episode':episode,**flat(groups['all']),
                    **{r+'_correct':v['greedy_exact'] for r,v in groups['role'].items()}})
        for label,groups in metrics['paired_contrasts'][model].items():
            for stratum,value in groups.items():
                if stratum=='by_role':
                    for role,v in value.items():tables['contrasts'].append({'model':model,'contrast':label,'stratum':role,**v})
                else:tables['contrasts'].append({'model':model,'contrast':label,'stratum':stratum,**value})
        for cell,episodes in metrics['before_after'][model].items():
            for episode,groups in episodes.items():
                for stratum,value in groups.items():
                    if stratum=='role':
                        for role,v in value.items():tables['before-after'].append({'model':model,'cell':cell,'episode':episode,'stratum':role,**v})
                    elif isinstance(value,dict):tables['before-after'].append({'model':model,'cell':cell,'episode':episode,'stratum':stratum,**value})
        for reader,values in metrics['nll_interactions'][model].items():
            for row in values['per_question']:
                q=queries[row['id']];tables['nll-interactions'].append({'model':model,'reader':reader,**row,
                    **{k:q[k] for k in ('seed','episode','role','question_style','prefix_evidence')}})
    expected={'primary':3264,'same-source-replays':1536,'donor-queries':9216,'replacement-calls':18432,'cells':20,'episodes':68,
        'strata':296,'contrasts':178,'before-after':160,'nll-interactions':768,'cache-fields':6912}
    if {k:len(v) for k,v in tables.items()}!=expected:raise ValueError('Complete fresh exports required')
    for name,rows in tables.items():csv_file(args.output/(name+'.csv'),rows)
    write_json(args.output/'metrics.json',metrics)
    lines=['# 新事件流上的递归状态与问题Query来源：完整结果','',
        'Qwen3.5-2B IT及已有GDN适配器，8条新事件流、40个上下文、240个问题。全部3264项主读出：1536项尾部后未替换基线、1536项交叉Query干预和192项接尾部前基线。保留所有失败。',
        '另1536项同来源真实替换重放与原五字段结果及读取概率完全一致；1536次问题专用捕获记录9216个实际Query张量。运行控制与主结果分开。',
        f'登记包SHA256：`{audit["packet_manifest_sha256"]}`。正式运行manifest SHA256：`{audit["manifest_sha256"]}`。',
        '284项测试、两轮相反模型顺序smoke、222份测量源码、16个前缀与128个尾部后缓存通过审计。状态N为正常尾部；H只恢复旧递归矩阵，卷积和全部KV保持正常尾部结果。',
        '标签第一位是接收方状态，第二位是问题Query供体：NN/HH未替换，NH/HN交叉替换。只在已知问题后缀的一次forward替换6层Attention实际post-normalization/post-RoPE Query；后续gold评分与自由生成不再替换。',
        '供体只使用已知问题文本；捕获产生的预测logits被丢弃，不做gold评分或生成。Full/Oracle供体严格分开，Oracle使用真实支持标注，仅作能力控制。',
        '', '## 全部20条件','',
        'Before每行48次读出，接尾部后每行192次；分母显式列出。Joint6对应流×尾部，独立事件流仅8条。',
        '', '| Model | Condition | Reader | Exact | Current | Initial | Protected | Joint3 | Joint6 | All /8 | NLL |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in tables['cells']:
        role_total=r['readouts']//3
        values=(r['model'],r['condition'],r['reader'],f'{r["greedy_exact"]}/{r["readouts"]}',f'{r["target_current_correct"]}/{role_total}',
            f'{r["target_initial_correct"]}/{role_total}',f'{r["protected_current_correct"]}/{role_total}',
            f'{r["joint3_correct"]}/{r["joint3_total"]}',f'{r["joint6_correct"]}/{r["joint6_total"]}',r['all_questions_per_stream_correct'],f'{r["answer_nll"]:.6f}')
        lines.append('| '+' | '.join(map(str,values))+' |')
    lines.extend(['','## 全部68个阶段与尾部条件','',
        '| Model | Cell | Episode | Current /16 | Initial /16 | Protected /16 | Exact /48 | Joint6 /8 | NLL |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|'])
    for r in tables['episodes']:
        lines.append('| '+' | '.join(map(str,(r['model'],r['cell'],r['episode'],r['target_current_correct'],r['target_initial_correct'],
            r['protected_current_correct'],r['greedy_exact'],r['joint6_correct'],f'{r["answer_nll"]:.6f}')))+' |')
    lines.extend(['','## 解释与恢复限制','',
        '全部角色、文体、流和证据来源见strata.csv。contrasts.csv保留Query修复、反向破坏、状态变化和Oracle读取的全部配对增益/退化；before-after.csv单列固定前缀证据，相关更新的当前答案变化不能当成同一事实遗忘。',
        'donor-queries.csv列出9216个供体张量的实际字节指纹、布局与来源，replacement-calls.csv列出18432个实际替换层调用及解码直通计数。cache-fields.csv记录6912个源缓存张量。完整张量不另存副本，复现需相同本地权重与后端重跑。',
        'Query包含上游递归与Attention计算，交叉激活可能偏离自然分布。恢复不唯一证明语义地址构造，失败也不证明Query无关；没有中介百分比、显著性、创新性或新架构收益声明。',
        '所有旧失败门槛不变；本轮不训练、不扩模、不下载新权重、不创建环境。'])
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharey=True)
    index={(r['model'],r['reader'],r['condition']):r for r in tables['cells']};order=('before',*CONDITIONS)
    for i,model in enumerate(MODELS):
        for j,reader in enumerate(READERS):
            ax=axes[i,j];values=[index[(model,reader,c)] for c in order]
            ax.bar(range(5),[v['accuracy'] for v in values],color=['#92999e' if c=='before' else '#d38c43' if c in ('NH','HN') else '#388496' for c in order])
            ax.set_xticks(range(5),order);ax.set_ylim(0,1.12);ax.set_title(model+' / '+reader);ax.set_ylabel('Exact accuracy');ax.grid(axis='y',alpha=.2)
            for x,v in enumerate(values):ax.text(x,v['accuracy']+.02,f'{v["greedy_exact"]}/{v["readouts"]}',ha='center',fontsize=9)
    fig.suptitle('Fresh recurrent-state / question-Query source factorial: all 20 cells')
    fig.text(.5,.018,'Before: 48; after: 192 readouts per panel. 8 fresh streams. Query donors are reader-matched and answer-free. No architecture claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.96))
    for suffix in ('png','svg'):fig.savefig(args.output/('all20cells.'+suffix),dpi=180)
    plt.close(fig);(args.output/'export-source.py').write_bytes(Path(__file__).read_bytes())
    receipt={'status':'complete_fresh_query_exports','rows':expected,'audit_sha256':streaming_hash(args.audit),'export_source_sha256':streaming_hash(Path(__file__)),
        'files_sha256':{p.name:streaming_hash(p) for p in args.output.iterdir()}}
    write_json(args.output/'export-receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
