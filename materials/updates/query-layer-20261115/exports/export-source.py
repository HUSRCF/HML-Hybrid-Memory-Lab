"""Posthoc full export of every audited single-layer condition and reference."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.query_layer_data import registered_inputs, MODELS, LAYERS, DIRECTIONS
from scripts.analyze_query_layer_corrected import audit_run
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash


def flat(value):
    return {k: v for k, v in value.items() if not isinstance(v, dict)} | {
        key + '_' + field: item for key, v in value.items() if isinstance(v, dict) for field, item in v.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Never overwrite an export')
    audit = json.loads(args.audit.read_text())
    run = Path(audit['source_run'])
    if audit['status'] != 'completed_and_audited_query_layer' or audit_run(run) != audit:
        raise ValueError('Complete independent primary audit required')
    _, _, _, queries, refs = registered_inputs()
    metrics = audit['metrics']
    manifest = json.loads((run / 'manifest.json').read_text())
    tables = {k: [] for k in ('primary', 'archived-references', 'reference-replays', 'same-source-replays',
        'donor-queries', 'question-calls', 'cells', 'episodes', 'strata', 'contrasts', 'cache-fields')}
    def row_fields(row):
        q = queries[row['id']]
        return {k: row[k] for k in ('id', 'model', 'condition', 'reader', 'measurement_origin',
            'answer', 'generated', 'greedy_exact', 'answer_nll', 'query_tokens')} | {
            'selected_layer': row.get('layer'), **{k: q[k] for k in
                ('seed', 'episode', 'role', 'question_style', 'prefix_evidence', 'support_event_index', 'question', 'source_key')}}
    for model in MODELS:
        state = manifest['models'][model]
        for row in refs['full_readouts'][model]:
            tables['archived-references'].append(row_fields(row) | {'table_origin': 'archived_full_reference'})
        for kind, table in (('primary', 'primary'), ('references', 'reference-replays'), ('same-source', 'same-source-replays')):
            for line in (run / f'{model}.{kind}.jsonl').read_text().splitlines():
                row = json.loads(line)
                tables[table].append(row_fields(row))
                if kind == 'references': continue
                for layer, value in row['proof']['query_exchange']['layers'].items():
                    tables['question-calls'].append({
                        'model': model, 'id': row['id'], 'condition': row['condition'], 'selected_layer': row['layer'],
                        'actual_layer': layer, 'measurement_origin': row['measurement_origin'],
                        'native_query_sha256': value['native_query']['sha256'], 'used_query_sha256': value['used_query']['sha256'],
                        **{k: value[k] for k in ('actual_replacement', 'original_query_object_preserved', 'eager_calls',
                            'question_calls', 'question_rows', 'passthrough_calls', 'non_query_tensor_checks', 'native_query_checks')}})
        for line in (run / f'{model}.donors.jsonl').read_text().splitlines():
            row = json.loads(line)
            q = queries[row['id']]
            for layer, value in row['bank']['tensors'].items():
                tables['donor-queries'].append({'model': model, 'id': row['id'], 'origin': row['origin'], 'reader': row['reader'],
                    'seed': q['seed'], 'episode': q['episode'], 'role': q['role'], 'question_style': q['question_style'], 'layer': layer,
                    'source_cache_identity': row['source_cache_identity'], 'guard_checks': row['bank']['guard_checks'],
                    **{k: json.dumps(v) if isinstance(v, list) else v for k, v in value.items()}})
        cache_records = [(key, 'before', record) for key, record in state['prefixes'].items()] + [
            (key, origin, record) for key, source in state['sources'].items() for origin, record in source['caches'].items()]
        for key, origin, record in cache_records:
            for field, value in record['tensors'].items():
                tables['cache-fields'].append({'model': model, 'context_or_prefix': key, 'origin': origin, 'field': field,
                    'sha256': value['sha256'], 'shape': json.dumps(value['shape']), 'dtype': value['dtype'],
                    'layout': json.dumps(record['layouts'][field]), 'guard_checks': record['source_guard_checks'],
                    'source_logits_sha256': record['logits_sha256']})
        for cell, value in metrics['cells'][model].items():
            tables['cells'].append({'model': model, 'cell': cell, **flat(value),
                **{r + '_correct': v['greedy_exact'] for r, v in metrics['strata'][model][cell]['role'].items()}})
            for kind, groups in metrics['strata'][model][cell].items():
                for group, v in groups.items(): tables['strata'].append({'model': model, 'cell': cell, 'stratum': kind, 'group': group, **v})
            for episode, value in metrics['episodes'][model][cell].items():
                tables['episodes'].append({'model': model, 'cell': cell, 'episode': episode, **flat(value['all']),
                    **{r + '_correct': v['greedy_exact'] for r, v in value['role'].items()}})
        for label, groups in metrics['paired_contrasts'][model].items():
            for stratum, value in groups.items():
                if stratum == 'by_role':
                    for role, v in value.items(): tables['contrasts'].append({'model': model, 'contrast': label, 'stratum': role, **v})
                else: tables['contrasts'].append({'model': model, 'contrast': label, 'stratum': stratum, **value})
    expected = {'primary': 4608, 'archived-references': 1536, 'reference-replays': 1536, 'same-source-replays': 384,
        'donor-queries': 4608, 'question-calls': 29952, 'cells': 32, 'episodes': 128, 'strata': 480, 'contrasts': 288, 'cache-fields': 6912}
    if {k: len(v) for k, v in tables.items()} != expected: raise ValueError('Complete single-layer export required')
    if sum(row['actual_replacement'] for row in tables['question-calls']) != 4992:
        raise ValueError('Exactly one actual copied Query per new or single-layer self readout required')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        path = args.output / (name + '.csv')
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        with path.open(newline='') as handle:
            if len(list(csv.DictReader(handle))) != len(rows): raise ValueError('CSV row roundtrip changed')
    write_json(args.output / 'metrics.json', metrics)
    lines = ['# 单层问题 Query 定位：完整结果', '',
        '复用八条事件流、32 个尾部上下文、192 个问题。4,608 项新读出与 1,536 项已标明来源的历史 Full 参考共 6,144 项；运行重放不重复计入新主结果。保留所有基线失败。',
        '原生及已有 GDN 适配器，六层分别单独替换，HN/NH 两方向，只使用 Full 供体。其余层使用当次真实原生 Query 对象，后续变化自然传播；答案及解码直通。',
        '1,536 项历史参考精确重放、384 项单层同来源真实替换、768 次问题专用捕获及 4,608 个实际供体张量独立审计通过。299 项测试、相反模型顺序检查、230 份原封存源码及四份审计修正源码、固定参数通过核验。原审计器局部变量覆盖的失败及单独登记修正保留；模型计算与门槛未变。',
        f'登记 manifest SHA256 `{audit["packet_manifest_sha256"]}`；正式 manifest SHA256 `{audit["manifest_sha256"]}`。', '',
        '## 全部 32 条件', '',
        'all6 是历史参考；NN/HH 为其未替换基线，HN/NH 为其六层同时交叉替换。Lxx 是本轮只替换 xx 层的读出。每行分母均为 192，三个角色各 64；新尾部证据分母 32，固定前缀证据分母 160。', '',
        '| Model | Cell | Origin | Exact /192 | Current /64 | Initial /64 | Protected /64 | New /32 | Prefix /160 | Joint6 /32 | NLL |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in tables['cells']:
        evidence = metrics['strata'][row['model']][row['cell']]['evidence']
        values = (row['model'], row['cell'], row['measurement_origin'], row['greedy_exact'], row['target_current_correct'],
            row['target_initial_correct'], row['protected_current_correct'], evidence['False']['greedy_exact'],
            evidence['True']['greedy_exact'], row['joint6_correct'], f'{row["answer_nll"]:.6f}')
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines.extend(['', '## 全部 128 个尾部条件', '',
        '| Model | Cell | Episode | Current /16 | Initial /16 | Protected /16 | Exact /48 | Joint6 /8 | NLL |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|'])
    for row in tables['episodes']:
        values = (row['model'], row['cell'], row['episode'], row['target_current_correct'], row['target_initial_correct'],
            row['protected_current_correct'], row['greedy_exact'], row['joint6_correct'], f'{row["answer_nll"]:.6f}')
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines.extend(['', '## 解释限制', '',
        '全部配对增益、退化及 NLL 差见 contrasts.csv；全部问题、文体、角色和事件流见 primary.csv 与 strata.csv。历史参考及其新重放分开列出；它们不增加独立事件流数量。',
        'question-calls.csv 记录 29,952 个问题阶段层调用，其中仅 4,992 个实际复制 Query，其余调用传入真实原生 Query 对象。供体张量只记录字节指纹与布局；重建实际激活需要记录的同环境和本地权重。',
        '这是复用队列的定位，不是另一轮未见数据确认。单层成功不证明必要性或可加性，失败不证明该层无关；交叉激活可能分布外，供体包含上游 Attention 和递归计算。',
        '没有中介百分比、显著性、创新性、架构有效性或扩模声明。所有旧失败判据保持原判，本轮没有新训练或环境。'])
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    for i, model in enumerate(MODELS):
        for j, direction in enumerate(DIRECTIONS):
            order = [direction[0] * 2 + '.all6', *[f'{direction}.L{layer:02d}' for layer in LAYERS], direction + '.all6']
            values = [metrics['cells'][model][cell] for cell in order]
            ax = axes[i, j]
            ax.bar(range(8), [v['accuracy'] for v in values], color=['#92999e', *(['#d38c43'] * 6), '#388496'])
            ax.set_xticks(range(8), ['Base', *[str(layer) for layer in LAYERS], 'All six'])
            ax.set_ylim(0, 1.12); ax.set_title(model + ' / ' + direction)
            ax.set_xlabel('Replaced Attention layer'); ax.set_ylabel('Full-history exact accuracy'); ax.grid(axis='y', alpha=.2)
            for x, v in enumerate(values): ax.text(x, v['accuracy'] + .02, str(v['greedy_exact']), ha='center', fontsize=9)
    fig.suptitle('Single question-Query layer: all 24 new cells and 8 archived references')
    fig.text(.5, .018, 'Each bar: 192 questions on 8 reused streams. Base and All six are archived references. Both intervention directions retained.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .96))
    for suffix in ('png', 'svg'): fig.savefig(args.output / ('all32cells.' + suffix), dpi=180)
    plt.close(fig)
    (args.output / 'export-source.py').write_bytes(Path(__file__).read_bytes())
    receipt = {'status': 'complete_query_layer_exports', 'rows': expected, 'actual_single_layer_replacements': 4992,
        'audit_sha256': streaming_hash(args.audit), 'export_source_sha256': streaming_hash(Path(__file__)),
        'files_sha256': {p.name: streaming_hash(p) for p in args.output.iterdir()}}
    write_json(args.output / 'export-receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
