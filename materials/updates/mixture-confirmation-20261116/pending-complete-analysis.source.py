"""Complete seven-model analysis after all16800 physical rows pass independent audit."""
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from unittest.mock import patch
import csv, hashlib, json, sys

ROOT = Path('/home/husrcf/Code/SAS'); sys.path.insert(0, str(ROOT))
BASE = Path('/media/990Pro/Code/SAS'); ART = BASE / 'artifacts/memory_update'
from scripts.mixture_confirmation_data import registered_inputs, MODELS
from scripts.analyze_mixture_confirmation import audit_run
from export_readout_confirmation import program, classify, pair

RUN = BASE / 'runs/memory_update/mixture-confirmation-primary-20261116-01'
AUDIT = ART / 'mixture-confirmation-primary-audit-20261116-01.json'
OUT = ART / 'mixture-confirmation-exports-20261116-01'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metadata(data):
    sets = {c['id']: c for c in data['set_cases']}
    copies = {c['id']: c for c in data['copy_cases']}
    semantics, rows = {}, []
    for qid, q in data['set_queries'].items():
        case = sets[q['case_id']]
        episode = next(e for e in case['episodes'] if e['id'] == q['episode'])
        key = q['source_key']; semantics.setdefault(key, program(case['prefix_events'] + episode['tail']))
        rows.append({'logical_id': 'set:' + qid, 'physical_id': qid, 'task': 'set', 'context_id': q['context_id'],
            'case_id': q['case_id'], 'base_case_id': 'set-s' + str(q['seed']), 'seed': q['seed'], 'naming': q['naming'],
            'question_style': q['question_style'], 'dependency_depth': 0, 'variant': '', 'condition': q['episode'],
            'kind': q['kind'], 'tail_records': q['tail_records'], 'active_entities': q['active_entities'], 'distractor_updates': 0,
            'role': q['role'], 'entity': case['protected'] if q['role'] == 'protected_current' else case['target'],
            'question': q['question'], 'gold': q['answer'], 'source_key': key})
    for b in data['copy_registry']['bindings']:
        q = data['copy_queries'][b['physical_id']]; case = copies[b['case_id']]
        variant = case['variants'][b['variant']]
        bq = next(v for v in variant['queries'] if v['role'] == b['role'])
        key = q['source_key']; state = program(case['prefix'] + variant['tail'])
        if key in semantics:
            assert semantics[key]['events_sha256'] == state['events_sha256']
        semantics.setdefault(key, state)
        rows.append({'logical_id': 'copy:' + b['case_id'] + ':' + b['variant'] + ':' + b['role'],
            'physical_id': q['id'], 'task': 'copy', 'context_id': q['context_id'], 'case_id': b['case_id'],
            'base_case_id': b['base_case_id'], 'seed': b['seed'], 'naming': '', 'question_style': b['question_style'],
            'dependency_depth': b['dependency_depth'], 'variant': b['variant'], 'condition': b['variant'], 'kind': 'copy',
            'tail_records': len(variant['tail']), 'active_entities': 0, 'distractor_updates': b['distractor_updates'],
            'role': b['role'], 'entity': bq['entity'], 'question': q['question'], 'gold': q['answer'], 'source_key': key})
    assert len(rows) == len({r['logical_id'] for r in rows}) == 1296 and len(semantics) == 200
    return rows, semantics

def main():
    with (patch('sas_memory.runtime.TextRuntime.__init__', side_effect=AssertionError('No model')),
          patch('torch.cuda._lazy_init', side_effect=AssertionError('No GPU'))):
        audit = json.loads(AUDIT.read_text()); assert audit == audit_run(RUN)
        p, packet, cohorts, lock = registered_inputs()
    assert audit['new_physical_readouts_audited'] == 10800 and audit['all_physical_rows_including_archived_audited'] == 16800
    logical, physical, pairs, physical_pairs, joint, joint_pairs, contrasts = [], [], [], [], [], [], []
    wording, versions, names, lengths, loads, relations, before_after = [], [], [], [], [], [], []
    summaries = {}
    for cohort in ('observed', 'fresh'):
        data = cohorts[cohort]; meta, semantics = metadata(data)
        raw = {}
        for model in MODELS:
            rows = [json.loads(line) for line in (RUN / cohort / (model + '.jsonl')).read_text().splitlines()]
            assert [r['id'] for r in rows] == data['query_order']
            raw.update({(model, r['id']): r for r in rows})
        rows = [{**classify(raw[model, r['physical_id']], r, semantics[r['source_key']]), 'cohort': cohort}
                for model in MODELS for r in meta]
        by_logical = {(r['model'], r['logical_id']): r for r in rows}
        reps = {r['physical_id']: r['logical_id'] for r in meta}
        phys = [by_logical[model, reps[qid]] for model in MODELS for qid in data['query_order']]
        assert len(phys) == 8400 and len(rows) == 9072
        logical.extend(rows); physical.extend(phys)
        for left, right in combinations(MODELS, 2):
            label = left + ' -> ' + right
            for r in meta:
                pairs.append({**pair(by_logical[left, r['logical_id']], by_logical[right, r['logical_id']], label), 'cohort': cohort})
            for qid in data['query_order']:
                lid = reps[qid]
                physical_pairs.append({**pair(by_logical[left, lid], by_logical[right, lid], label), 'cohort': cohort})
        groups = defaultdict(list)
        for r in rows:
            if r['task'] == 'copy' and r['variant'] == 'before':
                continue
            groups[r['model'], r['task'], r['context_id'] if r['task'] == 'set' else r['case_id']].append(r)
        jj = []
        for (model, task, unit), group in groups.items():
            assert len(group) == 6
            jj.append({'cohort': cohort, 'model': model, 'task': task, 'unit_id': unit, 'seed': group[0]['seed'],
                       'depth': group[0]['dependency_depth'], 'question_style': group[0]['question_style'] if task == 'copy' else 'both',
                       'condition': group[0]['condition'] if task == 'set' else 'control+treatment',
                       'answers_correct': sum(r['correct'] for r in group), 'answers_required': 6,
                       'joint_correct': int(all(r['correct'] for r in group))})
        joint.extend(jj); jby = {(r['model'], r['task'], r['unit_id']): r for r in jj}
        for left, right in combinations(MODELS, 2):
            for (model, task, unit), r in jby.items():
                if model != left:
                    continue
                other = jby[right, task, unit]
                joint_pairs.append({'cohort': cohort, 'model_left': left, 'model_right': right, 'task': task,
                    'unit_id': unit, 'seed': r['seed'], 'correct_left': r['joint_correct'], 'correct_right': other['joint_correct'],
                    'gain': int(not r['joint_correct'] and other['joint_correct']),
                    'regression': int(r['joint_correct'] and not other['joint_correct'])})
            for task in ('set', 'copy'):
                pp = [r for r in physical_pairs if r['cohort'] == cohort and r['model_left'] == left and r['model_right'] == right and r['task'] == task]
                delta = defaultdict(int)
                for r in pp:
                    delta[r['seed']] += r['gain'] - r['regression']
                assert len(delta) == 8
                contrasts.append({'cohort': cohort, 'model_left': left, 'model_right': right, 'task': task,
                    'physical_answers': len(pp), 'repairs': sum(r['gain'] for r in pp), 'regressions': sum(r['regression'] for r in pp),
                    'base_stream_improvements': sum(v > 0 for v in delta.values()),
                    'base_stream_regressions': sum(v < 0 for v in delta.values()), 'base_stream_ties': sum(v == 0 for v in delta.values())})
        summary = {}
        for model in MODELS:
            summary[model] = {}
            for task in ('set', 'copy'):
                rr = [r for r in rows if r['model'] == model and r['task'] == task]
                pp = [r for r in phys if r['model'] == model and r['task'] == task]
                ju = [r for r in jj if r['model'] == model and r['task'] == task]
                value = {'physical_correct': sum(r['correct'] for r in pp), 'physical_questions': len(pp),
                    'logical_correct': sum(r['correct'] for r in rr), 'logical_bindings': len(rr),
                    'protected_correct': sum(r['correct'] for r in pp if r['role'] == 'protected_current'),
                    'protected_questions': sum(r['role'] == 'protected_current' for r in pp),
                    'joint6_correct': sum(r['joint_correct'] for r in ju), 'joint6_units': len(ju),
                    'error_value_counts': {category: sum(r['error_value_category'] == category for r in pp) for category in
                        ('correct','non_integer','matches_requested_entity_other_seen_value','matches_other_entity_seen_value','unrecorded_value')}}
                assert value['physical_correct'] == audit['cohorts'][cohort][model]['set_correct' if task == 'set' else 'copy_physical_correct']
                if task == 'copy':
                    assert value['joint6_correct'] == audit['cohorts'][cohort][model]['copy_joint6_correct']
                    value['joint6_by_depth'] = {str(d): sum(r['joint_correct'] for r in ju if r['depth'] == d) for d in (1,2,3)}
                summary[model][task] = value
            own = [r for r in rows if r['model'] == model]
            s = {(r['seed'],r['naming'],r['condition'],r['question_style'],r['role']):r for r in own if r['task']=='set'}
            c = {(r['base_case_id'],r['dependency_depth'],r['variant'],r['question_style'],r['role']):r for r in own if r['task']=='copy'}
            def paired(left, right, label):
                return {**pair(left, right, label), 'cohort': cohort}
            for r in own:
                seed, style, role = r['seed'], r['question_style'], r['role']
                if r['task'] == 'set':
                    naming, cond = r['naming'], r['condition']
                    if style == 'state_term': wording.append(paired(r,s[seed,naming,cond,'event_term',role],'SET state -> event wording'))
                    if role == 'target_current': versions.append(paired(r,s[seed,naming,cond,style,'target_initial'],'current -> initial'))
                    if naming == 'original': names.append(paired(r,s[seed,'permuted',cond,style,role],'original -> permuted account names'))
                    if cond != 'before':
                        before_after.append(paired(s[seed,naming,'before',style,role],r,'before -> after normal updates'))
                        if r['tail_records'] == 64: lengths.append(paired(r,s[seed,naming,f"{r['kind']}-512-load{r['active_entities']}",style,role],'64 -> 512 tail records'))
                        if r['active_entities'] == 2: loads.append(paired(r,s[seed,naming,f"{r['kind']}-{r['tail_records']}-load16",style,role],'2 -> 16 active tail accounts'))
                        if r['kind'] == 'unrelated': relations.append(paired(r,s[seed,naming,f"related-{r['tail_records']}-load{r['active_entities']}",style,role],'unrelated -> related final update'))
                else:
                    base, depth, var = r['base_case_id'], r['dependency_depth'], r['variant']
                    if style == 'original': wording.append(paired(r,c[base,depth,var,'entity_first',role],'COPY original -> entity-first wording'))
                    if role == 'target_current': versions.append(paired(r,c[base,depth,var,style,'target_initial'],'current -> initial'))
        summaries[cohort] = summary
    assert len(physical)==16800 and len(logical)==18144 and len(physical_pairs)==50400 and len(pairs)==54432
    assert len(joint)==2688 and len(joint_pairs)==8064 and len(contrasts)==84
    fields=('cohort','model','task','naming','question_style','dependency_depth','variant','kind','tail_records','active_entities','distractor_updates','role')
    groups=defaultdict(list)
    for r in logical: groups[tuple(r[k] for k in fields)].append(r)
    cells=[{**dict(zip(fields,key)),'bindings':len(values),'correct':sum(r['correct'] for r in values),
            'physical_prompts':len({r['physical_id'] for r in values})} for key,values in groups.items()]
    seed_groups=defaultdict(list)
    for r in logical: seed_groups[r['cohort'],r['model'],r['task'],r['seed']].append(r)
    seeds=[{'cohort':cohort,'model':model,'task':task,'seed':seed,'bindings':len(values),
            'correct':sum(r['correct'] for r in values),'physical_prompts':len({r['physical_id'] for r in values})}
           for (cohort,model,task,seed),values in seed_groups.items()]
    assert len(seeds)==224
    assert not OUT.exists(); OUT.mkdir()
    tables={'physical-readouts.csv':physical,'all-logical-bindings.csv':logical,
            'all21-model-pairs-physical.csv':physical_pairs,'all21-model-pairs-logical.csv':pairs,
            'all-role-condition-cells.csv':cells,'all-base-streams.csv':seeds,'all-joint6-units.csv':joint,
            'all-joint6-model-pairs.csv':joint_pairs,'all-model-task-contrasts.csv':contrasts,
            'all-wording-pairs.csv':wording,'all-current-initial-pairs.csv':versions,'all-account-renaming-pairs.csv':names,
            'all-tail-length-pairs.csv':lengths,'all-active-account-pairs.csv':loads,
            'all-update-relevance-pairs.csv':relations,'all-before-after-pairs.csv':before_after}
    for name, values in tables.items():
        with (OUT/name).open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
        assert len(list(csv.DictReader((OUT/name).open())))==len(values)
    report={'status':'all16800physical_and18144logical_answers_independently_regraded',
            'created_utc':datetime.now(timezone.utc).isoformat(),'source_run':str(RUN),'source_audit_sha256':sha(AUDIT),
            'packet_manifest_sha256':sha(Path(p['dataset'])/'manifest.json'),'summary_by_separate_cohort':summaries,
            'fresh_functional_checks':audit['fresh_functional_checks'], 'all_model_task_contrasts':contrasts,
            'table_rows':{name:len(values) for name,values in tables.items()},'new_model_inference':False,
            'independent_base_streams_per_task_per_cohort':8,'training_initializations':1,
            'old_failed_criteria_unchanged':True,'architecture_superiority_claim':False,
            'error_categories_are_output_value_matches_not_internal_attention_evidence':True,
            'source_sha256':sha(Path(__file__)), 'reused_pure_analysis_source_sha256':sha(Path('/tmp/export_readout_confirmation.py'))}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'analysis.source.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'pure-analysis-helpers.source.py').write_bytes(Path('/tmp/export_readout_confirmation.py').read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=summaries['fresh']; colors=['#747b83','#704c99','#1b6ca8','#d47616','#25845a','#a04a77','#098278']
    labels=['Native','Fixed GDN','Query','Gate','Query+gate','Gate replay','Gate mix']
    fig, axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    for ax,task,title in ((axes[0],'set','Fresh SET: all864questions/model'),(axes[1],'copy','Fresh COPY: all336physical questions/model')):
        yy=[100*summary[m][task]['physical_correct']/summary[m][task]['physical_questions'] for m in MODELS]
        ax.bar(labels,yy,color=colors); ax.set(ylim=(0,111),ylabel='Exact numeric answers (%)',title=title)
        for i,m in enumerate(MODELS):ax.text(i,yy[i]+1.2,str(summary[m][task]['physical_correct'])+'/'+str(summary[m][task]['physical_questions']),ha='center',fontsize=7)
        ax.tick_params(axis='x',rotation=35,labelsize=8)
    for model,color,label in zip(MODELS,colors,labels):
        axes[2].plot([1,2,3],[100*summary[model]['copy']['joint6_by_depth'][str(d)]/16 for d in (1,2,3)],'o-',label=label,color=color,lw=1.3)
    axes[2].set(xlabel='COPY dependency depth',ylabel='All six answers correct (%)',ylim=(-3,103),xticks=[1,2,3],title='Fresh COPY: both orders and all three roles')
    axes[2].legend(fontsize=7,frameon=False)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Frozen checkpoints; 8 base streams/task; one initialization; paired views are dependent',fontsize=11)
    fig.savefig(OUT/'fresh-summary.png',dpi=180);fig.savefig(OUT/'fresh-summary.pdf');plt.close(fig)
    (OUT/'export-receipt.json').write_text(json.dumps({'status':report['status'],'source_audit_sha256':sha(AUDIT),
        'files_sha256':{f.name:sha(f) for f in sorted(OUT.iterdir())},'table_rows':report['table_rows']},indent=2)+'\n')
    print(json.dumps({'status':report['status'],'summary_by_separate_cohort':summaries,
                      'fresh_functional_checks':report['fresh_functional_checks'],'destination':str(OUT)}),flush=True)

if __name__=='__main__':
    main()
