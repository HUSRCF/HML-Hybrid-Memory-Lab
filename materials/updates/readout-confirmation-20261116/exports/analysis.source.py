"""Complete independent analysis after all6000 frozen confirmation readouts are audited."""
from collections import defaultdict
from datetime import datetime,timezone
from itertools import combinations
from pathlib import Path
import csv,hashlib,json,re,sys
from unittest.mock import patch
ROOT=Path('/home/husrcf/Code/SAS');sys.path.insert(0,str(ROOT))
BASE=Path('/media/990Pro/Code/SAS');ART=BASE/'artifacts/memory_update'
from scripts.readout_confirmation_data import registered_inputs,MODELS
from scripts.analyze_readout_confirmation import audit_run
from sas_memory.tasks import execute,Operation
from sas_memory.training_data import data_fingerprint
h=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
RUN=BASE/'runs/memory_update/readout-confirmation-primary-20261116-01'
AUDIT=ART/'readout-confirmation-primary-audit-20261116-01.json'
OUT=ART/'readout-confirmation-exports-20261116-01'


def program(events):
    current,initial,histories={},{},defaultdict(set)
    for e in events:
        if e['kind']=='SET':value=e['value']
        elif e['kind']=='COPY':value=current[e['source']]
        else:raise ValueError('Unexpected operation in fixed confirmation')
        current[e['target']]=value
        initial.setdefault(e['target'],value)
        histories[e['target']].add(str(value))
    truth=execute([Operation(**e) for e in events])
    if current!=truth['current'] or initial!=truth['initial']:raise ValueError('Independent program trace differs')
    return {'current':current,'initial':initial,'histories':histories,'events_sha256':data_fingerprint(events)}


def classify(row,metadata,semantics):
    expected=metadata['gold'];entity=metadata['entity']
    version='initial' if metadata['role']=='target_initial' else 'current'
    assert str(semantics[version][entity])==expected
    match=re.fullmatch(r'\s*(-?\d+)\s*',row['generated'])
    prediction=match[1] if match else ''
    correct=bool(match and prediction==expected)
    assert row['answer']==expected and row['greedy_exact']==correct
    matches=sorted(e for e,values in semantics['histories'].items() if match and prediction in values)
    category=('correct' if correct else 'non_integer' if match is None else
              'matches_requested_entity_other_seen_value' if entity in matches else
              'matches_other_entity_seen_value' if matches else 'unrecorded_value')
    return {**metadata,'model':row['model'],'prediction':prediction,'generated':row['generated'],
            'correct':int(correct),'answer_nll':row['answer_nll'],'normal_history_sha256':row['normal_history_sha256'],
            'error_value_category':category,'matching_value_entities':'|'.join(matches),
            'prediction_matches_requested_entity_current':int(bool(match and prediction==str(semantics['current'][entity]))),
            'prediction_matches_requested_entity_initial':int(bool(match and prediction==str(semantics['initial'][entity])))}


def pair(left,right,label):
    return {'comparison':label,'task':left['task'],'model_left':left['model'],'model_right':right['model'],
            'logical_id_left':left['logical_id'],'logical_id_right':right['logical_id'],
            'physical_id_left':left['physical_id'],'physical_id_right':right['physical_id'],
            'seed':left['seed'],'role':left['role'],'naming_left':left['naming'],'naming_right':right['naming'],
            'style_left':left['question_style'],'style_right':right['question_style'],
            'depth_left':left['dependency_depth'],'depth_right':right['dependency_depth'],
            'condition_left':left['condition'],'condition_right':right['condition'],
            'gold_left':left['gold'],'gold_right':right['gold'],'gold_equal':int(left['gold']==right['gold']),
            'prediction_left':left['prediction'],'prediction_right':right['prediction'],
            'correct_left':left['correct'],'correct_right':right['correct'],
            'gain':int(not left['correct'] and right['correct']),
            'regression':int(left['correct'] and not right['correct']),
            'same_normal_history':int(left['normal_history_sha256']==right['normal_history_sha256'])}


def main():
    with (patch('sas_memory.runtime.TextRuntime.__init__',side_effect=AssertionError('No model construction')),
          patch('torch.cuda._lazy_init',side_effect=AssertionError('No GPU initialization'))):
        audit=json.loads(AUDIT.read_text());assert audit==audit_run(RUN)
        assert audit['physical_readouts_audited']==6000 and audit['source_files_verified']==289
        p,packet,data,lock=registered_inputs()
    manifest=json.loads((RUN/'manifest.json').read_text())
    all_queries={**data['set_queries'],**data['copy_queries']}
    raw={}
    for model in MODELS:
        values=[json.loads(line) for line in (RUN/(model+'.jsonl')).read_text().splitlines()]
        assert [r['id'] for r in values]==data['query_order']
        for r in values:raw[model,r['id']]=r
    assert len(raw)==6000
    set_cases={c['id']:c for c in data['set_cases']};copy_cases={c['id']:c for c in data['copy_cases']}
    contexts={c['id']:c for c in data['set_registry']['contexts']}
    semantics={};logical_metadata=[]
    for qid,q in data['set_queries'].items():
        case=set_cases[q['case_id']];episode=next(e for e in case['episodes'] if e['id']==q['episode'])
        events=case['prefix_events']+episode['tail'];key=q['source_key']
        semantics.setdefault(key,program(events))
        logical_metadata.append({'logical_id':'set:'+qid,'physical_id':qid,'task':'set','context_id':q['context_id'],
            'case_id':q['case_id'],'base_case_id':'set-s'+str(q['seed']),'seed':q['seed'],'naming':q['naming'],
            'question_style':q['question_style'],'dependency_depth':0,'variant':'','condition':q['episode'],
            'kind':q['kind'],'tail_records':q['tail_records'],'active_entities':q['active_entities'],'distractor_updates':0,
            'role':q['role'],'entity':case['protected'] if q['role']=='protected_current' else case['target'],
            'question':q['question'],'gold':q['answer'],'source_key':key})
    for b in data['copy_registry']['bindings']:
        q=data['copy_queries'][b['physical_id']];case=copy_cases[b['case_id']];variant=case['variants'][b['variant']]
        binding_query=next(x for x in variant['queries'] if x['role']==b['role'])
        events=case['prefix']+variant['tail'];key=q['source_key'];compiled=program(events)
        if key in semantics:assert semantics[key]['events_sha256']==compiled['events_sha256']
        semantics.setdefault(key,compiled)
        assert binding_query['answer']==q['answer']
        logical_metadata.append({'logical_id':'copy:'+data_fingerprint([b['case_id'],b['variant'],b['role']]),
            'physical_id':q['id'],'task':'copy','context_id':q['context_id'],'case_id':b['case_id'],
            'base_case_id':b['base_case_id'],'seed':b['seed'],'naming':'','question_style':b['question_style'],
            'dependency_depth':b['dependency_depth'],'variant':b['variant'],'condition':b['variant'],'kind':'copy',
            'tail_records':len(variant['tail']),'active_entities':0,'distractor_updates':b['distractor_updates'],
            'role':b['role'],'entity':binding_query['entity'],'question':q['question'],'gold':q['answer'],'source_key':key})
    assert len(logical_metadata)==1296 and len({r['logical_id'] for r in logical_metadata})==1296
    assert len(semantics)==200
    rows=[classify(raw[model,r['physical_id']],r,semantics[r['source_key']]) for model in MODELS for r in logical_metadata]
    assert len(rows)==6480
    by_logical={(r['model'],r['logical_id']):r for r in rows}
    physical_representatives={r['physical_id']:r['logical_id'] for r in logical_metadata}
    physical=[by_logical[model,physical_representatives[key]] for model in MODELS for key in data['query_order']]
    assert len(physical)==6000
    pairs=[];physical_pairs=[]
    for left,right in combinations(MODELS,2):
        for r in logical_metadata:pairs.append(pair(by_logical[left,r['logical_id']],by_logical[right,r['logical_id']],left+' -> '+right))
        for key in data['query_order']:
            lid=physical_representatives[key]
            physical_pairs.append(pair(by_logical[left,lid],by_logical[right,lid],left+' -> '+right))
    assert len(pairs)==12960 and len(physical_pairs)==12000
    summary={}
    for model in MODELS:
        own=[r for r in rows if r['model']==model];phys=[r for r in physical if r['model']==model]
        value={}
        for task in ('set','copy'):
            rr=[r for r in own if r['task']==task];pp=[r for r in phys if r['task']==task];protected=[r for r in rr if r['role']=='protected_current']
            value[task]={'physical_correct':sum(r['correct'] for r in pp),'physical_questions':len(pp),
                         'logical_correct':sum(r['correct'] for r in rr),'logical_bindings':len(rr),
                         'protected_correct':sum(r['correct'] for r in protected),'protected_bindings':len(protected),
                         'error_value_counts':{name:sum(r['error_value_category']==name for r in rr) for name in
                            ('correct','non_integer','matches_requested_entity_other_seen_value','matches_other_entity_seen_value','unrecorded_value')}}
            assert value[task]['physical_correct']==audit['models'][model]['set_correct' if task=='set' else 'copy_physical_correct']
        summary[model]=value
    fields=('model','task','naming','question_style','dependency_depth','variant','kind','tail_records','active_entities','distractor_updates','role')
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in fields)].append(r)
    cells=[{**dict(zip(fields,k)),'bindings':len(v),'correct':sum(r['correct'] for r in v),
            'physical_prompts':len({r['physical_id'] for r in v}),'base_seeds':len({r['seed'] for r in v})} for k,v in grouped.items()]
    assert len(cells)==1080
    seed_groups=defaultdict(list)
    for r in rows:seed_groups[r['model'],r['task'],r['seed']].append(r)
    seeds=[{'model':m,'task':t,'seed':seed,'bindings':len(v),'correct':sum(r['correct'] for r in v),
            'physical_prompts':len({r['physical_id'] for r in v})} for (m,t,seed),v in seed_groups.items()]
    assert len(seeds)==80
    joint_groups=defaultdict(list)
    for r in rows:
        if r['task']=='copy' and r['variant']=='before':continue
        key=(r['model'],r['task'],r['context_id'] if r['task']=='set' else r['case_id'])
        joint_groups[key].append(r)
    joint=[]
    for (m,t,key),v in joint_groups.items():
        assert len(v)==6
        joint.append({'model':m,'task':t,'unit_id':key,'seed':v[0]['seed'],'naming':v[0]['naming'],
                      'depth':v[0]['dependency_depth'],'question_style':v[0]['question_style'] if t=='copy' else 'both',
                      'condition':v[0]['condition'] if t=='set' else 'control+treatment',
                      'answers_correct':sum(r['correct'] for r in v),'answers_required':6,'joint_correct':int(all(r['correct'] for r in v))})
    assert len(joint)==960
    jby={(r['model'],r['task'],r['unit_id']):r for r in joint};jpairs=[]
    for left,right in combinations(MODELS,2):
        for (model,task,unit),v in jby.items():
            if model!=left:continue
            other=jby[right,task,unit]
            jpairs.append({'model_left':left,'model_right':right,'task':task,'unit_id':unit,'seed':v['seed'],
                           'correct_left':v['joint_correct'],'correct_right':other['joint_correct'],
                           'gain':int(not v['joint_correct'] and other['joint_correct']),
                           'regression':int(v['joint_correct'] and not other['joint_correct'])})
    assert len(jpairs)==1920
    for model in MODELS:
        for task in ('set','copy'):
            values=[r for r in joint if r['model']==model and r['task']==task]
            summary[model][task].update(joint6_correct=sum(r['joint_correct'] for r in values),joint6_units=len(values))
    wording=[];versions=[];names=[];lengths=[];loads=[];relations=[];before_after=[]
    for model in MODELS:
        own=[r for r in rows if r['model']==model]
        s={(r['seed'],r['naming'],r['condition'],r['question_style'],r['role']):r for r in own if r['task']=='set'}
        c={(r['base_case_id'],r['dependency_depth'],r['variant'],r['question_style'],r['role']):r for r in own if r['task']=='copy'}
        for r in own:
            if r['task']=='set':
                seed,naming,cond,style,role=r['seed'],r['naming'],r['condition'],r['question_style'],r['role']
                if style=='state_term':wording.append(pair(r,s[seed,naming,cond,'event_term',role],'SET state -> event wording'))
                if role=='target_current':versions.append(pair(r,s[seed,naming,cond,style,'target_initial'],'current -> initial'))
                if naming=='original':names.append(pair(r,s[seed,'permuted',cond,style,role],'original -> permuted account names'))
                if cond!='before':
                    before_after.append(pair(s[seed,naming,'before',style,role],r,'before -> after normal updates'))
                    if r['tail_records']==64:lengths.append(pair(r,s[seed,naming,f"{r['kind']}-512-load{r['active_entities']}",style,role],'64 -> 512 tail records'))
                    if r['active_entities']==2:loads.append(pair(r,s[seed,naming,f"{r['kind']}-{r['tail_records']}-load16",style,role],'2 -> 16 active tail accounts'))
                    if r['kind']=='unrelated':relations.append(pair(r,s[seed,naming,f"related-{r['tail_records']}-load{r['active_entities']}",style,role],'unrelated -> related final update'))
            else:
                base,depth,var,style,role=r['base_case_id'],r['dependency_depth'],r['variant'],r['question_style'],r['role']
                if style=='original':wording.append(pair(r,c[base,depth,var,'entity_first',role],'COPY original -> entity-first wording'))
                if role=='target_current':versions.append(pair(r,c[base,depth,var,style,'target_initial'],'current -> initial'))
    assert [len(x) for x in (wording,versions,names,lengths,loads,relations,before_after)]==[3240,2160,2160,1920,1920,1920,3840]
    for comparison in sorted({r['comparison'] for r in pairs}):
        for task in ('set','copy'):
            rr=[r for r in pairs if r['comparison']==comparison and r['task']==task]
            assert sum(r['gain']-r['regression'] for r in rr)==sum(r['correct_right']-r['correct_left'] for r in rr)
    model_contrasts=[]
    for left,right in combinations(MODELS,2):
        for task in ('set','copy'):
            rr=[r for r in pairs if r['model_left']==left and r['model_right']==right and r['task']==task]
            ss=[]
            for seed in sorted({r['seed'] for r in rr}):
                group=[r for r in rr if r['seed']==seed];ss.append(sum(r['gain']-r['regression'] for r in group))
            model_contrasts.append({'model_left':left,'model_right':right,'task':task,'logical_bindings':len(rr),
                'gains':sum(r['gain'] for r in rr),'regressions':sum(r['regression'] for r in rr),
                'base_seeds':len(ss),'base_seed_improvements':sum(x>0 for x in ss),
                'base_seed_regressions':sum(x<0 for x in ss),'base_seed_ties':sum(x==0 for x in ss)})
    assert len(model_contrasts)==20
    assert not OUT.exists();OUT.mkdir()
    def table(name,values):
        with (OUT/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
        assert len(list(csv.DictReader((OUT/name).open())))==len(values)
    tables={'physical-readouts.csv':physical,'all-logical-bindings.csv':rows,'all-model-pairs-physical.csv':physical_pairs,
            'all-model-pairs-logical.csv':pairs,'all-role-condition-cells.csv':cells,'all-base-streams.csv':seeds,
            'all-joint6-units.csv':joint,'all-joint6-model-pairs.csv':jpairs,'all-wording-pairs.csv':wording,
            'all-current-initial-pairs.csv':versions,'all-account-renaming-pairs.csv':names,'all-tail-length-pairs.csv':lengths,
            'all-active-account-pairs.csv':loads,'all-update-relevance-pairs.csv':relations,'all-before-after-pairs.csv':before_after,
            'all-model-task-contrasts.csv':model_contrasts}
    for name,values in tables.items():table(name,values)
    report={'status':'all6000physical_and6480logical_answers_independently_regraded',
            'created_utc':datetime.now(timezone.utc).isoformat(),'source_run':str(RUN),'source_audit_sha256':h(AUDIT),
            'packet_manifest_sha256':audit['packet_manifest_sha256'],'summary':summary,'all_model_task_contrasts':model_contrasts,
            'table_rows':{name:len(values) for name,values in tables.items()},'new_model_inference':False,
            'independent_base_streams_per_task':8,'training_initializations':1,'old_failed_criteria_unchanged':True,
            'architecture_superiority_claim':False,'error_categories_are_value_matches_not_internal_attention_evidence':True,
            'source_sha256':h(Path(__file__))}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');(OUT/'analysis.source.py').write_bytes(Path(__file__).read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#747b83','#704c99','#1b6ca8','#d47616','#25845a'];labels=['Native','Fixed GDN','Query','Gate','Query + gate']
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    for ax,task,title in ((axes[0],'set','Fresh SET: all 864 questions/model'),(axes[1],'copy','Fresh COPY: all 336 physical questions/model')):
        y=[100*summary[m][task]['physical_correct']/summary[m][task]['physical_questions'] for m in MODELS]
        ax.bar(labels,y,color=colors);ax.set(ylim=(0,110),ylabel='Exact numeric answers (%)',title=title)
        for i,m in enumerate(MODELS):ax.text(i,y[i]+1.2,str(summary[m][task]['physical_correct'])+'/'+str(summary[m][task]['physical_questions']),ha='center',fontsize=8)
    xs=[1,2,3]
    for model,color,label in zip(MODELS,colors,labels):
        values=[]
        for depth in xs:
            jj=[r for r in joint if r['task']=='copy' and r['model']==model and r['depth']==depth]
            assert len(jj)==16;values.append(100*sum(r['joint_correct'] for r in jj)/16)
        axes[2].plot(xs,values,'o-',label=label,color=color,lw=1.5)
    axes[2].set(xlabel='COPY dependency depth',ylabel='All six answers correct (%)',ylim=(-3,103),xticks=xs,title='COPY: both orders, all three roles');axes[2].legend(fontsize=8,frameon=False)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for ax in axes[:2]:ax.tick_params(axis='x',rotation=25,labelsize=8)
    fig.suptitle('Frozen models; 8 base streams per task; paired views are dependent',fontsize=11)
    fig.savefig(OUT/'confirmation-summary.png',dpi=180);fig.savefig(OUT/'confirmation-summary.pdf');plt.close(fig)
    files={f.name:h(f) for f in sorted(OUT.iterdir())}
    (OUT/'export-receipt.json').write_text(json.dumps({'status':report['status'],'source_audit_sha256':h(AUDIT),'files_sha256':files,'table_rows':report['table_rows']},indent=2)+'\n')
    print(json.dumps({'status':report['status'],'summary':summary,'model_task_contrasts':model_contrasts,'destination':str(OUT)}),flush=True)


if __name__=='__main__':main()
