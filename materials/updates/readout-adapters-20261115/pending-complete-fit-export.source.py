"""Complete observed-fit reporting after independent three-arm training audit."""
import csv,hashlib,json,re,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/home/husrcf/Code/SAS');sys.path.insert(0,str(ROOT))
BASE=Path('/media/990Pro/Code/SAS');ART=BASE/'artifacts/memory_update'
from scripts.analyze_readout_adapter_cache_training import audit_run,correction_audit
from scripts.readout_adapter_data import registered_inputs
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
run=BASE/'runs/memory_update/readout-adapter-training-20261115-01'
auditpath=ART/'readout-adapter-training-audit-20261115-01.json'
audit=json.loads(auditpath.read_text());assert audit==audit_run(run)
assert json.loads(auditpath.with_suffix('.correction.json').read_text())==correction_audit(run,audit)
p,packet,old,cases,registry,queries,examples,references=registered_inputs()
contexts={c['id']:c for c in registry['contexts']};cases={c['id']:c for c in cases}
models={'fixed_gdn':references}
for arm in ('query','gate','query_gate'):
 models[arm]=[json.loads(line) for line in (run/arm/'final-fit.jsonl').read_text().splitlines()]
rows=[]
for model,values in models.items():
 assert [v['id'] for v in values]==list(queries)
 for v in values:
  q=queries[v['id']];c=contexts[q['context_id']];case=cases[q['case_id']]
  match=re.fullmatch(r'\s*(-?\d+)\s*',v['generated']);prediction=match[1] if match else ''
  correct=prediction==q['answer'];assert correct==v['greedy_exact']
  events=case['prefix_events']+next(e['tail'] for e in case['episodes'] if e['id']==q['episode'])
  entity=case['protected'] if q['role']=='protected_current' else case['target']
  own={str(e['value']) for e in events if e['target']==entity}
  all_values={str(e['value']) for e in events}
  category=('correct' if correct else 'non_integer' if match is None else
            'matches_same_entity_other_record_value' if prediction in own else
            'matches_other_entity_record_value' if prediction in all_values else 'unrecorded_value')
  rows.append({'model':model,'id':q['id'],'context_id':q['context_id'],'case_id':q['case_id'],
               'seed':q['seed'],'naming':q['naming'],'source_cohort':registry['cohort_by_context'][q['context_id']],
               'episode':q['episode'],'kind':q['kind'],'tail_records':q['tail_records'],'active_entities':q['active_entities'],
               'question_style':q['question_style'],'role':q['role'],'question':q['question'],'gold':q['answer'],
               'generated':v['generated'],'prediction':prediction,'correct':int(correct),'error_value_category':category,
               'answer_nll':v['answer_nll'],'teacher_nll_per_token':v.get('teacher_nll_per_token','')})
assert len(rows)==6912
by={(r['model'],r['id']):r for r in rows};pairs=[]
for arm in ('query','gate','query_gate'):
 for qid in queries:
  base=by['fixed_gdn',qid];new=by[arm,qid]
  pairs.append({'model':arm,'id':qid,'seed':base['seed'],'role':base['role'],'question_style':base['question_style'],
                'source_cohort':base['source_cohort'],'baseline_correct':base['correct'],'adapted_correct':new['correct'],
                'gain':int(not base['correct'] and new['correct']),'regression':int(base['correct'] and not new['correct']),
                'baseline_generated':base['generated'],'adapted_generated':new['generated']})
assert len(pairs)==5184
baseline_errors=[r['id'] for r in rows if r['model']=='fixed_gdn' and not r['correct']];assert len(baseline_errors)==95
summary={}
for model in models:
 own=[r for r in rows if r['model']==model];paired=[r for r in pairs if r['model']==model]
 summary[model]={'fit_correct':sum(r['correct'] for r in own),'fit_questions':1728,
                 'gain_vs_fixed_gdn':sum(r['gain'] for r in paired) if paired else 0,
                 'regression_vs_fixed_gdn':sum(r['regression'] for r in paired) if paired else 0,
                 'error_value_counts':{name:sum(r['error_value_category']==name for r in own) for name in (
                     'correct','non_integer','matches_same_entity_other_record_value','matches_other_entity_record_value','unrecorded_value')}}
key_fields=('model','source_cohort','question_style','role','kind','tail_records','active_entities')
groups=defaultdict(list)
for r in rows:groups[tuple(r[k] for k in key_fields)].append(r)
cells=[{**dict(zip(key_fields,key)),'questions':len(value),'correct':sum(r['correct'] for r in value)} for key,value in groups.items()]
steps=[]
for arm in ('query','gate','query_gate'):
 for line in (run/arm/'training.jsonl').read_text().splitlines():
  s=json.loads(line);steps.append({'model':arm,'step':s['step'],'loss_mean':sum(e['teacher_nll_per_token'] for e in s['examples'])/4,
                                  'gradient_norm':s['gradient_norm'],'elapsed_seconds':s['elapsed_seconds']})
assert len(steps)==1296
out=ART/'readout-adapter-fit-exports-20261115-01';assert not out.exists();out.mkdir()
def write_csv(name,values):
 with (out/name).open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
for name,values in (('readouts.csv',rows),('paired-vs-fixed-gdn.csv',pairs),('all-role-condition-cells.csv',cells),('all-training-steps.csv',steps)):
 write_csv(name,values)
report={'status':'all_observed_fit_outputs_independently_regraded','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_run':str(run),'source_audit_sha256':h(auditpath),'fit_not_confirmation':True,'new_native_inference':False,
        'readouts':6912,'paired_comparisons':5184,'all_training_steps':1296,'baseline_error_ids':baseline_errors,
        'summary':summary,'all_old_failed_scientific_criteria_unchanged':True,'new_architecture_claim':False,
        'analysis_source_sha256':h(Path(__file__))}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'analysis.source.py').write_bytes(Path(__file__).read_bytes())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
colors={'query':'#1b6ca8','gate':'#d47616','query_gate':'#25845a'}
for arm in colors:
 selected=[s for s in steps if s['model']==arm]
 axes[0].plot([s['step'] for s in selected],[s['loss_mean'] for s in selected],label=arm,color=colors[arm],lw=1,alpha=.8)
axes[0].set(xlabel='Optimizer step',ylabel='Teacher loss (4 answer tokens)',title='Complete fixed-budget fit pass');axes[0].legend(frameon=False)
labels=['fixed_gdn','query','gate','query_gate'];scores=[summary[a]['fit_correct']/1728*100 for a in labels]
axes[1].bar(labels,scores,color=['#747b83']+[colors[a] for a in labels[1:]])
axes[1].set(ylabel='Exact numeric answers (%)',ylim=(0,105),title='Observed fit only: all 1,728 questions')
for i,a in enumerate(labels):axes[1].text(i,scores[i]+1.3,str(summary[a]['fit_correct'])+'/1728',ha='center',fontsize=8)
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.savefig(out/'fit-summary.png',dpi=180);fig.savefig(out/'fit-summary.pdf');plt.close(fig)
files={str(f.relative_to(out)):h(f) for f in sorted(out.iterdir())}
(out/'export-receipt.json').write_text(json.dumps({'status':report['status'],'source_audit_sha256':h(auditpath),
                                                'files_sha256':files,'fit_not_confirmation':True},indent=2)+'\n')
print(json.dumps({'status':report['status'],'summary':summary,'destination':str(out)}),flush=True)
