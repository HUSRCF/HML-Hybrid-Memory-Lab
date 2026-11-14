"""Direct full-registry counts, independently of the registered metrics module."""
import hashlib
import json
from pathlib import Path

base=Path('/media/990Pro/Code/SAS');run=base/'runs/memory_update/trained-shared-timing-20261115-01'
m=json.loads((run/'manifest.json').read_text());data=Path(m['dataset']);registry=json.loads((data/'registry.json').read_text());cases=json.loads((data/'cases.json').read_text())
assert m['status']=='completed' and len(m['completed_models'])==16
keys=[q['id'] for q in registry['physical']];assert len(keys)==336
bindings=registry['bindings'];assert len(bindings)==432
roles=['target_current','target_initial','protected_current'];index={(b['case_id'],b['variant'],b['role']):b['physical_id'] for b in bindings}
rows={};table={};joint={}
for name in m['model_order']:
 p=run/f'{name}.jsonl';assert hashlib.sha256(p.read_bytes()).hexdigest()==m['models'][name]['results_sha256']
 values=[json.loads(line) for line in p.read_text().splitlines()];assert [v['id'] for v in values]==keys
 rows[name]={v['id']:v for v in values}
 joint[name]={c['id']:all(rows[name][index[c['id'],v,r]]['greedy_exact'] for v in ['control','treatment'] for r in roles) for c in cases}
 table[name]={'physical_correct':sum(v['greedy_exact'] for v in values),'physical_questions':336,'joint_six_after':sum(joint[name].values()),'cases':48,
  'post_role_counts':{r:sum(rows[name][index[c['id'],v,r]]['greedy_exact'] for c in cases for v in ['control','treatment']) for r in roles}}
q='shared_question_trained_question';contrasts={}
for ref in ['shared_question_timing','native']:
 pairs=[{'case_id':c['id'],'seed':c['seed'],'depth':c['dependency_depth'],
  'gain':joint[q][c['id']] and not joint[ref][c['id']],'loss':joint[ref][c['id']] and not joint[q][c['id']]} for c in cases]
 def count(group):
  a=sum(p['gain'] for p in group);b=sum(p['loss'] for p in group);return {'gains':a,'losses':b,'net':a-b}
 by_depth={str(d):count([p for p in pairs if p['depth']==d]) for d in [1,2,3]};by_seed={str(s):count([p for p in pairs if p['seed']==s]) for s in range(63000,63008)}
 contrasts[ref]={**count(pairs),'by_depth':by_depth,'by_seed':by_seed,
  'passed':count(pairs)['net']>0 and all(v['net']>0 for v in by_depth.values()) and sum(v['net']>0 for v in by_seed.values())>=3}
counts={'preserved':0,'regression':0,'repair':0,'both_wrong':0};physical_pairs=[]
for key in keys:
 a,b=rows[q][key]['greedy_exact'],rows['shared'][key]['greedy_exact'];label='preserved' if a and b else 'regression' if b else 'repair' if a else 'both_wrong';counts[label]+=1
 physical_pairs.append({'physical_id':key,'outcome':label})
result={'status':'raw_counts_crosscheck_pending_full_execution_audit','source_manifest_sha256':hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest(),
 'models':table,'matched_question_joint_contrasts':contrasts,'matched_question_vs_shared_all_physical':counts,'physical_pairs':physical_pairs}
output=base/'artifacts/memory_update/trained-shared-timing-table-crosscheck-20261115-01.json';assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='physical_pairs'},indent=2))
