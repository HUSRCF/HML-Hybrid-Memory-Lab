"""Export all new and archived cells, checking every CSV field against raw data."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.gate_component_data import MODELS,REGIMES,UNIFIED,registered_inputs,read_plan
from scripts.gate_component_runtime import SETTINGS
from scripts.analyze_gate_components import audit_run
from scripts.run_controller_trial import write_json
from sas_memory.write_locality_data import streaming_hash

def interactions(values):
    full=values['hold']-values['decay_full']-values['delta_full']+values['full']
    oracle=values['hold_oracle']-values['decay_oracle']-values['delta_oracle']+values['oracle']
    return {'gates_full':full,'gates_oracle':oracle,
        'reader_decay':values['decay_oracle']-values['decay_full']-values['oracle']+values['full'],
        'reader_delta':values['delta_oracle']-values['delta_full']-values['oracle']+values['full'],
        'three_way':oracle-full}

def export(audit,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    report=json.loads(audit.read_text());run=Path(report['source_run']);plan=read_plan()
    if report['status']!='completed_and_audited_gate_components' or audit_run(run)!=report:
        raise ValueError('Complete unchanged primary audit required')
    _,cases,registry,queries,policies,archived=registered_inputs()
    rows={n:{r:([json.loads(s) for s in (run/f'{n}.{r}.jsonl').read_text().splitlines()] if r in REGIMES else archived[n][r])
        for r in UNIFIED} for n in MODELS}
    indexed={n:{r:{v['id']:v for v in cell} for r,cell in cells.items()} for n,cells in rows.items()}
    metrics=report['metrics'];counts={};main=[];pairs=[];factorial=[]
    for n in MODELS:
        counts[n]={}
        for r in UNIFIED:
            bound={(b['case_id'],b['variant'],b['role']):indexed[n][r][b['physical_id']] for b in registry['bindings']}
            joint={c['id']:all(bound[c['id'],v,role]['greedy_exact'] for v in ('control','treatment')
                for role in ('target_current','target_initial','protected_current')) for c in cases}
            roles={role:sum(bound[c['id'],v,role]['greedy_exact'] for c in cases for v in ('control','treatment'))
                for role in ('target_current','target_initial','protected_current')}
            count={'joint':sum(joint.values()),'roles':roles,'physical':sum(v['greedy_exact'] for v in rows[n][r]),
                'joint_by_depth':{str(d):sum(joint[c['id']] for c in cases if c['dependency_depth']==d) for d in (1,2,3)}}
            s=metrics['models'][n][r]
            if (count['joint']!=s['joint_six_after'] or count['physical']!=s['physical_correct'] or
                any(roles[role]!=sum(s['role_variant_counts'][role][v]['correct'] for v in ('control','treatment')) for role in roles) or
                any(joint[c['case_id']]!=c['joint_after'] for c in s['case_outcomes'])):
                raise ValueError('Independent raw joint/role counts disagree')
            counts[n][r]=count
            for row in rows[n][r]:
                main.append({'model':n,'regime':r,'measurement_origin':'new_primary' if r in REGIMES else 'archived_reference',
                    'gate_component':SETTINGS[r][0],'reader_policy':'oracle' if SETTINGS[r][1] else 'full',
                    'targeted_token_count':policies[row['id']]['held_token_count'] if SETTINGS[r][0]!='none' else 0,
                    **{k:row[k] for k in ('id','token_policy','prompt_ids_sha256','answer','generated','greedy_exact','answer_nll','query_tokens')}})
        for candidate,reference in plan['contrasts']:
            values=[]
            for b in registry['bindings']:
                if b['variant']=='before' or b['role']!='target_current':continue
                c=indexed[n][candidate][b['physical_id']];r=indexed[n][reference][b['physical_id']]
                values.append({'model':n,'candidate':candidate,'reference':reference,**b,
                    'candidate_correct':c['greedy_exact'],'reference_correct':r['greedy_exact'],
                    'gain':c['greedy_exact'] and not r['greedy_exact'],'loss':r['greedy_exact'] and not c['greedy_exact'],
                    'nll_candidate_minus_reference':c['answer_nll']-r['answer_nll']})
            expected=metrics['paired_contrasts'][n][candidate+'_vs_'+reference]['current_target']
            if (sum(v['gain'] for v in values)!=expected['gains'] or sum(v['loss'] for v in values)!=expected['losses'] or
                [{k:v[k] for k in expected['individual_pairs'][0]} for v in values]!=expected['individual_pairs']):
                raise ValueError('Every independently computed repair/regression must match audit')
            pairs.extend(values)
        calculated=[]
        for q in registry['physical']:
            key=q['id'];values={r:indexed[n][r][key]['answer_nll'] for r in UNIFIED};delta=interactions(values)
            factorial.append({'model':n,'id':key,**{f'nll_{r}':v for r,v in values.items()},**delta})
            calculated.append({'id':key,**delta})
        if calculated!=metrics['loss_interactions'][n]['all_physical_values']:raise ValueError('All five raw NLL contrasts must match')
        by_id={v['id']:v for v in calculated}
        for key in calculated[0]:
            if key=='id':continue
            if (mean(v[key] for v in calculated)!=metrics['loss_interactions'][n]['physical_means'][key] or
                mean(by_id[b['physical_id']][key] for b in registry['bindings'] if b['role']=='target_current' and b['variant']!='before')
                !=metrics['loss_interactions'][n]['current_binding_means'][key]):raise ValueError('Interaction aggregation differs')
    sizes=[len(main),len(pairs),len(factorial)]
    if sizes!=[13440,4800,1680] or sum(v['measurement_origin']=='new_primary' for v in main)!=6720:
        raise ValueError('Explicit new/archive coverage required')
    suffixes=('.csv','.pairs.csv','.factorial.csv','.png','.pdf','.data.json','.source.py','.provenance.json')
    if any(prefix.with_suffix(s).exists() for s in suffixes):raise ValueError('Preserve prior exports')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    for suffix,data in zip(('.csv','.pairs.csv','.factorial.csv'),(main,pairs,factorial)):
        with prefix.with_suffix(suffix).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        with prefix.with_suffix(suffix).open(newline='') as f:reread=list(csv.DictReader(f))
        for expected,actual in zip(data,reread,strict=True):
            if any(actual[k]!=str(v) or (isinstance(v,float) and float(actual[k])!=v) for k,v in expected.items()):
                raise ValueError('CSV field or numeric roundtrip changed')
    colors=dict(zip(UNIFIED,['#778da0','#d39337','#b46558','#b49b68','#2b8a73','#56a4a0','#8266ad','#5369a8']))
    names=['Native','Shared all / all','Shared question / all','Shared question / question','GDN adapters all / all']
    cells=[(n,r) for n in MODELS for r in UNIFIED]
    labels=[f'{name} | {r}'+(' [new]' if r in REGIMES else ' [archive]') for name in names for r in UNIFIED]
    fig,axes=plt.subplots(1,3,figsize=(18,20))
    for ax,field,denom,title in zip(axes,['joint','current','physical'],[48,96,336],
        ['All six post-update answers','Current target after update','All original questions']):
        values=[counts[n][r]['roles']['target_current'] if field=='current' else counts[n][r][field] for n,r in cells]
        ax.barh(np.arange(40),values,color=[colors[r] for n,r in cells]);ax.invert_yaxis()
        ax.set(yticks=np.arange(40),yticklabels=labels if ax is axes[0] else [],xlim=(0,denom*1.12),title=title,xlabel=f'Correct / {denom}')
        for i,v in enumerate(values):ax.text(v+denom*.015,i,str(v),va='center',fontsize=8)
        for p in (7.5,15.5,23.5,31.5):ax.axhline(p,color='#ddd',lw=.7)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Gate-component decomposition — all five fixed configurations and eight conditions',fontsize=15)
    fig.text(.01,.035,'6720 new + 6720 archived readouts; reused 8 source streams. Original tokens / positions fixed. All six Attention layers remain.',fontsize=10)
    fig.text(.01,.018,'Query-dependent support is privileged. Beta removes both Delta erase and insertion. No semantic-specificity or architecture claim.',fontsize=10)
    fig.tight_layout(rect=[0,.055,1,.97]);fig.savefig(prefix.with_suffix('.png'),dpi=180);fig.savefig(prefix.with_suffix('.pdf'));plt.close(fig)
    write_json(prefix.with_suffix('.data.json'),{'counts':counts,'measurement_origin':metrics['measurement_origin'],
        'loss_interaction_means':{n:{k:v for k,v in metrics['loss_interactions'][n].items() if k!='all_physical_values'} for n in MODELS},
        'csv_rows_verified':{'main':13440,'paired':4800,'factorial':1680},'new_primary_readouts':6720,'archived_reference_readouts':6720})
    prefix.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    write_json(prefix.with_suffix('.provenance.json'),{'source_run':str(run),'audit_sha256':streaming_hash(audit),
        'manifest_sha256':streaming_hash(run/'manifest.json'),'source_sha256':streaming_hash(Path(__file__)),
        'outputs_sha256':{prefix.with_suffix(s).name:streaming_hash(prefix.with_suffix(s)) for s in suffixes if s!='.provenance.json'}})
    print(json.dumps({'status':'all40cells_exported','new_primary':6720,'archived_reference':6720,'csv_rows_verified':sizes}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--prefix',type=Path,required=True)
    a=p.parse_args();export(a.audit,a.prefix)

if __name__=='__main__':main()
