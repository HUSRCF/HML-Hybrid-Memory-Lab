"""Descriptive independent raw grading after complete registered audit."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash


def observe(audit_path):
    audit=json.loads(audit_path.read_text());run=Path(audit['source_run']);m=json.loads((run/'manifest.json').read_text())
    if (audit['status']!='completed_and_audited_version_query' or audit['primary_readouts_audited']!=1728 or
        audit['archived_full_references_audited']!=1728 or audit['total_source_files_verified']!=254 or
        streaming_hash(run/'manifest.json')!=audit['manifest_sha256']):raise ValueError('Complete audited phase required')
    packet=Path(m['packet']);pm=json.loads((packet/'manifest.json').read_text());qp=packet/'queries.json'
    if streaming_hash(qp)!=pm['payload_sha256']['queries.json']:raise ValueError('Question bytes changed')
    queries=json.loads(qp.read_text());attrib={(r['model'],r['id']):r for r in audit['metrics']['value_attribution_all_readouts'] if r['origin']=='new'}
    plan=json.loads((run/'plan.json').read_text());old_packet=Path(plan['source_packet'])
    old_manifest=json.loads((old_packet/'manifest.json').read_text());old_path=old_packet/'queries.json'
    if streaming_hash(old_path)!=old_manifest['payload_sha256']['queries.json']:raise ValueError('Archived question metadata changed')
    old_queries=json.loads(old_path.read_text())
    old_index={(q['context_id'],q['role'],q['question_style']):key for key,q in old_queries.items()}
    totals=defaultdict(lambda:[0,0]);roles=defaultdict(lambda:[0,0]);contrasts=defaultdict(lambda:[0,0,0]);joint=defaultdict(list)
    stable=[];old_all=[];pairs=[];sources={}
    for model,state in m['models'].items():
        raw=run/f'{model}.jsonl';archive=run/f'{model}.references.jsonl'
        if streaming_hash(raw)!=state['results_sha256'] or streaming_hash(archive)!=state['references_sha256']:
            raise ValueError('Raw bytes changed')
        sources[model]={'new_sha256':streaming_hash(raw),'archived_sha256':streaming_hash(archive)}
        rows={r['id']:r for r in map(json.loads,raw.read_text().splitlines())}
        archived={r['id']:r for r in map(json.loads,archive.read_text().splitlines())}
        def correct(row,answer):
            match=re.fullmatch(r'\s*(-?\d+)\s*',row['generated']);grade=bool(match and match[1]==answer)
            if grade!=row['greedy_exact']:raise ValueError('Independent exact grading differs')
            return grade
        for key,row in rows.items():
            q=queries[key];grade=correct(row,q['answer']);style=q['question_style'];role=q['role']
            totals[(model,style)][0]+=grade;totals[(model,style)][1]+=1
            roles[(model,style,role)][0]+=grade;roles[(model,style,role)][1]+=1
            joint[(model,style,q['context_id'])].append(grade)
            if style!='state_term':continue
            other=rows[q['paired_query_id']];event=correct(other,q['answer'])
            result={'model':model,**{k:q[k] for k in ('seed','naming','episode','role','answer')},'state_id':key,'event_id':q['paired_query_id'],
                'state_correct':grade,'event_correct':event,'state_generated':row['generated'],'event_generated':other['generated']}
            pairs.append(result);v=contrasts[(model,role)];v[0]+=int(not grade and event);v[1]+=int(grade and not event);v[2]+=1
            if not grade and not event:
                result={**result,'state_category':attrib[(model,key)]['category'],'event_category':attrib[(model,q['paired_query_id'])]['category']}
                stable.append(result)
                old=archived[q['source_truth_query_id']]
                # The archived original/other-syntax pair is descriptive only.
                original_key=old_index[(q['context_id'],q['role'],'original')]
                candidate=old_index[(q['context_id'],q['role'],'entity_first')]
                if original_key!=q['source_truth_query_id'] or candidate==original_key:raise ValueError('Explicit distinct archived syntax identities required')
                old_all.append({**result,'archived_original_correct':old['greedy_exact'],
                    'archived_original_id':original_key,'archived_other_syntax_id':candidate,
                    'archived_other_syntax_correct':archived[candidate]['greedy_exact']})
    for cell in audit['metrics']['cells']:
        if cell['origin']=='new' and totals[(cell['model'],cell['question_style'])][1]!=432:raise ValueError('Unfiltered full style grid required')
    combined=defaultdict(list);joint_summary=defaultdict(lambda:[0,0])
    for (model,style,cid),values in joint.items():
        if len(values)!=3:raise ValueError('All three roles required in each same-cache condition')
        joint_summary[(model,style)][0]+=all(values);joint_summary[(model,style)][1]+=1
        combined[(model,cid)].extend(values)
    joint_both={model:{'correct':sum(all(v) for (name,_),v in combined.items() if name==model),'contexts':144} for model in m['models']}
    return {'status':'independent_complete_raw_grading_and_descriptive_pairs_verified','created_utc':datetime.now(timezone.utc).isoformat(),
        'audit_sha256':streaming_hash(audit_path),'manifest_sha256':audit['manifest_sha256'],'source_sha256':streaming_hash(Path(__file__)),
        'raw_files':sources,'new_readouts':1728,'paired_comparisons':len(pairs),'all_pairs':pairs,
        'style_totals':[{'model':k[0],'style':k[1],'correct':v[0],'questions':v[1]} for k,v in sorted(totals.items())],
        'role_totals':[{'model':k[0],'style':k[1],'role':k[2],'correct':v[0],'questions':v[1]} for k,v in sorted(roles.items())],
        'paired_role_totals':[{'model':k[0],'role':k[1],'gains':v[0],'losses':v[1],'pairs':v[2]} for k,v in sorted(contrasts.items())],
        'same_cache_all_three_roles':[{'model':k[0],'style':k[1],'correct':v[0],'contexts':v[1]} for k,v in sorted(joint_summary.items())],
        'same_cache_all_six_answers_both_styles':joint_both,'both_wordings_failed_pairs':stable,'archived_comparison_for_stable_pairs':old_all,
        'interpretation':'Descriptive post-audit observations only. Both-wording failure subset is not a preregistered new evaluation cohort. All864pairs remain listed. Names/history conditions reuse8streams. Old/new syntax comparison is not token matched. No new inference,accuracy gate or architecture claim.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prior-observations',type=Path);args=p.parse_args()
    if args.output.exists():raise ValueError('Never overwrite observations')
    result=observe(args.audit)
    if args.prior_observations:
        previous=json.loads(args.prior_observations.read_text())
        for key in ('raw_files','all_pairs','style_totals','role_totals','paired_role_totals','same_cache_all_three_roles','same_cache_all_six_answers_both_styles','both_wordings_failed_pairs'):
            if result[key]!=previous[key]:raise ValueError('Supplementary archived metadata correction may not change any new result')
        result['superseded_operational_receipt']={'path':str(args.prior_observations),'sha256':streaming_hash(args.prior_observations),
            'correction':'Use explicit context/role/style metadata for archived hashed query IDs; previous supplementary other-syntax field duplicated original ID. All new grading,pairs,stable failures and joint counts unchanged. Both receipts retained.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','new_readouts','paired_comparisons','style_totals','paired_role_totals','same_cache_all_six_answers_both_styles')}),flush=True)


if __name__=='__main__':main()
