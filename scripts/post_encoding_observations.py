"""Post-hoc descriptive extraction; no model execution or new success criteria."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

AUDIT_SHA256='a1e86be89ca3ba4d3296f26659d8158c3c25727708df14b2428b7d9182260265'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or sha(a.audit)!=AUDIT_SHA256:raise ValueError('Pinned complete audit and new output required')
    audit=json.loads(a.audit.read_text());m=audit['metrics'];models=list(m['cell_metrics']);episodes=('before','unrelated-8','unrelated-64','related-8','related-64')
    native={e:{reader:{role:v['greedy_exact'] for role,v in m['strata']['native'][e+'.none.'+reader]['role'].items()}
        for reader in ('full','oracle')} for e in episodes}
    positive=[];prefix=[]
    for name in models:
        for episode in episodes[1:]:
            for mode in ('delta','both'):
                value=m['before_after'][name][episode+'.'+mode+'.oracle']['fixed_prefix_evidence']
                prefix.append({'model':name,'episode':episode,'mode':mode,**value})
                if not episode.startswith('related-'):continue
                for reader in ('full','oracle'):
                    value=m['strata'][name][episode+'.'+mode+'.'+reader]['role']['target_current']
                    positive.append({'model':name,'episode':episode,'mode':mode,'reader':reader,**value})
    result={'status':'posthoc_descriptive_observations','created_utc':datetime.now(timezone.utc).isoformat(),
        'audit_sha256':AUDIT_SHA256,'source_sha256':sha(Path(__file__)),'new_inference':False,'new_success_criteria':False,
        'independent_streams':8,'native_normal_updates_by_role_out_of16':native,
        'native_unrelated64_before_after_by_role':m['before_after']['native']['unrelated-64.none.full']['by_role'],
        'all_models_normal_full_joint6_out_of8':{n:{e:m['cell_metrics'][n][e+'.none.full']['joint6_streams'] for e in episodes} for n in models},
        'delta_and_both_new_value_positive_controls':positive,'delta_and_both_fixed_prefix_oracle':prefix,
        'old_prefix_oracle_correct':sum(v['both_correct']+v['gain'] for v in prefix),'old_prefix_oracle_readouts':sum(v['pairs'] for v in prefix),
        'new_value_correct':sum(v['greedy_exact'] for v in positive),'new_value_readouts':sum(v['readouts'] for v in positive),
        'scope':'Repeated diagnostic cells over8streams. Oracle access is privileged. Changes to tail KV, Query and convolution prevent unique storage localization. New-value evidence is newly encoded, unlike fixed prefix evidence.',
        'next_mechanistic_question':'At a fixed post-tail boundary, distinguish recurrent/convolution state effects from tail-KV encoding before training a new architecture.'}
    assert result['old_prefix_oracle_correct']==result['old_prefix_oracle_readouts']==1600
    assert result['new_value_correct']==0 and result['new_value_readouts']==640
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');a.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:result[k] for k in ('status','old_prefix_oracle_correct','old_prefix_oracle_readouts','new_value_correct','new_value_readouts')}))

if __name__=='__main__':main()
