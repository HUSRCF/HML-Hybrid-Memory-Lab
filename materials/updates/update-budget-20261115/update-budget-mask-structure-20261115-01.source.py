"""Inspect locked mask combinatorics without reading predictions or loading a model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from math import comb
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.source.py').exists():
        raise ValueError('Preserve existing structural receipts')
    p=args.dataset/'policies.json';policies=json.loads(p.read_text())
    cells=Counter();copy_matches=Counter();copy_single_choice=[]
    for q in policies.values():
        available=Counter(v['kind'] for v in q['record_blocks'])
        required=Counter({v['kind']:v['count'] for v in q['budget_signature']})
        for kind in ('SET','COPY'):
            cells[kind,available[kind],required[kind],comb(available[kind],required[kind])]+=1
        copy_single_choice.append(comb(available['COPY'],required['COPY'])==1)
        support=set(q['oracle_event_indices'])
        original_copy=[v['index'] for v in q['record_blocks'] if v['kind']=='COPY' and v['index'] in support]
        for family,indices in q['selected_event_indices'].items():
            actual=[v['index'] for v in q['record_blocks'] if v['kind']=='COPY' and v['index'] in set(indices)]
            copy_matches[family]+=actual==original_copy
    out={'status':'locked_data_only_structure_audited','created_utc':datetime.now(timezone.utc).isoformat(),
        'model_inference':False,'predictions_read':False,'policy_sha256':sha(p),
        'dataset_manifest_sha256':sha(args.dataset/'manifest.json'),'source_sha256':sha(Path(__file__)),
        'physical_queries':len(policies),'choices':[{'kind':k,'available':a,'selected':s,'possible_subsets':c,'queries':n}
            for (k,a,s,c),n in sorted(cells.items())],
        'all_copy_strata_have_one_possible_selection':all(copy_single_choice),
        'copy_selection_equals_original_support_per_family':dict(copy_matches),
        'interpretation':'All336queries select exactly one SET record. Controls vary SET identity among16/17/72/73 candidates. COPY is either fully included or excluded by the oracle-derived type budget; every family has exactly the original COPY selection. This does not test discovering a COPY dependency chain or choosing between alternative COPY paths.'}
    if len(policies)!=336 or not all(copy_single_choice) or set(copy_matches.values())!={336}:
        raise ValueError('Locked data structural expectations changed')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
