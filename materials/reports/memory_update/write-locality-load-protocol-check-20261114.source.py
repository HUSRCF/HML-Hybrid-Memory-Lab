"""Tokenizer/executable-truth feasibility only; no model or dataset registration."""
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT=Path('/home/husrcf/Code/SAS')
sys.path.insert(0,str(ROOT))
from sas_memory.tasks import Operation, execute, INSTRUCTIONS, render
from sas_memory.training_data import data_fingerprint
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.run_controller_trial import write_json

plan_path=ROOT/'configs/semantic_write_locality_load_plan.json'
plan=json.loads(plan_path.read_text())
grid=plan['development_grid']
model=json.loads((ROOT/'configs/models.json').read_text())['it']
runtime=tokenizer_runtime(model)
lengths=[]
by_cell={}
by_load={}
by_value={}
for seed in grid['base_seeds']:
    names=[f'E{i}' for i in range(700,900)]
    random.Random(seed).shuffle(names)
    a,b,c=names[:3]
    all_values=random.Random(seed+410009).sample(range(100,1000),4*162)
    for load in grid['active_accounts']:
        rng=random.Random(seed+310003+load)
        remaining=[]
        while len(remaining)<157:
            cycle=names[3:load].copy()
            rng.shuffle(cycle)
            remaining.extend(cycle)
        targets=names[:3]+remaining[:157]
        assert len(targets)==160 and len(set(targets))==load
        for version in range(4):
            values=all_values[version*162:(version+1)*162]
            prefix=[Operation('SET',target,value) for target,value in zip(targets,values)]
            for cell in grid['tail_cells']:
                tail=[Operation('SET',a,values[160] if cell[1]=='1' else values[0]),
                      Operation('SET',c,values[161] if cell[-1]=='1' else values[2])]
                events=prefix+tail
                truth=execute(events)
                assert truth['current'][b]==values[1] and truth['initial'][a]==values[0]
                ledger=runtime.prompt_start+INSTRUCTIONS+render([vars(event) for event in events])
                ledger_ids=runtime.ids(ledger)
                for role,entity,kind in [('target_current',a,'current'),('protected_current',b,'current'),('target_initial',a,'initial')]:
                    question=(f'What is the current balance of account {entity}?' if kind=='current'
                              else f'What was the initial balance of account {entity}?')
                    prompt=ledger+runtime.query_suffix(question)
                    tokens=runtime.tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
                    ids=tokens['input_ids']
                    offsets=tokens['offset_mapping']
                    assert ids[:len(ledger_ids)]==ledger_ids
                    assert len(ids)<=grid['maximum_prompt_tokens']
                    matches=list(re.finditer(r'\b\d{3}\b',ledger))
                    assert len(matches)==162
                    # Map each character to its token once; do not scan the
                    # entire token sequence separately for every literal.
                    characters={char:index for index,(left,right) in enumerate(offsets) for char in range(left,right)}
                    spans=[tuple(characters[match.start()+i] for i in range(3)) for match in matches]
                    assert all(len(set(span))==3 for span in spans)
                    assert all([tuple(offsets[index]) for index in span]==[(match.start()+i,match.start()+i+1) for i in range(3)] for match,span in zip(matches,spans))
                    positions={index for span in spans for index in span}
                    signature=(len(ledger_ids),len(ids),data_fingerprint(ids[len(ledger_ids):]))
                    key=(seed,version,role)
                    assert by_load.setdefault(key,signature)==signature
                    normalized=data_fingerprint([-1 if index in positions else token for index,token in enumerate(ids)])
                    key=(seed,load,cell,role)
                    assert by_value.setdefault(key,normalized)==normalized
                    assert len(runtime.continuation_ids(prompt,str(truth[kind][entity])))==3
                    by_cell.setdefault((seed,load,version,role),{})[cell]=(ids,spans)
                    lengths.append(len(ids))
for cells in by_cell.values():
    baseline,spans=cells['A0_C0']
    for cell,(ids,actual_spans) in cells.items():
        assert actual_spans==spans
        allowed=set(spans[160] if cell[1]=='1' else ()) | set(spans[161] if cell[-1]=='1' else ())
        changed={index for index,(left,right) in enumerate(zip(baseline,ids)) if left!=right}
        assert changed<=allowed
        if allowed:assert changed
assert len(lengths)==grid['expected_free_answers']==576
output=Path('/media/990Pro/Code/SAS/artifacts/memory_update/write-locality-load-protocol-check-20261114.json')
report={'status':'tokenizer_and_executable_truth_protocol_feasibility_passed',
        'model_inference':False,'dataset_registered':False,'planned_prompt_count':len(lengths),
        'planned_unique_prefixes':192,'minimum_prompt_tokens':min(lengths),'maximum_prompt_tokens':max(lengths),
        'all_load_and_tail_lengths_and_query_suffixes_matched':True,'tail_changes_confined_to_registered_value_digits':True,
        'four_value_siblings_differ_only_at_value_digit_tokens':True,'all_gold_answers_three_tokens':True,
        'protocol_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'interpretation':'Feasibility check only. No model inference, frozen dataset manifest, native results, architecture result or confirmation claim.'}
write_json(output,report)
output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(report))
