"""Pinned, paired training-timing protocol; no changes to sealed model code."""

from datetime import datetime
import json
from pathlib import Path

from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.run_controller_trial import state_fingerprint
from scripts.unseen_shared_timing_data import references as old_references,read_plan as old_plan
from scripts.unseen_shared_timing_runtime import sources as previous_sources

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'configs/matched_shared_training_timing_plan.json'
PROTOCOL_SHA256='027e1d8fcfde9478115e1d238b0528497e6285ca3b1a80086fe332389da5405f'
ARMS=('shared_question','shared_ledger')


def sources():
    return previous_sources()|{'configs/matched_shared_training_timing_plan.json',
        'scripts/matched_shared_timing_common.py','scripts/run_matched_shared_training_timing.py',
        'scripts/analyze_matched_shared_training_timing.py'}


def validate_plan(plan,base):
    changed={'status','registered_utc','purpose','profile','arms','initial_state_sha256','gate','requirements','interpretation'}
    expected={name:{'mode':'shared','rank':64,'token_policy':policy,'parameters':694272}
        for name,policy in zip(ARMS,('question','ledger'))}
    confirm=plan['confirmation']
    order=['native','shared','shared_question_timing','shared_ledger_timing',
        *[f'{arm}_trained_{p}' for arm in ARMS for p in ('all','question','ledger')],
        'independent','local','local_question','attention_only','gdn_all_tokens','gdn_question_only']
    if (any(plan.get(k)!=v for k,v in base.items() if k not in changed) or
        plan['profile']!='matched_shared_training_timing' or plan['status']!='registered_before_new_training_and_unseen_confirmation' or
        plan['arms']!=expected or list(plan['arms'])!=list(ARMS) or
        plan['initial_state_sha256']!={n:base['initial_state_sha256']['shared'] for n in ARMS} or
        plan['maximum_new_optimizer_updates']!=864 or not plan['new_training'] or
        any(plan[k] for k in ('new_backbone','new_environment','architecture_effectiveness_claim','explicit_semantic_controller_implemented','can_change_prior_gates')) or
        confirm['fresh_seeds']!=list(range(63000,63008)) or confirm['dependency_depths']!=[1,2,3] or
        confirm['question_styles']!=['original','entity_first'] or confirm['distractor_updates_by_seed_index']!=[8,64]*4 or
        confirm['evaluation_order']!=order or list(confirm['evaluation_arms'])!=order or
        confirm['counts']!={'base_streams':8,'related_cases':48,'physical_queries':336,'related_bindings':432,
            'evaluation_arms':16,'primary_physical_full_answer_readouts':5376,'primary_related_bindings':6912} or
        confirm['smoke_physical_readouts_per_order']!=288):
        raise ValueError('Paired training allocation, optimization budget or complete prospective confirmation changed')
    old=old_plan()
    for name,spec in old['evaluation_arms'].items():
        if confirm['evaluation_arms'][name]!=spec:raise ValueError('Existing baseline or shared deployment schedule changed')
    for arm in ARMS:
        for policy in ('all','question','ledger'):
            if confirm['evaluation_arms'][f'{arm}_trained_{policy}']!={'model':arm,'token_policy':policy,'new_training':True}:
                raise ValueError('Complete3x3training/deployment factorial required')


def read_plan():
    if streaming_hash(PROTOCOL)!=PROTOCOL_SHA256:raise ValueError('Pre-training protocol changed')
    plan=json.loads(PROTOCOL.read_text());base=json.loads((ROOT/'configs/order_augmented_semantic_plan.json').read_text())
    if streaming_hash(ROOT/'configs/order_augmented_semantic_plan.json')!=plan['augmentation_plan_sha256']:
        raise ValueError('Original training plan changed')
    validate_plan(plan,base);return plan


def references(plan):
    run=Path(plan['previous_timing_run']);m=json.loads((run/'manifest.json').read_text())
    a=Path(plan['previous_timing_audit']);r=json.loads(a.read_text());previous=old_plan();refs=old_references(previous)
    original=Path(plan['augmentation_reference_run']);training=json.loads((original/'manifest.json').read_text())
    if (streaming_hash(a)!=plan['previous_timing_audit_sha256'] or streaming_hash(run/'manifest.json')!=plan['previous_timing_manifest_sha256'] or
        r['status']!='completed_and_audited_unseen_shared_timing' or not r['operational_controls_passed'] or
        r['counts']['physical_readouts']!=3360 or any(r['metrics']['registered_criteria'].values()) or
        r['manifest_sha256']!=plan['previous_timing_manifest_sha256'] or m['references']!=refs or
        streaming_hash(ROOT/plan['previous_timing_plan'])!=plan['previous_timing_plan_sha256'] or
        set(m['source_sha256'])!=previous_sources() or len(m['source_sha256'])!=136 or
        plan['legacy_models']!=previous['models'] or any(plan[k]!=previous[k] for k in
            ('model_path','packages','native_loaded_weights_sha256','arithmetic_profile','precision_addendum_sha256')) or
        streaming_hash(original/'manifest.json')!=plan['augmentation_reference_manifest_sha256'] or
        streaming_hash(original/'plan.json')!=plan['augmentation_plan_sha256'] or
        streaming_hash(Path(plan['data_source'])/'manifest.json')!=plan['data_manifest_sha256'] or
        not datetime.fromisoformat(m['completed_utc'])<datetime.fromisoformat(plan['registered_utc'])):
        raise ValueError('Completed negative evidence, matched data/checkpoints or chronology changed')
    if {p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')}!=refs['generation_config_sha256']:
        raise ValueError('Original native generation/stopping configuration changed')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(run/'sources'/name)!=digest:raise ValueError('Preserve all136successful runtime sources')
    for name,state in m['models'].items():
        if streaming_hash(run/f'{name}.jsonl')!=state['results_sha256']:raise ValueError('Previous complete raw answers changed')
    native=training['native_reference'];native_path=Path(native['source_run'])
    if streaming_hash(native_path/'manifest.json')!=native['manifest_sha256'] or native['status']!='completed_and_audited':
        raise ValueError('Pinned native training reference changed')
    description=training['arms']['shared']['description']
    if (training['arms']['shared']['initial_state_sha256']!=plan['initial_state_sha256'][ARMS[0]] or
        description['trainable_parameters']!=694272 or description['mode']!='shared' or description['rank']!=64):
        raise ValueError('Original paired shared scope or initialization changed')
    return {'native':native,'previous_audit_sha256':streaming_hash(a),'previous_manifest_sha256':streaming_hash(run/'manifest.json'),
        'prior_failed_gates':refs['prior_fresh_gates'],'previous_timing_criteria':r['metrics']['registered_criteria'],
        'shared_description':description,'installed_modeling_source':refs['installed_modeling_source'],
        'installed_modeling_source_sha256':refs['installed_modeling_source_sha256'],'generation_config_sha256':refs['generation_config_sha256']}


def initial_fingerprint(description,seed):
    import torch
    from torch import nn
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed);module=nn.Module()
        for field,shapes in [('encoders','encoder_shapes'),('outputs','output_shapes')]:
            setattr(module,field,nn.ModuleDict({name:nn.Linear(shape[1],shape[0],bias=False,dtype=torch.float32,device='cpu')
                for name,shape in description[shapes].items()}))
        for output in module.outputs.values():nn.init.zeros_(output.weight)
        return state_fingerprint(module)


def scope_samples(data):
    selected={}
    for i,e in enumerate(data['validation']):selected.setdefault((e['variant'],e['query']['role']),i)
    return sorted(set(selected.values())|{len(data['validation'])-1})


def query(runtime,example,registration):
    from scripts.run_controller_trial import prompt_for
    ledger,prompt=prompt_for(runtime,example['case'],example['variant'],example['query'])
    q={**registration,**example['query'],'id':registration['prompt_ids_sha256'],
        'source_key':data_fingerprint(runtime.ids(ledger)),'question_tokens':len(runtime.continuation_ids(ledger,runtime.query_suffix(example['query']['question']))),
        'answer_ids':runtime.continuation_ids(prompt,example['query']['answer'])}
    if (runtime.ids(prompt)[:q['ledger_tokens']]!=runtime.ids(ledger) or data_fingerprint(runtime.ids(prompt))!=q['id']):
        raise ValueError('Actual development scope-control input changed')
    return ledger,q
