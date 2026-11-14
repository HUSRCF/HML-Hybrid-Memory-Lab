"""Pinned completed timing evidence; never rescan a past novelty inventory."""

from datetime import datetime
import copy
import json
from pathlib import Path

from sas_memory.training_data import data_fingerprint
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_blocked_reader_adaptation import tokenizer_runtime
from scripts.copy_readout_data import build_masks,sample_ids,safety_samples,selected_inputs
from scripts.copy_readout_runtime import readout
from scripts.trained_shared_timing_data import sources as previous_sources,unique_prefix_queries

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'configs/trained_timing_evidence_access_plan.json'
PROTOCOL_SHA256='c4a5e0e20c423c5a4764860ff4bf4934e739b73a0962143e86b518867141e8cc'
DESTINATION=Path('/media/990Pro/Code/SAS/datasets/memory_update/trained-timing-evidence-access-v1')
MODELS=('native','shared','shared_question_trained_all','shared_question_trained_question','gdn_all_tokens')
REGIMES=('full','oracle','nonslice');MASKED_REGIMES=REGIMES[1:]
DATA_SOURCES=previous_sources()|{'configs/trained_timing_evidence_access_plan.json',
    'scripts/trained_timing_evidence_access_data.py','scripts/prepare_trained_timing_evidence_access.py'}


def sources():
    return DATA_SOURCES|{'scripts/run_trained_timing_evidence_access.py',
        'scripts/trained_timing_evidence_access_metrics.py','scripts/analyze_trained_timing_evidence_access.py'}


def model_order(reverse=False):return list(MODELS[::-1] if reverse else MODELS)


def read_plan():
    if streaming_hash(PROTOCOL)!=PROTOCOL_SHA256:raise ValueError('Pre-mask diagnostic registration changed')
    p=json.loads(PROTOCOL.read_text())
    if (p['model_order']!=list(MODELS) or list(p['models'])!=list(MODELS) or p['regimes']!=list(REGIMES) or
        p['masked_regimes']!=list(MASKED_REGIMES) or Path(p['dataset_destination'])!=DESTINATION or
        any(p[k] for k in ('training','new_adapters','new_models','new_task_cases','new_environment',
            'architecture_effectiveness_claim','can_change_prior_gates','prospective_unseen_confirmation')) or
        not p['local_files_only'] or p['smoke_readouts_per_order']!=270 or
        p['counts']!={'base_streams':8,'cases':48,'physical_queries':336,'related_bindings':432,'models':5,
            'regimes':3,'primary_physical_readouts':5040,'primary_related_bindings':6480,'fresh_full_replays':1680,'masked_readouts':3360}):
        raise ValueError('Complete reused-cohort diagnosis or fixed model/regime allocation changed')
    return p


def source_inputs(plan):
    run=Path(plan['source_run']);m=json.loads((run/'manifest.json').read_text());ap=Path(plan['source_audit']);a=json.loads(ap.read_text())
    data=Path(plan['source_dataset']);reg=json.loads((data/'manifest.json').read_text())
    cases=json.loads((data/'cases.json').read_text());registry=json.loads((data/'registry.json').read_text())
    if (m['status']!='completed' or m['smoke'] or len(m['completed_models'])!=16 or
        a['status']!='completed_and_audited_trained_shared_timing' or not a['operational_controls_passed'] or
        a['counts']['physical_readouts']!=5376 or any(a['metrics']['registered_criteria'].values()) or
        streaming_hash(run/'manifest.json')!=plan['source_manifest_sha256'] or a['manifest_sha256']!=plan['source_manifest_sha256'] or
        streaming_hash(ap)!=plan['source_audit_sha256'] or streaming_hash(ap.with_suffix('.source.py'))!=a['analysis_source_sha256'] or
        streaming_hash(data/'manifest.json')!=plan['source_dataset_manifest_sha256'] or
        streaming_hash(ROOT/plan['source_protocol'])!=plan['source_protocol_sha256'] or
        data_fingerprint(cases)!=plan['source_cases_sha256'] or data_fingerprint(registry)!=plan['source_registry_sha256'] or
        reg['cases_sha256']!=plan['source_cases_sha256'] or reg['registry_sha256']!=plan['source_registry_sha256'] or
        set(m['source_sha256'])!=previous_sources() or len(m['source_sha256'])!=160 or
        m['weights_before']!=m['weights_after'] or m['weights_after']['sha256']!=plan['native_loaded_weights_sha256'] or
        m['model_path']!=plan['model_path'] or m['packages']!=plan['packages'] or m['arithmetic_profile']!=plan['arithmetic_profile'] or
        streaming_hash(Path(plan['precision_addendum']))!=plan['precision_addendum_sha256'] or
        not datetime.fromisoformat(m['completed_utc'])<datetime.fromisoformat(plan['registered_utc'])):
        raise ValueError('Completed full16-arm negative evidence, original inputs or chronology changed')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(run/'sources'/name)!=digest:raise ValueError('Preserve all160sealed sources')
    for name,digest in reg['source_sha256'].items():
        if streaming_hash(data/'sources'/name)!=digest or streaming_hash(ROOT/name)!=digest:raise ValueError('Original registered data source changed')
    source_plan=json.loads((run/'plan.json').read_text());refs={'source_audit_sha256':streaming_hash(ap),
        'source_manifest_sha256':streaming_hash(run/'manifest.json'),'prior_timing_criteria_unchanged':a['metrics']['registered_criteria'],
        'prior_fresh_gates_unchanged':a['prior_fresh_gates_unchanged'],'training':{},'all16_raw_sha256':{},
        'actual_eos_token_id':m['actual_eos_token_id'],'generation_config_sha256':m['generation_config_sha256'],
        'installed_modeling_source':m['references']['installed_modeling_source'],
        'installed_modeling_source_sha256':m['installed_modeling_source_sha256']}
    if (streaming_hash(Path(refs['installed_modeling_source']))!=refs['installed_modeling_source_sha256'] or
        streaming_hash(run/'installed_modeling_source.py')!=refs['installed_modeling_source_sha256'] or
        {p.name:streaming_hash(p) for p in Path(plan['model_path']).glob('generation_config.json')}!=refs['generation_config_sha256']):
        raise ValueError('Native installed implementation or generation configuration changed')
    rows={}
    for name,state in m['models'].items():
        path=run/f'{name}.jsonl';digest=streaming_hash(path)
        if digest!=state['results_sha256']:raise ValueError('Retain every complete16-arm raw reference')
        refs['all16_raw_sha256'][name]=digest
        if name not in MODELS:continue
        arm=source_plan['confirmation']['evaluation_arms'][name];spec=reg['models'][arm['model']]
        expected=None if spec is None else {**spec,'token_policy':arm['token_policy']}
        if expected!=plan['models'][name]:raise ValueError('Selected frozen model or deployment policy changed')
        if spec:
            if streaming_hash(Path(spec['checkpoint']))!=spec['checkpoint_sha256']:raise ValueError('Fixed checkpoint changed')
            refs['training'][name]={'checkpoint_state_sha256':state['adapter_state_sha256_before'],
                'checkpoint_sha256':state['checkpoint_sha256']}
        rows[name]=[json.loads(line) for line in path.read_text().splitlines()]
    rows={n:rows[n] for n in MODELS}
    return cases,registry,rows,refs


def registered_inputs(path=DESTINATION):
    path=Path(path);plan=read_plan();cases,registry,rows,refs=source_inputs(plan)
    m=json.loads((path/'manifest.json').read_text());masks=json.loads((path/'masks.json').read_text())
    if (path!=DESTINATION or m['status']!='registered_before_inference' or m['training'] or m['model_inference'] or
        m['protocol_sha256']!=PROTOCOL_SHA256 or m['counts']!=plan['counts'] or m['references']!=refs or
        json.loads((path/'plan.json').read_text())!=plan or m['cases_sha256']!=data_fingerprint(cases) or
        m['registry_sha256']!=data_fingerprint(registry) or m['mask_registry_sha256']!=data_fingerprint(masks) or
        m['sample_query_ids']!=sample_ids(cases,registry) or set(m['source_sha256'])!=DATA_SOURCES or
        not datetime.fromisoformat(plan['registered_utc'])<datetime.fromisoformat(m['generation_started_utc'])<datetime.fromisoformat(m['registered_utc'])):
        raise ValueError('Frozen original cohort, all masks or registered chronology changed')
    for name,digest in m['source_sha256'].items():
        if streaming_hash(ROOT/name)!=digest or streaming_hash(path/'sources'/name)!=digest:raise ValueError('Mask source closure changed')
    if masks!=build_masks(cases,registry,tokenizer_runtime(plan['model_path'])):raise ValueError('Executable truth or original-token mask reconstruction changed')
    return m,{'models':copy.deepcopy(plan['models'])},cases,registry,masks,rows


def require_exact_full(actual,expected,model,query_id):
    if readout(actual)!=readout(expected):raise ValueError(f'Original five-field canonical full readout drift: {model},{query_id}')


def checked_smoke_equivalence(path,dataset_digest):
    from scripts.analyze_trained_timing_evidence_access import audit_equivalence
    path=Path(path);r=json.loads(path.read_text())
    if (r!=audit_equivalence(Path(r['forward_audit']),Path(r['reverse_audit'])) or r['dataset_manifest_sha256']!=dataset_digest or
        streaming_hash(path.with_suffix('.source.py'))!=r['analysis_source_sha256']):raise ValueError('Exact opposite270-readout smoke audits required')
    return {'path':str(path),'sha256':streaming_hash(path)}
