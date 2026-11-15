"""Record successful remote byte verification and current scientific limits without new inference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil

ROOT=Path('/home/husrcf/Code/SAS')
BASE=Path('/media/990Pro/Code/SAS')
ART=BASE/'artifacts/memory_update'
DEST=ROOT/'materials/updates/readout-confirmation-20261116'
receipt_path=ART/'readout-confirmation-publication-20261116-01.json'
receipt=json.loads(receipt_path.read_text())
assert receipt['status']=='published_confirmation_backup_remote_assets_roundtrip_byte_verified'
assert len(receipt['remote_assets'])==3 and all(x['downloaded_bytes_verified'] for x in receipt['remote_assets'])
for f in (receipt_path,receipt_path.with_suffix('.source.py'),BASE/'logs/memory_update/readout-confirmation-publish-20261116-01.log'):
    shutil.copyfile(f,DEST/f.name)
(DEST/'finalize-publication.source.py').write_bytes(Path(__file__).read_bytes())
url=receipt['release_url']
p=DEST/'README.md';s=p.read_text()
old='Full backup packaging/publication is the next operational step. The broader hybrid-memory research goal remains active.'
assert old in s
s=s.replace(old,f"[The complete backup]({url}) is published at commit `{receipt['release_target_commit']}`. "
    f"All three remote assets were downloaded and byte-verified. The archive contains {receipt['archive_members']-1:,} files; "
    f"SHA256 `{receipt['archive_sha256']}`. The archive also contains its file manifest. "
    'The broader hybrid-memory research goal remains active.\n\n'
    'Next, register same-budget SET/COPY-mixture simple readout controls before further architecture changes. '
    'This will test whether the observed transfer/retention loss can be removed by task coverage, and establish the baseline a shared hybrid block must exceed. '
    'No next-stage training or data generation has started. The completed confirmation cohort is now observed; any later tuning on it must be treated as development.')
p.write_text(s)
p=ROOT/'README.md';s=p.read_text()
old='Complete raw tables, audits and plots are committed; full backup publication is in preparation.'
assert old in s
s=s.replace(old,'Complete raw tables, audits and plots are committed. The [complete backup]('+url+')\n'
    'is published; all three assets were downloaded and byte-verified. Next: same-budget SET/COPY-mixture\n'
    'simple controls before a shared-block method. The next stage is not yet trained or generated.')
p.write_text(s)
p=ROOT/'configs/experiment_progress.json';d=json.loads(p.read_text())
now=datetime.now(timezone.utc).isoformat()
next_step=('Register same-budget SET/COPY-mixture simple readout controls before new code,training or data; '
           'test task-coverage explanations for COPY joint regressions and protected-readout regressions. '
           'Preserve all289predecessor sources and fixed completed checkpoints;fresh fixed evaluation cohorts must be registered. '
           'No next-stage training or generation has started. Completed confirmation is now observed development for any future changes. '
           'Shared hybrid architecture advantage remains unestablished.')
entry=d['readout_confirmation']
entry.update(status='complete_audited_exported_and_published_remote_bytes_verified',
             publication_receipt=str(receipt_path),publication_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
             release_url=url,release_target_commit=receipt['release_target_commit'],
             committed_snapshot_files_verified=receipt['committed_snapshot_files_byte_verified'],
             remote_assets_verified=3,next_action=next_step)
d['learned_readout_controls']['training_progress_snapshot']={m:{'optimizer_steps':432,'fit_readouts':1728} for m in ('query','gate','query_gate')}
d.update(goal_status='active',current_phase='readout_confirmation_complete_published_next_baseline_registration_pending',
         latest_completed_phase='readout_confirmation',last_completed_phase='readout_confirmation',
         latest_published_phase='readout_confirmation',last_completed_utc=now,updated_utc=now,
         publication_receipt=str(receipt_path),next_research_step=next_step,
         current_conclusion='Fresh SET improves from814/864fixedGDN to859Query,856Gate,859joint across all8basestreams. '
            'Query/joint each add2protected errors;Gate preserves288/288. COPY joint falls14/48to9,11,10 despite similar aggregate accuracy. '
            'Same-family SET readout adaptation works;general relational transfer,necessary co-design and unique Query-address cause are unestablished. '
            'All prior failed criteria remain false.',
         latest_completed_diagnostic='6000fresh_physical_answers_and6480logical_bindings_independently_audited_all5frozen_models',
         previous_goal_turn_classification='progress',
         last_goal_turn_result='Completed all6000registered fresh readouts,independent grading,allpaired gains/regressions and alltask/depth/role/seed cells; '
            'published1606file complete backup with all3remote assets roundtrip byte verified. No scientific289source change oradditional training.')
p.write_text(json.dumps(d,indent=2)+'\n')
seal=json.loads((ART/'readout-confirmation-source-seal-20261115-01.json').read_text())
assert len(seal['source_sha256'])==289
for name,digest in seal['source_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
print(json.dumps({'status':entry['status'],'release_url':url,'all289scientific_sources_unchanged':True,
                  'goal_status':'active','next_stage_started':False}),flush=True)
