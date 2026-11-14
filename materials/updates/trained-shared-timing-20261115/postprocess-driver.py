"""Run deterministic exports and backup only after the live primary/audit pipeline ends."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('/home/husrcf/Code/SAS');base=Path('/media/990Pro/Code/SAS')
with Path('/tmp/hml-post-trained-confirmation-20261115.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent=Path('/proc/2108829/cmdline')
    print(json.dumps({'status':'waiting_for_live_primary_and_independent_audit_pipeline','parent_pid':2108829}),flush=True)
    while parent.exists():
        try:command=parent.read_bytes()
        except FileNotFoundError:break
        if b'run_trained_shared_timing.py' not in command:break
        time.sleep(5)
    audit=base/'artifacts/memory_update/trained-shared-timing-audit-20261115-01.json'
    report=json.loads(audit.read_text())
    if report['status']!='completed_and_audited_trained_shared_timing' or report['counts']['physical_readouts']!=5376:
        raise ValueError('Complete independent primary audit required before postprocessing')
    jobs=[(['scripts/export_trained_shared_timing.py',str(audit),'--prefix',str(base/'artifacts/memory_update/trained-shared-timing-figure-20261115-01')],base/'logs/trained-shared-timing-export-20261115-01.log'),
          (['scripts/package_trained_timing_confirmation.py','--output',str(base/'backups/trained-shared-timing-confirmation-20261115-01')],base/'logs/trained-shared-timing-backup-20261115-01.log')]
    for argv,log in jobs:
        print(json.dumps({'status':'starting','script':argv[0],'log':str(log)}),flush=True)
        with log.open('x') as out:subprocess.run([sys.executable,*argv],cwd=root,stdout=out,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'status':'completed_exports_and_byte_verified_backup','primary_audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest()}),flush=True)
