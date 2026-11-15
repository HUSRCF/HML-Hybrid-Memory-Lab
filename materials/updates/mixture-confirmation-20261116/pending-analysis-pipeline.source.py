"""Wait for the existing terminal successful audit; export once, never retry or infer."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess, sys, time
BASE = Path('/media/990Pro/Code/SAS'); ROOT = Path('/home/husrcf/Code/SAS')
ART = BASE / 'artifacts/memory_update'
receipt = ART / 'mixture-confirmation-analysis-pipeline-20261116-01.json'
parent = ART / 'mixture-confirmation-pipeline-20261116-01.json'
source = Path('/tmp/export_mixture_confirmation.py')
log = BASE / 'logs/memory_update/mixture-confirmation-complete-analysis-20261116-01.log'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert not receipt.exists() and not log.exists()
proof = {'status': 'waiting_for_all10800readouts_and_terminal16800row_audit',
         'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__)),
         'export_source_sha256': sha(source), 'parent_pipeline': str(parent), 'new_model_inference': False}
receipt.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
def save():
    receipt.write_text(json.dumps(proof, indent=2) + '\n')
try:
    save()
    while True:
        try:
            state = json.loads(parent.read_text())
        except json.JSONDecodeError:
            time.sleep(1)
            continue
        if state['status'] == 'failed_no_retry':
            raise RuntimeError('Parent failed; do not analyze partial outcomes or retry')
        if state['status'] == 'completed_full_seven_model_observed_and_fresh_confirmation_and_audit':
            break
        time.sleep(10)
    assert sha(source) == proof['export_source_sha256']
    proof.update(status='running_complete_independent_exports', parent_terminal_sha256=sha(parent),
                 analysis_started_utc=datetime.now(timezone.utc).isoformat())
    save()
    with log.open('x') as handle:
        result = subprocess.run([sys.executable, '-B', str(source)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    proof.update(returncode=result.returncode, log_sha256=sha(log), completed_utc=datetime.now(timezone.utc).isoformat())
    if result.returncode:
        raise RuntimeError('Complete analysis failed without retry')
    proof['status'] = 'completed_all16800physical_and18144logical_exports'
    save()
    print(json.dumps(proof), flush=True)
except BaseException as error:
    proof.update(status='failed_no_retry', error=f'{type(error).__name__}: {error}', completed_utc=datetime.now(timezone.utc).isoformat())
    save()
    raise
