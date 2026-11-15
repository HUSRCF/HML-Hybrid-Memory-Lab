"""No-retry388test/seal, complete training lock, seven-model replay and full confirmation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, subprocess, sys

ROOT = Path('/home/husrcf/Code/SAS')
BASE = Path('/media/990Pro/Code/SAS')
ART = BASE / 'artifacts/memory_update'
LOG = BASE / 'logs/memory_update'
RUN = BASE / 'runs/memory_update'
sys.path.insert(0, str(ROOT))
from scripts.mixture_confirmation_data import read_plan, sources, PROTOCOL_SHA256

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():
    return datetime.now(timezone.utc).isoformat()

plan = read_plan()
receipt = ART / 'mixture-confirmation-pipeline-20261116-01.json'
seal = Path(plan['source_seal'])
smoke = RUN / 'mixture-confirmation-smoke-20261116-01'
smoke_audit = ART / 'mixture-confirmation-smoke-audit-20261116-01.json'
primary = RUN / 'mixture-confirmation-primary-20261116-01'
primary_audit = ART / 'mixture-confirmation-primary-audit-20261116-01.json'
for path in (receipt, seal, Path(plan['dataset']), smoke, smoke_audit, primary, primary_audit):
    if path.exists():
        raise ValueError('Never overwrite or retry existing stage: ' + str(path))
expected = {name: sha(ROOT / name) for name in sorted(sources())}
assert len(expected) == 303
proof = {'status': 'running', 'started_utc': now(), 'source_sha256': sha(Path(__file__)),
         'scientific_sources_at_test_start': expected, 'stages': [], 'accuracy_gate': False,
         'prior_failed_criteria_unchanged': plan['prior_failed_criteria_unchanged']}
receipt.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())

def save():
    receipt.write_text(json.dumps(proof, indent=2) + '\n')

def stage(name, args, filename, script=False):
    path = LOG / filename
    if path.exists():
        raise ValueError('Never overwrite stage log')
    item = {'stage': name, 'started_utc': now(), 'command': [sys.executable, '-B', *args], 'log': str(path)}
    proof['stages'].append(item)
    save()
    print(json.dumps({'event': 'started', 'stage': name}), flush=True)
    with path.open('x') as handle:
        result = subprocess.run(item['command'], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    item.update(returncode=result.returncode, completed_utc=now(), log_sha256=sha(path))
    save()
    if result.returncode:
        raise RuntimeError('Stage failed without retry: ' + name)
    assert {n: sha(ROOT / n) for n in expected} == expected
    print(json.dumps({'event': 'completed', 'stage': name}), flush=True)
    return path

try:
    save()
    log = stage('full388_cpu_regression', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                'mixture-confirmation-full-tests-20261116-01.log')
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$', log.read_text())
    assert match and int(match[1]) == 388
    sealed = {'status': 'sealed303mixture_confirmation_sources_before_generation', 'created_utc': now(),
              'protocol_sha256': PROTOCOL_SHA256, 'source_sha256': expected,
              'parent_source_seal_sha256': plan['parent_source_seal_sha256'],
              'full_tests_log': str(log), 'full_tests_log_sha256': sha(log),
              'full_tests_passed': 388, 'full_tests_seconds': float(match[2]),
              'all296predecessor_sources_unchanged': True,
              'future_confirmation_generated': False, 'new_native_execution_started': False,
              'pipeline_source_sha256': sha(Path(__file__))}
    seal.write_text(json.dumps(sealed, indent=2) + '\n')
    proof['source_seal_sha256'] = sha(seal)
    save()
    stage('lock_complete_training_then_register_full_cohorts', ['scripts/prepare_mixture_confirmation.py'],
          'mixture-confirmation-registration-20261116-01.log')
    stage('independent_both_cohort_inputs_and_truths', ['/tmp/check_mixture_confirmation_inputs.py'],
          'mixture-confirmation-independent-inputs-20261116-01.log')
    stage('all42_observed_final_fit_replays', ['scripts/run_mixture_confirmation.py', '--smoke', '--output', str(smoke)],
          'mixture-confirmation-smoke-20261116-01.log')
    stage('independent42_replay_audit', ['scripts/analyze_mixture_confirmation.py', '--run', str(smoke),
          '--allow-smoke', '--output', str(smoke_audit)], 'mixture-confirmation-smoke-audit-20261116-01.log')
    stage('all10800_frozen_primary_readouts', ['scripts/run_mixture_confirmation.py', '--output', str(primary),
          '--smoke-audit', str(smoke_audit)], 'mixture-confirmation-primary-20261116-01.log')
    stage('independent_all16800_physical_answer_audit', ['scripts/analyze_mixture_confirmation.py', '--run', str(primary),
          '--output', str(primary_audit)], 'mixture-confirmation-primary-audit-20261116-01.log')
    proof.update(status='completed_full_seven_model_observed_and_fresh_confirmation_and_audit', completed_utc=now())
    save()
    print(json.dumps({'status': proof['status'], 'audit': str(primary_audit)}), flush=True)
except BaseException as error:
    proof.update(status='failed_no_retry', completed_utc=now(), error=f'{type(error).__name__}: {error}')
    save()
    raise
