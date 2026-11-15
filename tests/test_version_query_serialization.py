"""Real failed-smoke regression and unchanged complete audit on CPU only."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_version_query_corrected as runner
from scripts import analyze_version_query_corrected as analysis
from tests import test_version_query as base_tests
from scripts.version_query_data import sources
from scripts.run_controller_trial import write_json


class SerializationCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=json.loads(runner.CORRECTION.read_text())
        failed=json.loads((Path(cls.plan['failed_run'])/'manifest.json').read_text())
        cls.state=failed['models']['native']
        prior=json.loads((Path(base_tests.data.read_plan()['source_run'])/'manifest.json').read_text())
        cls.reference=prior['models']['native']
        base_tests.VersionQueryTests.setUpClass()
        cls.fixture=base_tests.VersionQueryTests()

    def live_state(self):
        state=deepcopy(self.state)
        for source in state['sources'].values():
            layers=source['tail_proof']['gates']['layers']
            source['tail_proof']['gates']['layers']={int(k):v for k,v in layers.items()}
        return state

    def test_real_failure_reproduced_and_only_key_representation_corrected(self):
        state=self.live_state();before=deepcopy(state)
        with self.assertRaisesRegex(ValueError,'tail or proof'):runner.BASE_VERIFY(state,self.reference)
        actual=runner.verify_model_sources(state,self.reference)
        self.assertEqual(actual,runner.BASE_VERIFY(self.state,self.reference))
        self.assertEqual(actual['normal_after_caches_exact'],8)
        self.assertEqual(state,before)
        self.assertIs(base_tests.runner.frozen.run_model.__code__,base_tests.runner.cloned_run_model().__code__)

    def test_numeric_proof_cache_and_key_corruption_remain_rejected(self):
        for kind in ('gate_value','tail_tokens','cache_sha','layer_collision','bad_key_type'):
            with self.subTest(kind=kind):
                state=self.live_state();source=next(iter(state['sources'].values()));proof=source['tail_proof']
                if kind=='gate_value':next(iter(proof['gates']['layers'].values()))['rows']+=1
                elif kind=='tail_tokens':proof['tail_tokens']+=1
                elif kind=='cache_sha':next(iter(source['cache']['tensors'].values()))['sha256']='0'*64
                elif kind=='layer_collision':proof['gates']['layers']['0']=proof['gates']['layers'][0]
                else:proof['gates']['layers'][(0,)]=proof['gates']['layers'].pop(0)
                with self.assertRaises(ValueError):runner.verify_model_sources(state,self.reference)

    def test_all246_original_sources_and_four_source_import_closure(self):
        descriptor=runner.correction_descriptor(False)
        closure=sources()|set(descriptor['source_sha256']);self.assertEqual(len(closure),250)
        for name in descriptor['source_sha256']:
            if not name.endswith('.py'):continue
            for node in ast.walk(ast.parse((runner.ROOT/name).read_text())):
                if isinstance(node,ast.ImportFrom) and node.module:
                    target=node.module.replace('.','/')+'.py'
                    if (runner.ROOT/target).is_file():self.assertIn(target,closure)

    def test_corrected_audit_reuses_exact_original_code_and_complete_both_models(self):
        clone=analysis.clone(analysis.original.audit_run,verify_model_sources=runner.verify_model_sources,__file__=analysis.__file__)
        self.assertIs(clone.__code__,analysis.original.audit_run.__code__)
        with tempfile.TemporaryDirectory() as d:
            run,packet,values=self.fixture.fixture(d);manifest=json.loads((run/'manifest.json').read_text())
            descriptor=runner.correction_descriptor(False);manifest['serialization_correction']=descriptor
            for name in descriptor['source_sha256']:
                target=run/'serialization-correction-sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((runner.ROOT/name).read_bytes())
            write_json(run/'manifest.json',manifest)
            with patch.object(analysis.original,'DESTINATION',packet),patch.object(analysis.original,'registered_inputs',return_value=values), \
                 patch('sas_memory.runtime.TextRuntime.__init__',side_effect=AssertionError('No model')):
                report=analysis.audit_run(run,True,True)
                self.assertEqual(report['primary_readouts_audited'],108)
                self.assertEqual(report['archived_full_references_audited'],1728)
                self.assertEqual(report['total_source_files_verified'],250)
                raw=run/'gdn_all_tokens.jsonl';raw.write_text('\n'.join(raw.read_text().splitlines()[:-1])+'\n')
                manifest['models']['gdn_all_tokens']['results_sha256']=base_tests.streaming_hash(raw)
                write_json(run/'manifest.json',manifest)
                with self.assertRaisesRegex(ValueError,'unfiltered'):analysis.audit_run(run,True,True)

    def test_missing_correction_provenance_rejected(self):
        with self.assertRaisesRegex(ValueError,'declare source'):runner.check_annotation({},Path('/tmp/unused-version-fix'),True)

    def test_wrapper_restores_all_three_original_bindings_on_failure(self):
        saved={k:getattr(runner.original,k) for k in ('verify_model_sources','checked_gate','write_json')}
        state=self.live_state()
        def fail(args):
            self.assertEqual(runner.original.verify_model_sources(state,self.reference)['normal_after_caches_exact'],8)
            raise RuntimeError('intentional wrapper failure')
        with patch.object(runner,'correction_descriptor',return_value={'source_sha256':{}}), \
             patch.object(runner.original,'execute_run',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'intentional'):
                runner.execute_run(type('Args',(),{'output':Path('/tmp/unused-version-fix')})())
        for key,value in saved.items():self.assertIs(getattr(runner.original,key),value)


if __name__=='__main__':unittest.main()
