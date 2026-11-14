"""The source-only addendum preserves the original actual safety execution."""

import json
from pathlib import Path
import unittest

from scripts.analyze_matched_shared_training_timing_verified import registered_addendum,runtime_sources
from scripts.matched_shared_timing_common import read_plan,sources,PROTOCOL_SHA256


class CompleteTrainingInventoryTests(unittest.TestCase):
    def test_every_actual_dependency_is_preserved_not_dropped_to_match_declared_core(self):
        a=registered_addendum();m=json.loads((Path(a['original_smoke_run'])/'manifest.json').read_text())
        self.assertEqual(set(m['source_sha256'])-sources(),set(a['six_original_trainer_dependencies']))
        self.assertEqual(len(m['source_sha256']),146);self.assertEqual(len(runtime_sources()),155)
        self.assertTrue(set(m['source_sha256'])<runtime_sources())

    def test_inventory_correction_cannot_change_scientific_protocol_or_training_budget(self):
        a=registered_addendum();p=read_plan()
        self.assertEqual(a['scientific_protocol_sha256'],PROTOCOL_SHA256)
        self.assertFalse(a['changes_model_data_loss_budget_or_hypotheses'])
        self.assertTrue(a['preserve_scientific_plan'])
        self.assertEqual((p['maximum_optimizer_steps'],p['maximum_new_optimizer_updates']),(432,864))
        self.assertEqual(p['confirmation']['counts']['primary_physical_full_answer_readouts'],5376)

    def test_verified_inventory_is_closed_under_first_party_imports(self):
        import ast
        names=runtime_sources()
        for name in names:
            if not name.endswith('.py'):continue
            for node in ast.walk(ast.parse(Path(name).read_text())):
                modules=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                    [alias.name for alias in node.names] if isinstance(node,ast.Import) else [])
                for module in modules:
                    target=module.replace('.','/')+'.py'
                    if module.startswith(('scripts.','sas_memory.')) and Path(target).is_file():self.assertIn(target,names)


if __name__=='__main__':unittest.main()
