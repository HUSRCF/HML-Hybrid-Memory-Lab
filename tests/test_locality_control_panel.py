from copy import deepcopy
import unittest

from scripts.locality_panel_data import MODELS, read_plan
from scripts.locality_panel_metrics import summarize
from scripts.run_write_locality_control_panel import bound_engine, engine_binding


class LocalityControlPanelTests(unittest.TestCase):
    def test_five_final_checkpoints_and_local_timing_pair_are_fixed(self):
        plan = read_plan()
        self.assertEqual(list(plan['models']), list(MODELS))
        self.assertEqual({spec['final_step'] for spec in plan['models'].values()}, {432})
        self.assertEqual(plan['counts']['total_physical_queries'], 1920)
        self.assertEqual(plan['models']['local']['initial_state_sha256'],
                         plan['models']['local_question']['initial_state_sha256'])
        self.assertLess(plan['models']['attention_only']['teacher_train_fit']['teacher_numeric_tokens_correct'],
                        plan['models']['shared']['teacher_train_fit']['teacher_numeric_tokens_correct'])

    def test_private_engine_binding_preserves_sealed_two_model_module(self):
        import scripts.run_write_locality_adapter_transfer as original
        original_models, original_protocol, original_sources = original.MODELS, original.PROTOCOL, original.RUNTIME_SOURCES
        engine = bound_engine()
        self.assertEqual(engine.MODELS, MODELS)
        self.assertEqual(engine.execute.__code__.co_code, original.execute.__code__.co_code)
        self.assertIs(engine.execute.__globals__, vars(engine))
        self.assertIsNot(engine.execute.__globals__, original.execute.__globals__)
        self.assertEqual((original.MODELS, original.PROTOCOL, original.RUNTIME_SOURCES),
                         (original_models, original_protocol, original_sources))
        self.assertEqual(engine_binding()['bound_models'], list(MODELS))

    @staticmethod
    def fixture():
        pairs, bindings, physical = [], [], []
        models = {name: [] for name in MODELS}
        for number in range(2):
            pair = {'id': f'p{number}', 'cohort': 'replica', 'stratum': 'B', 'case_id': f'case{number}',
                    'base_stream_id': f'stream{number}', 'source_keys': {'left': f'L{number}', 'right': f'R{number}'}}
            pairs.append(pair)
            for side in ('left', 'right'):
                for label in ('preservation', 'current_A'):
                    key = f'{number}/{side}/{label}'
                    native_ok = label == 'current_A' or (side == 'left' and number == 0)
                    physical.append({'id': key, 'native_full_readout': {'greedy_exact': native_ok, 'answer_nll': .1}})
                    bindings.append({'pair_id': pair['id'], 'side': side, 'label': label, 'physical_id': key})
                    for model in MODELS:
                        ok = not (model == 'shared' and side == 'right' and label == 'current_A')
                        models[model].append({'id': key, 'greedy_exact': ok, 'answer_nll': .05})
        return {'pairs': pairs, 'bindings': bindings, 'physical_queries': physical}, models

    def test_shared_preservation_repair_losing_current_update_is_not_joint_success(self):
        registry, models = self.fixture()
        group = summarize(registry, models)['cohort_stratum_results'][0]
        self.assertEqual(group['native_full_update_regression_cases'], ['case0'])
        self.assertEqual(group['native_full_update_regression_repairs']['shared']['preservation_repaired'], ['case0'])
        self.assertEqual(group['native_full_update_regression_repairs']['shared']['all_roles_joint_repaired'], [])
        contrast = group['architecture_comparisons']['shared_versus_independent']
        self.assertEqual((contrast['reference'], contrast['candidate']), ('independent', 'shared'))
        self.assertEqual(contrast['outcomes']['right']['all_roles_joint']['regressed'], 2)
        self.assertEqual(group['counts']['shared']['right']['all_roles_joint']['correct'], 0)

    def test_metrics_reject_missing_models_missing_queries_and_duplicate_rows(self):
        registry, models = self.fixture()
        for altered in ({key: value for key, value in models.items() if key != 'attention_only'},
                        {**models, 'shared': models['shared'][:-1]},
                        {**models, 'shared': models['shared'] + [models['shared'][0]]}):
            with self.assertRaises(ValueError):
                summarize(registry, deepcopy(altered))


if __name__ == '__main__':
    unittest.main()
