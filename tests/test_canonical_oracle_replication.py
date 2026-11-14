"""Replicate denominators/gates and reject exact-reference or preservation failures."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.analyze_canonical_oracle_replication import audit_proofs
from scripts.canonical_oracle_common import MODELS, REGIMES, model_order, read_plan, require_exact_full
from scripts.canonical_oracle_metrics import model_stats, summarize
from scripts.compare_canonical_oracle_smokes import compare_observations
from scripts.copy_transfer_data import ROLES


class CanonicalOracleReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = read_plan()
        old = json.loads(Path(cls.plan['existing_oracle_protocol']).read_text())
        source = Path(old['source_dataset'])
        cls.cases = json.loads((source / 'cases.json').read_text())
        cls.registry = json.loads((source / 'registry.json').read_text())
        cls.report = json.loads(Path(cls.plan['completed_necessity_audit']).read_text())
        run = Path(cls.plan['completed_necessity_source_run'])
        cls.full = {m: [json.loads(line) for line in (run / f'{m}.full.jsonl').read_text().splitlines()] for m in MODELS}

    def test_two_model_statistics_match_sealed_full_denominators(self):
        for model in MODELS:
            actual = model_stats(self.full[model], self.cases, self.registry)
            old = self.report['metrics']['models_by_regime']['full'][model]
            self.assertEqual(actual['physical_queries'], 336)
            self.assertEqual(actual['bindings'], 432)
            for field, value in actual.items():
                if field != 'case_outcomes': self.assertEqual(value, old[field], (model, field))
            for a, b in zip(actual['case_outcomes'], old['case_outcomes']):
                self.assertEqual(a, {k: b[k] for k in a})

    def synthetic_oracle(self):
        return {m: {'full': deepcopy(self.full[m]), 'oracle': [
            {**row, 'greedy_exact': True, 'answer_nll': .01} for row in self.full[m]]} for m in MODELS}

    def test_exact_existing_sufficiency_gate_requires_both_models(self):
        rows = self.synthetic_oracle()
        result = summarize(rows, self.cases, self.registry, self.plan)
        self.assertTrue(result['frozen_prefix_sufficiency']['passed'])
        self.assertEqual(result['cross_profile_rows_in_primary'], 0)
        self.assertTrue(result['prior_shared_history_repeat_primaries_remain_failed'])
        targets = {b['physical_id'] for b in self.registry['bindings'] if b['variant'] != 'before' and b['role'] == ROLES[0]}
        rejected = set(sorted(targets)[:7])
        for row in rows['gdn_question_only']['oracle']:
            if row['id'] in rejected: row.update(greedy_exact=False, answer_nll=2.)
        failed = summarize(rows, self.cases, self.registry, self.plan)
        self.assertFalse(failed['frozen_prefix_sufficiency']['passed'])
        self.assertFalse(failed['frozen_prefix_sufficiency']['model_checks']['gdn_question_only']['oracle_current_at_least90_of96'])
        self.assertTrue(all(failed['frozen_prefix_sufficiency']['model_checks']['local_question'].values()))
        with self.assertRaises(ValueError): summarize({MODELS[0]: rows[MODELS[0]]}, self.cases, self.registry, self.plan)
        rows = self.synthetic_oracle(); rows[MODELS[1]]['oracle'].pop()
        with self.assertRaises(ValueError): summarize(rows, self.cases, self.registry, self.plan)

    def test_five_field_full_replay_has_no_numeric_tolerance(self):
        row = self.full[MODELS[0]][0]
        require_exact_full(row, row, MODELS[0], row['id'])
        for key, value in [('answer_nll', row['answer_nll'] + 1e-12), ('generated', row['generated'] + ' '),
                           ('greedy_exact', not row['greedy_exact']), ('answer', 'bad'), ('query_tokens', row['query_tokens'] + 1)]:
            with self.assertRaises(ValueError): require_exact_full({**row, key: value}, row, MODELS[0], row['id'])

    def test_oracle_memory_proof_covers_complete_state_and_actual_probabilities(self):
        from tests.test_copy_readout_sufficiency import ReadoutSufficiencyTests
        p, full, rows = ReadoutSufficiencyTests().proof_fixture()
        p['regimes'] = {'oracle': p['regimes']['oracle']}
        p['unmasked_weights'] = {'oracle': p['unmasked_weights']['oracle']}
        audit_proofs([p], ['q'], full, rows['oracle'])
        wrong = deepcopy(p); wrong['regimes']['oracle']['historical_kv_fields_exact'] = 11
        with self.assertRaises(ValueError): audit_proofs([wrong], ['q'], full, rows['oracle'])
        wrong = deepcopy(p); wrong['regimes']['oracle']['masked_prefix_sha256'] = 'b' * 64
        with self.assertRaises(ValueError): audit_proofs([wrong], ['q'], full, rows['oracle'])
        with self.assertRaises(ValueError): audit_proofs([], ['q'], full, rows['oracle'])

    def order_fixture(self):
        ids = [f'q{i}' for i in range(18)]
        template = self.full[MODELS[0]][0]
        rows = {m: {r: [{**template, 'id': q, 'regime': r, 'model': m} for q in ids] for r in REGIMES} for m in MODELS}
        first = {'model_order': model_order(), 'selected_physical_ids': ids, 'native_anchors': [{'id': q} for q in ids[:10]],
            'models': {m: {'prefix_checks': [{'source_key': q, 'prefix_sha256': 'a' * 64} for q in ids[:6]],
                'proofs': [{'id': q, 'regimes': {'oracle': {'prefix_sha256': 'a' * 64, 'masked_prefix_sha256': 'a' * 64}}} for q in ids[:10]]} for m in MODELS}}
        second = deepcopy(first); second['model_order'] = model_order(True)
        return first, second, rows, deepcopy(rows)

    def test_all72_cross_order_readouts_and20_prefixes_required(self):
        a, b, x, y = self.order_fixture()
        self.assertEqual(compare_observations(a, b, x, y)['exact_readouts'], 72)
        y[MODELS[1]]['oracle'][-1]['answer_nll'] += 1e-12
        with self.assertRaises(ValueError): compare_observations(a, b, x, y)
        a, b, x, y = self.order_fixture()
        b['models'][MODELS[1]]['proofs'][-1]['regimes']['oracle']['masked_prefix_sha256'] = 'b' * 64
        with self.assertRaises(ValueError): compare_observations(a, b, x, y)
        a, b, x, y = self.order_fixture(); y[MODELS[0]]['full'].pop()
        with self.assertRaises(ValueError): compare_observations(a, b, x, y)
        a, b, x, y = self.order_fixture(); b['model_order'] = model_order()
        with self.assertRaises(ValueError): compare_observations(a, b, x, y)


if __name__ == '__main__':
    unittest.main()
