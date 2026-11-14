from copy import deepcopy
import json
from pathlib import Path
import unittest

from sas_memory.locality_replication import primary_recurrence, read_plan, summarize
from sas_memory.write_locality_data import build_data, validate_data
from sas_memory.write_locality_metrics import summarize as discovery_metrics

ROOT = Path(__file__).resolve().parents[1]


def observations(plan):
    rows = []
    for example in build_data(plan)['validation']:
        case = example['case']
        rows.append({**{key:case[key] for key in ('seed','active_accounts','base_case_id','base_stream_id','value_version')},
                     'case_id':case['id'],'variant':example['variant'],'role':example['query']['role'],
                     'answer':example['query']['answer'],'greedy_exact':True,'answer_nll':0.1})
    return rows


class LocalityReplicationTests(unittest.TestCase):
    def test_parameterized_metrics_preserve_discovery_definitions(self):
        plan = json.loads((ROOT/'configs/semantic_write_locality_load_plan.json').read_text())
        rows = observations(plan)
        for row in rows:
            if row['value_version']==0 and row['variant']=='A1_C1' and row['role']=='protected_current':
                row['greedy_exact']=False
        self.assertEqual(summarize(rows,plan),discovery_metrics(rows))

    def test_actual_replica_identity_and_smoke_use_new_seeds(self):
        plan = read_plan()
        data = build_data(plan)
        self.assertEqual(validate_data(data,plan)['examples'],576)
        self.assertEqual({e['case']['seed'] for e in data['validation']},{97100,97101,97102,97103})
        self.assertTrue(all(f"s{e['case']['seed']}-" in e['case']['id'] for e in data['validation']))
        from sas_memory.locality_replication import evaluated_members
        smoke = evaluated_members(data,True)
        self.assertEqual(len(smoke),12)
        self.assertEqual({e['case']['seed'] for e in smoke},{97100})
        rows = observations(plan)
        rows[0]['seed']=97000
        with self.assertRaises(ValueError):summarize(rows,plan)

    def test_secondary_signal_cannot_replace_failed_primary(self):
        plan = read_plan()
        rows = observations(plan)
        for row in rows:
            if (row['seed'] in (97100,97101) and row['value_version']==0 and row['active_accounts']==8 and
                row['variant']=='A1_C0' and row['role']=='protected_current'):
                row['greedy_exact']=False
        metrics = summarize(rows,plan)
        self.assertTrue(metrics['screen_signal']['present'])
        self.assertFalse(primary_recurrence(metrics,plan)['event_path_gate_passed'])
        for row in rows:
            if (row['seed'] in (97102,97103) and row['value_version']==0 and row['active_accounts']==32 and
                row['variant']=='A1_C1' and row['role']=='protected_current'):
                row['greedy_exact']=False
        result = primary_recurrence(summarize(rows,plan),plan)
        self.assertTrue(result['event_path_gate_passed'])
        self.assertTrue(result['contrasts'][0]['recurs'])
        self.assertFalse(result['contrasts'][1]['recurs'])
        self.assertTrue(all('971' in value for value in result['contrasts'][0]['regressed_base_streams']))


if __name__=='__main__':unittest.main()
