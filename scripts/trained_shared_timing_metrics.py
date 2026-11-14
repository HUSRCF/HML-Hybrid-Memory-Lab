"""Pre-registered3x3 timing contrasts, with all physical regressions retained."""

from scripts.canonical_oracle_metrics import model_stats
from scripts.copy_transfer_metrics import paired_outcomes
from scripts.trained_shared_timing_data import ORDER
from scripts.unseen_shared_timing_metrics import pointwise

QUESTION='shared_question_trained_question'
SHARED_CELLS=('shared','shared_question_timing','shared_ledger_timing',*ORDER[4:10])
NATURAL=('native','shared','independent','local','local_question','attention_only','gdn_all_tokens','gdn_question_only')


def summarize(all_rows,cases,registry,registered):
    if list(all_rows)!=list(ORDER):raise ValueError('All16 pre-registered arms required; no selected cells')
    models={name:model_stats(all_rows[name],cases,registry) for name in ORDER}
    comparisons=list(dict.fromkeys([
        (QUESTION,'shared_question_timing'),(QUESTION,'native'),(QUESTION,'shared'),
        ('shared_ledger_trained_ledger','shared_ledger_timing'),
        *[(c,r) for c in SHARED_CELLS for r in NATURAL if c!=r],
        *[(c,r) for start in (4,7) for c in ORDER[start:start+3] for r in ORDER[start:start+3] if c!=r],
    ]))
    joint=[{'candidate':c,'reference':r,**paired_outcomes(models[c]['case_outcomes'],models[r]['case_outcomes'])}
        for c,r in comparisons]
    lookup={(v['candidate'],v['reference']):v for v in joint}
    preservation=pointwise(all_rows[QUESTION],all_rows['shared'],cases,registry)
    physical=[{'candidate':c,'reference':r,**pointwise(all_rows[c],all_rows[r],cases,registry)} for c,r in comparisons]
    return {'models':models,'joint_paired_comparisons':joint,'all_pointwise_comparisons':physical,
        'matched_question_vs_deployment_ablation':lookup[QUESTION,'shared_question_timing'],
        'matched_question_vs_native':lookup[QUESTION,'native'],
        'matched_question_functional_preservation':preservation,
        'registered_criteria':{
            'matched_question_positive_joint_gain_over_all_trained_question_deployed':lookup[QUESTION,'shared_question_timing']['candidate_passed'],
            'matched_question_positive_joint_gain_over_native':lookup[QUESTION,'native']['candidate_passed'],
            'matched_question_no_all_trained_all_deployed_correct_regression':preservation['regression']==0},
        'training_teacher_fits':{name:spec['teacher_train_fit'] for name,spec in registered['models'].items() if spec},
        'prior_fresh_gates_unchanged':registered['references']['prior_fresh_gates'],
        'scope':'One paired initialization and one untouched executable-task cohort. Training fit is retained. Timing effects cannot establish explicit semantic scope, recurrent damage, an architecture benefit or independent training replication.'}
