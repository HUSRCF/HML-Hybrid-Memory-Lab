# HML — Hybrid Memory Lab

Research on dependency-aware memory updates and evidence access in existing
Gated DeltaNet / Attention hybrid language models. The working directory keeps
its historical name `SAS`; the current research scope is broader than sparse
attention routing.

**Status: active; the complete text-competence factorial diagnostic is audited.**
All 6,720 registered readouts and 223 regression tests passed operational checks.
Native rises from 1/48 joint cases on full history to 48/48 with original COPY
logic on program-selected support text; compiling COPY to SET on full history
reaches 21/48. These are privileged competence controls on a reused eight-stream
cohort. Text, positions and prefix computation change; they do not uniquely
localize GDN damage or establish a new architecture. All earlier failed
scientific criteria remain failed. GDN-only means adapter scope; Attention stays.

- [All twenty text-control cells, limitations and audit](materials/updates/text-competence-factorial-20261115/README.md)
- [Previous frozen-prefix evidence-access diagnostic](materials/updates/trained-timing-evidence-access-20261115/README.md)
- [Previous matched-training confirmation](materials/updates/trained-shared-timing-20261115/README.md)
- [Research status and archived materials](materials/README.md)
- [Registered text-control protocol](configs/text_competence_factorial_plan.json)
- [Machine-readable progress](configs/experiment_progress.json)
- [Full backup files](https://github.com/HUSRCF/HML-Hybrid-Memory-Lab/releases)

The fixed backup release records the explicit pause; subsequent work resumes
only after the user's instruction. The repository contains code, plans, reports, figures and execution logs. Release
attachments preserve complete synthetic datasets and experiment run files,
including adapter checkpoints, optimizer states and raw logits. Pretrained LLM
weights, environments and compiler/download caches remain local.

## Local research runtime

Use the existing **Conda base** at `/home/husrcf/anaconda3`. This project does not
create a Conda environment or a virtual environment. Frozen-model diagnostics now
study dependency-aware updates and evidence access in the existing hybrid model.

## Enter the runtime

```bash
cd /home/husrcf/Code/SAS
source scripts/activate.sh
```

The activation script selects `conda base` and places model/data and compiler
caches on `/media/990Pro/Code/SAS`. Models remain at the paths recorded in
`configs/models.json`; no model copies are made.

## Dependencies

The existing base provides ROCm PyTorch 2.9.1, Triton 3.5.1, Transformers 5.14.1,
Accelerate 1.14.0, NumPy 1.26.4, PyArrow 21.0.0, and Pandas 2.3.3.
Only the five small additions in `requirements.txt` are installed. The setup
command uses `--no-deps` to avoid replacing shared packages, then checks their
direct dependency versions. It does not resolve or upgrade the host stack.

```bash
bash scripts/setup_env.sh
python scripts/check_dependencies.py
```

`environment/host-constraints.txt` records the protected host versions;
`environment/project-packages.lock.txt` records the five additions. If dependency
checks fail on a different host, review that host's versions before installing.

## Runtime acceptance

```bash
python scripts/check_environment.py --smoke --mode reference \
  --output "$SAS_STORAGE_ROOT/artifacts/environment/reference.json"
python scripts/check_environment.py --smoke --kernels --mode auto \
  --output "$SAS_STORAGE_ROOT/artifacts/environment/accelerated.json"
```

The checks load both checkpoints as `Qwen3_5ForCausalLM`, without remote code or
the vision tower, and check finite prefill/cached decode and short generation.
FLA chunk/recurrent outputs are checked against Torch, including finite, nonzero
Q/K/V backward gradients. Data checks exercise a small in-memory Arrow round trip.

The reference mode uses Torch GDN and convolution; auto mode uses FLA for GDN
and Torch for convolution. No `causal-conv1d` binary extension is installed.
Weight verification checks shard lengths and index coverage, not full-file hashes.
These are environment checks, not retrieval accuracy or throughput measurements.

Reports are saved under `/media/990Pro/Code/SAS/artifacts/environment/`.
Retain reusable Triton compiler caches on the new disk. Disposable package
download caches and project-specific virtual environments are unnecessary.

## Memory-update pilot

```bash
python -m unittest discover -s tests -v
python scripts/run_diagnostics.py --model it \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/it-pilot-001"
```

The interpreter generates assignment, independent/dependent swap, duplicate
assignment, repeated addition, and snapshot-copy tasks. Each case has a shared
prefix and before/control/treatment variants. Current target, protected account,
and initial-version questions have executable ground truth. The pilot config is
`configs/diagnostic_pilot.json`; all generated cases and hashes are saved with a
run. Output directories must be new, preventing accidental result replacement.

The default natural baseline prefills each complete question independently and
scores candidates serially. Optional `--prefill cached` and `--score-batch` trade
numerical equivalence for speed. The runner first checks full versus split
prefill, serial versus batched candidate scoring, and isolation of all mixed
cache tensors. Small BF16 numerical differences are measured explicitly.

`results.jsonl` preserves raw greedy answers, exact integer correctness, answer
NLL, and separately labeled candidate-ranking results. `summary.json` reports
paired preservation only when the gold answer is unchanged, and records whether
control/treatment token lengths match. These are small diagnostic experiments,
not evidence of an architectural benefit or a localized causal mechanism.

For a paired duration/entity-load screen (64/512 updates, 4/128 distractor
accounts), use `--config configs/diagnostic_stress_screen.json`. Target values and
final updates match across these conditions. `configs/diagnostic_stress.json`
reserves additional seeds for confirmation. The original pilot repeats prefixes
across task families; its 216 questions are not 216 independent samples.

```bash
python scripts/analyze_diagnostics.py /path/to/it-run /path/to/base-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/comparison.json"
python scripts/run_cache_interventions.py --cases /path/to/run/cases.jsonl \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/transplant-001"
python scripts/analyze_interventions.py /path/to/transplant-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/transplant-summary.json"
python scripts/run_oracle_control.py --cases /path/to/run/cases.jsonl \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/oracle-001"
```

Cache transplants use equal-length control/treatment ledgers, replace recurrent
and convolution states together, and test both directions for each of six groups.
Self-transplants must preserve tensors; replacing every path must reproduce the
donor's candidate NLL. Mixed caches are counterfactual and may be out of distribution.
Full/cached discrepancies are recorded as a numerical comparison, not assumed
away. A full/serial baseline can continue when an unused speed optimization fails
its tolerance; requesting that optimization explicitly still fails validation.

The oracle control computes a backward program slice for each current-balance
question, including snapshot-copy dependencies and repeated addition. It is a
reasoning/readout upper bound using oracle dependencies. Filtering also changes
positions and computation, so it cannot alone identify retrieval or GDN failure.
Base models can emit reasoning tags instead of a bare integer; reports distinguish
format validity, free generation, and restricted-candidate ranking.

## Frozen-backbone controller feasibility trial

```bash
python scripts/run_controller_trial.py --smoke \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/controller-smoke-001"
python scripts/run_controller_trial.py \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/controller-pilot-001"
```

`configs/controller_training_pilot.json` fixes the first small trial: one group,
64 optimizer steps with two microbatches per step, answer-only loss, and four
roughly matched parameter budgets. Shared and independent controllers receive the
same causal group-entry representation; the latter has separate encoders for GDN
and attention. The other controls adapt only GDN or only attention. GDN residuals
enter before the native gate nonlinearities. Query residuals preserve every
head's output-gate projection. No backbone weights or cache dimensions change.

Each run checks exact zero-init identity, adapter gradients, absence of backbone
gradients, and restoration after removing hooks. Train/validation/test have
disjoint seeds and entity namespaces. All modes use the same ordered, saved
microbatches and supervised tokens. Only final checkpoints are evaluated, with no
test-based selection. Raw unrestricted greedy answers, true-answer NLL, source
snapshots and controller-only safetensors are saved. The trial establishes
feasibility and compares sharing; it does not impose semantic dependency labels
or establish an architectural contribution. Group-entry hooks currently require
gradient checkpointing to be disabled and one controlled group.

`configs/controller_training_local.json` adds an ordinary linear low-rank
control at the same projection targets, using each projection's local input.
`configs/controller_training_keys.json` also adapts the three GDN Key slices.
Those residuals enter before native depthwise convolution; Q/V slices and native
Q/K normalization are retained. Five modes then have approximately 0.69M
parameters. Both protocols use the identical saved data and token budget.

```bash
python scripts/check_training_gradients.py \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/gradient-reference.json"
python scripts/run_controller_trial.py --config configs/controller_training_keys.json \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/controller-keys-001"
python scripts/analyze_controller_trial.py /path/to/trial-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/controller-summary.json" \
  --plot "$SAS_STORAGE_ROOT/artifacts/memory_update/controller-summary.png"
python scripts/run_controller_ablations.py --run /path/to/trial-run --mode shared \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/controller-ablation-001"
```

The gradient reference includes q/k/v and both decay/write gates across two
chunks. Branch ablations disable trained outputs without retraining. Enabling all
outputs must reproduce the saved checkpoint; disabling all must reproduce the
native baseline, including raw greedy strings and NLL. Summaries audit source/data
fingerprints, complete evaluation labels and optimizer-step coverage. They report
joint competence: current, initial and protected readouts all correct for the same
ledger, so gains that sacrifice historical evidence remain visible. The current
small held-out set is diagnostic; larger confirmation needs fresh seeds and more
than one optimization initialization.

## Frozen-checkpoint confirmation and timing controls

```bash
python scripts/run_controller_ablations.py --run /path/to/trial-run --mode shared \
  --temporal --output "$SAS_STORAGE_ROOT/runs/memory_update/controller-temporal-001"
python scripts/analyze_controller_ablations.py /path/to/temporal-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/controller-temporal.json"
python scripts/run_controller_confirmation.py \
  --config configs/controller_locked_confirmation.json \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/controller-confirm-001"
python scripts/analyze_controller_confirmation.py /path/to/confirmation-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/controller-confirm.json"
```

Timing controls apply a token-position mask during full prefill: enable adapters
throughout, during the ledger, during the question/answer, or nowhere. Full/disabled
endpoints must reproduce the trained/native raw answers and NLL. These are
post-training interventions. Ledger-only changes both recurrent states and KV
construction; question-only includes downstream question and answer computation.
Neither condition alone isolates a recurrent write mechanism, and masking can
introduce a distribution shift.

`configs/controller_locked_confirmation.json` fixes 48 new seeds before evaluation:
32 COPY and 16 assignment cases, with both operation orders and three readout roles.
Native and all five final checkpoints are frozen; this set receives no optimization
or checkpoint selection. The analyzer verifies labels, raw correctness, data/source
snapshots, checkpoint hashes, and complete model coverage. Joint-readout differences
use paired bootstrap resampling by case, averaging the two related operation orders.
Intervals are exploratory and are not adjusted for multiple comparisons. A single
training initialization cannot establish robust architectural superiority.

The first full-Key pilot used the same 128 microbatches and approximately 0.69M
trainable parameters per mode. COPY joint current/initial/protected correctness was
7/16 native, 13/16 shared, 13/16 independent, 9/16 attention-only, 12/16 GDN-only,
and 16/16 ordinary local low-rank adaptation. In the timing controls, shared/local
joint scores were 9/16 with ledger-only adaptation and 10/16 with question-only
adaptation. This pilot supports conventional adaptation as a strong baseline;
it does not establish a benefit from a shared semantic interface or prove that
unrelated memory was damaged by an update. Full records and current job status are
indexed in `configs/experiment_progress.json`.

The locked fresh-seed confirmation completed all six models and 2,592 questions.
Across 32 COPY cases and two operation orders, joint correctness was 21/64 native,
57/64 shared, 60/64 independent, 42/64 attention-only, 55/64 GDN-only, and 62/64
local low-rank. Shared/local current-value accuracy was 62/64 for both, while
initial-version accuracy was 59/64 versus 64/64. These results support adaptation
on this executable task, with no demonstrated benefit from sharing. They remain
single-initialization results. Full tables, case-bootstrap intervals and figures
are under `artifacts/memory_update/controller-keys-confirm-20261114.*` on the data
disk. Different length strata use different seeds and are descriptive, not a
matched causal comparison of duration.

Native protected-account answers included one new order-related regression
(`test-copy-s40029-n64`), plus one case already wrong before the final operations.
All five trained models answered the protected queries correctly. The rare
regression is a candidate functional symptom; it is not localized to recurrent
state damage or evidence that a shared interface is necessary.

To distinguish disablement effects from a genuine need to change ledger
processing, `configs/controller_training_ledger.json` and
`configs/controller_training_question.json` specify matched local-adapter training
with the corresponding token policy active from the first optimizer step through
evaluation. Both retain the full-Key protocol's 128 examples, 64 steps, seed and
689,216 parameters. `configs/controller_timing_training_plan.json` records the
comparison before these jobs start. Exact ledger/question tokenizer boundaries
are checked during encoding. Question-only training preserves pre-question
computation, while still allowing GDN and attention adaptation during the question
and answer. The results should not be described as a pure attention-read control.

Both matched-training timing runs completed on the development split: joint COPY
correctness was 16/16 full-token, 15/16 ledger-only, and 16/16 question-only. This
motivates a separate frozen-checkpoint confirmation, rather than an equivalence
claim from a small development sample:

```bash
python scripts/run_controller_confirmation.py \
  --config configs/controller_timing_locked_confirmation.json \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/controller-timing-confirm-001"
```

This second protocol fixes another 48 seeds (50000–50047), excluding all earlier
train/validation/development and confirmation cases. It compares three separately
trained token policies and the native model. Before evaluating question-only
adaptation, the runner checks its actual final checkpoint against the native
ledger KV, recurrent and convolution tensors and logits. The analyzer requires the
saved identity check; the earlier timing training jobs predate that additional
check and should not be cited as having executed it.

The second confirmation completed: joint COPY correctness was 24/64 native,
62/64 full-token, 63/64 ledger-only, and 64/64 question-only. All three readout roles
were correct for question-only adaptation. Its final checkpoint preserved every
checked native ledger KV/recurrent/convolution tensor and logits on the recorded
sample prefix. A separate real-model smoke run also passed the newly added cache
identity check after one optimizer update. Results and figures are saved as
`artifacts/memory_update/controller-timing-confirm-20261114.*` on the data disk.

This is evidence that changing history construction is unnecessary for high
competence on these particular tasks. Question-stage adaptation still changes GDN
and attention computation; the original dense historical attention remains
available. It does not prove that GDN alone retained every fact, that all native
errors are read errors, or that this result generalizes beyond one initialization
and the present operation templates. Architecture expansion should wait for
evidence that ordinary local adaptation cannot meet the functional constraints.

## Initialization replication and question-stage localization

`configs/controller_initialization_replication.json` locks local-adapter seeds
1701, 2718 and 31415, with full-token/question-only policies paired within each
seed. The four new checkpoints keep all 128 ordered training examples, 512 answer
tokens, 64 optimizer steps and 689,216 parameters. Evaluation deliberately reuses
the frozen 50000–50047 cohort. It is a fixed-benchmark initialization replication,
not a new-data confirmation or seed/checkpoint selection.

```bash
python scripts/run_controller_confirmation.py \
  --config configs/controller_replication_evaluation.json \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/controller-initialization-eval-001"
python scripts/analyze_initialization_replication.py \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/controller-replication.json"
python scripts/run_controller_ablations.py --run /path/to/question-trained-run \
  --mode local --gdn-components \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/controller-components-001"
```

The replication analyzer uses the explicit source-run paths in
`configs/controller_replication_analysis.json`; adjust those paths when
reproducing with different output names. It audits both source evaluations,
training inputs/answers, optimizer settings, actual source checkpoint identities,
and native replay. It reports every seed separately. Its case bootstrap for the
three-seed average is conditional on those seeds, rather than treating repeated
questions/orders/initializations as independent cases.

On development cases, disabling branches in the first question-only checkpoint
gave joint COPY scores of 16/16 with all adapters, 15/16 with GDN adapters, 8/16
with attention adapters, and 7/16 with none. These are interventions in one trained
computation, not matched-budget retraining. The finer component switch separates
Key residuals from decay/write-gate residuals, with and without attention adapters.
GDN processing of question tokens itself updates recurrent state; retaining the
original ledger computation does not isolate address construction from changes
to question-time state updates. Later native layers may mediate the output.

The fixed-cohort replication completed. Joint COPY correctness for full-token
versus question-only adaptation was 62/64 versus 64/64 (seed 1701), 64/64 versus
64/64 (2718), and 63/64 versus 64/64 (31415). Native replay between source
evaluations had zero NLL difference, and both new question-only checkpoints passed
their recorded native-prefix cache/logit checks. The three-seed average difference
was 1.56 percentage points, with a case-bootstrap interval including zero. This
replicates high question-only competence on this fixed cohort; it does not
establish superiority, equivalence, or generalization beyond the tested templates.

The finer component ablation, still only on seed 1701's development cases,
gave 15/16 with Key adapters alone, 8/16 with gate adapters alone, 16/16 with
Key plus attention adapters, and 9/16 with gate plus attention adapters. Both
full/disabled replay endpoints had zero NLL error. Key residuals account for most
of this checkpoint's improvement, but the intervention does not distinguish
query formation from question-time memory writing. "Key-only" identifies the
enabled learned residuals; downstream native queries, gates and values may still
change. Full evidence is in `controller-initialization-replication-20261114.json`
and `controller-question-components-20261114.json` under the memory-update artifacts.

## Depth/wording generalization and matched update controls

`sas_memory/generalization_tasks.py` constructs matched COPY depths 1/2/3 and
original/entity-first questions from eight held-out base cases. The generator
tests snapshot values against analytic constants, verifies backward-slice truth,
and reproduces the original task exactly at depth one with original wording.
`configs/copy_generalization_screen.json` points to 48 validated cases and
432 questions on the data disk. Frozen native, full-token and question-only local
checkpoints have completed evaluation, with no training on this cohort.
All 48 control/treatment ledger token lengths match; tokenizer boundaries
are exact and the longest teacher-forced input is 1,419 tokens.

Conditions share `base_case_id`, leaving eight independent base cases for pooled
uncertainty. Chain-depth changes also change text and reasoning requirements;
question reordering changes wording. Neither intervention alone identifies a
memory-duration or gate mechanism. Every condition is reported separately;
all experiments here use IT and adapter initialization 1701.

```bash
python scripts/run_controller_confirmation.py \
  --config configs/copy_generalization_screen.json \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/controller-generalization-001"
python scripts/run_copy_update_controls.py \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/controller-update-controls-001"
python scripts/analyze_copy_update_controls.py /path/to/update-control-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/update-controls.json"
```

Joint current/initial/protected accuracy across both orders was 9/96 native,
41/96 full-token adaptation and 53/96 question-only adaptation. For original
wording at COPY depth 1/2/3, question-only joint scores were 16/16, 8/16 and 6/16;
at depth three with entity-first wording it reached 2/16. One-hop success does
not establish broader semantic competence. Minimal executable slices gave every
model 96/96 current-target answers, but remove intervening/conflicting records
and rebuild the entire computation.

The short-program follow-up adds source or unrelated-account updates after a
snapshot COPY, with a shared prefix and matched input lengths. Target truth stays
fixed. Current-target accuracy (48 questions per regime) was:

| Checkpoint | Minimal slice | Initialize unrelated account too | Source update | Unrelated update |
| --- | ---: | ---: | ---: | ---: |
| Native | 48 | 48 | 32 | 40 |
| Full-token local | 48 | 46 | 25 | 27 |
| Question-only local | 48 | 41 | 30 | 21 |

For question-only adaptation, 25 unrelated-update errors output the unrelated
account's initial value. This is functional fragility, not localization to GDN
state corruption. Results are in `controller-generalization-20261114.json` and
`controller-update-controls-20261114.json` under the data-disk artifacts.

## Historical KV intervention with the original prefix retained

```bash
python scripts/run_oracle_kv_readout.py --smoke \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/oracle-kv-smoke-001"
python scripts/run_oracle_kv_readout.py \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/oracle-kv-001"
python scripts/analyze_oracle_kv_readout.py /path/to/oracle-kv-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/oracle-kv.json" \
  --plot "$SAS_STORAGE_ROOT/artifacts/memory_update/oracle-kv.png"
```

This intervention keeps the complete original input, token positions and prefix
computations. At all six attention layers, only question/answer rows restrict
historical keys to original token spans in the executable backward slice, plus
the instruction/chat prelude. Question and answer tokens retain causal access;
GDN runs without a mask. A root forward hook applies the restriction to both
full prefill and every cached scoring/generation step.

On a sampled query per checkpoint, an all-history mask must exactly reproduce
the original logits, complete cache, NLL and greedy output. The restricted mask
must preserve prefix logits/cache, and all 12 historical K/V fields within full
prefill. All checks passed, with zero unrestricted replay NLL error.

| Current-target answers / 96 | Full original prefix | Same prefix, oracle KV | Rebuilt minimal slice |
| --- | ---: | ---: | ---: |
| Native | 10 | 44 | 96 |
| Full-token local | 41 | 96 | 96 |
| Question-only local | 53 | 96 | 96 |

The adapted checkpoints can recover every tested answer without reconstructing
history. This is a stronger access intervention than deleting context, but uses
oracle dependencies and changes question/answer computation across multiple
layers. It does not show that recurrence alone retained the facts, identify a
single causal layer, or establish a learned sparse router. The implementation
uses dense eager masks and makes no latency claim. Results and figures are under
`artifacts/memory_update/oracle-kv-readout-20261114.*` on the data disk.

`configs/oracle_kv_competing_versions.json` and
`configs/oracle_kv_matched_unrelated.json` define equal-budget controls:
retain necessary evidence and add two SET records containing unnecessary versions
or unrelated accounts. The full input and retained historical token counts match
for all 96 queries. Extra records differ in content and original position. Both
controls use oracle dependency identities and are diagnostics, not learned
selectors. Their collector uses the same `run_oracle_kv_readout.py --config ...`
path with fresh output directories.

Both controls completed, with all per-query input and token-budget checks passing:

| Current-target answers / 96 | Necessary evidence | + two unnecessary version SETs | + two unrelated SETs |
| --- | ---: | ---: | ---: |
| Native | 44 | 15 | 39 |
| Full-token local | 96 | 50 | 89 |
| Question-only local | 96 | 70 | 91 |

The adapted checkpoints are more sensitive to the competing-version additions.
Position and numeric-content differences remain; counterbalance them on fresh
facts before attributing the difference specifically to semantic version conflict.
The present results do not establish a need to change recurrent prefix updates.

```bash
python scripts/analyze_readout_control_comparison.py \
  --oracle /path/to/oracle-kv-run \
  --competing /path/to/competing-version-run \
  --unrelated /path/to/unrelated-record-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/kv-controls.json"
```

The comparison reruns every individual audit, checks source snapshots/checkpoint
hashes, original token identities, retained evidence and equal token budgets, and
reports paired outcomes with bootstrap resampling of the eight base cases. Saved
audit outputs accompany `kv-readout-controls-20261114.json` on the data disk.
`configs/readout_counterbalance_plan.json` records the next frozen diagnostic.

## Fresh position- and value-balanced readout controls

```bash
python scripts/prepare_readout_counterbalance.py \
  --output "$SAS_STORAGE_ROOT/datasets/memory_update/readout-counterbalance-001"
python scripts/run_readout_counterbalance.py --smoke \
  --data /path/to/registered-data \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/readout-counterbalance-smoke-001"
python scripts/run_readout_counterbalance.py \
  --data /path/to/registered-data \
  --output "$SAS_STORAGE_ROOT/runs/memory_update/readout-counterbalance-001"
python scripts/analyze_readout_counterbalance.py /path/to/completed-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/readout-counterbalance.json" \
  --plot "$SAS_STORAGE_ROOT/artifacts/memory_update/readout-counterbalance.png"
```

Preparation has frozen `readout-counterbalance-v1` on the data disk: 24 new base
facts (seeds 70000–70023), COPY depths 1/2/3, two question styles, and two late SET
orders. There are 144 cases and 288 complete question contexts, with at most
1,433 prompt tokens. After the snapshot COPY chain, both the original source and
an unrelated account are assigned the **same** new integer. The copied target
must retain its snapshot value.

Every query is evaluated with full historical access, the executable oracle
slice, oracle plus the late source SET, and oracle plus the late unrelated SET.
These masks use the identical original prompt and prefix computation. The two
extra-record masks have exactly equal token counts. Swapping late SET roles
places source/unrelated extras in the exact same absolute token intervals.
Both late records stay in the full input under every mask; no prefix is deleted
or retokenized.

Native and question-only local-adapter seeds 1701/2718/31415 are frozen. Their
actual checkpoint hashes, training manifests/data and supervised token budgets
are checked; no new optimization occurs. The dataset stores executable cases,
complete keep-bit masks, token identities and source snapshots before inference.
The collector checks the runtime encoding against this registration for every
query. All-history identity, original-prefix identity and historical-KV identity
are checked on the first short and last long case for each mask/checkpoint;
adapter weights must remain unchanged and removal must restore native behavior.

The four-checkpoint smoke run passed, with 64 records independently audited.
Use `analyze_readout_counterbalance.py --allow-smoke` for operational audits only;
smoke data produce no cohort effect estimates. Full evaluation completed all
4,608 records, with every record independently audited:

| Current-target answers / 288 | Full historical access | Necessary evidence | + late source SET | + late unrelated SET |
| --- | ---: | ---: | ---: | ---: |
| Native | 5 | 93 | 28 | 73 |
| Question-only, seed 1701 | 122 | 286 | 286 | 286 |
| Question-only, seed 2718 | 101 | 287 | 285 | 286 |
| Question-only, seed 31415 | 130 | 288 | 287 | 286 |

The large source-specific accuracy penalty did not replicate in these frozen
adapted models. Both extra-record conditions averaged 99.31% across the three
fixed initializations; their average paired difference was zero, with a
base-fact bootstrap interval of approximately -0.46 to +0.58 percentage points.
These near-ceiling observations do not establish population-level equivalence.
Native performance remains much weaker and cannot alone establish an
architectural necessity. Full tables and PNG/PDF figures are saved as
`artifacts/memory_update/readout-counterbalance-20261114.*` on the data disk.

This fresh test uses post-COPY updates and one extra SET, whereas the earlier
screen bundles the target's initial SET with an unnecessary source SET, across
both source-before/after-COPY orders. A changed result cannot be attributed only
to value/position balancing. The singleton decomposition configs
`oracle_kv_target_initial_only.json` and `oracle_kv_source_other_only.json` separate
the two earlier extras on every old case. Both singleton masks keep the same
number of tokens; their intersection is the necessary-evidence mask and their
union is exactly the old two-extra mask. They still differ in positions/values.

The full analyzer regenerates all cases and token masks, audits raw correctness
and provenance, and reports every depth/wording/late-order condition. Uncertainty
resamples base facts; averaging the three fixed initializations does not imply a
population-level confidence interval over training runs. Same-position contrasts
also compare the swapped full-input orders, so their inputs differ in those two
account labels. This remains a templated oracle-access mechanism diagnostic,
with no learned selector, recurrent-damage claim or sparse-kernel speed claim.

The audit also rejects a deliberately altered keep-mask digest and a failed
historical-KV proof; these checks were exercised on disposable copies of the
smoke run, without modifying original results. Late-order consistency reports
both-correct and both-incorrect pairs separately, so constant wrong predictions
are never treated as functional preservation.

## Decomposing the earlier bundled version intervention

The two singleton runs completed for question-only seed 1701, using all 96 old
60000-series questions (eight shared base facts). The two singleton masks have
identical retained-token budgets and exactly reconstruct the prior bundled mask:

| Original-prefix mask | Current-target correct / 96 |
| --- | ---: |
| Necessary evidence | 96 |
| + target's initial SET | 75 |
| + unnecessary source SET | 92 |
| + both SETs | 70 |

All 21 target-singleton errors output the target's initial value. In the
source-after-COPY stratum, the source singleton retains 48/48, while the target
singleton reaches 35/48. This localizes a larger component of the earlier decline
to allowing the old target record to be read in this checkpoint; it does not
isolate a general semantic mechanism because record positions/values differ.
The exploratory NLL second-difference interval includes zero, so these results
also do not establish a robust pairwise interaction requiring a set router.

```bash
python scripts/run_oracle_kv_readout.py \
  --config configs/oracle_kv_target_initial_only.json --output /path/to/target-run
python scripts/run_oracle_kv_readout.py \
  --config configs/oracle_kv_source_other_only.json --output /path/to/source-run
python scripts/analyze_version_split_controls.py \
  --target /path/to/target-run --source /path/to/source-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/version-split.json"
```

The audit reruns individual source/checkpoint/token-selection audits before
pairing every mask. Results are in `kv-version-split-20261114.json` on the data
disk. These are oracle readout diagnostics, not a deployable selector or proof of
irreversible GDN damage. The next planned carrier ablation is
`configs/record_kv_ablation_plan.json`: block question-time historical record KV
access while retaining normal prefix processing, recurrent/convolution state,
instruction context, and causal question/answer access. This tests compressed
carrier readout without assuming which mechanism or method must win. Oracle
access success does not eliminate possible matched-cost benefits of write-side
co-design; practical adaptation/access controls remain required before that claim.

## Blocking question-time access to all ledger records

```bash
python scripts/prepare_record_kv_ablation.py \
  --output "$SAS_STORAGE_ROOT/datasets/memory_update/record-kv-ablation-001"
python scripts/run_record_kv_ablation.py --smoke \
  --data /path/to/registered-record-masks --output /path/to/smoke-run
python scripts/run_record_kv_ablation.py \
  --data /path/to/registered-record-masks --output /path/to/full-run
python scripts/analyze_record_kv_ablation.py /path/to/completed-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/record-kv.json" \
  --plot "$SAS_STORAGE_ROOT/artifacts/memory_update/record-kv.png"
python scripts/check_record_attention_weights.py /path/to/completed-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/record-kv-weights.json"
```

The completed frozen trial reuses all 288 contexts and four checkpoints. Masks
retain 85 historical instruction/chat tokens and block **every** ledger-record
column during question/answer processing, including cached scoring/generation.
They depend only on prelude/ledger token boundaries, not answer or dependency
labels. Original token positions, prefix computations, recurrent/convolution
states and historical KV construction remain present.

| Current-target correct / 288 | Full historical access | Oracle evidence | Ledger records blocked |
| --- | ---: | ---: | ---: |
| Native | 5 | 93 | 0 |
| Question-only, seed 1701 | 122 | 286 | 0 |
| Question-only, seed 2718 | 101 | 287 | 0 |
| Question-only, seed 31415 | 130 | 288 | 0 |

All 1,152 new records passed input/mask/source/checkpoint audits, plus sampled
raw/oracle replay and prefix/KV identity checks. In an independent probe of native
and seed1701 (first/last cases, both late orders), all six layers had exactly zero
query-to-record softmax probability and probability mass on every observed
prefill/scoring/generation call. Unmasked controls were nonzero, and observing
weights preserved the saved outputs. This probability check covers eight
sampled model/context pairs, rather than every full-run query.

Under the blocked intervention, the three adapted checkpoints output the query
account's numeric identifier in 253, 256 and 194 of 288 answers. All their outputs
still have integer format. Thus the observed failure is not solely malformed
generation. The frozen readers depend on historical record KV on this benchmark;
this does **not** establish that recurrent state lacks the information. Blocking
a path used during training can also prevent an otherwise recoverable readout.
The retained carriers include convolution state and recurrent states constructed
with normal hybrid prefix attention, so this is not an isolated GDN-matrix test.

Results and figures are under `record-kv-ablation-20261114.*`; direct probability
checks are in `record-kv-weights-20261114.json`, with source copies beside them.
The operational smoke is independently audited without cohort effect estimates.

`configs/blocked_reader_adaptation_plan.json` specifies the constructive
probe: train ordinary local adapters with the blocked reader policy, comparing
question-only against all-token adaptation. Both use the same zero initialization,
689,216 parameters, original 128 examples, 512 supervised answer tokens and 64
updates. Reader-policy support now covers native comparisons, zero-init/detached
identity checks, supervised loss and every cached evaluation forward. Boundary-only
mask registries are saved before inference. A registered adaptation plan is required
before output creation; original data, optimized token identities and optimizer
settings must match. Each training microbatch logs its masked-forward count.

Both real-model smoke checks passed, including finite nonzero adapter gradients,
no backbone gradients, and exact question-only prefix caches/logits after a nonzero
update. Actual probabilities at all six attention layers were zero on ledger
records in teacher forcing and cached answer decoding; unmasked controls were
nonzero. The unit suite has 30 passing tests. Mutation checks confirm the auditor
rejects missing microbatch masks, an opened record column and missing result rows.

```bash
python scripts/run_controller_trial.py \
  --config configs/controller_blocked_reader_question_only.json \
  --adaptation-plan configs/blocked_reader_adaptation_plan.json \
  --output /path/to/new-question-only-run --smoke
# Repeat without --smoke for the fixed training run. The other arm uses
# configs/controller_blocked_reader_all_tokens.json with the same plan.
python scripts/analyze_blocked_reader_adaptation.py \
  /path/to/question-only-run /path/to/all-token-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/blocked-adaptation.json"
python scripts/run_blocked_reader_generalization.py --output /path/to/new-frozen-run
python scripts/analyze_blocked_reader_generalization.py /path/to/completed-frozen-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/blocked-generalization.json" \
  --plot "$SAS_STORAGE_ROOT/artifacts/memory_update/blocked-generalization.png"
python scripts/check_blocked_reader_training_fit.py \
  /path/to/question-only-run /path/to/all-token-run --output /path/to/new-fit-check
python scripts/analyze_blocked_reader_training_fit.py /path/to/completed-fit-check \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/blocked-training-fit.json"
```

Both registered step-64 training runs have completed and passed the independent
audit (`blocked-reader-adaptation-20261114.json`). The original held-out COPY grid
has 8 cases and two orders, excluding the pre-update variant from these counts:

| Correct / 16 | Question-only | All-token |
| --- | ---: | ---: |
| Current target | 0 | 0 |
| Initial target | 1 | 0 |
| Unrelated protected target | 0 | 0 |
| All three correct | 0 | 0 |

This budget has not produced functional recovery. Training losses at steps 1/64
were 4.461/1.452 and 4.461/1.515 respectively, but different steps use different
examples; these endpoints do not establish final training-set fit. The separate
post-hoc fit probe evaluates the exact 128 training examples using only the final
checkpoints, with no new updates or checkpoint selection. It distinguishes
teacher-forced loss including the newline from free-generation answer accuracy.
The fixed 24-base-fact generalization evaluation is a reused diagnostic, not
untouched architectural confirmation. Failed generic adapters do not establish
absent state information, a state-capacity limit or GDN write damage. The blocked
reader is a carrier diagnostic; practical semantic write/read co-design with
historical access remains part of the main research question.

The completed frozen generalization run (`blocked-reader-generalization-20261114-02`)
has 576 audited rows. Question-only answers 12/288 correctly; all-token answers
4/288. Correct outputs in both arms occur on just two of the 24 underlying facts,
with correlated depth/wording/order variants. This is not broad recovery. The
report includes a descriptive hindsight constant-answer reference (12/288), not
a deployable baseline or random-chance calibration. Sampled short/long final
checkpoints pass prefix/historical-KV checks and actual blocked-probability
inspection at all six attention layers. The failed `-01` run is retained: an
incorrect positional call to the verification helper stopped it before any result
row; the corrected run uses keyword arguments and the same trained checkpoints.

The post-hoc training-fit audit covers all 256 rows from both final checkpoints:

| Exact original training data | Question-only | All-token |
| --- | ---: | ---: |
| Unrestricted greedy answers | 0/128 | 0/128 |
| Teacher-forced numeric tokens | 78/384 | 88/384 |
| Teacher-forced final newline | 128/128 | 128/128 |
| Mean teacher loss, including newline | 1.539 | 1.509 |

Thus even the exact training data have not been fitted at this budget; the loss
drop alone is not evidence of factual recovery. Optimization budget and adapter
expressivity remain unresolved. Artifacts are
`blocked-reader-adaptation-20261114.json`,
`blocked-reader-generalization-20261114.{json,png,pdf}` and
`blocked-reader-training-fit-20261114.json`, with audit-source copies alongside.

`configs/blocked_reader_budget_calibration_plan.json` registers the next step:
repeat the same ordered 128 examples, using fixed checkpoints at 64/256/1024
updates and matched question-only/all-token arms. The first 64 updates must
exactly replay the completed adapters. Calibration will use training and original
validation data; the original test grid and 70000-series cohort will not be used
again to tune this budget. The standalone runner supports deterministic cycling,
atomic adapter/optimizer/RNG snapshots and exact step-64 replay validation. The
33-test suite includes cyclic order, next-update equivalence after optimizer/RNG
restore, and preservation of interrupted log tails. The two-arm real-model smoke
passed independent audit (`blocked-reader-budget-smoke-20261114.json`).

```bash
python scripts/run_blocked_reader_budget_calibration.py \
  --output /path/to/new-calibration-smoke --smoke
python scripts/run_blocked_reader_budget_calibration.py \
  --output /path/to/new-calibration-run
python scripts/analyze_blocked_reader_budget_calibration.py /path/to/completed-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/budget-calibration.json" \
  --plot "$SAS_STORAGE_ROOT/artifacts/memory_update/budget-calibration.png"
```

Only after confirming the original process has stopped, resume an interrupted
run with the same command plus `--resume`. A process lock rejects a second live
writer. Resume requires unchanged plan/source/data, verifies the saved log prefix,
restores AdamW moments and RNG, and preserves any uncommitted log tail separately.
Committed model checkpoints are not overwritten. Evaluation also verifies that
it does not change adapter weights or training RNG.

The registered full run is `blocked-reader-budget-calibration-20261114-01`.
Its question-only step-64 weights, all 64 losses/gradient norms and validation
outputs exactly replayed the earlier run before longer optimization proceeded.
Calibration outcomes inform the subsequent shared semantic write/read controls,
without treating failed generic adapters as proof that GDN destroyed information.

The completed budget calibration audited all 1,092 train/validation result rows,
all 2,048 optimizer steps, and all six adapter/AdamW/RNG snapshots. Both arms
exactly replayed step-64 weights, losses, gradient norms and validation outputs.
Actual record probabilities and trained question-only short/long prefix identity
were checked at every fixed checkpoint. No new original-test or 70000-series
inference was performed.

| Updates | Question-only train / 128 | All-token train / 128 | Validation COPY joint, both arms |
| --- | ---: | ---: | ---: |
| 64 | 0 | 0 | 0/8 |
| 256 | 1 | 4 | 0/8 |
| 1024 | 40 | 54 | 0/8 |

Larger budgets improve fitting but do not solve this reused validation task.
The single initialization and training/validation gap do not establish a useful
write-side or shared-interface advantage. Audit mutation checks reject an AdamW
moment at the wrong step and an opened record column even after the corresponding
file hashes are recomputed. Results and figures are in
`blocked-reader-budget-calibration-20261114.{json,png,pdf}`.

A post-hoc data audit found 118 distinct question texts among the 128 original
training examples. A deterministic lookup built with hindsight from question text
alone can answer 118/128. This is a descriptive restricted-class ceiling, not a
trained baseline or a universal bound on models that can also use positions.
Training fit alone therefore cannot establish that the model uses prefix memory.
The audit is `blocked-reader-question-lookup-20261114.json`.

The new `single-fact-counterfactual-v1` data are registered before inference on
the external storage drive: 32 training families (128 examples) and 16 validation
families (64 examples). Four siblings share the exact question IDs, positions,
all non-value input IDs and allowed-history columns; only three fact-value token
IDs change, and all four answers differ. The primary metric is all four free
answers correct in a family. A deterministic prefix-independent answer can get
at most one of four. These immediate single-fact controls calibrate carrier
readability; they do not replace long-memory or multi-entity semantic-update
experiments. The native full-history versus records-blocked evaluation registered
in `configs/single_fact_native_probe.json` is now complete and independently audited.
Full-history readout answers 128/128 training and 64/64 validation examples,
including all four siblings in all 48 families. Record-blocked readout answers
0/128 and 0/64, with no successful family. The 384-row audit reproduces the token
registration, checks all free-generation labels, and verifies actual zero record
probabilities at all six attention layers on four sampled contexts, with exact
prefix cache/logit and historical-KV preservation. This immediate positive
control shows a readout gap under the intervention; it does not establish absent
state information, long-term forgetting or damage from semantic updates.
The artifact is `single-fact-native-probe-20261114.json`.

```bash
python scripts/prepare_single_fact_control.py \
  --output "$SAS_STORAGE_ROOT/datasets/memory_update/new-single-fact-data"
```

The conditional follow-up in `configs/gdn_query_readout_control_plan.json` tests
direct GDN query-path adaptation if needed after the native positive control.
The query-only rank-56 control has 688,128 parameters, versus the existing
689,216 (a 0.16% difference). Actual projection shapes and counts have passed a
real-2B safety run and independent audit (`gdn-query-capacity-smoke-20261114.json`).
It is implemented separately in `sas_memory/gdn_query_control.py`, leaving the
archived runtime and GroupController unchanged. It preserves native
pre-question computation, changes only query slices before convolution, and
compares fresh controllers on the same registered paired data/budgets using
training and validation only. The old budget cohort is retained as context,
rather than being treated as a matched-data positive-control baseline. Query changes
can affect downstream question-time updates, so this is not a pure read-only
matrix probe. Nonzero one-update safety checks verify direct K/V channel identity,
prefix KV/recurrent/convolution/logit identity, finite query/encoder gradients,
frozen backbone gradients, and zero record probabilities during teacher forcing
and cached answers. The full unit suite has 38 passing tests, including cached
query offsets and rejection of incomplete counterfactual families.

Matched fresh local/query training completed as `gdn-query-capacity-20261114-01`.
Both arms use the exact paired examples, cyclic order, accumulation of two,
seed 1701, learning rate 0.001, and fixed 64/256/1024-step evaluations. No original
test or 70000-series benchmark is used to choose the budget. This capacity
diagnostic remains subordinate to the multi-entity semantic write/read question.

The independent audit checks all 1,152 evaluation rows, 2,048 optimizer steps and
six adapter/AdamW/RNG snapshots, and re-audits the original native and safety
gates. All sampled trained prefixes remain exact, including recurrent and
convolution state, historical KV and logits. Actual record probabilities are
zero during teacher forcing and cached answers at all six attention layers.

| Fixed updates | Local train / 128 | Query train / 128 | Local validation / 64 | Query validation / 64 |
| --- | ---: | ---: | ---: | ---: |
| 64 | 3 | 5 | 2 | 1 |
| 256 | 10 | 22 | 2 | 7 |
| 1024 | 77 | 97 | 27 | 29 |

At 1024 updates, complete-four train families are 5/32 (local) and 10/32
(query); validation families are 0/16 and 1/16. More than one value is recovered
in 9/16 local and 8/16 query validation families. Such families exceed the
one-of-four bound for any deterministic prefix-independent answer with these
identical questions and positions. This supplies constructive evidence of
partial prefix-conditioned information use without changing pre-question
computation. It does not show broad reliable readout, pure GDN-matrix recovery,
write damage, or a shared-interface advantage. Query and local target different
projections; their two-answer validation gap on one initialization does not
establish query superiority. No further generic budget increase is scheduled.
Results and standalone descriptive figures are
`gdn-query-capacity-20261114.{json,png,pdf}`. Semantic mutation checks reject a
wrong optimizer step and an opened record column even with recomputed hashes.

```bash
python scripts/run_single_fact_probe.py --output /path/to/new-native-probe
python scripts/analyze_single_fact_probe.py /path/to/completed-native-probe \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/native-probe.json"
python scripts/run_gdn_query_capacity.py --output /path/to/new-capacity-run
python scripts/analyze_gdn_query_capacity.py /path/to/completed-capacity-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/query-capacity.json"
```

Each fixed checkpoint saves only adapter weights, AdamW moments and RNG. The
runner audits its optimized input/answer budgets, complete four-sibling readouts,
actual blocked probabilities and preserved prefix before continuing. Resume is
available with `--resume` only after the original process has stopped, and
requires unchanged plan, source, data and safety gates.

The return-to-mainline `paired-semantic-updates-v1` data are registered before
inference: 864 train and 432 validation question examples, from 24/12 underlying
four-value families. They cover overwrite, independent/dependent swaps, repeated
SET, repeated ADD and snapshot COPY, with before/control/treatment and
current/protected/initial readouts. Every fixed question/variant has four distinct
answers; executable truth verifies required order, repeated-operation semantics
and protected/initial preservation. Registered token IDs and positions match
outside SET value literals; ADD amounts stay constant. These data use full
historical access and disjoint train/validation entity namespaces. The native
validation probe is complete and audited (`paired-semantic-native-20261114.json`):
384/432 answers, 86/108 complete-four query groups, 98/144 three-role contexts,
and 17/36 complete-twelve value/variant groups are correct. There are five
unchanged-fact regressions across four case IDs from three base event families.
These are functional regressions, not isolated recurrent-write effects.

All 16 updated ADD target answers are wrong in the native full ledger. The
pre-registered competence control (`semantic-add-competence-20261114.json`)
retains the same executable truth: a rebuilt one-account SET/ADD ledger answers
6/16; a direct integer expression answers 13/16. Rebuilding changes state,
positions and context, while expressions change the question. Neither gap can be
assigned to one mechanism, and later ADD gains must not be called memory
protection by themselves.

`configs/paired_semantic_controller_plan.json` fixes four fresh approximately
0.69M-parameter controls: shared, independent, local, and question-only local.
The four-arm real-model safety smoke and independent audit passed. The full run
`paired-semantic-controllers-20261114-01` is complete and independently audited:
every arm saw exactly one ordered pass through 864 examples, 432 optimizer
updates, 527,808 input tokens
and 3,456 supervised answer tokens. Checkpoints at 144/288/432 preserve adapter,
AdamW and RNG; only the final checkpoint gets semantic validation. Final
teacher-forced training fit is measured separately from unrestricted validation
answers. All modes use the same answer LM loss, without auxiliary consistency
losses or oracle scope labels. This tests a generic shared bottleneck, not an
already established semantic interface.

The semantic comparisons require all relevant answers to be correct. Stable
unchanged units include before/control/treatment together, so making the before
answer wrong cannot fake improved preservation. Equal scores retain their paired
rescues and regressions; ADD is also separated from other current-state updates.
The unit suite has 54 passing tests. Same-question value pairing excludes a
fixed question-only lookup, but does not exclude a fixed record-position
shortcut. The registered initial-order diagnostic below tests this limitation;
wording and fresh-stream confirmation remain required before an architecture claim.

| Frozen 2B IT plus adapter | Answers / 432 | Three-role joint / 144 | Complete-twelve groups / 36 | Stable unchanged case/role / 96 |
| --- | ---: | ---: | ---: | ---: |
| Native | 384 | 98 | 17 | 86 |
| Shared | 429 | 141 | 33 | 96 |
| Independent | 429 | 141 | 34 | 96 |
| Local | 391 | 103 | 23 | 94 |
| Question-only local | 427 | 139 | 33 | 96 |

The audit covers 1,728 optimizer steps, all twelve adapter/AdamW/RNG snapshots,
3,456 final teacher-forced training observations, and 1,728 free validation
answers. Native outputs and source/data registration were re-audited. Final
question-only sampled short/long pre-question KV, recurrent/convolution state
and logits remain exact; final full-history access is observed at all six
attention layers. No new original-test or reused 70000-series inference ran.

Shared, independent and question-only local all solve 80/80 updated target
questions outside ADD, and all 96 unchanged case/role units. Shared's two-answer
advantage over question-only local occurs entirely in ADD; shared and independent
have equal answer totals but trade one corrected and one regressed answer.
There is no demonstrated shared-interface advantage. Repairing these canonical
semantic readouts does not require changing pre-question writes. This is a
constructive task-specific alternative, not general proof that recurrent write
damage never occurs. Local all-token training also fits less well: final numeric
teacher-token correctness is 2,528/2,592, versus 2,591 for shared/independent and
2,586 for question-only local. That optimization difference limits structural
interpretation of its lower validation score.

Artifacts are `paired-semantic-controllers-20261114.json` and
`paired-semantic-comparison-20261114.{json,png,pdf}`; the exported figure has been
visually checked. Results use one initialization and twelve underlying event
families. They are feasibility diagnostics, not an architecture result.

The pre-inference `initial-order-derangement-v1` data move all eight independent
initial SET records for each of the same 48 validation cases. Executable current
and initial truth, all tails, question IDs and absolute query positions remain
exactly unchanged; all four value siblings remain matched outside value tokens.
The condition is a reused-validation functional order diagnostic, not untouched
confirmation or isolated recurrent-write damage.
`configs/initial_order_control_plan.json` pins all four trained adapters and
requires native plus all four frozen controls. The runner and independent auditor
are implemented. The 20-output operational smoke and all 2,160 full frozen
readouts passed independent audit. No order-control training or checkpoint
selection was performed. All four loaded adapter state hashes match the original
final checkpoints; sampled detach/native restoration and six-layer full-history
access pass. Question-only sampled short/long permuted prefixes retain exact
native KV, recurrent/convolution state and logits.

| Frozen model | Canonical answers / 432 | Permuted answers / 432 | Permuted three-role joint / 144 | Permuted stable unchanged / 96 | Permuted non-ADD updates / 80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 384 | 367 | 88 | 81 | 60 |
| Shared | 429 | 419 | 131 | 96 | 74 |
| Independent | 429 | 429 | 141 | 96 | 78 |
| Local | 391 | 396 | 109 | 89 | 67 |
| Question-only local | 427 | 423 | 135 | 91 | 80 |

Independent has ten correct answers that shared misses, with no shared-only
correct answers. Its unchanged overall total hides two rescues and two
regressions across orders; it is not exact order invariance. Both shared and
independent retain all 96 stable unchanged units. Question-only retains all
80 non-ADD updated targets, while losing five stable unchanged units. Overall
accuracy alone would miss this distinction. These are twelve underlying event
streams and one initialization; no universal superiority or shared-semantic
interface claim follows.

A descriptive error check finds that 9/26 native and 7/10 local unchanged-fact
errors equal the initial value now occupying the queried entity's old record
slot. None of the eight question-only unchanged-fact errors has that form.
There is no chance correction or unique causal attribution: legacy-slot lookup
does not explain every error, and recomputing the prefix changes both recurrent
state and contextual KV. The artifacts are
`initial-order-control-20261114.{json,png,pdf}` and
`initial-order-shortcuts-20261114.json`; the exported figure has been visually
checked.

The ordinary initial-order augmentation control below keeps the same training
example/token/optimizer budgets. Both already-observed validation orders remain
development diagnostics. Fresh event streams and query templates are needed for
later confirmation.

```bash
python scripts/run_paired_semantic_probe.py --output /path/to/new-native-semantic-run
python scripts/analyze_paired_semantic_probe.py /path/to/completed-native-semantic-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/native-semantic.json"
python scripts/run_paired_semantic_controllers.py --output /path/to/new-semantic-controllers
python scripts/analyze_paired_semantic_controllers.py /path/to/completed-semantic-controllers \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/semantic-controllers.json"
python scripts/run_initial_order_control.py --output /path/to/new-frozen-order-control
python scripts/analyze_initial_order_control.py /path/to/completed-frozen-order-control \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/initial-order-control.json"
python scripts/plot_initial_order_control.py "$SAS_STORAGE_ROOT/artifacts/memory_update/initial-order-control.json" \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/initial-order-control.png"
```

Ordinary order augmentation is now registered in
`configs/order_augmented_semantic_plan.json` and
`paired-semantic-order-augmented-v1`. Stream indices 0/3 retain canonical order;
indices 1/2 deterministically derange all eight independent initial records.
Each task and each short/long context length has one canonical and one deranged
base train stream. All four value siblings share the same order. Every train
identity, tail, question, answer, absolute query position and optimized token
length stays matched; canonical validation remains byte-for-byte unchanged.
There are still 864 examples, 527,808 input tokens, 3,456 supervised answer tokens
and 432 optimizer updates per arm. Original initial controller state hashes are
pinned, in addition to the unchanged architecture, loss, optimizer and seed.

The original training engine accepts an optional final validation callback
and extra source snapshots; default training behavior is unchanged. This trial
adds the separate registered permuted validation after final canonical
validation and teacher-fit measurement. The 49-test suite and all four real-model
one-update safety checks passed; the safety run is independently audited. The
full run `order-augmented-semantic-20261114-01` is complete and independently
audited: 1,728 optimizer updates, twelve adapter/AdamW/RNG checkpoints, 3,456
final teacher-forced training observations and 3,456 free validation answers.
Every arm's actual initial state hash matches its original canonical-trained
counterpart. Evaluation weights/RNG stay unchanged; sampled six-layer full
historical access, native restoration after detach and short/long question-only
native-prefix identity in both orders pass. Both validation orders remain
observed development diagnostics, not fresh confirmation.

| Controller | Canonical old → augmented / 432 | Permuted old → augmented / 432 | Augmented stable unchanged: canonical / permuted, each of 96 | Augmented non-ADD updates: canonical / permuted, each of 80 |
| --- | ---: | ---: | ---: | ---: |
| Shared | 429 → 429 | 419 → 428 | 95 / 95 | 80 / 78 |
| Independent | 429 → 429 | 429 → 428 | 96 / 96 | 80 / 80 |
| Local | 391 → 428 | 396 → 427 | 96 / 96 | 80 / 78 |
| Question-only local | 427 → 428 | 423 → 418 | 96 / 96 | 79 / 74 |

Ordinary augmentation closes most of the earlier shared-versus-local gap.
Local gains 38 rescued/1 regressed canonical answers and 33 rescued/2 regressed
permuted answers. Shared gains 10 rescued/1 regressed permuted answers.
Shared and independent have equal totals after augmentation but trade three
correct answers in each direction in the permuted condition. Independent also
retains more unchanged and non-ADD target units there. Equal totals are not
functional equivalence, and no unique shared-interface gain is established.

Question-only improves stable unchanged units from 91/96 to 96/96 under the
permuted order, yet loses target-update competence: its overall change is
9 rescued/14 regressed answers. Thus lower overall accuracy here does not mean
worse preservation. Its sampled pre-question KV/recurrent/convolution/logit
identity stays exact; question-phase adaptation can still change GDN and
Attention during question processing. This is not a pure Q-projection ablation.

Final numeric teacher-token fit out of 2,592 is shared 2,585, independent 2,591,
local 2,578 and question-only 2,585; all 864 newlines per arm are correct. Local's
fit improves from the original 2,528, so optimization/data effects remain part
of the structural comparison. These results use one paired initialization and
twelve underlying validation event streams, with no fresh-cohort claim.
Artifacts are `order-augmented-semantic-20261114.json` and
`order-augmented-comparison-20261114.{json,png,pdf}`. The exported figure has
been visually checked; its report/source/image hashes are recorded separately.

```bash
python scripts/run_order_augmented_semantics.py --output /path/to/new-order-augmented-run
python scripts/analyze_order_augmented_semantics.py /path/to/completed-order-augmented-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/order-augmented-semantic.json"
python scripts/compare_order_augmented_semantics.py "$SAS_STORAGE_ROOT/artifacts/memory_update/order-augmented-semantic.json" \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/order-augmented-comparison.json" \
  --plot "$SAS_STORAGE_ROOT/artifacts/memory_update/order-augmented-comparison.png"
```

`configs/semantic_order_path_probe_plan.json` registers the next native-only
mechanism diagnostic, gated on independently audited completion of the entire
ordinary-augmentation trial. It retains every original validation readout and
crosses canonical/permuted recurrent-plus-convolution state with
canonical/permuted attention KV across all six groups. Its 1,728 cached query
readouts include both natural split-prefill baselines; the old full-prefill
outputs are separately labeled calibration references. Self/all-path replay and
source-cache immutability are required. Mixed caches are counterfactuals that
may be out of distribution, so rescue is not proof of unique storage or GDN
write damage. The augmentation audit gate is now passed. The runner
`scripts/run_semantic_order_path_probe.py` and independent auditor
`scripts/analyze_semantic_order_path_probe.py` are implemented. CPU reconstruction
verifies all 432 query positions and both natural input origins. New controls
detect bookkeeping changes, wrong convolution/recurrent/KV provenance and
unpaired NLL contrasts; the full suite has 54 passing tests. Natural prefixes
are recomputed per query to preserve the registered evaluation order without
retaining every cache on the GPU. Repeated prefix hashes must agree. Self and
all-path replay, actual cached historical access at every Attention layer,
source immutability and full-vs-cached calibration remain required before any
mechanism interpretation. The operational smoke and complete native path run
are now independently audited; the results are recorded below.

`configs/semantic_single_path_controls_plan.json` registers the two component
controls explicitly required by the co-design proposal: selected-group GDN-only
rank 84 (expected 696,192 parameters) and Attention-only rank 150 (expected
691,200). The original augmented data, seed, loss, optimizer, 432-step schedule,
three checkpoint boundaries and both complete development orders are fixed.
Loaded-model shape/count/gradient/native-identity safety checks must precede
fresh training. These modes change the scope of trainable projection adapters;
they do not imply that downstream hidden states or other memory paths remain
unchanged. Adequate training fit is required before interpreting a failed
component-only model. The four-arm augmentation audit gate has passed; these
runners and independent auditor are implemented. Seeded initialization
reconstruction and rejection of mixed projection scopes pass. Both real-model
safety checks pass with the expected 696,192/691,200 counts and are independently
audited. Formal run `semantic-single-path-20261114-01` is complete and independently
audited: 864 optimizer updates, six adapter/AdamW/RNG checkpoints, 1,728 teacher
training observations and 1,728 free validation answers. Actual initial states
are reconstructed independently from seed 1701. Frozen backbone gradients,
sampled six-layer full historical access and exact native restoration after
detach pass.

| Adapted projections | Canonical / 432 | Permuted / 432 | Stable unchanged: canonical / permuted, each of 96 | Non-ADD updates: canonical / permuted, each of 80 |
| --- | ---: | ---: | ---: | ---: |
| GDN only | 426 | 429 | 96 / 96 | 80 / 80 |
| Attention only | 408 | 405 | 94 / 95 | 74 / 72 |
| Shared reference | 429 | 428 | 95 / 95 | 80 / 78 |
| Independent reference | 429 | 428 | 96 / 96 | 80 / 80 |

The full hybrid backbone is used by every row; these are trainable-parameter
scopes, not isolated memory pathways. GDN-only's six/three errors are entirely
ADD target updates (10/16 and 13/16 correct). Shared versus GDN-only rescues
four/regresses one canonical answer and rescues two/regresses three permuted
answers. Shared loses one stable unchanged unit in each order and two non-ADD
target units in the permuted order. This development cohort therefore does not
establish a need to adapt both component types or a unique shared-interface
advantage.

Numeric teacher-token fit is GDN-only 2,584/2,592, Attention-only 2,551/2,592;
all 864 newlines per arm are correct. GDN-only is close to shared's 2,585,
whereas Attention-only has poorer fit (loss 0.03906 versus 0.00632 for GDN-only).
Attention-only failure cannot establish joint necessity. Counts/ranks differ
slightly across near-matched parameter allocations. One initialization and the
same twelve underlying development streams do not establish superiority or
equivalence. The shared controller is generic low-rank adaptation trained only
on answer loss; its latent is not yet demonstrated to encode semantic scope.
Artifacts are `semantic-single-path-20261114.json` and
`semantic-single-path-comparison-20261114.{json,png,pdf}`. The figure is visually
checked and its source/report/image hashes are recorded separately. The native
cache-source diagnostic remains next; B/C structures and larger-scale training
are deferred pending mechanism evidence.

```bash
python scripts/run_semantic_single_path_controls.py --output /path/to/new-component-controls
python scripts/analyze_semantic_single_path_controls.py /path/to/completed-component-controls \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/semantic-single-path.json"
python scripts/compare_semantic_single_path_controls.py "$SAS_STORAGE_ROOT/artifacts/memory_update/semantic-single-path.json" \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/semantic-single-path-comparison.json" \
  --plot "$SAS_STORAGE_ROOT/artifacts/memory_update/semantic-single-path-comparison.png"
```

```bash
python scripts/run_semantic_order_path_probe.py --smoke --output /path/to/new-native-path-smoke
python scripts/analyze_semantic_order_path_probe.py /path/to/completed-native-path-smoke --allow-smoke \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/semantic-order-path-smoke.json"
```

Native run `semantic-order-path-20261114-01` is complete and independently
audited: all 1,728 registered free answers and gold NLLs, with no training or
adapters. The 432 query-level controls verify tensor/metadata source provenance
and source-cache immutability. Each natural origin has 144 distinct prefixes,
recomputed three times for its three query roles (864 total natural prefills).
Repeated prefix hashes match. Eight prescribed observations show actual cached
historical access at all six Attention layers; eight self/all-path donor
replays match generation with zero NLL error. All backbone parameters remain
frozen and observer hooks are restored.

Here C means canonical initial-record order and P means its registered
independent-record permutation. State includes all recurrent/convolution paths.
Every primary readout uses cached question processing.

| Recurrent/conv origin | KV origin | Answers / 432 | Stable unchanged / 96 | Non-ADD updates / 80 |
| --- | --- | ---: | ---: | ---: |
| C | C | 385 | 86 | 61 |
| P | P | 368 | 81 | 61 |
| C | P | 370 | 84 | 58 |
| P | C | 374 | 87 | 55 |

The old full-prefill natural scores were 384/367. Cached natural processing
rescues two/regresses one answer in each order; mean absolute NLL differences
are 0.00671/0.00992 and maxima 0.07908/0.09766. The canonical-versus-permuted
17-answer gap remains. Equal 61/80 non-ADD totals hide eight rescues/eight
regressions between the two natural orders.

Compared with P/P, changing only KV to C rescues 34/regresses 28 individual
answers, improves stable unchanged units by 15 rescues/9 regressions, but loses
non-ADD updates by 5 rescues/11 regressions. Changing only recurrent/conv origin
to C yields 8 rescues/6 regressions overall, gains three stable units and loses
three non-ADD targets. Neither mixture recovers the canonical natural result
across all functions. These are counterfactual cache-source effects, not a new
method or proof of a uniformly damaged memory pathway.

Mean gold-NLL effects of P rather than C recurrent/conv origin are +0.03108 at
C KV and -0.00538 at P KV. KV-origin effects are +0.11063 at C state and
+0.07417 at P state; the difference-in-differences is -0.03646. Family/role
tables expose substantial heterogeneity, including larger interactions for ADD
and COPY current-target readouts. These effects can include incompatibility of
mixed representations; they do not establish semantic complementarity or
locate irrecoverable write damage. The twelve reused underlying streams remain
development diagnostics.

Artifacts are `semantic-order-path-20261114.json` and
`semantic-order-path-diagnostics-20261114.{json,png,pdf}`. The exporter retains
both natural calibration comparisons, all family/role cells and all twelve
base-stream NLL averages. Gray plot points are related-readout cluster means,
not confidence intervals. The exported figure has been visually checked and
its source/report/image hashes are recorded separately.

```bash
python scripts/run_semantic_order_path_probe.py --output /path/to/new-native-path-run
python scripts/analyze_semantic_order_path_probe.py /path/to/completed-native-path-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/semantic-order-path.json"
python scripts/export_semantic_order_path_diagnostics.py "$SAS_STORAGE_ROOT/artifacts/memory_update/semantic-order-path.json" \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/semantic-order-path-diagnostics.json" \
  --plot "$SAS_STORAGE_ROOT/artifacts/memory_update/semantic-order-path-diagnostics.png"
```

Run `gdn-question-timing-20261114-01` completes the matched control registered
in `configs/gdn_question_timing_control_plan.json`: GDN-only rank84 adaptation
restricted to question/answer processing, with exactly the same initial weights,
696,192 trainable parameters, augmented examples, loss and 432-step budget as
the all-token GDN baseline. The independent audit verifies all three checkpoints,
864 teacher-fit rows and 864 validation answers, together with the full native
path and component reference audits. Zero-initialization identity, actual
projection scope, finite adapter gradients, frozen backbone and native restoration
checks pass. Prescribed shortest/longest prefixes in both orders have exactly
native KV, recurrent state, convolution cache and logits after adaptation.

| GDN adaptation timing | Canonical answers /432 | Permuted answers /432 | Stable unchanged /96 C/P | Non-ADD updates /80 C/P |
| --- | ---: | ---: | ---: | ---: |
| All tokens | 426 | 429 | 96 / 96 | 80 / 80 |
| Question and answer | 427 | 430 | 95 / 96 | 79 / 80 |

Canonical question-stage adaptation rescues four/regresses three individual
answers relative to all-token adaptation. Its one stable-unit regression is a
protected query already wrong in the before variant, not newly damaged by the
controlled update; its non-ADD loss is one COPY target. Permuted question-stage
adaptation rescues one/regresses none. Both arms retain the same number of fully
correct four-value and twelve-readout units in each order. Small total-score
differences do not establish superiority or equivalence.

Teacher numeric-token fit is 2,588/2,592 for question-stage adaptation versus
2,584/2,592 for all-token adaptation; all 864 newlines are correct in each arm.
The corresponding mean losses are 0.004157 and 0.006321. All remaining
question-stage teacher errors belong to ADD current-target readouts (140/144
numeric tokens versus the all-token arm's 141/144); its COPY teacher fit is
144/144 versus 139/144. The export includes all eighteen family/role fit cells,
so aggregate fit does not hide these differences.

These twelve reused development streams support substantial semantic competence
while leaving the entire historical prefix native. They do not establish that
modifying historical writes is needed for the all-token baseline's high scores.
The full hybrid model and all historical Attention remain active, and
question/answer processing can create fresh state and KV; this does not isolate
Attention queries or prove absence of recurrent damage. It remains a falsification
control for the whole hybrid-block update question. Shared semantic scope and a
need to adapt both component types have still not been demonstrated.

Artifacts are `gdn-question-timing-20261114.json` and
`gdn-question-timing-comparison-20261114.{json,png,pdf}`, with source snapshots
and a provenance file recording the visually checked figure. The full suite
passed 56 tests and 8 subtests after the timing-control additions; the exporter
also checks archived teacher observations against the independent audit.

```bash
python scripts/run_gdn_question_timing_control.py --output /path/to/new-timing-control
python scripts/analyze_gdn_question_timing_control.py /path/to/completed-timing-control \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/gdn-question-timing.json"
python scripts/compare_gdn_question_timing_control.py "$SAS_STORAGE_ROOT/artifacts/memory_update/gdn-question-timing.json" \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/gdn-question-timing-comparison.json" \
  --plot "$SAS_STORAGE_ROOT/artifacts/memory_update/gdn-question-timing-comparison.png"
```

The existing semantic tasks contain only eight distinct accounts per stream;
their longest ledger is 803 tokens. The next bounded native diagnostic is
registered in `configs/semantic_write_locality_load_plan.json`. It fixes 160
prefix records and two tail records while varying active accounts 8/32/128.
Four tail cells toggle A and C between their old and new values. This preserves
B and A's initial value; toggling C also preserves A's current value. All
576 planned prompts pass a tokenizer/executable-truth feasibility check:
each has exactly 2,865 tokens, identical query suffixes and aligned positions,
with tail differences confined to the specified value digits. Four value
siblings prevent a fixed-question answer shortcut. The samples share four
underlying base streams, not 576 independent replications.

The complete screen is now implemented, executed and independently audited as
`write-locality-load-20261114-01`. Data are frozen in
`datasets/memory_update/fixed-length-write-locality-load-v1`, with 48 value cases,
192 distinct ledger prefixes and 576 ordered queries. The complete numeric
and record-boundary token registry is reconstructed by the auditor, along with
executable truths, value-sibling and tail-toggle controls. Local tokenizer,
model configuration, checkpoint and implementation hashes are pinned. The
12-readout operational smoke passed before the full run; the existing
environment and model files were reused.

All loaded model state tensors have identical full byte hashes before and after
both runs. Both prescribed native replays agree exactly and actual historical
prefill/cached access is observed at all six Attention layers. No adapters or
training are involved. The expanded suite passes 60 tests and 11 subtests;
targeted data/statistics checks also pass after the final registration guards.

In the next table, C-only means A0_C1 and A+C means A1_C1. Thus the comparison
changes only A's three registered value-digit tokens, keeping C updated in
both conditions. Every role/cell count is out of 16 related value readouts.

| Active accounts | All answers /192 | Current A across cells /64 | B current: C-only to A+C /16 | Initial A: C-only to A+C /16 |
| --- | ---: | ---: | ---: | ---: |
| 8 | 187 | 64 | 15 to 15 | 16 to 15 |
| 32 | 172 | 64 | 13 to 11 | 16 to 13 |
| 128 | 169 | 64 | 13 to 12 | 16 to 11 |

The full result is 528/576. At load 32, the two B-current regressions occur in
two distinct base streams, satisfying the registered repeated functional-symptom
rule. At load 128, the five initial-A regressions span all four base streams.
Toggling C after A has updated also regresses initial-A from 13/16 to 11/16 in
two streams. All 30 preservation contrasts, all six target-update comparisons,
all 36 cell/role counts and all twelve load/base-stream clusters are exported;
baseline failures and rescues remain visible. Discovery is four base streams,
not 576 independent replications.

The 48 errors comprise 34 B-current and 14 initial-A errors. Every initial-A
error outputs the current A value. Of the B errors, 22 output an A value and 12
output an initial/current C value. These exact value matches are descriptive
entity/version clues, not an attribution to Attention or a memory erase
operator. Many B errors are already present in the no-op baseline and cannot
be counted as new update-induced damage.

Initial-A needs an additional caveat: A0 repeats the old value near the question,
whereas A1 replaces that redundant recent evidence. Losing an initial-A answer
can therefore reflect version selection or dependence on recent evidence.
B's evidence and age stay identical across cells, making the matched B
regression the cleaner unrelated-fact symptom. Neither result yet establishes
irreversible recurrent-state damage or a need for a shared controller.

Artifacts are `write-locality-load-20261114.json` and
`write-locality-load-diagnostics-20261114.{json,csv,png,pdf}`, with archived
analysis source and visually verified figure provenance. The earlier tokenizer
feasibility check remains separately archived and is not treated as model
evidence.

```bash
python scripts/prepare_write_locality_load_screen.py --output /path/to/new-locality-data
python scripts/run_write_locality_load_screen.py --dataset /path/to/new-locality-data \
  --smoke --output /path/to/new-locality-smoke
python scripts/analyze_write_locality_load_screen.py /path/to/new-locality-smoke --allow-smoke \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/new-locality-smoke.json"
python scripts/run_write_locality_load_screen.py --dataset /path/to/new-locality-data \
  --smoke-audit "$SAS_STORAGE_ROOT/artifacts/memory_update/new-locality-smoke.json" \
  --output /path/to/new-locality-run
python scripts/analyze_write_locality_load_screen.py /path/to/new-locality-run \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/new-locality.json"
python scripts/export_write_locality_load_screen.py "$SAS_STORAGE_ROOT/artifacts/memory_update/new-locality.json" \
  --output "$SAS_STORAGE_ROOT/artifacts/memory_update/new-locality-diagnostics.json" \
  --plot "$SAS_STORAGE_ROOT/artifacts/memory_update/new-locality-diagnostics.png"
```

The independent replica is complete and separately audited as
`write-locality-replication-20261114-01`. Its protocol fixes four genuinely new
seeds 97100--97103 and the unchanged complete 576-query grid; no cases are
relabeled from discovery. The full discovery audit was reexecuted before data
registration. The 12-readout operational smoke passed its safety audit, then
the complete replica ran with identical loaded native weights and packages.
No adapters, training, extra environments or model copies were used.

Both registered primary effects recur. B-current at load 32 goes from 12/16
to 10/16, with two regressions in two new base streams and no rescues.
Initial-A at load 128 goes from 16/16 to 6/16, with ten regressions across
three new streams and no rescues. Discovery and replica remain separate
four-stream cohorts; the related value readouts are not independent samples.
All secondary comparisons, all cell/role counts, base-stream results and error
value matches are exported, including baseline failures.

The replica totals 486/576, broken down as 180/192, 158/192 and 148/192 at
loads 8, 32 and 128. Current A is 191/192, B-current 140/192 and initial-A
155/192. All 37 initial-A errors output the current A value. These descriptive
matches and the recurring preservation symptoms do not establish recurrent
memory destruction or shared-interface necessity. Initial-A retains the
recent-redundant-evidence caveat above. The native engines are sealed; separate
provider bindings parameterize the real new identities without changing the
historical implementation or statistics.

Artifacts are `write-locality-replication-20261114.json` and
`write-locality-replication-comparison-20261114.{json,csv,png,pdf}`, with source
snapshots and visually checked figure provenance. The replica additions passed
63 tests and 11 subtests, plus exact reproduction of discovery's metrics from
its archived actual observations.

```bash
python scripts/prepare_write_locality_replication.py
python scripts/run_write_locality_replication.py --dataset "$SAS_STORAGE_ROOT/datasets/memory_update/fixed-length-write-locality-replication-v1" \
  --smoke --output /path/to/new-replica-smoke
python scripts/analyze_write_locality_replication.py /path/to/new-replica-smoke --allow-smoke \
  --output /path/to/new-replica-smoke.json
python scripts/run_write_locality_replication.py --dataset "$SAS_STORAGE_ROOT/datasets/memory_update/fixed-length-write-locality-replication-v1" \
  --smoke-audit /path/to/new-replica-smoke.json --output /path/to/new-replica-run
python scripts/analyze_write_locality_replication.py /path/to/new-replica-run \
  --output /path/to/new-replica-audit.json
```

Data registration refuses to overwrite existing immutable registrations. Its
command documents first creation rather than a way to rerun an existing path.

`configs/semantic_write_locality_event_paths_plan.json` is conditional on at
least one primary recurring in two new base streams. It registers aligned
ledger-only recurrent/convolution versus KV source interventions, cached/full
calibration, immutable-source and self/all-path controls, plus separately
labeled oracle-evidence access. The companion requirement in
`configs/semantic_write_locality_event_paths_addendum.json` also checks A's
legitimate current update, so reverting an update cannot masquerade as a
successful preservation intervention. The additional
`configs/semantic_write_locality_event_paths_commit_guard.json` checks C's
legitimate current value for C-axis interventions, including separately
registered full-prefill C-query baselines. Mixed-cache compatibility, oracle
readout limitations and natural-endpoint score differences must be reported.
The replica meets this diagnostic gate. Both complete native audits were
reexecuted before event-data registration. The provider/preparer binds all 96
pair/role units across both cohorts, including the additional C prompts and
both natural-world targets. Independent input reconstruction reproduces the
entire token registry, ledger-only boundaries, identical supporting evidence
and exact token masks. No error-only case selection is allowed.

The event-path runner and separate auditor are implemented. The operational
smoke completed and independently audited 96 readouts and 28 cached/full
calibrations across six pair/role units. Loaded model states remain identical;
all six supporting-record KV layers match exactly between natural tail worlds.
Thirty-six actual full/oracle Attention probability observations, complete
recurrent/convolution/KV provenance, source immutability and self/all-path clone
replays pass. The suite passes 70 tests and 11 subtests. Smoke performance does
not gate execution.

The complete diagnostic is now independently audited as
`write-locality-event-paths-20261114-01`: 1,200 actual readouts and 448
cached/full calibration observations. There are 96 pair/role units and 192
physical ledger prefills, covering 160 unique natural source prefixes. Repeated
prefix fingerprints agree. Every paired supporting record's K/V is exactly
identical at all six Attention layers. All 72 actual full/oracle probability
checks and prescribed 48 self/all-path clone replays pass. Complete loaded
native state hashes remain unchanged.

L and R below mean matched no-op/update worlds at identical ledger positions,
not states taken before/after a chronological step. Each includes all 162
ledger records. GDN state replacement includes all 18 recurrent matrices and
18 convolution caches. KV replacement covers all six Attention layers.

| Cohort / preservation query | Natural R state / R KV | L state / R KV | Joint preservation + right-world updates under L/R |
| --- | ---: | ---: | ---: |
| Discovery: B, 32 accounts | 11/16 | 14/16 | 14/16 |
| Replica: B, 32 accounts | 10/16 | 12/16 | 12/16 |
| Discovery: initial A, 128 accounts | 11/16 | 13/16 | 13/16 |
| Replica: initial A, 128 accounts | 6/16 | 6/16 | 6/16 |
| Discovery: initial A after C update, 128 accounts | 11/16 | 13/16 | 12/16 |
| Replica: initial A after C update, 128 accounts | 6/16 | 6/16 | 6/16 |

At B/32, L-state/R-KV rescues three discovery answers across two streams and
two replica answers across two new streams, with no regressions. Current A
remains 16/16 in both cohorts. This identifies a recurrent/convolution path
contribution under the fixed-KV intervention, not irreversible storage damage
or a requirement for joint adaptation. Restoration does not repair all B
errors, and the initial-A benefit fails to recur in the replica.
The discovery B rescues were already wrong at the cached L endpoint; they
cannot be labeled reversals of newly induced damage. The replica B rescues
are the two L-correct/R-wrong cases. Individual and base-stream exports mark
this distinction explicitly. Uniform mediation of update-induced errors
across cohorts has not been established.

R-state/L-KV gets initial A to 16/16 in the A-axis stratum but current A to
0/16 in both cohorts. It is a version rollback, not a successful semantic
commit. In the C-axis discovery stratum, one of the two L/R preservation
rescues remains non-joint because its C-current answer was already wrong in
the natural R baseline; the intervention does not introduce a C regression.
All 192 oracle-preservation reads are correct while keeping their original
ledger computation and source caches. These controls provide the evidence
address; they do not establish uncorrupted compressed state or a joint-update
repair method.

Cached and full execution differ on five of 448 reported comparisons: two
rescues and three regressions, including repeated source/question bindings
across strata. Full totals are 377/448 and cached totals 376/448. Equal
cell totals can also hide changed individual cases. All source-path effects
above compare cached execution with cached execution; they are not silently
equated with the earlier full-prefill screen. Both cohorts still contain four
base streams, not hundreds of independent replications.

Artifacts are `write-locality-event-paths-20261114.json` and
`write-locality-event-paths-diagnostics-20261114.{json,csv,png,pdf}`, plus the
`-commit.{png,pdf}` figure, analysis snapshots and visually inspected figure
provenance. Exports include all 24 cohort/stratum/base-stream cells and all
384 individual condition/preservation transitions, separately reporting
current A/C and joint outcomes.

```bash
python scripts/prepare_write_locality_event_paths.py
python scripts/run_write_locality_event_paths.py --smoke --output /path/to/new-event-smoke
python scripts/analyze_write_locality_event_paths.py /path/to/new-event-smoke --allow-smoke \
  --output /path/to/new-event-smoke.json
python scripts/run_write_locality_event_paths.py --smoke-audit /path/to/new-event-smoke.json \
  --output /path/to/new-event-run
python scripts/analyze_write_locality_event_paths.py /path/to/new-event-run \
  --output /path/to/new-event-audit.json
python scripts/export_write_locality_event_paths.py /path/to/new-event-audit.json \
  --output /path/to/new-event-diagnostics.json --plot /path/to/new-event-diagnostics.png
```

B/C structures and 4B/9B scaling remain deferred; this SET-only diagnosis
supplements the existing order/COPY/history tests and matched adaptation
baselines. A mixed-cache rescue that loses a legitimate current A/C update is
reported explicitly, rather than presented as a successful semantic commit.

The next bounded control is registered in
`configs/write_locality_adapter_transfer_plan.json`. It uses the existing
paired group3/rank84 GDN-only final step432 checkpoints: all-token versus
question-only adaptation, 696,192 parameters each. No new training, checkpoint
selection or environment changes. All 448 natural pair/endpoint readout
bindings per mode are retained; deduplication yields 384 actual full-prefill
questions per mode. References use full-prefill native observations, with the
C baselines above. New long-prefix identity and actual-access smoke checks
are required before the complete fixed-checkpoint transfer. A failed transfer
can reflect distribution shift from the eight-account/832-token training
tasks and does not establish a need for prefix-writing or a shared interface.

The transfer provider, deduplicated runner and independent auditor are now
implemented. The complete event diagnosis audit was reexecuted before
registration; the two older training audits are pinned and checked against
their actual source/checkpoint artifacts, without claiming to rerun training
or its complete nested audits. The frozen registry reproduces all own-world
targets, full-prefill references and 448-to-384 query bindings per model.
The operational cohort has six pair/role units, 28 reported bindings and 24
actual queries per model. Ten long source-prefix checks and ten historical
Attention/detached-native-replay samples per model are fixed before inference;
the full cohort uses twenty of each. The suite passes 75 tests and 11 subtests.
The operational model run and independent audit completed: 48 actual adapter
queries, 56 reported bindings, ten complete native prefix/cache/logit identity
checks, twenty actual six-layer historical-access observations and twenty
exact detached native replays. Native and adapter state hashes are unchanged.
The full two-checkpoint transfer and independent audit completed: 768 actual
queries, 896 related bindings and 96 pair units per model. Twenty complete
native-prefix checks, forty actual historical-access observations and forty
detached native replays passed. All native and adapter weights are unchanged.

Unique full-prefill accuracy is native 328/384, all-token GDN 370/384 and
question-phase GDN 366/384. At the updated natural B endpoints, the respective
counts are 11/16, 16/16, 16/16 in discovery and 10/16, 16/16, 16/16 in the
replica. Both adapters repair all four native full-prefill update-induced B
regressions while retaining current A at 16/16. Question-phase adaptation
leaves ledger computation native, so these B symptoms do not establish a
requirement to rewrite prefix memories.

Historical A remains unresolved: updated-world initial-A counts are 11/16,
16/16, 15/16 in discovery and 6/16, 10/16, 8/16 in the replica. The all-token
adapter retains thirteen unique historical-A errors and one current-C error;
question-phase retains eighteen historical-A errors. These checkpoints were
trained on shorter eight-account tasks, not on this 2865-token distribution.
Each cohort contains four base streams with related value variants. The small
aggregate timing difference establishes neither superiority nor a need for a
shared hybrid interface.

Complete audit: `write-locality-adapter-transfer-20261114.json`. Every binding,
all paired/base-stream outcomes and the visually checked figure are exported
as `write-locality-adapter-transfer-comparison-20261114.{json,csv,png,pdf}`,
with analysis snapshots and hash provenance. The next bounded comparison uses
the already trained final shared, independent, local, local-question and
Attention-only checkpoints on the same complete natural endpoints, before
new architecture training or scaling.

```bash
python scripts/prepare_write_locality_adapter_transfer.py
python scripts/run_write_locality_adapter_transfer.py --smoke --output /path/to/new-transfer-smoke
python scripts/analyze_write_locality_adapter_transfer.py /path/to/new-transfer-smoke --allow-smoke \
  --output /path/to/new-transfer-smoke.json
python scripts/run_write_locality_adapter_transfer.py --smoke-audit /path/to/new-transfer-smoke.json \
  --output /path/to/new-transfer-run
python scripts/analyze_write_locality_adapter_transfer.py /path/to/new-transfer-run \
  --output /path/to/new-transfer-audit.json
python scripts/export_write_locality_adapter_transfer.py /path/to/new-transfer-audit.json \
  --output /path/to/new-transfer-comparison.json --plot /path/to/new-transfer-comparison.png
```

The additional control panel is fixed in
`configs/write_locality_control_panel_plan.json`. It reuses the identical
384-query registry for final-step432 shared, independent, local,
local-question and Attention-only checkpoints: 1920 actual queries and 2240
related bindings. Registration reexecutes the prior complete transfer audit;
older training audits are pinned and checked against their actual artifacts.
The runner loads a private instance of the sealed transfer execution engine,
explicitly binding the panel model list, protocol and source set. Original
two-model source files, outputs and model weights remain sealed.

This is a generic shared-control baseline, not yet a learned semantic
dependency interface. Shared control must provide additional benefit over
independent and ordinary local adaptation before more elaborate co-design is
justified. All checkpoints use one training initialization and short-task
training; approximately matched parameter counts use different ranks.
Attention-only has weaker original teacher fit. The native/GDN outcomes have
already been observed, so the panel is an additional development comparison,
not a fresh confirmation cohort. Its 120-query operational smoke uses safety
and coverage gates, never a performance threshold. The complete unit suite
passes 79 tests. The 120-query operational smoke and complete 1920-query
panel are independently audited. Twenty complete native-prefix checks,
100 actual history-access observations and 100 exact detached native replays
passed; native and adapter weights are unchanged.

All models below use the same 384 unique natural full-prefill questions;
448 reported bindings per model include reused observations.

| Fixed model | Correct /384 | Historical A /96 |
| --- | ---: | ---: |
| Native | 328 | 68 |
| GDN all tokens (prior audited transfer) | 370 | 83 |
| GDN question (prior audited transfer) | 366 | 78 |
| Shared | 366 | 78 |
| Independent | 368 | 80 |
| Local all tokens | 348 | 60 |
| Local question | 366 | 78 |
| Attention only | 356 | 68 |

Every new panel checkpoint gets protected B 64/64, current A 160/160 and
current C 64/64; all remaining errors concern historical A. Shared control
does not improve over independent control in this panel. A post-hoc
descriptive check also finds shared and local-question generated outputs
identical on all 384 questions. This is output agreement on these inputs,
not evidence of equal representations or mechanisms. Local-all transfer
loses historical-A accuracy despite its stronger short-task fit; this is not
a general architecture ranking.

Audit: `write-locality-control-panel-20261114.json`. Complete metrics,
deduplicated role counts, all 3584 eight-model reported bindings and the
visually checked figure are exported as
`write-locality-control-panel-comparison-20261114.{json,csv,png,pdf}`, with
analysis snapshots and hash provenance. These limited development results
do not justify adding a shared semantic interface or scaling the model.

The next native diagnostic is registered, before data implementation or
inference, in `configs/semantic_history_no_recent_repeat_plan.json`.
The earlier A0 no-op control repeats the initial answer in its final A
record, so historical-A regressions can mix loss of that recent shortcut
with historical retrieval difficulty. The new common prefix first writes
initial A and then a distinct intermediate A. The final record either
repeats that intermediate value or writes a third value. Both worlds now
require the identical distant initial-A evidence. New prescribed development
and replica seeds, 32/128-account loads and current A/C commit guards yield
384 native questions. Its new provider, runner, commit-aware metrics and
independent auditor are implemented; the complete suite passes 87 tests.
The prior panel audit was reexecuted before data registration. All actual
prompts are 2865 tokens. The 24-query smoke and complete 384-query native
evaluation are independently audited, including eight/sixteen actual
history-access checks and exact native replays. Complete loaded-state hashes
are unchanged. This focused check does not invalidate the earlier B symptoms
or replace the independent-order/COPY controls.

The no-repeat generator places the distinct intermediate A at common record3.
Only the final A value at record160 changes between worlds; record161 updates
C identically. The initial numeric answer occurs in exactly one SET value
field, at record0. Both final A values differ from that answer. Full paired
truths and evidence positions are reconstructed from executable events and
the actual local tokenizer. The prescribed operational sample has 24 queries
and eight history-access/replay checks; the complete run has 384 queries and
sixteen such checks. No model adaptation is involved.

```bash
python scripts/prepare_semantic_history_no_repeat.py
python scripts/run_semantic_history_no_repeat.py --smoke --output /path/to/new-history-smoke
python scripts/analyze_semantic_history_no_repeat.py /path/to/new-history-smoke --allow-smoke \
  --output /path/to/new-history-smoke.json
python scripts/run_semantic_history_no_repeat.py --smoke-audit /path/to/new-history-smoke.json \
  --output /path/to/new-history-run
python scripts/analyze_semantic_history_no_repeat.py /path/to/new-history-run \
  --output /path/to/new-history-audit.json
python scripts/export_semantic_history_no_repeat.py /path/to/new-history-audit.json \
  --output /path/to/new-history-diagnostics.json --plot /path/to/new-history-diagnostics.png
```

The complete native screen gets 365/384 correct: historical A 127/128,
current A 113/128 and current C 125/128. Each table entry below is hold-middle
to update-new, with sixteen related value cases per side and four base streams
per cohort.

| Cohort | Accounts | Historical A | Current A | Current C | Joint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Discovery | 32 | 16→16 | 13→15 | 16→16 | 13→15 |
| Discovery | 128 | 16→15 | 14→16 | 15→15 | 13→14 |
| Replica | 32 | 16→16 | 11→16 | 16→16 | 11→16 |
| Replica | 128 | 16→16 | 12→16 | 15→16 | 12→16 |

The registered history-preservation primary **fails**: only one discovery
stream and zero replica streams have a commit-qualified historical regression
at 128 accounts. Both hold-world history baselines are 16/16, so this is not
the registered low-baseline floor. Neither the secondary load nor NLL-only
changes replace the failed primary. Adding the common intermediate assignment
also changes the event history; this is not a one-factor ablation proving that
the earlier failures came solely from the recent-answer shortcut.

A separate exploratory observation is current A: repeated intermediate values
get 50/64 versus 63/64 for novel final values. Ten of the fourteen repeat-world
errors numerically match initial A. These are output matches, not evidence
that a specific memory state has been damaged. The repeated and novel answer
codes were not counterbalanced in this screen.

Audit: `history-no-repeat-20261114.json`. All 384 raw outcomes, 64 paired cases,
base-stream/commit results and the visually checked figure are exported as
`history-no-repeat-diagnostics-20261114.{json,csv,png,pdf}` and `-pairs.csv`,
with source snapshots and hash provenance. Post-hoc error matches are labeled
separately and do not alter the original primary decision.

The new hypothesis is registered before data implementation/inference in
`configs/semantic_overwrite_history_plan.json`: a 128-account, four-cell
early-value/final-value factorial on entirely new 97600/97700-series streams.
Comparisons YX→XX and XY→YY have identical final answers and complete
executable initial/current states, differing only in the earlier A value at
record3. This tests sensitivity to obsolete overwrite history while
counterbalancing which number is repeated. The planned 384 native queries
retain initial-A/current-C guards, signed losses and rescues, both cohorts and
all value siblings. A positive result must satisfy the new preregistered
baseline and per-stream direction checks; it cannot turn the failed history
primary into a success or automatically justify architecture training. The provider, native runner, signed paired metrics and independent auditor
are implemented; 97 tests pass. The complete earlier 384-query audit was
reexecuted before freezing the new data. The compact dataset stores 32
case templates and 384 query selectors; every actual prompt is 2865 tokens.
The 24-query operational smoke is independently audited, including eight
actual history-access checks and exact native replays. The complete 384-query
native evaluation is independently audited, including sixteen actual
history-access checks and exact native replays. Loaded weights are unchanged.
No model adaptation or new environment is involved.

```bash
python scripts/prepare_write_locality_control_panel.py
python scripts/run_write_locality_control_panel.py --smoke --output /path/to/new-panel-smoke
python scripts/analyze_write_locality_control_panel.py /path/to/new-panel-smoke --allow-smoke \
  --output /path/to/new-panel-smoke.json
python scripts/run_write_locality_control_panel.py --smoke-audit /path/to/new-panel-smoke.json \
  --output /path/to/new-panel-run
python scripts/analyze_write_locality_control_panel.py /path/to/new-panel-run \
  --output /path/to/new-panel-audit.json
python scripts/export_write_locality_control_panel.py /path/to/new-panel-audit.json \
  --output /path/to/new-panel-comparison.json --plot /path/to/new-panel-comparison.png
```


```bash
python scripts/prepare_semantic_overwrite_history.py
python scripts/run_semantic_overwrite_history.py --smoke --output /path/to/new-overwrite-smoke
python scripts/analyze_semantic_overwrite_history.py /path/to/new-overwrite-smoke --allow-smoke \
  --output /path/to/new-overwrite-smoke.json
python scripts/run_semantic_overwrite_history.py --smoke-audit /path/to/new-overwrite-smoke.json \
  --output /path/to/new-overwrite-run
python scripts/analyze_semantic_overwrite_history.py /path/to/new-overwrite-run \
  --output /path/to/new-overwrite-audit.json
python scripts/export_semantic_overwrite_history.py /path/to/new-overwrite-audit.json \
  --output /path/to/new-overwrite-diagnostics.json --plot /path/to/new-overwrite-diagnostics.png
```


The complete counterbalanced screen gets 368/384 correct: historical A
128/128, current A 124/128 and current C 116/128. Below, novel versus repeated
final writes have identical final executable state maps and own-world answers.
Each cohort contains four base streams; the four value siblings and two final
identities within a stream are related observations.

| Cohort | Novel current A | Repeated current A | Raw losses / rescues | Guarded losses / rescues | Positive guarded streams |
| --- | ---: | ---: | ---: | ---: | ---: |
| Discovery | 32/32 | 29/32 | 3/0 | 2/0 | 1/4 |
| Replica | 32/32 | 31/32 | 1/0 | 1/0 | 1/4 |

The new registered primary **fails in both cohorts**: each needs positive
guard-qualified repeat loss in at least two distinct base streams, but each
has only one. Both novel baselines meet the specified floor; raw and guarded
net losses are positive, so the failure is the preregistered stream-coverage
criterion. The qualified comparison includes 27 discovery and 30 replica
pairs; all other pairs, including guard failures, remain in the raw export.
Do not promote the subgroup or adjust the threshold. The earlier failed
history-preservation primary remains failed.

These data retain a small functional dependence on obsolete overwrite
history, without establishing a robust general repeated-write deficit,
recurrent-state damage, or the necessity of a shared GDN/Attention interface.
This branch does not trigger architecture training, another load search, or
scaling. The main semantic-update question remains open; any later proposal
must still establish a reproducible failure and beat ordinary adaptation,
question-stage adaptation and independent-controller controls.

Audit: `overwrite-history-20261114.json`. All 384 raw outcomes, 64 same-final
pairs, all four cells, final identities, preservation guards and base-stream
results are exported as `overwrite-history-diagnostics-20261114.{json,csv,png,pdf}`
and `-pairs.csv`, with an analysis source snapshot. Artifacts are under
`/media/990Pro/Code/SAS/artifacts/memory_update/`.


The fixed COPY dependency transfer is registered before its implementation or
new inference in `configs/copy_dependency_transfer_plan.json`. It fills a
coverage gap for the seven final order-augmented adapters, rather than adding
another COPY cohort or training a new interface. The original 48-case
`copy-generalization-v1` dataset is unchanged: eight existing base streams,
COPY depths 1/2/3, original/entity-first questions, before/COPY-then-SET/
SET-then-COPY variants, and current/initial/unrelated readouts. All executable
truths and the actual existing case-file hash are verified. Existing native
and older-adapter results have been observed, so this is a reused development
transfer control, not fresh confirmation.

All seven final432-step adapters are fixed before evaluation: shared,
independent, local-all, local-question, Attention-only, GDN-all and
GDN-question. Their original training identities and budgets match; parameter
counts are near-matched and Attention-only has weaker teacher fit. Native is
rerun as the eighth arm. All432related bindings/model are retained; merging
only identical before-context prompts leaves336physical queries/model,
2688physical readouts and3456bindings overall. Actual prompts are383–1416tokens.
No oracle, router, new adapter, optimizer, environment or model copy is added.

The complete suite passes105tests, with a separate positive/negative gate
check. The previous full384-query negative native audit was reexecuted and all
seven actual checkpoint files/training references verified before the token
registry was frozen. The144-query operational smoke is independently audited:
48actual historical-access checks,48exact native replays and12complete native
prefix identity checks passed. The complete2688physical-query/3456-binding
evaluation is also independently audited, with48actual historical-access
checks,48exact native replays and12complete native-prefix checks. All native
and adapter weights remain frozen and unchanged; all hooks are restored.

The descriptive shared-candidate rule requires positive paired six-answer
joint gains over EACH of independent, local-question and GDN-question,
in at least three distinct streams and at every depth. Every role, baseline,
order, depth and wording remains visible. This development screen cannot
establish architecture success or recurrent-memory damage. Its failure does
not trigger threshold/rank/checkpoint tuning, favorable-subset searches or
training; its success would require a separate fresh-cohort mechanism test.

```bash
python scripts/prepare_copy_dependency_transfer.py
python scripts/run_copy_dependency_transfer.py --smoke --output /path/to/new-copy-smoke
python scripts/analyze_copy_dependency_transfer.py /path/to/new-copy-smoke --allow-smoke \
  --output /path/to/new-copy-smoke.json
python scripts/run_copy_dependency_transfer.py --smoke-audit /path/to/new-copy-smoke.json \
  --output /path/to/new-copy-run
python scripts/analyze_copy_dependency_transfer.py /path/to/new-copy-run \
  --output /path/to/new-copy-audit.json
python scripts/export_copy_dependency_transfer.py /path/to/new-copy-audit.json \
  --output /path/to/new-copy-transfer.json --plot /path/to/new-copy-transfer.png
```


The full eight-arm transfer does not pass the registered shared-candidate
screen. Counts below are related depth/wording cases out of48, built from only
eight base streams. Six-answer joint success requires current target,
unrelated current and target initial answers correct in BOTH COPY/SET orders.

| Fixed checkpoint | Both current-target orders /48 | All six after-update answers /48 | Historical target after /96 | Unrelated current after /96 |
| --- | ---: | ---: | ---: | ---: |
| Native | 1 | 1 | 96 | 81 |
| Shared | 17 | 14 | 91 | 96 |
| Independent | 14 | 13 | 95 | 96 |
| Local all tokens | 17 | 17 | 96 | 96 |
| Local question only | 15 | 15 | 96 | 96 |
| Attention only | 4 | 4 | 96 | 96 |
| GDN all tokens | 12 | 12 | 96 | 96 |
| GDN question only | 13 | 13 | 96 | 96 |

Shared joint wins/losses are3/2 versus independent (net+1),3/4 versus local
question (net−1), and5/4 versus GDN question (net+1). Each required comparison
fails the registered stream/depth coverage checks; no comparator passes the
whole descriptive rule. At depth1, shared joint success is11/16 versus13/16
independent and14/16 local-question. At depth2, shared is3/16 and local-all
4/16. At depth3, all arms are0/16 except GDN-question1/16. These prespecified
strata remain descriptive; a favorable depth does not replace the failed gate.

Every adapted model gets the before readouts correct for all48related cases.
All adapted unrelated-current answers remain96/96 after updates; historical readout
errors occur in shared(5) and independent(1), with all other adapted arms
96/96. Most remaining errors concern current-target answers in the dependency task.
The queried unrelated/history facts are largely preserved. Poor multi-hop
transfer is not itself evidence of recurrent-memory destruction: the fixed
training distribution uses one-hop COPY/original wording, whereas this reused
development cohort also changes depth, wording and context length. The tiny
paired total differences do not establish superiority or equivalence, and
Attention-only's weaker original teacher fit remains a limitation.

Current evidence does not justify a new shared-interface training run or
larger-model scaling. The tested shared adapters are generic controls without
explicit semantic-dependency supervision. Keep the original coordination
question open, preserve all previous negative decisions, and distinguish
remaining evidence-addressing failures from dependency-execution failures
before proposing a write/update mechanism. Ordinary data-augmentation and
question-stage controls remain necessary comparisons.

Audit: `copy-dependency-transfer-20261114.json`. All2688unique physical results
and3456related bindings, all per-role/order/depth/wording counts, case outcomes
and paired stream results are exported as
`copy-dependency-transfer-comparison-20261114.{json,csv,png,pdf}` and
`-bindings.csv`, under `/media/990Pro/Code/SAS/artifacts/memory_update/`.


The frozen-prefix readout follow-up is registered before its implementation
and inference in `configs/copy_readout_sufficiency_plan.json`. It reuses the
same eight fixed models and all336physical prompts/model, including all roles,
orders, depths, wordings and before baselines. No new training is performed.
All2,688full-history reference readouts remain sealed. Each query gets two
new readouts: an executable oracle event-slice mask and a deterministic nearby
non-slice mask with exactly the same retained-record token budget. Original
prefix tokens and positions are retained; masks apply only to question/answer
historical columns in all six softmax layers. The non-slice KV may contain
contextualized answer information and is not a no-information control.

The114-test suite passes; nine targeted tests also pass after diagnostic-only
exception-message clarification. The first288-readout operational run stopped
at an unmasked exact-reference replay/observer check in Attention-only. Its
failed run and source snapshots remain intact. A dedicated diagnostic replayed
six existing Attention-only queries three times each with and without the
observer: all36replays and hook restorations were exact. The same288-readout
smoke was repeated without changing masks, cases, models or tolerances, and
passed independent complete audit. The original intermittent failure has no
established cause; it is not a scientific result. No score-based smoke filter
is used.

The successful smoke verifies160actual six-layer masked-probability checks,
160complete ledger-memory/historical-KV preservation checks,80exact all-history
controls,80detached-native replays and12question-only native-prefix checks.
The complete5,376new-readout evaluation passed independent audit, including
reexecution of the successful operational smoke audit, with the same full
preservation/access/replay counts. Every native and adapter tensor remained
frozen and unchanged; all hooks were restored.

Two descriptive decisions are fixed separately. For BOTH local-question and
GDN-question, frozen-prefix sufficiency requires at least90/96oracle current
answers, no historical/unrelated loss in aggregate, and positive repair in at
least four of eight base streams. Address-specific gain separately requires
oracle superiority over equal-budget non-slice KV in at least four streams.
Success of one decision does not imply the other. Oracle repair with preserved
prefix memories can compensate for compressed-memory damage using KV; it
neither proves every recurrent fact intact nor establishes a feasible router.
Every earlier shared/history/repeat primary remains failed. This is a reused
development-cohort dense-eager diagnostic, not a new architecture result.

Both registered readout decisions pass. Current-target answers and joint
six-answer outcomes are below; every after-update history and unrelated answer
is retained in the separate complete exports. Only eight base streams underlie
these related depth, wording and order observations.

| Fixed checkpoint | Full current /96 | Oracle current /96 | Non-slice current /96 | Full joint /48 | Oracle joint /48 | Non-slice joint /48 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 10 | 44 | 0 | 1 | 17 | 0 |
| Shared | 43 | 96 | 0 | 14 | 48 | 0 |
| Independent | 35 | 95 | 0 | 13 | 47 | 0 |
| Local all | 43 | 82 | 0 | 17 | 37 | 0 |
| Local question | 37 | 96 | 0 | 15 | 48 | 0 |
| Attention only | 35 | 96 | 0 | 4 | 48 | 0 |
| GDN all | 35 | 94 | 0 | 12 | 46 | 0 |
| GDN question | 37 | 96 | 0 | 13 | 48 | 0 |

For both question-only models, oracle-versus-full current-target gains/losses
are59/0, with positive signed gains in all eight base streams. Oracle accuracy
is32/32 at each dependency depth for both models. Their historical and
unrelated after-update answers are96/96 under both full and oracle access.
Oracle-versus-non-slice gains/losses are96/0, also positive in all eight streams.
The non-slice control collapses on every after-update role for every model;
most sampled outputs are still numeric, so this is not just answer-format
failure. Nevertheless, its severe degradation and possibly partial-record
masks limit the interpretation: passing that comparison does not isolate a
unique causal mechanism, establish absent information in contextualized KV,
or show that a practical learned selector can reproduce the oracle.

These data demonstrate complete functional repair of the two existing
question-only adapters on this reused COPY cohort while their prefix memories
remain unchanged. Prefix write modification is therefore not required for
this particular oracle repair. Recovery could use precise KV to compensate
for compressed-memory damage; it does not prove every GDN memory intact.
Nor does it locate the original failure uniquely in Attention-Q or address
formation: masking also changes normalization and downstream question-stage
computation. Native oracle current accuracy is only44/96, so oracle addresses
alone do not solve the native model's computation/answering limitations.

The earlier generic shared-interface screen remains failed. This follow-up
does not justify another shared-adapter training run, scale-up, a router
project, or the proposed cross-layer feedback mechanism. Keep explicit
semantic-dependency coordination as a hypothesis. Before changing prefix
writes, establish functional collateral damage that survives strong readout
controls, and then test whether shared constraints beat same-data,
same-budget independent and question-stage alternatives. A generic shared
latent without explicit dependency supervision is not that full hypothesis.

Audit: `copy-readout-sufficiency-20261114.json`. The complete export is
`copy-readout-sufficiency-comparison-20261114.{json,csv,png,pdf}` plus
`-bindings.csv`, `-pairs.csv`, analysis source, layout checks and artifact/source
provenance, under `/media/990Pro/Code/SAS/artifacts/memory_update/`. It contains
8,064physical rows (5,376new and2,688explicitly reused full references),
10,368related bindings and1,536paired current-target comparisons. The
analysis-only figure legend was moved below its axis label after visual
inspection; inference code, masks, checkpoints and all numerical decisions
were unchanged.


The next bounded diagnostic tests whole-record evidence necessity, registered
before new implementation/inference in
`configs/copy_evidence_necessity_plan.json`. It does not add a long-context
cohort or search for write damage after the failed history/repeat screens.
It addresses the previous control's severe degradation and partial-record
selection before making a stronger memory-update claim.

For each unchanged physical query, remove question/answer KV access to exactly
one complete original SET event: the unique earliest SET leaf in its executable
evidence slice. A matched intervention instead removes a complete SET for an
entity outside all read/write entities of that slice, with exactly equal token
length and nearest original token-center position (event-index tie break).
All other history remains accessible; every original prefix computation,
position, recurrent/convolution state and historical KV is retained. The
symbolic slice without its leaf must lose the queried truth, but masking direct
access does not remove that fact from GDN or other contextualized KV.

All336queries have complete17-token event removals; the matched center is at
most34tokens away. The121-test suite passes, including boundary-crossing
rejection, deterministic scope/budget matching, positive/negative necessity
gates and complete memory/observation proofs for both new regimes. The complete
previous5,376-query audit was reexecuted before all masks were frozen under
`copy-evidence-necessity-v1`; the mask registry hash is
`e48ba67ed22f47c506d721f178b3174f27965d74f87c577b6c5ea7790720e497`.

The same eight fixed checkpoints,336physical questions and432related
bindings/model are used. The operational288-readout smoke must pass independent
audit before5,376newreadouts. Both question-only models must show positive
matched-other-versus-support-removal current loss overall and in at least four
base streams; matched-other removal must also keep current/history/unrelated
after counts and before joint success no lower than the full reference. This
strict descriptive guard prevents a broadly degraded control from qualifying.
All models, roles, wordings, depths, variants, baselines, gains and losses remain
visible. The earlier sufficiency result and failed architecture/history/repeat
decisions remain separate and unchanged. No training, new model or practical
sparse kernel is introduced.


The first whole-record operational smoke stopped before full inference at an
exact unmasked-reference check. For the same Attention-only query, the sealed
FP32-adapter NLL is0.4226212501525879, but a multi-arm run produces
0.44792552789052326 with the same free generated answer. Native and adapter
weight fingerprints are unchanged. A diagnostic arm-sequence replay reproduced
the discrepancy; synchronization, allocator cleanup, deterministic flags, and
rebuilding the controller or model inside that process did not remove it.
An independent process matches the sealed reference.

Activation traces match through the preceding decoder layers and adapter
encoder. The first differing tensors are the FP32 adapter output projections.
A small fixed-input matrix replay reproduces the complete failed Q/K output
hashes; maximum differences8.344650268554688e-7 and3.5762786865234375e-7
change23and4elements after BF16 rounding. The numerical path can therefore
explain downstream drift without changed parameter bytes. This does not
establish a vendor-library defect or fully identify its kernel-selection cause.
A second instrumented288-readout diagnostic run completed without the failure;
its wrapper/source set makes it ineligible as a scientific smoke gate. Failed
and diagnostic runs are retained; no tolerance or sample was relaxed.

The explicit numerical follow-up is registered before canonical model
inference in `configs/copy_evidence_necessity_precision_addendum.json`.
`scripts/stable_adapter_linears.py` recomputes only frozen controller Linear
affines in FP64 and rounds their output to FP32 before the existing nonlinear
and BF16 delta path. Parameters keep the same FP32 storage and bytes; native
model modules are not wrapped. The20known-projection GPU layout/offset products
agree exactly with CPUFP64 products rounded toFP32. The complete123-test suite
passes, including cancellation, cleanup and frozen-inference checks.

An actual-model arithmetic diagnostic is independently audited: three existing
controllers, six prescribed queries each, both model orders, three repeats,
108exact readouts;24complete question-only native-prefix checks and6exact
detached-native replays. Every parameter remains byte-identical. All18unique
model/query readouts differ from their legacy FP32 record in at least one
field, which reinforces the need for fresh references. This is a bounded
arithmetic check, not the required scientific smoke or proof of universal
determinism. Audit: `canonical-adapter-readout-check-audit-20261114-01.json`.

At the end of the arithmetic-check phase, neither the original5,376masked-only
full experiment nor the canonical scientific evaluation had started. The canonical plan requires two independent
432-readout operational smoke audits, with forward/reverse model order and
exact cross-order readout/prefix equivalence. Only then run8,064fresh primary
readouts:2,688new full-history references plus5,376whole-record interventions.
Do not mix the numerical profiles or reuse old FP32 NLLs as canonical baselines.
The original tasks, masks, models and necessity decision thresholds are
unchanged. All old scientific decisions remain recorded under their original
profile; no architecture training or load/cohort search follows from the
numerical repair.

## Fresh-reference canonical whole-record necessity follow-up

The separate `scripts/run_canonical_copy_necessity.py` implements the registered
precision addendum without editing any sealed FP32 runner, checkpoint, dataset
or analysis source. It computes fresh full-history references under the same
arithmetic as both whole-record masks. Its three regimes are `full`,
`remove_support` and `remove_matched_other`: 8,064 new physical observations,
including 2,688 new full references, with 10,368 related bindings. No old FP32
observation enters the primary contrasts. All 2,688 old-versus-canonical full
pairs, including unchanged pairs, are reserved for a separate descriptive CSV.

The numerical profile is limited to frozen controller Linear affines: FP64
accumulation followed by FP32 rounding, before the existing activation/scaling
and BF16 residual path. Adapter storage and native model arithmetic are
unchanged. This is diagnostic execution, not a training implementation or a
speed claim. The original numerical failure and all historical decisions remain
preserved. A passing arithmetic check does not establish universal determinism.

Before primary inference, both operational runs completed and were independently
audited: `canonical-copy-necessity-smoke-forward-20261114-01` and
`canonical-copy-necessity-smoke-reverse-20261114-01`. Across opposite model orders,
all 432 corresponding readouts matched exactly, including gold-answer NLL and
unrestricted generated text; all 80 sampled complete-prefix hashes, 160 masked
prefix-hash pairs, 12 question-only prefix checks and 10 pre-controller native
anchors also matched. Neither smoke used an accuracy gate. Each independently
verified all-history controls, actual six-layer masked probabilities, recurrent
and convolution caches, 12 historical K/V slices per sampled mask, native
weight hashes, frozen adapter tensors, hook removal and detached native replays.
The complete unit suite passed 129 tests. The equivalence report is
`/media/990Pro/Code/SAS/artifacts/memory_update/canonical-copy-necessity-smoke-equivalence-20261114-01.json`.

The primary scientific gate remains the original one, now relative to fresh
same-profile full references. For **both** `local_question` and
`gdn_question_only`, matched-other removal must preserve current, historical and
unrelated after-update counts plus before-update joint success. Evidence-leaf
removal must cause positive signed current-answer harm relative to matched-other
removal overall and in at least four of the eight reused base streams. No
favorable query, layer or checkpoint filtering is allowed.

These masks restrict direct historical K/V access during the question and its
continuation. They do not remove records from prefix computation or erase them
from recurrent states, and their information can also survive in contextualized
K/V at other positions. A failed necessity screen cannot demonstrate loss of
GDN information; a passed screen cannot establish a shared-controller advantage.
Live execution state remains in `configs/experiment_progress.json`.

The full canonical run completed and independently audited all **8,064 new
physical readouts /10,368 related bindings**. The registered whole-record
necessity screen **passed for both prescribed question-only models**. The audit
also verified 160 actual six-layer masked-attention checks, 160 complete-prefix
and historical-K/V proofs, 80 all-history controls, 80 detached-native exact
replays, 12 question-only native-prefix checks and unchanged native/adapter
weights. The native loaded-state hash remains
`2fcdfb2c15878ad12794dcff913b5d43fe0787850a4b4d86185087bc9dcd6158`.
Audit: `/media/990Pro/Code/SAS/artifacts/memory_update/canonical-copy-necessity-audit-20261114-01.json`.

| Fixed model | Fresh full current /96 | Remove evidence current /96 | Remove other current /96 | Six-answer joint: full /evidence /other, each /48 |
| --- | ---: | ---: | ---: | --- |
| Native | 10 | 0 | 11 | 1 /0 /1 |
| Shared | 43 | 0 | 44 | 14 /0 /18 |
| Independent | 37 | 0 | 37 | 14 /0 /15 |
| Local all | 43 | 0 | 44 | 17 /0 /18 |
| Local question | 37 | 0 | 38 | 15 /0 /15 |
| Attention only | 35 | 0 | 35 | 4 /0 /5 |
| GDN all | 35 | 0 | 37 | 12 /0 /12 |
| GDN question | 36 | 0 | 37 | 13 /0 /13 |

For Local question and GDN question, matched-other removal preserves historical
and unrelated after-update answers at96/96 and before-update joint success at
48/48, all no lower than their fresh full references. Relative to that control,
evidence removal causes38losses/0rescues and37losses/0rescues, respectively,
with positive net harm in **all8reused base streams**. The respective harms at
depths1/2/3 are30/7/1 and28/4/5 out of32related current-answer bindings perdepth;
the low deep-task baseline limits sensitivity there. The preservation statement
is specific to these registered primary models: native unrelated-after answers
fall81→79 under matched-other removal. Shared historical answers improve91→96
and independent95→96 under the same control. All eight models and all outcomes
are retained rather than filtering to the primary successes.

All eight evidence-removal arms have0/336correct physical answers across their
own role-dependent masks. A separately labelled **posthoc descriptive** output
analysis finds324/336native,336shared,336independent,336local,330local-question,
326attention-only,331GDN-all and332GDN-question outputs are still integers.
Thus the zero-correct result is not mainly a formatting failure. These counts
were not a new gate and do not establish absence of information. The report
and archived source are `canonical-copy-necessity-output-format-20261114-01.json`
and its `.source.py` under the artifact directory.

The supported statement is narrow: **under these fixed readouts, access to the
query-specific complete SET leaf matters even while the original prefix
memories and all other history remain available**. This separates the finding
from the earlier partial-record non-slice control. It does not show GDN has
forgotten the value, identify a damaged recurrent layer, prove irreversibility,
or establish that sharing semantic control beats independent adapters. The
mask acts at all six softmax layers, not just the adapted group. These are
48related cases from8reused streams, not48independent replications or a general
long-context benchmark.

The full old-versus-canonical descriptive comparison includes all2,688pairs:
2,270differ in at least one readout field,3change greedy correctness and15change
generated text; all336native pairs are exact. This reinforces the need for
fresh references. It does not reopen the old failed shared/history/repeat
primaries. The earlier oracle-sufficiency result remains specific to its FP32
execution: do not combine it with this new-profile necessity screen as if both
were a single numerical-profile experiment. The next bounded scientific step
should reproduce oracle readout sufficiency under the canonical profile,
using these fresh full references and the already fixed primary models, before
claiming a joint sufficiency/necessity result or training a new architecture.

All observations and comparisons are exported as
`canonical-copy-necessity-comparison-20261114-01.{json,csv,png,pdf}` plus
`-bindings.csv` (10,368rows), `-pairs.csv` (2,304paired current contrasts),
`-legacy-full.csv` (2,688descriptive pairs), `.source.py`, `.layout.json` and
`.provenance.json`. The2720×1920figure was visually inspected; all visible text
fits and the legend does not overlap its axis label. All artifact hashes and
95live/archived runtime sources were verified. No additional training, new
model weights, downloads, environment or favorable cohort search was used.

## Canonical oracle-sufficiency replication

`configs/canonical_oracle_sufficiency_replication_plan.json` was registered
before any new-profile oracle inference, SHA256
`2444974c7a9e55e50cb60e132abd12b971c9c1e4903c5e827558bf6c3fd92c66`.
It uses exactly the already prescribed Local question and GDN question models,
all336physical queries/432related bindings permodel, and the existing
complete-event oracle masks. The previous partial-record nonslice condition is
not executed. There is no new training, checkpoint, task case or model.

A separate `scripts/run_canonical_oracle_replication.py` recomputes every full
reference and every oracle observation:1,344new readouts,672perregime. Every
new full observation must exactly match all five readout fields of the
completed canonical necessity phase. Any drift stops interpretation. Both
primary contrasts use this phase own fresh full rows, with no FP32-profile
reference mixing. The source closure includes101files and leaves all sealed
previous runtimes unchanged.

Both independent operational smokes completed and audited72readouts each.
Opposite model orders matched all72readouts,20sampled complete prefixes,
12question-only nativeprefix checks and10pre-controller nativeanchors exactly.
Their source and data scope are identical and neither smoke uses an accuracy
gate. The134-test suite also checks that the two-model statistics exactly
reproduce the sealed full-reference denominators and aggregate counts; it does
not manufacture six additional arms to satisfy the old eight-model summary.

The primary sufficiency gate reuses the existing five checks unchanged:
for each of both models, oracle current-after correctness>=90/96, historical
and unrelated after-update counts no lower than fresh full, positive paired
current-answer net repair overall and in at least four of the eight streams.
This is a same-corpus, same-checkpoint numerical-profile replication, not
independent-data replication or architecture success. The prior failed shared,
history and repeat primaries remain failed. Live primary execution state is in
`configs/experiment_progress.json`.

The canonical oracle replication completed and independently audited all
**1,344new physical readouts /1,728related bindings**. All672new full-history
observations exactly replay the completed canonical necessity phase, including
NLL. The unchanged five-check sufficiency screen **passed for both models**.
Oracle answers are336/336correct physical queries permodel; all six after-update
answers are correct in48/48related cases. Native and adapter weights remain
unchanged;20all-history controls,20actual masked-probability checks,20complete
prefix/historical-KV proofs,20detached-native replays and12native question-prefix
checks passed. The loaded-state hash remains
`2fcdfb2c15878ad12794dcff913b5d43fe0787850a4b4d86185087bc9dcd6158`.

| Fixed model | Fresh full current /96 | Oracle current /96 | Whole evidence removal current /96, prior canonical | Whole other removal current /96, prior canonical |
| --- | ---: | ---: | ---: | ---: |
| Local question | 37 | 96 | 0 | 38 |
| GDN question | 36 | 96 | 0 | 37 |

Oracle repairs59current answers with0regressions for Local question and60with
0regressions for GDN question. Both have positive repair in all8reused streams,
current correctness32/32at each of depths1/2/3, historical and unrelated answers
96/96, and before-update joint success48/48. Current full counts bydepth are
30/6/1 and28/4/4, respectively. The same-profile necessity effects remain the
recorded38losses/0rescues and37losses/0rescues against matched-other removal;
there is no cross-profile combination or new re-selection of models.

This supports a functional statement: on these fixed tasks and readouts,
important errors can be repaired by restricting question-time evidence access
while preserving the original prefix computation and memories; access to the
relevant complete SET also matters. **These errors cannot by themselves prove
that original memory writes destroyed information.** This does not show that
GDN alone retains every fact, locate the fault uniquely in Attention Q, or
validate shared block control. Oracle addresses come from executable program
truth and can simplify evidence selection and subsequent reasoning. A usable
learned mechanism and an architecture advantage are still unproven. The48cases
are related within8reused streams; this numerical-profile replication is not
new-data confirmation. All earlier shared/history/repeat failures remain.

Audit:
`/media/990Pro/Code/SAS/artifacts/memory_update/canonical-oracle-replication-audit-20261114-01.json`.
The comparison prefix in the same directory is
`canonical-oracle-replication-comparison-20261114-01`: primary CSV1,344rows,
primary bindings1,728, combined CSV2,688rows and bindings3,456, new current-answer
pairs192and combined pairs768. The combined export explicitly identifies
1,344reused canonical necessity mask observations; the new full/oracle rows are
all fresh. JSON, PNG, PDF, source archive, layout and artifact/source provenance
are included. The2400×1760figure was visually inspected; all text fits and the
figure legend is disjoint from all plot bounds. All11artifact hashes and
101live/archived runtime sources were checked. The full suite passed134tests.

The next stage is registered before fresh data generation in
`configs/fresh_copy_memory_confirmation_plan.json`, SHA256
`e310e4690db41d0c673c93f0785301bd8503c33cfd91c176416010ad672c54da`.
Seeds61000..61007 are fixed now, with the same alternating8/64load and complete
depth/wording grid. Eligibility checks must inspect actual previous dataset
inputs; no seed, case, full prompt or full prefix reuse is allowed, and there
is no resampling after scores. The new cohort retains the same templates and
sampling distribution, so it is not out-of-domain validation.

All eight fixed models retain natural full-history evaluation; the two
preselected question-only models additionally receive oracle, whole-evidence
removal and matched-other removal. The stage plans4,704new physical readouts
and6,048related bindings, preceded by independent forward/reverse252-readout
operational smokes. The existing sufficiency and necessity gates remain
separate from the original generic shared-controller versus independent,
Local-question and GDN-question joint-case gate. This tests whether the
functional access pattern replicates without dropping the strong architecture
controls. A successful access intervention still does not demonstrate an
architecture advantage, and a new-cohort generic shared pass cannot erase
previous failed comparisons. At registration, no new case generation or inference
had started. No architecture training is triggered automatically.

The fresh cohort has now been generated once, frozen and checked against actual
prior inputs:17registered dataset directories,69data files,6,384complete prompts
and1,912complete prefixes. Seeds, case identifiers, complete prompts and complete
prefixes have no overlap. The48related cases provide336physical questions and
432bindings; the dependency depths, both wordings and both update orders are all
retained. This is independent-history confirmation under the same generator,
not new linguistic templates or independent training initializations. Whole
records are retained or removed; no partial-token record mask is used.

The140-test suite passed. Both252-observation operational smokes completed and
passed independent audit. Opposite model orders reproduce all252readouts,
60masked complete-prefix hash pairs,12question-only native-prefix controls and
10pre-controller native anchors exactly. The4704-readout execution completed
and independently audited under this gate. Runtime and data closures contain109and104source files,
respectively; all previous sealed sources and scientific outcomes remain
unchanged. All eight natural controls remain in the primary, with three extra
readout conditions only for the two preselected question-only models. Progress
is recorded in `configs/experiment_progress.json`; the standalone exporter
`scripts/export_fresh_copy_confirmation.py` is outside the runtime closure.

The current controller experiments test generic shared versus independent
adaptation, not an implemented semantic-dependency guarantee. A shared latent
vector is not by itself a semantic scope. Question-time oracle access provides
addresses from the executable program and can simplify reasoning; successful
repair with unchanged prefix memories cannot establish that a recurrent state
contains every fact or that Attention-Q is the unique cause. These distinctions
remain necessary when assessing whether a new four-layer block is warranted.

Fresh confirmation completed **4,704new physical readouts /6,048related
bindings** with no reused observations or numerical-profile mixing. All eight
models completed all336natural full-history questions. Current, history and
unrelated counts below each use96after-update bindings; joint success requires
all six after-update answers in a related case. There are48related cases from
8new streams, not48independent streams.

| Fixed model | Current /96 | History /96 | Unrelated /96 | Joint /48 |
| --- | ---: | ---: | ---: | ---: |
| Native | 17 | 96 | 81 | 0 |
| Shared | 53 | 96 | 96 | 22 |
| Independent | 41 | 96 | 96 | 16 |
| Local all tokens | 40 | 96 | 96 | 13 |
| Local question | 38 | 96 | 96 | 12 |
| Attention only | 38 | 94 | 96 | 3 |
| GDN all tokens | 40 | 96 | 96 | 16 |
| GDN question | 45 | 96 | 96 | 17 |

The generic shared candidate screen **did not pass**. Shared versus independent
has6joint-case gains and0losses across3positive streams, but depth1ties; its
depth1/2/3net changes are0/+3/+3. Shared versus Local-question has11gains and
1loss across6positive streams and passes its comparison, with depth nets+4/+3/+3.
Shared versus GDN-question has6gains and1loss across3positive streams, but
depth2ties; its depth nets are+2/0/+3. All three comparisons were required.
These are positive descriptive signals, not a uniformly successful screen,
absence of benefit, or permission to erase prior failed cohorts. The shared
code still has no explicit dependency guarantee, and all training uses the
fixed original initialization and checkpoints.

| Preselected model | Full current /96 | Oracle current /96 | Drop evidence current /96 | Drop other current /96 |
| --- | ---: | ---: | ---: | ---: |
| Local question | 38 | 96 | 0 | 37 |
| GDN question | 45 | 96 | 0 | 45 |

Oracle repairs58and51current answers, respectively, with0current regressions
and positive repair in all8streams. GDN-question passes both its unchanged
sufficiency and necessity checks, with Oracle336/336physical answers and48/48
joint cases. Local-question does not: Oracle gives335/336physical answers and
47/48joint cases, losing one unrelated answer (96to95after-update), while
matched-other removal loses one current answer (38to37). Therefore **both
overall two-model readout screens also did not pass**. No one-example tolerance,
cohort replacement or model exclusion is introduced.

Even the GDN-question matched-other aggregate tie is not pointwise harmless:
one correct current answer becomes wrong and one wrong answer becomes correct.
The three observed control harms and this repair are all retained in a separate
posthoc descriptive report. These are question-time changes with the original
prefix memories unchanged, not evidence of new prefix-write damage. Program
Oracle addresses also reduce evidence-selection and reasoning difficulty.
Neither a successful repair nor a control harm by itself establishes whether
the recurrent state stores every needed fact. The older same-corpus canonical
readout passes remain recorded as passes; this fresh result limits the scope
of their generalization.

Audit: `/media/990Pro/Code/SAS/artifacts/memory_update/fresh-copy-confirmation-audit-20261114-01.json`,
SHA256 `d28f7568ed6d1d6ddf8314e694c12f11b87b3a7544adeec27f82ce706268d2e9`.
The comparison prefix in that directory is
`fresh-copy-confirmation-comparison-20261114-01`: CSV4,704rows, bindings6,048,
readout pairs768, natural joint-case pairs336and complete cells756, plus JSON,
PNG, PDF, source archive, layout and provenance. All10primary artifact hashes
and109live/archived runtime sources were checked. The2720x2000figure was
visually inspected; all visible text fits, the figure legend is disjoint from
plot content, and the image has803unique colors. The140-test suite passed.
All native and adapter weights remain frozen and unchanged;80all-history
controls,60actual masked-attention checks,60complete-prefix/historical-KV
proofs,80detached-native replays and12question-only native-prefix checks passed.

One additional direct prior is
[How Linear Attention Remembers](https://arxiv.org/html/2609.33093v1),
especially sections3.3and4. It already reports cross-fact causal coupling and
write-overlap interference in pretrained recurrent models, and KV-dominated
recalled-identity effects in the tested hybrids. Those controlled recall tasks
and base checkpoints differ from this2B-IT executable-program experiment.
Cross-fact interference alone is therefore insufficient novelty. A proposed
semantic block must implement and validate dependency-specific coordination,
and exceed the simple controls, rather than only sharing a latent vector.
The current mixed confirmation does not trigger new architecture training,
rank/checkpoint tuning, a larger model or a new environment.

The reproducible posthoc control-change report is
`/media/990Pro/Code/SAS/artifacts/memory_update/fresh-copy-confirmation-control-changes-20261114-02.json`.
It includes all three control harms and the GDN-question matched-other repair,
with original/intervened readouts, bindings and frozen masks. The first version
and its source are retained; only an explanatory wording error was corrected.
This descriptive report does not change the primary scores or screens.

Before any targeted replay, the next operational check is registered in
`configs/fresh_control_change_replay_plan.json`, SHA256
`1a3ba5fa85c2547ab5e915f874fc27d9908ec8b05a844cbe6f4f13259b55b1c1`.
It retains all four correctness flips, including the repair, and prescribes
three exact full/intervened replays pertuple with complete-prefix and actual
masked-attention proofs. This is a posthoc reproducibility check on reused
questions, not new scientific confirmation. At registration its runner and
auditor had not yet been implemented and no replay inference had started.

The four-change operational replay has now completed and independently
audited: **24recorded full/intervened readouts,60measure calls including
controls**. Every one of the three harms and the repair exactly reproduces its
original answer,NLL and token boundary in all3repetitions, including reversed
query order. Twelve unmasked repeats,12all-history controls,12actual six-layer
masked-probability/complete-prefix/historical-KV proofs,4native-prefix checks,
4pre-controller native anchors and8detached-native replays passed. The complete
suite passed145tests. All113live/archived runtime sources and all3audit/source/
CSVartifact hashes were checked; native and adapter weights remain unchanged.

| Frozen readout | Condition | Gold | Full generation | Intervened generation | Three exact repetitions |
| --- | --- | ---: | ---: | ---: | --- |
| Local question | Oracle, unrelated query | 784 | 784 | 1584 | harm |
| Local question | Drop matched other | 375 | 375 | 410 | harm |
| GDN question | Drop matched other | 312 | 703 | 312 | repair |
| GDN question | Drop matched other | 703 | 703 | 343 | harm |

In the last case, mean gold-answer NLL decreases from0.5702381531397501to
0.5652628739674886despite the greedy answer becoming wrong. This retained
example shows why a token-averaged loss alone cannot stand in for the proposed
functional answer-preservation criterion. These observations are posthoc
replays on the same four questions, not new-data confirmation, evidence of
prefix-write destruction or a semantic-block method. All three original
overall scientific screens remain failed.

Audit: `/media/990Pro/Code/SAS/artifacts/memory_update/fresh-control-replay-audit-20261114-01.json`,
SHA256 `6542987b6f9149ec096e8a1c565b631e24027044e1f38688552054e6698c5902`.
The same prefix provides the24-row CSV, auditor source archive and provenance.
All execution sessions for this phase are confirmed terminal.

The next mechanism diagnostic is registered before any layer-localization
inference in `configs/fresh_control_layer_localization_plan.json`, SHA256
`9cdde60299233fe414332f69d71c45fba8fa13a1a70717dc8bbf5f3333dceb1f`.
It retains all4questions and14conditions:full history,all6attention layers
restricted,each of6layers restricted alone,and each of6layers omitted from
the full restriction,for56scored observations. Full/all-layer operator fidelity,
actual mask receipt/probabilities,prefix and historicalKV identity remain
required. The only-layer23condition additionally requires the entire
full-question cache and input to Attention23to match unmasked execution;
first-answer-token logits are compared separately. This can test a direct
final-readout route without assuming that every effect requires intervening
GDN updates. All layer outcomes must be reported, with no accuracy gate or
favorable-subset selection. At registration the runtime was not implemented
and no layer-localization inference had started.

The layer-localization diagnostic has now completed and independently audited:
**56 scored readouts, 76 measure calls including controls**. All 56 actual
mask/probability and complete-prefix/historical-KV proofs passed. Four original
global-operator equivalence checks, four all-history controls, four native
prefix checks, four pre-controller anchors and eight detached-native replays
passed. The 155-test suite passed. All 117 live/archived runtime sources and
eight primary audit/figure artifacts were verified; native and adapter weights
remain unchanged. All execution sessions are terminal.

The following table retains all four cases. Layers are zero-based. A
single-layer correctness flip need not reproduce the same incorrect number.
Leave-one-out necessity is conditional on masking the other five layers.

| Frozen readout | All-layer change | Single layers reproducing correctness direction | Omitted layers removing that direction |
| --- | --- | --- | --- |
| Local question, Oracle | 784 to 1584 (harm) | 11 gives 264 | 3, 7, 11, 15, 19 |
| Local question, matched other | 375 to 410 (harm) | 19 gives 410 | 3, 11, 19 |
| GDN question, matched other | 703 to 312 (repair) | 7 or 11 gives 312 | none |
| GDN question, matched other | 703 to 343 (harm) | 15 gives 784; 19 gives 343 | 19 |

Restricting only Attention23 preserves the entire full-question cache and
actual input to that layer in all four cases. It changes the first-token
logits, but changes neither the first-token argmax nor the full generated
answer in any case. Thus this diagnostic does **not** demonstrate a direct
final-reader answer flip. Earlier-layer interventions can cause flips, but
their effects can propagate through residuals, MLPs, later Attention, question
KV and GDN; the present experiment does not identify GDN uniquely.

First-token correctness also remains distinct from full-answer correctness:
the GDN-question only-layer15 case generates 784 instead of 703 despite a
positive gold-first-token margin. The repaired case reaches a zero gold-versus-
strongest-alternative first-token margin under only-layer7 or only-layer11.
All ties and small margins remain in the raw report, with the original frozen
arithmetic profile. No precision tuning, case exclusion or prior gate
relabeling is introduced. These are four reused posthoc cases, not independent
confirmation or evidence that original-prefix writes damaged memory.

Audit: /media/990Pro/Code/SAS/artifacts/memory_update/fresh-layer-localization-audit-20261115-01.json,
SHA256 d58e1973ee787f925e953d44d74da16b08499161896afd96423c2c1de17d6f33.
The same prefix provides the 56-row CSV, auditor source and provenance.
The complete comparison prefix is fresh-layer-localization-comparison-20261115-01
in the same directory: PNG, PDF, all-cell JSON, layout, exporter source and
provenance. The 2880x972 figure was visually inspected with all 56 cells
retained and all text inside the canvas. An initial figure-only layout check
rejected an automatic colorbar tick outside the canvas; that attempt's log
is retained and bounded colorbar ticks corrected the export. Inference and
the independently audited scores were unaffected.

The next diagnostic is registered before inference in
configs/fresh_control_token_timing_plan.json, SHA256
0e6bc1bd3cd6b619af3ee565b757d6a233827c4a6cd1936dd9bd36c63e7c112b.
It retains all four cases and all six single layers, plus full/all-layer
conditions, across three question-row policies: complete question, final
question token only, and earlier question tokens only. All 96 first-token
observations must be reported; 32 complete-question references must exactly
replay the completed localization and 64 are new timing counterfactuals.
For final-token-only masking, the complete cache through the preceding
question token must equal the same frozen controller's unmasked reference.
This separates a current-token depth effect from effects propagated through
earlier question tokens, while still not identifying GDN alone. It does not
score full generations or introduce an accuracy gate. At registration its
runtime was not implemented and no timing inference had begun.

The question-token timing diagnostic has now completed and independently
audited: **96 first-token observations**, comprising 32 exact completed-layer
references and 64 new timing counterfactuals. The 165-test suite passed.
All 96 actual row-mask/probability and complete-ledger-prefix/historical-KV
proofs passed, together with 32 complete preceding-question memory/logit/KV
checks for final-token-only masking, 12 only-final-layer question-cache/input
checks, 12 all-history controls, four native-prefix checks, four pre-controller
anchors and eight detached-native replays. All 121 live/archived runtime
sources and ten primary audit/figure/descriptive-analysis artifacts were
verified. Weights remain unchanged and all execution sessions are terminal.

The table below reports only the first answer token, restricting all six
Attention layers in the three timing policies. The harm/repair labels refer
to the original completed full-answer changes; the new timing cells do not
score whole generated answers.

| Original case | Unmasked | All question rows | Final question token only | Earlier question tokens only |
| --- | ---: | ---: | ---: | ---: |
| Local question, Oracle harm, gold 784 | 7 | 1 | 7 | 7 |
| Local question, matched-other harm, gold 375 | 3 | 4 | 3 | 3 |
| GDN question, matched-other repair, gold 312 | 7 | 3 | 3 | 3 |
| GDN question, matched-other harm, gold 703 | 7 | 3 | 3 | 7 |

Across all 32 case/layer cells, nine complete-question restrictions change
the first-token argmax. Final-token-only masking reproduces that argmax in
four of these nine cells, earlier-token-only masking in three. These overlap
and are descriptive counts, not a scientific gate. The final-token-only
effects occur under exact identity of the complete preceding-question memory,
so prior-question state changes are not necessary for those particular
effects. They can still propagate through GDN, residuals, MLPs and Attention
within the final token. Other cases have different timing patterns. None
identifies a recurrent-state-only mechanism or original-prefix write damage.
Earlier-token-only Attention23 also leaves first logits exactly unmasked in
all four cases, as expected for this last-layer causal cut.

One descriptive check prevents a misleading architectural interpretation.
For the gold-375 case at only-layer19, the gold-3 versus fixed competitor-4
margin is +0.125 unmasked, 0 under last-only masking, 0 under earlier-only
masking, and -0.125 under both. Thus both together flip the argmax although
the four-point fixed-pair margin interaction is exactly zero:
(-0.125) - 0 - 0 + 0.125 = 0. A combined-only discrete flip need not imply
positive interaction or semantic complementarity. The posthoc factorial
report retains all 32 cells; it computes fixed-pair differences only in the
28 cells whose strongest non-gold token stays identical in all four
conditions. Other cells are retained with that interaction field unset.
There is no tolerance tuning or prior gate relabeling.

Audit: /media/990Pro/Code/SAS/artifacts/memory_update/fresh-token-timing-audit-20261115-01.json,
SHA256 17ca8a1279550fa5bc35c6795a7b0cb99709e495ef2395be560f93dd19ba032e.
The same prefix provides the 96-row CSV, auditor source and provenance.
The comparison prefix fresh-token-timing-comparison-20261115-01 provides PNG,
PDF, complete 96-cell JSON, exporter source, layout and provenance. The
2484x1908 figure was visually inspected; all text fits and all cells remain.
The descriptive factorial prefix is fresh-token-timing-factorial-20261115-01,
with JSON, source and provenance. These remain reused-case diagnostics; no
new architecture performance or independent scientific confirmation is claimed.

The next conditional path diagnostic is registered before inference in
configs/fresh_question_path_mediation_plan.json, SHA256
f2eadecbeb209e95b2dc58f546853b548e6bb8a43e87063438508b8ef3f08146.
It retains all four queries, comparing unmasked and earlier-question-only
Attention19 contexts with four oracle activation patches: none, final-row
GDN20-22 branch outputs, earlier-question Attention23 K/V, and both.
All 32 first-token vectors will be archived, including eight exact prior
references and 12 unmasked patch shams. The cut is a posthoc choice motivated
by the observed earlier-only19 effect and the tractable terminal GDN triplet;
it does not establish a universal layer role. Patching GDN branch outputs
does not patch recurrent or convolution cache writes, and includes other
GDN computation. Even exact restoration of first logits would not establish
future-state preservation, semantic write locality or a deployable method.
At registration the runtime was not implemented and no mediation inference
had started. The completed results below supersede that execution status.

The conditional path diagnostic has now completed and independently audited:
**32 first-token observations and 32 archived full-vocabulary vectors**. The
173-test suite passed. All eight unpatched prior references, 12 unmasked
oracle-patch shams, four combined-path exact-logit closures, 32 actual
row-mask/prefix/KV proofs and 160 actual tensor-patch receipts passed. Four
native-prefix checks, four pre-controller native anchors and eight detached
native replays passed. All 125 live and archived runtime sources, the
installed modeling-source archive, two raw-vector files and eight exported
audit/figure artifacts were verified. Native and controller weights remain
unchanged; all execution sessions are terminal.

The table reports the gold-first-token logit minus the **fixed unmasked
strongest non-gold competitor**, in the earlier-question-only Attention19
context. Patches restore same-query unmasked activations. The last column is
the four-point margin contrast, not an accuracy improvement. All four reused
queries remain in the report.

| Original case | No patch | GDN20-22 final-row outputs | Attention23 earlier-question K/V | Both | Margin contrast |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local question, Oracle harm, gold 784 | 7.875 | 7.125 | 7.750 | 7.000 | 0 |
| Local question, matched-other harm, gold 375 | 0 | 0.125 | 0 | 0.125 | 0 |
| GDN question, matched-other repair, gold 312 | 0 | -0.125 | -0.125 | -0.125 | 0.125 |
| GDN question, matched-other harm, gold 703 | 0.125 | 0.500 | 0.125 | 0.500 | 0 |

Only the gold-312 case changes first-token argmax under these restoration
patches: no patch gives 3, while either branch restoration or both gives 7.
Both branches therefore participate in this conditional first-token change.
It is a single retained BF16 tie-boundary case, not evidence for a general
semantic interaction or shared-controller advantage. The other three cases
retain the same first-token argmax across all four patch choices; these null
changes are retained. This phase does not score full generated answers.

Combined restoration makes the entire first-logit vector exactly unmasked in
all four cases. This is an operator closure control and can restore an
incorrect reference. It does not restore the complete cache. Comparing the
actual recorded state hashes, precisely six linear-cache tensors still
differ: the convolution and recurrent states in each of layers 20, 21 and 22.
All other recorded linear-state hashes and all 12 earlier-position K/V hashes
match the unmasked run. Those K/V checks stop before the final question row;
full-cache equality, including that row and bookkeeping, must be directly
checked in the next phase before interpreting a state-only continuation.

The GDN patches replace branch outputs, which include recurrence, short
convolution, gating, normalization and projection. They do not overwrite the
already-written convolution or recurrent caches. Thus this result does not
locate original fact-prefix write damage, establish recurrent-memory loss,
demonstrate autoregressive recovery or provide a deployable update method.
The three original fresh-confirmation overall scientific gates remain failed.
No new semantic-block training is triggered.

Audit: /media/990Pro/Code/SAS/artifacts/memory_update/question-path-mediation-audit-20261115-01.json,
SHA256 abb5808dfd3fc5081c62ccca4ad82ed8fa0d0d5fb3dfcabd05f37b30f5c5dd32.
The matching prefix provides the complete 32-row CSV, auditor source and
provenance. Comparison prefix question-path-mediation-comparison-20261115-01
provides PNG, PDF, all-32-cell JSON, layout, exporter source and provenance.
The 2070x1404 figure was visually inspected with all cells retained and text
inside the canvas. Both models' actual full-vocabulary vectors are retained
in the source run's logits.safetensors files.

The next state-continuation diagnostic is registered before implementation
and inference in configs/fresh_state_continuation_plan.json, SHA256
62f2684e0ad9315facb2e56cec76633519b07b48ff496f4374530d355ee779d9.
It will exactly replay all 32 source observations and retain all four queries.
For each query it compares five cached continuations: unmasked reference,
both-paths-restored state, restore convolution only, restore recurrent state
only, and restore both. This gives 20 full greedy-answer and gold-NLL
readouts. All use identical unmasked starting logits; only the six registered
state tensors may differ. All complete K/V tensors, every other cache tensor
and bookkeeping must first be verified exactly unmasked. All temporary masks
and activation patches are removed before continuation; the same frozen
question-only controller remains active under its original policy.

The native-reference readouts must exactly replay the original full answers
and NLLs, and restoring both state types must give complete-cache and full
readout closure. Source caches must remain unchanged. Every actual greedy
step and teacher-forced continuation vector will be archived and independently
checked, with first-token and continuation-only losses reported separately.
No accuracy gate, precision tuning, new checkpoint, new query or training is
introduced. At this registration the new runtime is not implemented and no
state-continuation inference has started.

This was the final planned localization of these four reused cases. The
completed results follow; additional favorable posthoc cases cannot substitute
for a separately registered unseen-stream test.

The state-continuation diagnostic has now completed and independently
audited: **32 exact source replays and 20 full-answer continuations**. All
183 tests passed, including ten new state-origin/continuation tests. The
runner called the existing TextRuntime scoring and greedy functions with
passive observations, preserving the original shapes, arithmetic and stopping
rules. It archived 112 full-logit tensors: 32 source first-token vectors,
60 actual greedy-forward outputs and 20 teacher-forced continuation outputs.
The two raw tensor files total 131,131,616 bytes; no model copies, new
environment, installation or training were introduced.

All four source pairs have identical **complete** K/V, including the final
question row, identical complete bookkeeping, identical initial logits and
identical tensors outside the six registered state fields. All 960 starting
cache tensor-origin receipts passed. Start caches and both reused origin
caches remain unchanged after each scoring and decoding operation. Actual
cached input IDs, forward lengths and native RoPE positions were checked.
All four original full-answer/NLL references replay exactly. Restoring both
state types restores the entire cache and every later gold/greedy logit
exactly. Four native-prefix checks, four pre-controller anchors and eight
detached-native replays passed; all weights and hooks remain unchanged.

**None of the 20 generated strings differs from its query's unmasked
reference.** The four strings remain 784, 375, 703 and 703 in every condition;
the third is still wrong for gold 312. However, all twelve partially restored
or unrestored state conditions change the later teacher-forced full-logit
tensor. Thus a state difference can affect the subsequent prediction
distribution without changing this short generated answer.

| Model / original case | Native mean gold NLL | Retain altered states | Restore convolution | Restore recurrent | Restore both |
| --- | ---: | ---: | ---: | ---: | ---: |
| LocalQ / Oracle harm, gold 784 | 0.000550434 | 0.000558618 | 0.000552023 | 0.000550314 | 0.000550434 |
| LocalQ / matched-other harm, gold 375 | 0.366128604 | 0.366133372 | 0.359286110 | 0.359315872 | 0.366128604 |
| GDNQ / matched-other repair, gold 312 | 0.328503450 | 0.330503782 | 0.328560929 | 0.332509021 | 0.328503450 |
| GDNQ / matched-other harm, gold 703 | 0.570238153 | 0.599184513 | 0.598967632 | 0.597141782 | 0.570238153 |

These rounded display values do not replace the full-precision raw losses.
The unmasked first-token loss is identical across all five conditions for
each query; the report separately retains continuation-only losses and token
losses. Four-point loss contrasts have mixed signs. They do not establish
semantic complementarity, a general memory-protection benefit or an
architecture contribution. At this conditional cut, the remaining six state
differences are insufficient to change the four complete answers. The result
does not establish that original fact-prefix writes damaged memory, nor that
GDN states are unimportant in other contexts. All three earlier overall
fresh-confirmation scientific gates remain failed in their original scope.

Audit: /media/990Pro/Code/SAS/artifacts/memory_update/state-continuation-audit-20261115-01.json,
SHA256 d5b99f3b5e69369508f86b694cb230e06ca40ad9038322214ea74686c174b303.
The matching prefix contains the complete 20-row CSV, auditor source and
provenance. Run: /media/990Pro/Code/SAS/runs/memory_update/state-continuation-20261115-01.
All 129 live/archived runtime sources and installed modeling-source archive,
both raw tensor files and eight final audit/figure artifacts were verified.
All jobs are terminal. The comparison prefix is
state-continuation-comparison-20261115-02: PNG, PDF, all-20-cell JSON, layout,
source and provenance. The 2196x1368 figure was visually inspected. The first
successful export is retained at suffix01; suffix02 displays tiny nonzero
loss changes in scientific notation rather than rounding them to signed zero.
Neither export changes scientific results.

The four-case localization series is now closed. The next prospective test
is registered before data generation or inference in
configs/unseen_shared_timing_plan.json, SHA256
1fe0317cb6ab1563d16ed44a1fa130dcb42453b7d96606bb6ba075d80c5b316a.
It uses one fixed new eight-stream cohort (seeds62000-62007), with the original
three dependency depths, two question styles and alternating 8/64 update loads.
The complete 48-case grid has 336 physical queries and 432 related bindings.
All eight existing natural baseline configurations remain, plus question-only
and ledger-only activation of the **same frozen shared-controller checkpoint**:
ten arms, 3360 primary full-answer readouts and 4320 related bindings.

This tests whether prefix adaptation is needed for the existing shared
controller's gains; it does not implement explicit semantic scope. The new
question-only shared arm must preserve complete native prefix states/logits
for every distinct registered ledger. Full historical Attention access,
checkpoint bytes, enabled outputs, arithmetic and answer stopping remain
fixed. Two 180-readout operational smoke orders must agree exactly before
primary evaluation. Novelty is checked against all actual previous datasets,
including the first fresh confirmation; no cohort resampling is allowed.

The prospective functional criterion reports every shared-correct query that
regresses or is preserved under question-only activation. Retained positive
joint gains versus native must hold overall and at every depth, across at
least three positive base streams. All natural baseline outcomes, losses,
nulls and role/seed/depth/style/load breakdowns remain visible. These timing
ablations change the deployment distribution of an all-token-trained
controller: a regression cannot by itself prove necessary semantic writes.
Any eventual explicit semantic-controller method requires a separate matched
training plan; no new training is triggered. At registration the new data and
runtime do not exist and no new-cohort inference has started. The broader
research goal remains active.

The subsequent unseen-stream run completed and passed its independent audit:
3360 physical full-answer readouts, 4320 related bindings, 68 exact native-prefix
checks and 17472 observed forwards. Both registered question-only criteria
failed. All ten outcomes, the 33 regressions and two repairs, stronger local
baseline, raw answer table, visualized counts and incremental raw snapshot are
reported in [the completed phase](materials/updates/unseen-shared-timing-20261115/README.md).
The broader goal remains active; this result does not trigger new training.

最新完成：[匹配训练/部署时机的完整新样本结果](materials/updates/trained-shared-timing-20261115/README.md)。5376次读出与独立审计完成；三项研究门槛均未通过，匹配训练的收益集中于一跳任务。

后续完成：[冻结历史状态的证据访问诊断](materials/updates/trained-timing-evidence-access-20261115/README.md)。5040次正式读出、独立审计及全量导出完成；旧失败门槛保持原判，研究继续。
