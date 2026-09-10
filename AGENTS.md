# Standing project instructions

These instructions record the owner's September 10, 2026 execution requirements.
Read `docs/COMPETITION_REBUILD.md` and the relevant `reports/checkpoints/` receipt,
then inspect current GitHub state before changing or launching work.

## Bounded execution and cost

- Use research → hypothesis → implement → test → bounded experiment → inspect →
  decide → checkpoint → report. Finish one measurable milestone before starting
  another. Do not chain model searches into hours of unattended work.
- Minimize ChatGPT Work/API consumption as well as GPU charges. Existing tests,
  predictions, models and completed checkpoints are assets to reuse. Do not
  rerun valid work, repeatedly print large logs, or poll without a purpose.
- Before a costly run verify data schemas, provenance, required artifacts,
  local/small-sample correctness, recovery, a specific hypothesis, fixed decision
  criteria, and explicit queue/runtime/spend limits. Use staged smoke, sample,
  fold and full comparisons; preserve successful stages independently.
- Use UTC heartbeats, progress counters, source/config/data hashes and durable
  completion markers. A disconnection must not erase completed computation.
- Stop directions that underperform, violate assumptions, duplicate evidence or
  cannot justify their cost. Diagnose the cause before a changed retry; sunk
  cost does not justify continuation. Never loosen a gate after seeing results.
- Report meaningful progress while working. After each milestone state the
  attempted and completed work, passes, failures, actual metric, saved artifacts,
  GitHub state, learning, next step and why its information gain merits compute.
  Activity alone is not progress. Say explicitly when no experiment is running.

## Feature and representation research stays open

- Investigate plausible high-value families using domain literature, academic
  methods, leading competition solutions, statistical methods and permitted
  external sources. Translate findings into reproducible hypotheses and tests.
- Examine what rule adjudication needs: intent, negation, quotations, exceptions,
  rule/comment interactions, positive-versus-negative comparisons, contextual
  examples and representation geometry. Do not manufacture absent thread,
  temporal or entity history from identifiers.
- For each material family document rationale, inference availability and
  leakage assumptions; implement and test it; screen within training only;
  measure matched additions/removals and stability across purged policy folds.
  Retain only supported improvements. Hundreds or thousands of coordinates are
  justified only by a meaningful representation, not a feature-count target.
- Historical research closure does not close the reopened competition rebuild.
  Keep an explicit coverage/gap record. Do not call features research-grade or
  exhausted without evidence across the realistic high-value avenues.
- Target the strongest historical score without guaranteeing a record. Separate
  measured improvements from hypotheses about representation, modeling or
  ensembles. Employer-facing quality requires honest evidence and reproducible
  delivery; a subjective rating is not a test gate.

## Data, evaluation and publication boundaries

- Competition-eligible labels are original training labels and supplied labeled
  support examples. Never use organizer-released targets for competition
  training/tuning or reopen the consumed protected research cohort.
- Purge query bodies from every fitted/support source in each validation fold.
  Fit screens/retrievers only within eligible training. Do not send query targets
  to GPU feature/inference workers. Never infer a negative label for another rule.
- The repeatedly inspected 881-comment benchmark is exploratory development,
  not an independent holdout. Preview rows and local AUC are not Kaggle scores.
  Preserve exact Kaggle version/receipt/public/private scores. Inspect submissions
  before an authorized entry to avoid duplicates. Keep notebooks private.
- Keep code, public aggregates and reports in Git; private comments, labels,
  models, predictions and checkpoints in the approved project S3 area or private
  Kaggle outputs. Preserve canonical notebook sources/outputs and Plotly reports.
- Publish tested bytes with source fingerprints and UTC receipts. Use connected
  GitHub tools when shell push has no credentials, verify remote/local tree
  equality, and require current-head quality checks before merge. Do not present
  draft work or running checks as merged/passed.
- Necessary project work is already authorized through connected services; ask
  only for a specific missing authorization or direct user interaction. This is
  not permission for unlimited spend, automatic model sweeps or wider access.

