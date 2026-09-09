# Research plan and acceptance gates

**Current status — competition gate reopened:** the original lexical entry scored 0.61956 private. The approximately 0.92 performance objective is unmet. [Competition rebuild](COMPETITION_REBUILD.md) supersedes the earlier overall closeout language below; completed historical experiments and their frozen evidence remain preserved.

The goal is a defensible, high-performing rule-conditioned classifier and an understandable engineering portfolio. Medal-level performance is an ambition, not a guarantee or a current result.

| Phase | Work | Evidence required before moving on |
| --- | --- | --- |
| 0 · Foundation | Dedicated AWS space and S3 bucket, environment lock, contracts, observability, CI | Quality gate and actual data download; real backup from Studio |
| 1 · Reference experiments | Lexical baselines, seen-rule and held-out-rule splits, diagnostics, offline inference | Reproducible real-data report, reviewed split artifacts, exact submission schema |
| 2 · Feature research complete | 323 fixed fits, frozen controls, semantic formatting/intent, source-checked error audit and fixed fusion | Passed scoped stopping rule; original centroid retained for unseen policies |
| 3 · Final model and confirmation complete | Fixed familiar/unseen route and nested calibration; 43,509 eligible protected rows | All 12 fixed checks passed; macro AUC 0.7770 versus 0.6801; prediction freeze published before target access |
| 4 · Local inference product complete | Promote the accepted representation, calibration/triage policy and model artifacts | Exact feature/model lineage, offline parity, missing-support behavior, latency/memory budget and reproducible restore |
| 5 · Notebook and portfolio complete | Offline weights and dependencies, inference budget test, versioned Kaggle notebook, model card and demonstration | Successful offline run; scored late submission only if enabled; public report with accurate claims |

## Candidate families

Phase 2A uses the pinned Qwen3-Embedding-0.6B model and locked CPU neural dependencies. `docs/PHASE_2.md` records its design, integration gate, and evidence. The frozen NLI, low-rank and fixed-template instruction-likelihood studies are **complete**; the latter two are not future deliverables. None justifies a final model by itself. Check [FEATURE_RESEARCH.md](FEATURE_RESEARCH.md) for measured outcomes.

The host's post-competition release resolves the earlier lack of additional labeled policies. [RELEASED_DATA.md](RELEASED_DATA.md) locks the next boundary: original training plus 9,106 retained Public rows for four development policies, with 43,576 rows reserved. The [expanded study](EXPANDED_STUDY.md) is complete: the leading frozen centroid reaches 0.7042 transfer AUC; reserve targets were subsequently used only for the frozen confirmation. The full source has 54,059 rows; 1,323 research rows are excluded for crossing the protected comment boundary and 54 reserved rows for historical exposure. The preparation code interprets only retained research targets.

## Completed deliverables

1. **A validated final model.** Implement the fixed training-policy-membership route with the validated familiar-policy feature model and unseen-policy centroid. Replay OOF behavior, develop calibration with another validation level, and freeze the exact candidate/reference and acceptance rules before one reserved comparison. Report familiar and unseen policies separately. The route, nine nested calibration fits and full-development artifacts are complete. Familiar calibration passes; unseen calibration fails and is discarded. A target-blind copy audit leaves 43,509 eligible confirmation rows. The candidate/reference and 43,509-row cohort are now preregistered in [the confirmation protocol](CONFIRMATION.md). Target-free inference and scoring guards are implemented and tested; the completed protected comparison passed all 12 fixed checks, with a +0.0969 macro-AUC gain and 95% interval [0.0898, 0.1051]. This first deliverable is complete. [Detailed next milestone](FINAL_MODEL_PLAN.md).
2. **A verified inference product — complete.** The exact accepted artifact and encoder are packaged with source and locked dependencies. Offline/cloud parity, order/batch invariance, missing-support rejection, corruption checks and resume behavior pass. The bounded CPU measurement and a separate environment restore ran with zero network calls. [Artifact, budgets and instructions](DELIVERY.md).
3. **An employer release someone can review quickly — complete.** The README leads with results and limits; `03` adds four authored examples, while `02` retains research detail. Model/data cards, current executed notebooks, original-data submission checks and independently restored S3 artifacts close the defined portfolio scope. The requested late Kaggle entry is complete: Version 2 succeeded on September 9, 2026 with 0.59191 public / 0.61956 private. It is the original-training-only lexical reference, separate from the accepted research model. [Submission receipt and deliverables](DELIVERY.md#your-kaggle-submission).

These are acceptance milestones, not a promise of a numerical employer rating. A four-policy development result and a two-policy reserve are substantially better evidence than repeated selection on two policies; they still do not establish performance on every possible community norm. Fine-tuning or a larger model remains optional and needs a specific development-set justification after the representation work. The reserved comparison cannot become another tuning loop.

The expensive model is not automatically the best model. Compare ranking accuracy, calibration, inference latency, peak RAM/VRAM, and cost per evaluated comment. Distillation may give a better employer-facing deployment story than running a large ensemble everywhere.

## Methodological constraints

- Keep all validation labels outside vocabulary fitting, supervised example expansion, tuning, and calibration fitting for the fold they evaluate.
- Disclose which experiments are inductive and which, if later allowed, adapt to unlabeled test inputs. Never present a transductive result as untouched unseen-domain generalization.
- Do not assume a positive example for one rule is negative for another. A comment may violate multiple rules.
- Do not use public leaderboard probing to reconstruct hidden labels.
- Preserve all OOF predictions and fold assignments. Any learned ensemble or calibrator needs a separate level of validation.
- Use rule-aware grouped resampling for uncertainty, but state that four observed development policies do not estimate variability across all future policies.
- Keep an untuned benchmark after architecture selection. Reusing held-out rules repeatedly for model selection weakens their role as final independent tests.
- A model score supports triage. Before demonstrating automated moderation, evaluate selective coverage versus error and clarify the role of human review.

## Cloud execution for neural phases

Before launching a training job, record a bounded runtime and storage plan; verify GPU availability, quotas, and the selected container/model compatibility. Persist checkpoints to S3 during training, including optimizer, scheduler, random generator state, data-order position, and progress. Resume only when the dataset, model revision, configuration, and software contract match. This phase's CPU fold checkpoints do not yet implement neural optimizer recovery.

## Git workflow

Work on a feature branch for each coherent phase. Commit the problem, behavior, and relevant evidence. Open a pull request with checks, limitations, and an artifact link. Merge after CI is green and record the merge commit for the milestone. Never hide exceptions by adding files named `fixed`, `repair`, or `final_v2`; update the canonical files and use Git history for versions.
