# Research plan and acceptance gates

The goal is a defensible, high-performing rule-conditioned classifier and an understandable engineering portfolio. Medal-level performance is an ambition, not a guarantee or a current result.

| Phase | Work | Evidence required before moving on |
| --- | --- | --- |
| 0 · Foundation | Dedicated AWS space and S3 bucket, environment lock, contracts, observability, CI | Quality gate and actual data download; real backup from Studio |
| 1 · Reference experiments | Lexical baselines, seen-rule and held-out-rule splits, diagnostics, offline inference | Reproducible real-data report, reviewed split artifacts, exact submission schema |
| 2 · Feature research complete | 323 fixed fits, frozen controls, semantic formatting/intent, source-checked error audit and fixed fusion | Passed scoped stopping rule; original centroid retained for unseen policies |
| 3 · Final model and locked confirmation | Verify fixed familiar/unseen-policy route and development calibration; reserve 43,576 rows including financial advice and spoilers | Commit candidate/reference specification before opening; paired official-metric comparison, per-policy behavior and probability quality; record a rejection without searching the reserve |
| 4 · Production pipeline | Promote the accepted representation, calibration/triage policy and model artifacts | Exact feature/model lineage, offline parity, missing-support behavior, latency/memory budget and reproducible restore |
| 5 · Submission and portfolio | Offline weights and dependencies, inference budget test, versioned Kaggle notebook, model card and demonstration | Successful offline run; scored late submission only if enabled; public report with accurate claims |

## Candidate families

Phase 2A uses the pinned Qwen3-Embedding-0.6B model and locked CPU neural dependencies. `docs/PHASE_2.md` records its design, integration gate, and evidence. The frozen NLI, low-rank and fixed-template instruction-likelihood studies are **complete**; the latter two are not future deliverables. None justifies a final model by itself. Check [FEATURE_RESEARCH.md](FEATURE_RESEARCH.md) for measured outcomes.

The host's post-competition release resolves the earlier lack of additional labeled policies. [RELEASED_DATA.md](RELEASED_DATA.md) locks the next boundary: original training plus 9,106 retained Public rows for four development policies, with 43,576 rows reserved. The [expanded study](EXPANDED_STUDY.md) is complete: the leading frozen centroid reaches 0.7042 transfer AUC; reserve targets remain untouched. The full source has 54,059 rows; 1,323 research rows are excluded for crossing the protected comment boundary and 54 reserved rows for historical exposure. The preparation code interprets only retained research targets.

## Next three deliverables

1. **A validated final model.** Implement the fixed training-policy-membership route with the validated familiar-policy feature model and unseen-policy centroid. Replay OOF behavior, develop calibration with another validation level, and freeze the exact candidate/reference and acceptance rules before one reserved comparison. Report familiar and unseen policies separately. The route, nine nested calibration fits and full-development artifacts are complete. Familiar calibration passes; unseen calibration fails and is discarded. A target-blind copy audit leaves 43,509 eligible confirmation rows. The next gate is the frozen reserved comparison, not another feature or model search. [Detailed next milestone](FINAL_MODEL_PLAN.md).
2. **A verified inference product.** Connect the accepted representation to the canonical offline pipeline and submission notebook. Demonstrate artifact lineage, feature parity, batch independence, resumability, missing-support behavior and named-hardware latency/memory measurements. Add a small example-driven interface and probability/triage diagnostics. No automatic Kaggle submission is part of this work.
3. **An employer release someone can review quickly.** Lead the README with the accepted result and limitations; keep `03` as the short evidence tour and `02` as the research detail. Add model/data cards, measured serving results and a concise reproducibility route. Tag the release after clean-environment restoration, offline inference and actual notebook execution pass.

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
