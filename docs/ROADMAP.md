# Research plan and acceptance gates

The goal is a defensible, high-performing rule-conditioned classifier and an understandable engineering portfolio. Medal-level performance is an ambition, not a guarantee or a current result.

| Phase | Work | Evidence required before moving on |
| --- | --- | --- |
| 0 · Foundation | Dedicated AWS space and S3 bucket, environment lock, contracts, observability, CI | Quality gate and actual data download; real backup from Studio |
| 1 · Reference experiments | Lexical baselines, seen-rule and held-out-rule splits, diagnostics, offline inference | Reproducible real-data report, reviewed split artifacts, exact submission schema |
| 2 · Finish feature research | 295 fixed fits plus six frozen-resolution controls completed on four policies; historical two-policy studies preserved | Finish bounded semantic formatting and policy-intent work; freeze representation and stopping rationale |
| 3 · Locked confirmation | Reserve 43,576 rows, including financial-advice and spoiler policies excluded from research | Commit candidate/reference specification before opening; paired official-metric comparison, per-policy behavior and probability quality; record a rejection without searching the reserve |
| 4 · Production pipeline | Promote the accepted representation, calibration/triage policy and model artifacts | Exact feature/model lineage, offline parity, missing-support behavior, latency/memory budget and reproducible restore |
| 5 · Submission and portfolio | Offline weights and dependencies, inference budget test, versioned Kaggle notebook, model card and demonstration | Successful offline run; scored late submission only if enabled; public report with accurate claims |

## Candidate families

Phase 2A uses the pinned Qwen3-Embedding-0.6B model and locked CPU neural dependencies. `docs/PHASE_2.md` records its design, integration gate, and evidence. The frozen NLI, low-rank and fixed-template instruction-likelihood studies are **complete**; the latter two are not future deliverables. None justifies a final model by itself. Check [FEATURE_RESEARCH.md](FEATURE_RESEARCH.md) for measured outcomes.

The host's post-competition release resolves the earlier lack of additional labeled policies. [RELEASED_DATA.md](RELEASED_DATA.md) locks the next boundary: original training plus 9,106 retained Public rows for four development policies, with 43,576 rows reserved. The [expanded study](EXPANDED_STUDY.md) is complete: the leading frozen centroid reaches 0.7042 transfer AUC; reserve targets remain untouched. The full source has 54,059 rows; 1,323 research rows are excluded for crossing the protected comment boundary and 54 reserved rows for historical exposure. The preparation code interprets only retained research targets.

## Next three deliverables

1. **A defensible end to feature research.** The 295-fit four-policy experiment, conflict/near-copy/support sensitivities, 309-column retrieval extension and six embedding-resolution controls are complete. The remaining bounded work is a preregistered asymmetric query/document comparison (2,217 unique plain supports) and a structured legal/medical error audit leading to one justified policy-intent representation check. Require matched transfer gains, per-policy behavior, probability quality and cost evidence; retain a negative result. Finish notebook `02` with a selected representation and a reasoned stopping decision. Do not repeat generic feature expansion or tune a larger model to conceal weak signals.
2. **A verified inference product.** Run the locked confirmation comparison with explicit seen/unseen-policy results. If the feature gate passes, promote the accepted artifact rather than leaving inference on the historical baseline. Deliver a tested offline package, measured inference budget, calibration/selective-review diagnostics, and the user's canonical submission notebook. No automatic submission is part of this work.
3. **An employer release someone can review quickly.** Lead the README with the problem, accepted result and limitations; make `03` the short evidence tour and `02` the research detail. Add a small example-driven demonstration, model/data cards and a concise reproducibility route. Tag a tested release after clean-environment restoration and notebook execution.

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
