# Research plan and acceptance gates

The goal is a defensible, high-performing rule-conditioned classifier and an understandable engineering portfolio. Medal-level performance is an ambition, not a guarantee or a current result.

| Phase | Work | Evidence required before moving on |
| --- | --- | --- |
| 0 · Foundation | Dedicated AWS space and S3 bucket, environment lock, contracts, observability, CI | Quality gate and actual data download; real backup from Studio |
| 1 · Reference experiments | Lexical baselines, seen-rule and held-out-rule splits, diagnostics, offline inference | Reproducible real-data report, reviewed split artifacts, exact submission schema |
| 2 · Feature research gate | Broad lexical, structural, rank, nested target/context, frozen semantic/NLI and low-rank studies executed | Independent confirmation, stable feature-family selection and traceable final representation; gate remains open |
| 3 · Instruction model | Small-to-medium open instruction model, classification token scoring, LoRA fine-tuning where justified | Pinned model revision and license; GPU smoke test; resumable optimizer/scheduler/RNG checkpoints; full validation |
| 4 · Robustness and calibration | Exact/approximate-copy audit and grouped intervals executed; nested calibration, thresholds and final ensemble remain gated | OOF-only selection, frozen final evaluation, per-rule error analysis, bounded confidence claims |
| 5 · Submission and portfolio | Offline weights and dependencies, inference budget test, versioned Kaggle notebook, model card and demonstration | Successful offline run; scored late submission only if enabled; public report with accurate claims |

## Candidate families

Phase 2A uses the pinned Qwen3-Embedding-0.6B model and locked CPU neural dependencies. `docs/PHASE_2.md` records its design, integration gate, and evidence. The frozen NLI feature study and additional representation controls are executed. A fixed-template Qwen instruction-likelihood probe is the next bounded feature study; no final fine-tuning is triggered. Check [FEATURE_RESEARCH.md](FEATURE_RESEARCH.md) for current results and source limitations.

The expensive model is not automatically the best model. Compare ranking accuracy, calibration, inference latency, peak RAM/VRAM, and cost per evaluated comment. Distillation may give a better employer-facing deployment story than running a large ensemble everywhere.

## Methodological constraints

- Keep all validation labels outside vocabulary fitting, supervised example expansion, tuning, and calibration fitting for the fold they evaluate.
- Disclose which experiments are inductive and which, if later allowed, adapt to unlabeled test inputs. Never present a transductive result as untouched unseen-domain generalization.
- Do not assume a positive example for one rule is negative for another. A comment may violate multiple rules.
- Do not use public leaderboard probing to reconstruct hidden labels.
- Preserve all OOF predictions and fold assignments. Any learned ensemble or calibrator needs a separate level of validation.
- Use rule-aware grouped resampling for uncertainty, but state that two rules are insufficient to estimate variability across future policies.
- Keep an untuned benchmark after architecture selection. Reusing held-out rules repeatedly for model selection weakens their role as final independent tests.
- A model score supports triage. Before demonstrating automated moderation, evaluate selective coverage versus error and clarify the role of human review.

## Cloud execution for neural phases

Before launching a training job, record a bounded runtime and storage plan; verify GPU availability, quotas, and the selected container/model compatibility. Persist checkpoints to S3 during training, including optimizer, scheduler, random generator state, data-order position, and progress. Resume only when the dataset, model revision, configuration, and software contract match. This phase's CPU fold checkpoints do not yet implement neural optimizer recovery.

## Git workflow

Work on a feature branch for each coherent phase. Commit the problem, behavior, and relevant evidence. Open a pull request with checks, limitations, and an artifact link. Merge after CI is green and record the merge commit for the milestone. Never hide exceptions by adding files named `fixed`, `repair`, or `final_v2`; update the canonical files and use Git history for versions.
