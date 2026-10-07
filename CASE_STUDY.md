# Case study · Rule-conditioned moderation NLP

**Alvaro Mendizabal · end-to-end ML research and engineering**

Python · PyTorch · Transformers · LoRA · scikit-learn · AWS SageMaker · Jupyter

**Delivered result: 0.91808 public / 0.91425 private ROC AUC**. **Measured improvement:** **+0.29469 private AUC** over the lexical reference. These are successful late Kaggle evaluations, not an original competition placement. The research-and-engineering portfolio is complete; optional frontier work is documented separately.

[Executed current review](notebooks/31_complete_project_review.ipynb) · [Model card](MODEL_CARD.md) · [Reproduce the public evidence](docs/REPRODUCIBILITY.md)

## Problem and delivered system

Community moderation is conditional: the same comment may be acceptable under one rule and violate another. The task is to rank comments against a supplied natural-language policy, using legitimate positive and negative examples as context.

I built a support-adapted Qwen3-4B system with LoRA, decision-position supervision, forward-only answer scoring, length-sorted batches, restored row order, and within-policy ranking. Original training labels and supplied support labels are eligible; organizer-released hidden targets are excluded from the competition path.

| Successful late evaluation | Public ROC AUC | Private ROC AUC |
| --- | ---: | ---: |
| Lexical reference | 0.59191 | 0.61956 |
| Retained support-adapted 4B | **0.91808** | **0.91425** |
| Absolute improvement | +0.32617 | +0.29469 |

The [scored receipt](reports/checkpoints/kaggle_adaptation.json) records the exact notebook version and verification date. This end-to-end comparison changes both backbone and training method. A separate matched 4B study isolates support adaptation: policy-macro AUC rose from **0.61460 to 0.71989** on the same development cohort.

## The difficult part: evaluating transfer

A familiar-policy score can hide poor generalization. A 323-fit feature campaign measured **0.7989 familiar-policy AUC versus 0.5515 held-out-policy AUC**, exposing that bottleneck.

The validation framework separates familiar-policy and whole-policy views, purges query text from fitted/support sources, preserves grouped out-of-fold predictions, and compares fixed controls with grouped uncertainty. The later **881-comment, two-policy** cohort has been repeatedly inspected. These development results are not leaderboard-equivalent evidence or an untouched holdout. The separate post-competition research artifact also remains explicitly labeled.

## Model selection with evidence

Larger standalone models were not consistently better. Multi-backbone diversity produced a five-model development candidate at **0.740351 policy-macro AUC**, but it has no official Kaggle score. Exact private ensemble construction remains unpublished.

Later cross-model studies tested materially different hypotheses against matched native controls:

| Study | Development outcome | Decision |
| --- | --- | --- |
| Fixed-teacher transfer | 0.743455 versus 0.743973 native control | Valid negative |
| Label-anchored transfer | 0.743644 versus the same native control | Valid negative |
| Reciprocal learning | 0.742539 versus 0.742236 peer control; interval crosses zero | Valid negative |

A higher point estimate does not automatically replace the incumbent. Preserving negative results prevents repeated spending on unsupported directions. [Current receipt](reports/checkpoints/cross_model_frontier_20261006.json) · [Five-model evidence](notebooks/29_five_model_frontier_review.ipynb)

## Engineering ownership

AWS SageMaker is the canonical research environment. I implemented immutable source/data/model identities, checksum validation, resumable optimizer and inference state, atomic publication, bounded runtime gates, heartbeats, GPU memory/throughput benchmarks, and regression tests for observed failures. Completed predictions and checkpoints are reused.

GitHub provides locked dependencies, reusable validation/runtime modules, aggregate receipts, executed Plotly/SVG notebooks, and CI. The current review verifies saved numerical output against source receipts, rejects stale or incomplete execution, and re-executes in real Jupyter kernels with save/reopen checks.

## Reproduction, attribution and limits

An employer can reproduce public component checks and aggregate evidence without AWS credentials or model downloads. Full competitive reproduction needs privately retained raw data, row predictions, model state and ensemble construction. Existing public source/configuration remains available; this boundary is not a claim that every implementation detail is secret.

Public leading-solution ideas informed support adaptation and representation comparisons. [The attribution record](docs/TOP_SOLUTION_INTEGRATION.md) distinguishes those sources from my validation, implementation, controlled experiments, recovery system and delivery work.

The delivered system is a research classifier, not a deployed autonomous moderation service. Its rank scores are not calibrated probabilities, and no production threshold, multilingual guarantee or fairness certification is claimed. The latest receipt marks E44 paired-demonstration adaptation as planned; it is not required to review the completed deliverable.
