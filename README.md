# Jigsaw · Rule-conditioned NLP

**A complete machine-learning research portfolio: from lexical baselines to a support-adapted language model, with verified Kaggle results and reproducible evidence.**

Built by [Alvaro Mendizabal](https://github.com/alvaromendizabal).

[![Quality](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml/badge.svg)](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml)

**0.91425 private ROC AUC · 0.91808 public ROC AUC · +0.29469 private AUC over the original baseline**

[Project review notebook](notebooks/27_latest_system_checkpoint.ipynb) · [Start here](START_HERE.md) · [Model card](MODEL_CARD.md) · [Closeout and evidence](docs/PROJECT_CLOSEOUT.md)

## The problem

Community moderation is not a fixed toxicity classification task. A comment can be acceptable under one policy and violate another. This project predicts whether an English-language comment violates a supplied community rule, using permitted and prohibited examples to adapt to that rule. The difficult part is transferring beyond familiar policies without learning shortcuts from duplicated comments or support examples.

## Results that matter

| System | Public Kaggle AUC | Private Kaggle AUC | Role |
| --- | ---: | ---: | --- |
| Original lexical reference | 0.59191 | 0.61956 | Reproducible baseline |
| **Support-adapted Qwen3-4B** | **0.91808** | **0.91425** | **Retained final system** |
| Historical competition winner | — | 0.92930 | External comparison, not our model |

The retained system improves private AUC by **0.29469** and is **0.01505 AUC units (1.505 percentage points)** below the documented winning private score. This is a strong ranking result for the completed project, not a claim of winning or a leaderboard percentile. AUC measures ranking, not classification accuracy. These are verified **late submissions**, not an original competition placement or medal. [Scored-version receipt](reports/checkpoints/kaggle_adaptation.json) · [Historical winning benchmark](configs/top_solution_integration.json).

A separate matched study isolates the mechanism: support adaptation raises the same 4B backbone from **0.61460 to 0.71989 policy-macro AUC** on **881 novel development comments**. The simultaneous 95% gain interval is **[0.06124, 0.14935]**, conditional on the observed policies and fixed predictions. This development result explains the method; it is not substituted for the Kaggle score. [Matched results](reports/support_adaptation/results.json) · [Uncertainty](reports/support_adaptation/uncertainty.json).

## What I built

**A task-adapted neural classifier.** The final path uses a pinned Qwen3-4B-Instruct-2507 model, LoRA adaptation from original training labels and supplied support labels, decision-position loss, efficient last-token scoring, length-sorted inference, and within-policy rank normalization. Output schemas, identifiers, ordering, model assets, and numerical validity are checked explicitly.

**A substantial feature and generalization investigation.** Lexical, semantic, retrieval, behavioral, policy-intent, and representation-geometry families were examined through controlled comparisons. The separate four-policy research campaign records **323 fixed fits**. Its full feature model reached **0.7989 familiar-policy AUC but 0.5515 held-out-policy AUC**, exposing why more features alone were not enough. Negative results, transfer failures, calibration choices, and stopping decisions remain visible. [Feature research](notebooks/02_baseline_and_review.ipynb) · [Expanded study](docs/EXPANDED_STUDY.md).

**Reproducible ML engineering.** Immutable input identities, training-only transformations, query/support separation, paired uncertainty, saved optimizer and random state, content-addressed checkpoints, and replay checks preserve the connection between code and evidence. GitHub Actions checks software quality, notebook execution and reuse, offline inference, and the pinned encoder. The public review needs no AWS account or private data.

## Architecture

Original training labels + supplied labeled examples → audited support pairs → pinned 4B model + LoRA → decision-token scores → within-rule ranks → validated submission.

The final model uses direct adapted decisions; it does **not** concatenate every historical feature bank. The separate post-competition research route is documented independently in the model card. [Inference implementation](scripts/kaggle_adaptation.py) · [Training](scripts/decision_training.py) · [Offline notebook](kaggle/submission.ipynb).

## Review the work

| Start with | What it demonstrates |
| --- | --- |
| [27 · Project review](notebooks/27_latest_system_checkpoint.ipynb) | Final score, matched evidence, feature-transfer lesson, latest experiment, and project conclusion |
| [03 · Results and examples](notebooks/03_saved_results.ipynb) | Detailed model comparisons and documented decisions |
| [02 · Feature research](notebooks/02_baseline_and_review.ipynb) | Feature contributions, ablations, and negative results |
| [01 · Validation](notebooks/01_data_and_validation.ipynb) | Data boundaries and leakage controls |
| [26 · Public-method map](notebooks/26_top_solution_integration.ipynb) | Attribution and comparison with leading methods |

Saved notebooks include visible evidence; the project-review notebook can also be rerun from public aggregates without model loading or cloud access. [Reproduction guide](START_HERE.md).

## Completed scope

**The research-and-engineering portfolio is complete, with the scored support-adapted 4B system retained.** Further backbone scaling and ensemble research are optional future work, not unfinished requirements for this release.

The latest supplementary majority-supervision experiment recovered ten conflicting pairs and completed its preview. Submission **56444879** was last observed pending at **2026-09-21 23:08 UTC**; no later score is verified in this release. It is preserved as an unpromoted experiment and does not replace the final system. [Experiment snapshot](reports/majority_submission/summary.json).

AWS remains the private data, checkpoint, and recovery workspace. GitHub contains reviewable code, configurations, tests, executed notebooks, and compact aggregate evidence—not raw comments, row-level predictions, model weights, environments, credentials, or a mirror of AWS. No deployment, autonomous moderation, fairness certification, or production-load claim is made.

[Competition](https://www.kaggle.com/competitions/jigsaw-agile-community-rules) · [Data card](DATA_CARD.md) · [License](LICENSE) · [Project conclusion](docs/PROJECT_CLOSEOUT.md)
