# Jigsaw · Rule-conditioned comment classification

Predict whether a comment violates a supplied community rule, using the rule text and examples of permitted and prohibited comments.

**Start with [03 · Results and model decision](notebooks/03_saved_results.ipynb), then [04 · Semantic benchmark](notebooks/04_semantic_benchmark.ipynb).** All five portfolio notebooks have executed outputs. Reading them requires no AWS account, private dataset, or model download.

[![Quality](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml/badge.svg)](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml)

Built by Alvaro Mendizabal. The project combines leakage-aware validation, interpretable reference models, a pinned frozen Qwen3 encoder, probability diagnostics, and resumable experiments. It does **not** claim a leaderboard score, medal, or demonstrated state-of-the-art performance.

## Measured results

Both recorded experiments use **2,029 competition training rows**, with one out-of-fold prediction per row for each model and validation protocol. These are local cross-validation results, not Kaggle scores.

| Model | Familiar-rule macro AUC ↑ | Held-out-rule macro AUC ↑ | Held-out log loss ↓ | Held-out Brier ↓ |
| --- | ---: | ---: | ---: | ---: |
| Comment-only TF-IDF | 0.7281 | 0.6041 | 0.6731 | 0.2400 |
| Rule/example TF-IDF | 0.7287 | 0.6156 | 0.6736 | 0.2405 |
| Frozen semantic margin | 0.6351 | 0.6351 | 0.6786 | 0.2423 |
| Fold-fitted semantic classifier | 0.6013 | 0.5858 | 0.8264 | 0.2976 |

![Recorded semantic and lexical comparison](reports/semantic/comparison.svg)

**Decision: retain the lexical rule/example reference.** The frozen margin improves held-out AUC by **+0.0195**, but its paired 95% bootstrap interval is **−0.0132 to +0.0491** and probability losses worsen slightly. The learned semantic classifier performs worse. Neither semantic candidate establishes an overall replacement.

The frozen margin fits no fold labels, so its identical predictions in both protocols are not independent replications. Only two labeled rules are observed. The audit found 162 duplicate training bodies and overlap between training and all 10 preview test comments; that preview checks inference plumbing, not independent performance.

The frozen **Qwen3-Embedding-0.6B** experiment took **1,004.7 seconds** on CPU after weights were downloaded, peaked at **3.854 GiB** process memory, and truncated **1 of 1,875** unique inputs at 256 tokens. Those are original experiment measurements, not notebook-rendering times. Original runs: lexical `c15c2c2318fc0ed619c6`; semantic `4e7e6c00d269c451c0a3`. [Lexical evidence](reports/baseline/README.md) · [Semantic evidence](reports/semantic/README.md).

## Continue from the saved work

The next controlled feature experiment and its analysis live in the existing [02 · Baseline and review](notebooks/02_baseline_and_review.ipynb). Four CPU candidates test word/character rule similarity, positive/negative support contrasts, writing structure, and their combination on the preserved, purged reference splits. The combined candidate adds 26 dense features to the sparse comment representation. New competition-data results are not claimed until that experiment is run and its checksummed aggregates are published.

Submission generation belongs to you: run the canonical [Kaggle notebook](kaggle/submission.ipynb) to generate, validate, and click to download your own CSV. Fitted-model and prediction-batch checkpoints survive notebook restarts; no Kaggle upload is automatic. [START_HERE.md](START_HERE.md) contains the exact existing-SageMaker continuation and guarded AWS-to-GitHub results-push command. The five-notebook employer review path is unchanged.

## Notebook review path

| Notebook | Purpose |
| --- | --- |
| [00 · Environment](notebooks/00_environment_and_data.ipynb) | Recorded data identity and reproduction environment |
| [01 · Validation](notebooks/01_data_and_validation.ipynb) | Saved audit, leakage controls, and limitations |
| [02 · Baseline](notebooks/02_baseline_and_review.ipynb) | Lexical feature comparison |
| [03 · Results](notebooks/03_saved_results.ipynb) | Consolidated evidence and model decision |
| [04 · Semantic benchmark](notebooks/04_semantic_benchmark.ipynb) | Per-rule diagnostics, probability quality, runtime |

The notebooks verify committed aggregate hashes and provenance. **They do not recompute metrics from private out-of-fold predictions.** That stricter verification is available through `uv run jigsaw review` after restoring the saved runs. It never trains a model.

In an existing locked environment:

```bash
# Read-only execution: no fitting or model downloads.
.venv/bin/python scripts/execute_notebooks.py --notebook 03 --notebook 04
# Execute all five and atomically refresh their canonical files.
.venv/bin/python scripts/execute_notebooks.py --publish
```

[START_HERE.md](START_HERE.md) gives the exact SageMaker continuation steps. Training is separate and explicit: `uv run jigsaw baseline --cloud` or `uv run --extra semantic jigsaw semantic --cloud`. Viewing historical results never requires rerunning those experiments.

## Evaluation and methodology

The official overview names **column-averaged AUC**. The project implements **rule macro ROC AUC**, consistent with published descriptions of rule-specific averaging, and reports **pooled ROC AUC** separately. Metric tests distinguish the two. The overview does not expose executable scoring code; no external Kaggle score has validated the implementation yet.

Essential diagnostics include average precision, log loss, Brier score, calibration error, confusion matrix, and precision/recall/F1 at a predeclared 0.5 threshold. Average precision is not mislabeled as trapezoidal PR AUC. Calibration fitting and threshold tuning require nested validation and are not claimed as completed.

**Familiar-rule CV** stratifies by rule and target while grouping normalized duplicate comments. **Held-out-rule CV** excludes the evaluated rule from training. Both purge training rows whose body or supplied examples contain a validation body. Vocabulary, scalers, and learned classifiers use only retained training-fold rows. Qwen weights remain frozen; semantic experiments reuse the original splits. Exact-text isolation does not establish near-duplicate or shared-origin isolation, and two rules do not establish broad policy transfer.

## Reliability and verification

Python 3.12 and direct dependencies are pinned in `uv.lock`. Operations emit UTC timestamps, 15-second heartbeats, cell progress, and stage/total elapsed time. Notebook checkpoints require matching source, evidence, environment, and output hashes. Publication rejects errors, stderr, changed source, and synthetic/private results; failed execution preserves the prior canonical file.

Completed notebooks, model folds, and embedding shards are reusable. An interrupted active notebook, CPU solver fold, or embedding shard restarts; GPU optimizer-state resume is not implemented. Data/code/configuration fingerprints prevent treating changed experiments as the same run.

S3 snapshots store immutable content objects and publish their manifest last. Restore verifies hashes and refuses conflicting local overwrites. Snapshots assume a single writer. Source and reviewed public outputs remain in Git; private data, predictions, credentials, and weights do not.

`uv run python scripts/verify.py` checks compilation, Ruff, formatting, tests, and canonical notebook sources. CI then executes all five public notebooks in encrypted Jupyter kernels, verifies checkpoint reuse, tests standalone Kaggle inference on explicitly synthetic data, and checks the real pinned encoder. [Verification record](docs/VALIDATION.md) · [PR #3](https://github.com/alvaromendizabal/jigsaw-rule-classifier/pull/3).

## Kaggle and next model experiment

[kaggle/submission.ipynb](kaggle/submission.ipynb) is a self-contained offline lexical reference. It fits on the supplied training data, predicts the current test rows, and validates `submission.csv` with columns `row_id,rule_violation`. A preview CSV is not a scored submission. Kaggle-hosted execution and authenticated late-submission eligibility remain unverified for this completed 2025 event.

The next controlled experiment is joint rule/comment encoding with comment-only, rule-text, and support-example ablations. Cross-encoder or parameter-efficient fine-tuning candidates must beat the unchanged reference under the saved validation design before promotion. [Experiment plan](docs/PHASE_2.md) · [Roadmap](docs/ROADMAP.md). Paid compute requires an approved hardware/runtime/spending limit; notebook review does not launch it.

## Sources and license

[Official overview, metric and submission requirements](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/overview) · [Data schema](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/data) · [Competition rules](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/rules) · [Published challenge description and per-rule AUC](https://arxiv.org/abs/2511.17592).

Code: MIT. Competition data and third-party models retain their own terms; redistribution and augmentation require checking the relevant licenses and competition rules.
