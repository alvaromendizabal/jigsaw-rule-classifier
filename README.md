# Jigsaw · Rule-conditioned comment classification

Predict whether a comment violates a supplied community rule, using the rule text and examples of permitted and prohibited comments.

**Start with [03 · Results and model decision](notebooks/03_saved_results.ipynb), then [04 · Semantic benchmark](notebooks/04_semantic_benchmark.ipynb).** All five portfolio notebooks have executed reference outputs. Reading them requires no AWS account, private dataset, or model download. The newly executed CPU feature study is summarized below; its refreshed notebook outputs are not yet published.

[![Quality](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml/badge.svg)](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml)

Built by Alvaro Mendizabal. The project combines leakage-aware validation, interpretable reference models, a pinned frozen Qwen3 encoder, probability diagnostics, and resumable experiments. It does **not** claim a leaderboard score, medal, or demonstrated state-of-the-art performance.

## Measured reference results

Both reference experiments use **2,029 competition training rows**, with one out-of-fold prediction per row for each model and validation protocol. These are local cross-validation results, not Kaggle scores.

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

## Executed CPU feature study

On **2026-09-08 UTC**, four predeclared candidates completed on the actual 2,029-row competition dataset in an isolated SageMaker Processing environment. The experiment reused the preserved purged reference splits: three familiar-rule folds and two held-out-rule folds, producing **20 fitted candidate/fold models** and eight model/protocol evaluations. Training-only vocabularies and scalers were retained. The original lexical models and frozen Qwen embeddings were not retrained.

| Candidate | Familiar-rule macro AUC ↑ | Held-out-rule macro AUC ↑ | Held-out log loss ↓ | Held-out Brier ↓ |
| --- | ---: | ---: | ---: | ---: |
| Rule-text similarity | 0.7137 | 0.6228 | 0.6939 | 0.2433 |
| Positive/negative support contrasts | 0.6424 | 0.5768 | 0.8201 | 0.2873 |
| Writing structure | 0.6813 | 0.4625 | 0.7496 | 0.2745 |
| Combined candidate | 0.6528 | 0.5373 | 0.8853 | 0.3047 |

**No candidate is promoted.** Rule-text similarity improved held-out AUC by only **+0.00724** over the rule/example reference. Its paired 95% normalized-comment-group bootstrap interval was **−0.02341 to +0.03556**; log loss and Brier score worsened. The other three candidates reduced held-out AUC. The combined candidate adds **26 dense features** to sparse comment TF-IDF; this is a small controlled study, not evidence that a thousands-feature search has been completed.

The study completed as run `e9091086b7bc7ed137ce` using source commit `9d9e1f72656d2e27fb65d435f71b1792fe6a6cb0`, with Python 3.12.13 and locked dependencies. Training CSV SHA-256: `83948d06a1e4b16421b738add60ef489cf1d44a2349ca711958fbb41c6207a0a`. Checksummed fold models, out-of-fold predictions, aggregate metrics, uncertainty intervals and coefficient audits are preserved in the project's private S3 snapshot. The completed workspace snapshot contains 375 paths referencing 357 unique content objects. The restored original snapshot's 306 paths and 289 unique objects were checked against their SHA-256 digests before training.

**Publication boundary:** this README records the inspected experiment outputs. The machine-readable feature aggregates and refreshed canonical notebook outputs still require a verified publication pass; they are not represented as already present in the public notebooks. Broader training-only screened structural, lexical and frozen-embedding feature research remains under development and is not an executed benchmark. This project is not complete against that broader research objective.

## Continue from the saved work

The controlled feature experiment and its analysis live in the existing [02 · Baseline and review](notebooks/02_baseline_and_review.ipynb). Reuse the completed study above rather than confusing the earlier reference notebooks with a newly trained model. A changed source/data/configuration fingerprint deliberately creates a different run; do not relabel historical results as new evidence.

Submission generation belongs to you: run the canonical [Kaggle notebook](kaggle/submission.ipynb) to generate, validate, and click to download your own CSV. Fitted-model and prediction-batch checkpoints survive notebook restarts; no Kaggle upload is automatic. [START_HERE.md](START_HERE.md) contains the existing-SageMaker workflow and guarded AWS-to-GitHub results-push command. Its feature execution section predates the completed study above; restore and inspect the recorded run before starting another experiment. The five-notebook employer review path is unchanged.

## Notebook review path

| Notebook | Purpose |
| --- | --- |
| [00 · Environment](notebooks/00_environment_and_data.ipynb) | Recorded data identity and reproduction environment |
| [01 · Validation](notebooks/01_data_and_validation.ipynb) | Saved audit, leakage controls, and limitations |
| [02 · Baseline](notebooks/02_baseline_and_review.ipynb) | Lexical feature comparison and feature-study workbench |
| [03 · Results](notebooks/03_saved_results.ipynb) | Consolidated reference evidence and model decision |
| [04 · Semantic benchmark](notebooks/04_semantic_benchmark.ipynb) | Per-rule diagnostics, probability quality, runtime |

The notebooks verify committed aggregate hashes and provenance. **They do not recompute metrics from private out-of-fold predictions.** That stricter verification is available through `uv run jigsaw review` after restoring the saved runs. It never trains a model.

In an existing locked environment:

```bash
# Read-only execution: no fitting or model downloads.
.venv/bin/python scripts/execute_notebooks.py --notebook 03 --notebook 04
# Execute all five and atomically refresh their canonical files.
.venv/bin/python scripts/execute_notebooks.py --publish
```

[START_HERE.md](START_HERE.md) gives the SageMaker workflow. Training is separate and explicit: `uv run jigsaw baseline --cloud` or `uv run --extra semantic jigsaw semantic --cloud`. Viewing historical results never requires rerunning those experiments.

## Evaluation and methodology

The official overview names **column-averaged AUC**. The project implements **rule macro ROC AUC**, consistent with published descriptions of rule-specific averaging, and reports **pooled ROC AUC** separately. Metric tests distinguish the two. The overview does not expose executable scoring code; no external Kaggle score has validated the implementation yet.

Essential diagnostics include average precision, log loss, Brier score, calibration error, confusion matrix, and precision/recall/F1 at a predeclared 0.5 threshold. Average precision is not mislabeled as trapezoidal PR AUC. Calibration fitting and threshold tuning require nested validation and are not claimed as completed.

**Familiar-rule CV** stratifies by rule and target while grouping normalized duplicate comments. **Held-out-rule CV** excludes the evaluated rule from training. Both purge training rows whose body or supplied examples contain a validation body. Vocabulary, scalers, and learned classifiers use only retained training-fold rows. Qwen weights remain frozen; semantic experiments reuse the original splits. Exact-text isolation does not establish near-duplicate or shared-origin isolation, and two rules do not establish broad policy transfer.

## Reliability and verification

The interpreter requirement is Python 3.12; direct dependencies are pinned in `uv.lock`, and each experiment records its exact resolved interpreter and package versions. The clean AWS verification used Python 3.12.13. `.python-version` selects the supported 3.12 minor release because the installed uv distribution could not obtain the previously requested 3.12.14. Git 2.28 or newer is required by the repository's branch-initialization tests; the verified isolated AWS toolchain used Git 2.55.0. A fresh run of the existing quality gate passed **139 tests**, compilation, Ruff, formatting and canonical notebook source checks before the four-candidate study.

Operations emit UTC timestamps, 15-second heartbeats, cell progress, and stage/total elapsed time. Notebook checkpoints require matching source, evidence, environment, and output hashes. Publication rejects errors, stderr, changed source, and synthetic/private results; failed execution preserves the prior canonical file.

Completed notebooks, model folds, and embedding shards are reusable. An interrupted active notebook, CPU solver fold, or embedding shard restarts; GPU optimizer-state resume is not implemented. Data/code/configuration fingerprints prevent treating changed experiments as the same run.

S3 snapshots store immutable content objects and publish their manifest last. Restore verifies hashes and refuses conflicting local overwrites. Conditional latest-manifest writes reject competing publishers, and incomplete workspaces cannot replace complete snapshots. Source and reviewed public outputs remain in Git; private data, predictions, credentials, and weights do not.

`uv run python scripts/verify.py` checks compilation, Ruff, formatting, tests, and canonical notebook sources. CI then executes all five public notebooks in encrypted Jupyter kernels, verifies checkpoint reuse, tests standalone Kaggle inference on explicitly synthetic data, and checks the real pinned encoder. [Verification record](docs/VALIDATION.md) · [PR #3](https://github.com/alvaromendizabal/jigsaw-rule-classifier/pull/3).

## Kaggle and next model experiment

[kaggle/submission.ipynb](kaggle/submission.ipynb) is a self-contained offline lexical reference. It fits on the supplied training data, predicts the current test rows, and validates `submission.csv` with columns `row_id,rule_violation`. A preview CSV is not a scored submission. Kaggle-hosted execution and authenticated late-submission eligibility remain unverified for this completed 2025 event.

Joint rule/comment encoding with comment-only, rule-text and support-example ablations remains unexecuted. Cross-encoder or parameter-efficient fine-tuning candidates must beat the unchanged reference under the saved validation design before promotion. [Experiment plan](docs/PHASE_2.md) · [Roadmap](docs/ROADMAP.md). Paid compute requires an approved hardware/runtime/spending limit; notebook review does not launch it.

## Sources and license

[Official overview, metric and submission requirements](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/overview) · [Data schema](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/data) · [Competition rules](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/rules) · [Published challenge description and per-rule AUC](https://arxiv.org/abs/2511.17592).

Code: MIT. Competition data and third-party models retain their own terms; redistribution and augmentation require checking the relevant licenses and competition rules.
