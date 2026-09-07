# Jigsaw · Rule-conditioned comment classification

Predict whether a comment violates a supplied community rule, using the rule text and examples of permitted and prohibited comments.

This project studies how text models behave when policies change. It combines explicit validation, auditable probability metrics, resumable experiments, and portable offline inference. Built by Alvaro Mendizabal for an employer-facing NLP portfolio.

**Current milestone:** The pinned Qwen3 embedding benchmark is complete and compared with the preserved lexical baseline. See [Phase 2](docs/PHASE_2.md) for its implementation, verification, and measured evidence. This repository does not claim a medal or leaderboard result.

[![Quality](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml/badge.svg)](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml)

## Recorded baseline

Run `c15c2c2318fc0ed619c6` evaluated **2,029 competition training rows**. Each model has one out-of-fold prediction per row in each protocol. Results are local cross-validation, **not leaderboard scores**.

| Model | Familiar-rule macro AUC | Held-out-rule macro AUC | Held-out log loss | Held-out Brier |
| --- | ---: | ---: | ---: | ---: |
| Comment-only TF-IDF | 0.7281 | 0.6041 | 0.6731 | 0.2400 |
| Rule/example TF-IDF | 0.7287 | 0.6156 | 0.6736 | 0.2405 |

![Baseline generalization comparison](reports/baseline/comparison.svg)

Adding lexical example features changes familiar-rule AUC very little and gives a modest descriptive improvement in held-out ranking. Probability losses do not improve. The lexical reference remains the comparison point for the semantic experiments below.

The audit found 162 duplicate training bodies and overlap between training and all 10 preview test comments. The preview tests submission plumbing; it cannot establish independent model performance. [Inspect the recorded evidence](reports/baseline/README.md) and [the Phase 2 experiment plan](docs/PHASE_2.md).

## Recorded semantic experiment

Run `4e7e6c00d269c451c0a3` uses frozen **Qwen3-Embedding-0.6B**, pinned to an immutable Hub revision, with the exact original validation assignments. Only the similarity classifier learns from fold labels; Qwen weights remain frozen.

| Model | Familiar-rule macro AUC | Held-out-rule macro AUC | Held-out log loss | Held-out Brier |
| --- | ---: | ---: | ---: | ---: |
| Lexical rule/example reference | 0.7287 | 0.6156 | 0.6736 | 0.2405 |
| Frozen semantic margin | 0.6351 | 0.6351 | 0.6786 | 0.2423 |
| Fold-fitted semantic classifier | 0.6013 | 0.5858 | 0.8264 | 0.2976 |

![Semantic and lexical comparison](reports/semantic/comparison.svg)

The margin's held-out AUC gain is **+0.0195**, with a paired 95% bootstrap interval of **−0.0132 to +0.0491**. This does not establish a reliable improvement. Its advertising AUC improves, legal-advice AUC declines, and probability losses worsen slightly. The learned similarity classifier performs worse. **The lexical model remains the reference; neither semantic candidate is promoted as an overall replacement.**

The frozen margin produces the same predictions in both protocols because it fits no fold labels. The two columns are not independent replications. Only two labeled rules are observed. The next experiment will test joint rule/comment encoding and context ablations. [Detailed experiment record](reports/semantic/README.md).

The full run took **1,004.7 seconds** on CPU after weights were already downloaded, with **3.854 GiB** peak process memory and **1 of 1,875** unique inputs truncated at 256 tokens. Runtime, original source hashes, paired intervals, and the negative results are retained. **Notebook 04 is committed with real outputs already displayed.**

## Start here

For the existing AWS project, follow [START_HERE.md](START_HERE.md): update the existing checkout, restore the saved snapshot, and open the semantic comparison. Existing Kaggle authentication remains valid.

```bash
bash bootstrap.sh
uv run jigsaw restore
uv run jigsaw review
```

`review` validates saved checksums and recomputes metrics from saved predictions. It never trains a model. Its exports identify the run, dataset kind, row count, and training file hash. By default it refuses synthetic results. Open **notebooks/04_semantic_benchmark.ipynb** for the semantic comparison, or notebook 03 for general saved-run review.

| Notebook | Purpose |
| --- | --- |
| `00_environment_and_data.ipynb` | Verify environment and data contracts |
| `01_data_and_validation.ipynb` | Inspect labels, duplicates, and validation splits |
| `02_baseline_and_review.ipynb` | Run or resume a baseline under the current source fingerprint |
| `03_saved_results.ipynb` | Review an already completed run without retraining |
| `04_semantic_benchmark.ipynb` | Compare saved semantic and lexical evidence without loading model weights |
| `kaggle/submission.ipynb` | Regenerate an exact-format submission offline |

A new baseline is `uv run jigsaw baseline --cloud`. Rerun that command after an interruption to reuse matching completed stages. A source or environment change deliberately creates a new experiment fingerprint; use `review` to inspect historical runs without recomputing them.

## Evaluation

The official overview names **column-averaged AUC**. Published descriptions of the challenge identify this as averaging rule-specific AUCs. We implement **rule macro ROC AUC**, and separately report **pooled ROC AUC**. The metric tests deliberately use examples where these disagree. The overview does not provide executable scorer code; a future actual Kaggle score is the authoritative external check.

Secondary diagnostics: average precision (AP), log loss, Brier score, equal-width calibration error, confusion matrix, and precision/recall/F1 at 0.5. AP is identified precisely rather than equated with trapezoidal PR AUC. Threshold metrics are diagnostics; threshold selection and calibration fitting are deferred to nested validation.

Two complementary protocols:

- **Seen-rule grouped CV:** stratification by rule and target, grouping normalized duplicate bodies.
- **Held-out-rule CV:** no training rows from the evaluated rule.

Both purge training rows whose body or example fields contain a validation body. Vocabulary is fitted only on retained training rows. Provided validation examples remain legitimate inputs, but their labels are not added to the training set. Near duplicates and shared origins require further audits. Only two labeled rules means only two rule-transfer experiments; it is not broad evidence of generalization.

## Models and phase boundaries

The two initial models are deliberately interpretable CPU references:

1. TF-IDF comment features with logistic regression.
2. The same features plus comment-to-rule/example similarities, positive/negative maximum similarities, and their margin.

The lexical baseline does not establish deep semantic rule understanding. Phase 2 adds frozen Qwen3 embeddings, example comparison, and a classifier fitted within each training fold. The semantic benchmark uses pinned CPU PyTorch and Transformers dependencies, hashed model assets, and atomic embedding shards. [ROADMAP.md](docs/ROADMAP.md) specifies the embedding, encoder, LoRA, ensemble, calibration, and deployment phases and their acceptance gates.

## Reliability

- Python 3.12, pinned direct packages, complete `uv.lock`, isolated environment.
- UTC timestamps, 15-second heartbeats, per-stage elapsed times and explicit failure events.
- Content fingerprints include data, code, configuration, and modeling library versions.
- Fold outputs commit only after the action succeeds; checkpoints require matching SHA-256 hashes.
- Completed stages survive interruptions. The active CPU fold restarts; it does not resume inside a solver iteration.
- A process lock prevents concurrent baseline runs on the same local project.
- S3 backup uses immutable content objects and publishes its manifest last. A failed backup retains the prior committed snapshot.
- Restore verifies hashes and refuses to overwrite differing local work. Cloud snapshots assume a single writer.
- No arbitrary unpickling during resume. The exported `model.joblib` is for trusted local use only.
- Raw comments, credentials, per-row predictions, and weights are excluded from Git. Only reviewed aggregate evidence is published.

`uv run python scripts/verify.py` runs compilation, Ruff, formatting, pytest, and notebook source-consistency checks. CI also executes all six notebooks in a Jupyter kernel and retains logs and executed synthetic notebooks as downloadable artifacts for 30 days. Source and reviewed evidence remain in Git; experiment artifacts remain in S3. `--engine inprocess` is an explicit option for environments that cannot open Jupyter sockets; it tests cell logic and rich outputs, not kernel integration.

## Kaggle compatibility and competition status

The event ended **October 23, 2025**. Its official page requires notebook submissions, internet disabled, CPU or GPU runtime at most 12 hours, and an output named `submission.csv` containing `row_id,rule_violation`. The standalone notebook trains the reference model and uses the current test file, so it does not assume the preview test size or IDs. It makes no network calls or package installations.

A disabled Late Submission button was visible while signed out on September 7, 2026. Authenticated late-submission eligibility has not been verified. A finished competition cannot award a new competitive medal for this work. Published winning scores are comparison targets, not results attributable to this project.

## Sources

- [Official overview, metric, timeline, and notebook requirements](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/overview)
- [Official data schema and two-rule training limitation](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/data)
- [Competition rules](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/rules)
- [First-place solution index](https://www.kaggle.com/c/jigsaw-agile-community-rules/writeups/1st-place-solution)
- [GigaEvo paper, challenge description and per-rule AUC](https://arxiv.org/pdf/2511.17592)

The data page labels the dataset CC0. Competition terms and third-party model licenses must also be reviewed for later data augmentation, model redistribution, and deployment.

Code license: MIT. Data and third-party models retain their own terms.
