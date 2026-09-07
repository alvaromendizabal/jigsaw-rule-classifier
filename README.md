# Jigsaw · Rule-conditioned comment classification

Predict whether a comment violates a supplied community rule, using the rule text and examples of permitted and prohibited comments.

This project studies how text models behave when policies change. It combines explicit validation, auditable probability metrics, resumable experiments, and portable offline inference. Built by Alvaro Mendizabal for an employer-facing NLP portfolio.

**Current milestone:** Phase 0 infrastructure and Phase 1 reference implementation. Software checks use labeled synthetic fixtures; real competition training and leaderboard scores are not yet established. No medal-level result is claimed.

## Start here

See [START_HERE.md](START_HERE.md) for the exact SageMaker setup and notebook order.

```bash
export PATH="$HOME/.local/bin:$PATH"
bash bootstrap.sh
uv run kaggle auth login
uv run jigsaw download
uv run jigsaw backup
```

The Kaggle login step is needed only if this new space has no working Kaggle credentials. Existing supported Kaggle credentials are reused by the official CLI. Never put tokens in a notebook, command history, Git, or chat. Linked GitHub/Hugging Face/AWS accounts do not authenticate Kaggle automatically.

Select **Python (Jigsaw Rules)** and run these notebooks in order:

| Notebook | Question and output |
| --- | --- |
| `00_environment_and_data.ipynb` | Is the environment reproducible and the data schema correct? |
| `01_data_and_validation.ipynb` | What do the labels and rules look like, and how will validation prevent leakage? |
| `02_baseline_and_review.ipynb` | Does example context help, and how does performance change on a held-out rule? |
| `kaggle/submission.ipynb` | Can the baseline regenerate an exact-format submission offline? |

The equivalent terminal experiment is `uv run jigsaw baseline --cloud`. Rerun the same command after an interruption. Outputs include an offline HTML review, OOF predictions, split memberships, metrics, coefficients, a local model, a preview submission, and UTC JSONL logs.

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

The baseline does not establish deep semantic rule understanding. [ROADMAP.md](docs/ROADMAP.md) specifies the embedding, encoder, LoRA, ensemble, calibration, and deployment phases and their acceptance gates.

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
- No raw comments, credentials, run outputs, or weights are included by default in Git commits.

`uv run python scripts/verify.py` runs compilation, Ruff, formatting, pytest, and notebook source-consistency checks. CI also runs notebook execution in a Jupyter kernel. `--engine inprocess` is an explicit option for environments that cannot open Jupyter sockets; it tests cell logic and rich outputs, not kernel integration.

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
