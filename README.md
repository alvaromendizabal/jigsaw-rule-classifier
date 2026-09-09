# Jigsaw · Rule-conditioned comment classification

Predict whether a comment violates a supplied community rule, using the rule and examples of permitted and prohibited comments.

**Start with [03 · Results and examples](notebooks/03_saved_results.ipynb), then [02 · Feature research](notebooks/02_baseline_and_review.ipynb).** The historical research model passed its protected comparison on 43,509 rows. **Competition performance is being rebuilt:** the submitted lexical reference scored only **0.61956 private**, well below the approximately **0.92** objective. [Failure analysis and reopened feature gate](docs/COMPETITION_REBUILD.md).

**Original baseline submission:** Version 2 succeeded as a late submission: **0.59191 public / 0.61956 private**. These are the original-training lexical reference scores. [Submission record](reports/checkpoints/kaggle_submission.json) · [Kaggle result, signed-in account](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/submissions#). **Get the submission notebook:** [`kaggle/submission.ipynb`](kaggle/submission.ipynb) · [download](https://github.com/alvaromendizabal/jigsaw-rule-classifier/raw/refs/heads/main/kaggle/submission.ipynb). [Run the accepted model and find the AWS backups](docs/DELIVERY.md) · [Model card](MODEL_CARD.md) · [Data card](DATA_CARD.md).

[![Quality](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml/badge.svg)](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml)

Built by Alvaro Mendizabal. This project combines rule-conditioned NLP, training-only screening, cross-fitted target features, frozen representations, matched ablations and resumable AWS experiments. Its central finding is that **features which work on familiar policies can fail on a new policy**. The late Kaggle lexical-reference scores are reported separately from the post-competition research. No medal or state-of-the-art claim is made.

**Current rebuild evidence:** support adaptation improves the same 4B model from **0.6146 to 0.7199 AUC on 881 novel comments**, with a paired simultaneous 95% gain interval of **[0.0612, 0.1493]**. Adapted coordinates and a fixed prototype blend do not improve its direct score. The earlier full 2,029-row study rejected a 15,369-column frozen feature bank. Both experiments, optimizer recovery and evaluation replay are complete. These are development results, not new Kaggle scores; the **0.92 objective remains open**. [Measured comparisons and next execution gate](docs/COMPETITION_REBUILD.md).

## Protected result: the fixed candidate is accepted

Predictions were frozen in [commit 530ad799](https://github.com/alvaromendizabal/jigsaw-rule-classifier/commit/530ad79929012e807cb42a5253f7b92b090682ec) before eligible targets were first interpreted on September 9, 2026. All **12 preregistered acceptance checks passed**, with no new model fitting or candidate selection.

| Protected cohort | Lexical reference AUC | Accepted route AUC |
| --- | ---: | ---: |
| All six policies, 43,509 rows | 0.6801 | **0.7770** |
| Four familiar policies, 25,485 rows | 0.7581 | **0.8276** |
| Two unseen policies, 18,024 rows | 0.5241 | **0.6757** |

Primary policy-macro AUC improves by **0.0969**, with a paired 95% bootstrap interval of **[0.0898, 0.1051]**. Every policy improves. Overall log loss falls from **0.6685 to 0.5121**, and Brier from **0.2360 to 0.1739**. The interval conditions on six observed policies and fixed predictions. This post-competition benchmark is not a Kaggle leaderboard score or proof of performance on arbitrary future policies. [Protocol, recovery and full results](docs/CONFIRMATION.md).

![Protected policy-level comparison](reports/confirmation/policy_gains.svg)

## What the development research found

The expanded study uses **11,135 development rows across four policies** and seven strict comment/support-purged splits. Financial-advice and spoiler policies were wholly excluded from development. The original 43,576-row reserve yielded **43,509 eligible confirmation rows** after 67 target-blind near-copy exclusions. This is a post-competition benchmark built from the host's released data.

| Representation | Held-out policy AUC ↑ | Log loss ↓ | Brier ↓ |
| --- | ---: | ---: | ---: |
| Rule/example lexical reference | 0.4728 | 0.8126 | 0.2992 |
| Words + compact semantic comparisons | 0.5761 | 0.8156 | 0.2969 |
| Compact semantic comparisons alone | 0.6876 | 0.6891 | 0.2456 |
| Frozen semantic centroid | **0.7042** | **0.6237** | **0.2177** |
| All transferable feature families | 0.5515 | 1.4700 | 0.4503 |
| Semantic comparisons + labeled retrieval | 0.5590 | 1.4763 | 0.3613 |
| Plain-document support centroid | 0.6865 | 0.6214 | 0.2170 |
| Fixed centroid/intent average (rejected) | 0.7086 | 0.6549 | 0.2320 |

The centroid's matched development improvement is **+0.2314 AUC**, with a within-study simultaneous 95% interval of **[0.1889, 0.2740]**. Compact semantic features add **+0.1385** to the same screened-word control. These intervals condition on fixed predictions and four observed policies; they do not establish independent generalization.

![Feature contributions under matched validation](reports/expanded/ablation.svg)

Most of the centroid's average gain comes from avoiding reversed lexical ranking on illegal-activity promotion. Legal-advice AUC remains **0.5702** and medical-advice AUC **0.5969**. The full feature model reaches **0.7989 on familiar policies**, but only **0.5515 on held-out policies**. Reporting only familiar-policy performance would conceal the project's central failure mode.

## What makes the feature work substantive

- **188,595–188,598 candidate columns per fold**, including 309 retrieval and 77 semantic-formatting/intent candidates; **9,281–9,660 retained** across banks before final family selection. Counts are fold-specific, not independent hypotheses or one promoted model's width.
- **323 fixed fits** in the four-policy campaign: 238 primary comparisons, 22 changed-training sensitivities, 35 retrieval and 28 formatting/intent fits. Six identical sensitivity cases reuse saved fits. Six embedding-resolution scores and one fixed fusion require no additional inference or fitting; three new frozen formatting scores use the separately recorded encoder caches.
- Matched additions/removals, per-policy results, group permutation, coefficient contributions, selection stability and paired uncertainty. Classifier hyperparameters stay fixed.
- Exact body/support isolation; training-only vocabulary, scaling, screening, percentiles, NB and SVD; inner cross-fitting for target context; query-policy exclusion for labeled retrieval.
- Explicit negative results: adding retrieval damages the compact semantic model; no shorter embedding prefix improves the centroid; larger combined banks do not solve transfer. Historical NLI/instruction failures remain documented on their original two-policy cohort.
- Conflict, near-copy and support-dependence sensitivities. Removing all 18 self-support matches leaves centroid AUC **0.7034**. Missing timestamps, authors and threads are treated as unavailable data, not invented features.
- A source-checked 48-row qualitative error audit, completed before new scores, motivates a falsifiable policy-intent comparison. No labels were changed; one assistant's stratified review cannot estimate a population label-error rate.

[Expanded study and results](docs/EXPANDED_STUDY.md) · [Retrieval experiment](docs/RETRIEVAL_STUDY.md) · [Embedding resolution](docs/RESOLUTION_STUDY.md) · [Historical methods and feature provenance](docs/FEATURE_RESEARCH.md).

## Historical research scope and deliverables

**Feature research: COMPLETE for the declared four-policy scope.** Retain the original frozen centroid for unseen policies. Seven new semantic candidates and one fixed average fail the predeclared replacement criteria. The average's higher AUC is uncertain, advertising regresses and probability losses worsen. [Stopping evidence](reports/feature_decision/decision.json) · [Coverage and exclusions](docs/FEATURE_COVERAGE.md) · [Semantic results](docs/SEMANTIC_FORMATTING.md).

**Model fitting and development calibration: COMPLETE.** Nine inner fits validate calibration without sharing outer validation labels. Familiar-policy log loss improves from **0.4761 to 0.4689**; transfer calibration worsens log loss to 0.6976 and is rejected. The fitted route uses calibrated familiar-policy features and the raw unseen-policy centroid. Its familiar pipeline retains **9,263 of 181,958 candidate columns**. [Protocol, results and lineage](docs/MODEL_VALIDATION.md).

**Protected confirmation and offline delivery: COMPLETE.** The exact accepted artifact is packaged with its pinned encoder and locked runtime. Real offline predictions match saved cloud probabilities within **0.00000122**; batch/order parity, missing-support rejection and restart reuse pass. A separate environment restored the package and ran with zero network calls. On the tested CPU, peak memory was **3.88 GiB** and warm mean latency **1.11 seconds per authored comment**. These are small-run measurements, not production-load estimates. [Download and measured budgets](docs/DELIVERY.md).

**Historical portfolio artifacts: delivered.** Five executed evidence notebooks, four authored inference examples, model/data cards and verified private S3 recovery make that scoped study reviewable and runnable. The original-training-only Kaggle notebook is preserved separately from the model trained with post-competition labels. CI verifies its 2,029-row training / 10-row preview workflow and replay. Kaggle Version 2 passed its offline preview run and its submitted hidden-test run: **0.59191 public / 0.61956 private**, recorded on September 9, 2026 as a late entry. The competition performance objective remains open.

The historical research and local inference release remains preserved. The original baseline was submitted and scored, but the competition performance goal is open. [Current rebuild](docs/COMPETITION_REBUILD.md). [Acceptance record](docs/ROADMAP.md#completed-deliverables) · [Quality and execution evidence](docs/VALIDATION.md).

## Review or reproduce

The five executed notebooks need no private data, AWS account or model download to read. They render verified public aggregates and Plotly figures with SVG fallbacks.

| Notebook | Question |
| --- | --- |
| [03 · Results](notebooks/03_saved_results.ipynb) | What improved, what failed, and what can we claim? |
| [02 · Feature research](notebooks/02_baseline_and_review.ipynb) | Which families contribute, and how were they screened? |
| [01 · Validation](notebooks/01_data_and_validation.ipynb) | What prevents leakage and misleading validation? |
| [04 · Diagnostics](notebooks/04_semantic_benchmark.ipynb) | Which policies, probability errors and support conditions matter? |
| [00 · Environment](notebooks/00_environment_and_data.ipynb) | Where did the data and artifacts come from? |

```bash
uv sync --locked --extra semantic --group dev
uv run python scripts/verify.py
uv run python scripts/execute_notebooks.py --publish
uv run jigsaw gate
# After restoring private artifacts; no fitting:
uv run python scripts/verify_expanded.py
```

Automated tests, locked dependencies and CI cover software contracts. Private verification checks all 100 expanded/retrieval metric records and 20 new semantic metric records, replays 14 earlier and all 28 new fitted models, and rebuilds all 28 new feature transforms. Source/configuration/data identities, completed-stage hashes and encrypted S3 archives preserve expensive work. Interrupted active stages restart; intact completed work is reused. The historical frozen-model runners do not have neural optimizer state. The new support-adaptation runner saves adapter, optimizer, scheduler and RNG state; its cloud verification status is recorded in the rebuild receipt. [Actual quality and execution record](docs/VALIDATION.md).

[START_HERE.md](START_HERE.md) covers restoration and the user's offline submission workflow. [VALIDATION.md](docs/VALIDATION.md) records actual executions, checksums and limits. Public aggregates do not substitute for private OOF verification.

## Metric, provenance and limits

The primary metric is equal-weight **policy-macro ROC AUC**; pooled AUC is separate. The official column-averaged AUC description and [host per-rule release](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/641121) strongly corroborate this aggregation. Executable scorer parity is not claimed. The separately recorded late Kaggle result belongs to the original-training lexical reference. [Arithmetic audit](reports/released/metric.json) · [Released-data boundary](docs/RELEASED_DATA.md).

[Competition](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/overview) · [Host data release](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/641107) · [Research and model sources](docs/FEATURE_RESEARCH.md#research-sources-and-external-data-feasibility).

Code: MIT. Data and third-party models retain their own terms. The canonical [Kaggle notebook](kaggle/submission.ipynb) supports the user's own offline CSV generation; notebook execution generates files locally, and the separate completed Kaggle entry is recorded above.
