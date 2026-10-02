# Five-model frontier: diversity beats standalone scaling

## Purpose

This frontier extension asks whether a genuinely different backbone can add useful ranking
diversity after simple parameter scaling plateaued. The retained leaderboard system remains
the support-adapted Qwen3-4B submission at **0.91808 public / 0.91425 private ROC AUC**.
No frontier development result is substituted for that official score.

The work stayed in the canonical AWS workspace. GitHub records aggregate evidence only:
model-family identities, validation design, decisions, and public-safe metrics. Raw comments,
row-level predictions, exact deployment weights, adapter weights, optimizer state, private
caches, credentials, and full cloud logs remain outside the repository.

## Why this direction

The previous Qwen3-14B study showed that a larger backbone was weaker by itself but useful
in a rank ensemble. That shifted the bottleneck from pure capacity to **complementary error
structure**.

The next candidate therefore introduced **Qwen2.5-14B-Instruct** as a distinct backbone
family and tested it first as a bounded held-out-policy screen, then as full two-policy OOF
evidence. The experiment reused completed incumbent predictions rather than retraining the
existing 4B/8B/14B/Phi models.

## Stage 1 · bounded diversity screen

On the 234-row Advertising policy holdout:

| Candidate | AUC | Change vs incumbent |
| --- | ---: | ---: |
| Policy-transfer incumbent | 0.689627 | — |
| Qwen2.5-14B standalone | 0.687910 | -0.001717 |
| **Fixed incumbent + Qwen2.5 insertion** | **0.696754** | **+0.007127** |

The candidate's Spearman correlation with the incumbent was **0.7656**, materially lower than
the correlations among the existing Qwen-family components. Grouped bootstrap estimated a
mean gain of **+0.006703**, with **0.89** probability of a positive gain. This was enough to
promote the backbone to a full OOF confirmation, not enough to claim hidden-test improvement.

## Stage 2 · full two-policy OOF confirmation

The full Qwen2.5 run produced predictions for all **881** development rows. The frozen public
prediction checksum is:

`a7ad39113a7c3309665ec0625fdc95c89b9a0fb4a8d1384e15922510ac4848dd`

| Candidate | Policy-macro AUC | Change vs policy-transfer incumbent |
| --- | ---: | ---: |
| Policy-transfer incumbent | 0.732076 | — |
| Qwen2.5-14B standalone | 0.730772 | -0.001304 |
| **Fixed Qwen2.5 insertion** | **0.738231** | **+0.006155** |

The fixed insertion improved both observed policies:

- Advertising: **+0.009478**
- Legal advice: **+0.002833**

Grouped bootstrap for that fixed candidate produced:

- mean gain: **+0.006020**
- 95% interval: **[+0.000735, +0.011509]**
- probability of positive gain: **0.984**

An aggressively optimized leave-one-policy-out candidate reached **0.735226** macro AUC but
regressed one policy by about **-0.00922**, so it was rejected. That is an important part of
the result: the strongest descriptive fit was not automatically promoted.

## Stage 3 · deployment compression was a valid negative result

A separate compression study asked whether the promoted signal could be carried by only two
or three backbones. The strongest compact candidate reached **0.738900** macro AUC, but its
bootstrap probability of a positive gain was only **0.8095** and Legal Advice regressed by
**-0.00419**.

The compact route therefore failed the preregistered stability gates. The project preserves
that negative result rather than weakening the model purely for deployment convenience.

## Stage 4 · fixed global deployment validation

The prior four-model global deployment reference was **0.734595** policy-macro AUC.

The final frozen five-model prior candidate reached:

- **0.740351 policy-macro AUC**
- **+0.005757** over the prior global deployment
- Advertising gain: **+0.008060**
- Legal-advice gain: **+0.003454**
- grouped-bootstrap 95% interval: **[+0.000128, +0.011706]**
- probability of positive gain: **0.9783**

The candidate passed the public development promotion gate. Its exact ensemble weights remain
private until official scoring; the public repository intentionally exposes the validation
logic and aggregate evidence without publishing competitive deployment details.

## What did not become a scientific result

Deep Mutual Learning remains an open mechanism. Several implementation attempts were stopped
before a valid DML experiment because the historical Qwen3-14B source/prompt contract could
not be reproduced with sufficient parity. That is an **implementation/source-state blocker**,
not evidence that DML is ineffective.

This distinction matters: execution failures and negative modeling results are tracked
separately.

## Engineering notes

The Qwen2.5 full OOF milestone:

- resumed the first fold from an existing 48-step checkpoint instead of restarting;
- completed both policy folds;
- ran on an AWS SageMaker `ml.g6e.8xlarge` / NVIDIA L40S;
- stayed around **14.13 GiB** peak framework GPU allocation;
- completed the full OOF milestone in about **1,206.8 seconds**;
- used no Kaggle public/private score for model selection.

This is the intended AWS/GitHub boundary: model development stays in AWS, while GitHub
publishes reviewable aggregate evidence.

## Decision and next step

The five-model candidate is **development-promoted but not yet leaderboard-scored**.

The official retained result is still **0.91808 public / 0.91425 private ROC AUC**. The next
competition action is to build one immutable submission notebook outside Kaggle, statically
validate it, and use Kaggle only for the actual scored submission. No further Kaggle
preflight/development notebooks are part of the planned workflow.

Machine-readable aggregate evidence:
[`reports/checkpoints/five_model_frontier.json`](../reports/checkpoints/five_model_frontier.json)

Public review notebook:
[`notebooks/29_five_model_frontier_review.ipynb`](../notebooks/29_five_model_frontier_review.ipynb)
