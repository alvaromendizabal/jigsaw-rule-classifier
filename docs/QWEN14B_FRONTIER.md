# Qwen3-14B frontier study

## Purpose

This post-closeout extension tests a winner-aligned Qwen3-14B route in the project's
canonical AWS workspace. The experiment was designed to answer two questions without using
private leaderboard scores for model selection:

1. does a 14B backbone improve the fixed support-adaptation protocol by itself?
2. does it add useful ranking diversity to the retained 4B model?

The public repository records aggregate metrics, configuration, checksums, and an executed
review notebook. Raw comments, row-level predictions, adapter weights, optimizer state,
private caches, and full cloud logs remain in AWS.

## Executed system

The study used `unsloth/Qwen3-14B-bnb-4bit` at revision
`b455922269e4678c84f854fe38a6993d73614a1e` with one LoRA epoch, rank 16 / alpha 32,
decision-position training, conflict-drop supervision, support-example weight 2,
expanded Yes/No/Y/N verbalizers, forward-only final-token scoring, and within-policy ranks.

The run executed directly on the existing AWS SageMaker Studio `ml.g6e.8xlarge` workspace
with an NVIDIA L40S. It created no new Kaggle experiment and used no Kaggle score to choose
the recipe.

## Development result

| Candidate | Policy-macro AUC | Advertising | Legal advice | Pooled AUC |
| --- | ---: | ---: | ---: | ---: |
| Retained Qwen3-4B | 0.719893 | 0.679254 | 0.760533 | 0.738959 |
| Qwen3-14B | 0.708455 | 0.677649 | 0.739261 | 0.722877 |
| **Fixed 50/50 4B + 14B rank blend** | **0.730175** | **0.689627** | **0.770724** | **0.748871** |

The standalone 14B model trails 4B by
**-0.011438** policy-macro AUC.
The fixed blend improves 4B by **+0.010282**
and improves both observed policies.

The grouped-bootstrap 95% interval for the fixed-blend gain is
**[-0.00429,
0.02450]**. Because it crosses zero
and the cohort contains only 881 repeatedly examined comments across two policies, the
blend is promising evidence of complementarity—not proof of hidden-test improvement.

## Decision

The standalone 14B model is **not promoted** over the retained 4B model.

The 14B predictions are retained as an ensemble component because the fixed blend produced
the largest development blend gain measured so far in this project. The next research
milestone is leakage-safe four-model OOF ensemble selection across the preserved 4B, 8B,
14B, and Phi prediction sets. A Kaggle submission is reserved for a fixed candidate after
that AWS evidence is available.

## Reproducibility boundary

Machine-readable aggregate evidence:
[`reports/checkpoints/qwen14b_frontier.json`](../reports/checkpoints/qwen14b_frontier.json)

Executed public review:
[`notebooks/28_qwen14b_frontier_review.ipynb`](../notebooks/28_qwen14b_frontier_review.ipynb)

Public configuration:
[`configs/qwen3_14b_frontier.json`](../configs/qwen3_14b_frontier.json)

The retained scored system remains the Qwen3-4B submission at **0.91808 public / 0.91425
private AUC**. No 14B Kaggle leaderboard score is claimed here.
