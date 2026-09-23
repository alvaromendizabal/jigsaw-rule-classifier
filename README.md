# Jigsaw · Rule-conditioned NLP

**A reproducible rule-conditioned NLP portfolio: lexical baselines → support-adapted Qwen3-4B → transfer analysis → AWS backbone/diversity research.**

Built by [Alvaro Mendizabal](https://github.com/alvaromendizabal).

[![Quality](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml/badge.svg)](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml)

**0.91425 private ROC AUC · 0.91808 public ROC AUC · +0.29469 private AUC over the lexical baseline**

[Latest 14B frontier review](notebooks/28_qwen14b_frontier_review.ipynb) · [Project review](notebooks/27_latest_system_checkpoint.ipynb) · [Start here](START_HERE.md) · [Model card](MODEL_CARD.md)

## The problem

Community moderation is not a fixed toxicity task. A comment can be acceptable under one policy and violate another. This project ranks English-language comments by whether they violate a supplied community rule, using permitted/prohibited support examples while explicitly testing transfer beyond familiar policies.

## Verified competition results

| System | Public Kaggle AUC | Private Kaggle AUC | Decision |
| --- | ---: | ---: | --- |
| Original lexical reference | 0.59191 | 0.61956 | Baseline |
| **Support-adapted Qwen3-4B** | **0.91808** | **0.91425** | **Retained scored system** |
| Strict-majority 4B candidate | 0.91720 | 0.91288 | Rejected regression |
| Historical competition winner | — | 0.92930 | External benchmark |

The retained 4B system improves private AUC by **0.29469** over the lexical baseline and remains **0.01505 AUC** below the documented historical winning private score. These were successful **late submissions**; no original placement, medal, or leaderboard percentile is claimed. AUC is a ranking metric, not classification accuracy.

## Latest frontier result: 14B adds diversity

The latest AWS-only experiment independently implemented a winner-aligned `Qwen3-14B` route with one LoRA epoch, decision-position loss, support weighting, expanded decision verbalizers, conflict-drop supervision, and within-policy ranks.

On the fixed 881-comment / two-policy development cohort:

| Candidate | Policy-macro AUC | Change vs 4B |
| --- | ---: | ---: |
| Qwen3-4B | 0.719893 | — |
| Qwen3-14B | 0.708455 | −0.011438 |
| **Fixed 50/50 4B + 14B rank blend** | **0.730175** | **+0.010282** |

The standalone 14B model is **not promoted**. The blend improves both observed policies and is the largest development blend gain measured so far, but its grouped-bootstrap interval crosses zero. The result is evidence of **complementarity**, not proof of hidden-test improvement. [Frontier report](docs/QWEN14B_FRONTIER.md) · [Machine-readable checkpoint](reports/checkpoints/qwen14b_frontier.json).

## What I built

**Task-adapted neural ranking.** The retained scored path uses Qwen3-4B-Instruct-2507, LoRA adaptation from original training labels plus legitimate supplied support labels, one-position decision loss, forward-only final-token scoring, length-sorted inference, restored row ordering, and within-rule rank normalization.

**Transfer-aware model research.** A separate fixed 881-comment study isolates support adaptation: policy-macro AUC rises from **0.61460 to 0.71989** on the same 4B backbone. Subsequent Phi, 8B, and 14B studies preserve negative results and model diversity rather than selecting on private leaderboard scores.

**A large feature/generalization campaign.** The separate four-policy feature campaign records **323 fixed fits**. The full feature model reached **0.7989 familiar-policy AUC but 0.5515 held-out-policy AUC**, demonstrating why more engineered features alone did not solve policy transfer.

**Reproducible ML engineering.** Immutable data/model identities, query/support separation, content-addressed checkpoints, optimizer-state recovery, paired/grouped uncertainty, executable notebooks, and CI preserve the connection between code and evidence.

## Architecture

Retained scored path:

`original labels + supplied support labels → audited pairs → Qwen3-4B + LoRA → decision logits → within-rule ranks → validated submission`

Frontier research path:

`preserved OOF predictions → backbone/diversity studies in AWS → group-safe ensemble evidence → fixed submission candidate only after promotion`

AWS remains the canonical private workspace for raw data, model weights, row-level predictions, optimizer state, caches, and operational logs. GitHub publishes source, compact configurations, aggregate evidence, tests, and executed notebooks—not a mirror of AWS.

## Review the work

| Start with | What it demonstrates |
| --- | --- |
| [28 · Qwen3-14B frontier](notebooks/28_qwen14b_frontier_review.ipynb) | Latest backbone/diversity result and next ensemble direction |
| [27 · Project review](notebooks/27_latest_system_checkpoint.ipynb) | Retained Kaggle result, matched adaptation evidence, and feature-transfer lesson |
| [26 · Public-method map](notebooks/26_top_solution_integration.ipynb) | Leading-solution mechanisms and independent implementation plan |
| [03 · Results](notebooks/03_saved_results.ipynb) | Detailed model comparisons and decisions |
| [02 · Feature research](notebooks/02_baseline_and_review.ipynb) | Ablations, transfer failures, and negative results |
| [01 · Validation](notebooks/01_data_and_validation.ipynb) | Leakage controls and data boundaries |

## Current research state

The public portfolio remains complete and reviewable, while competitive frontier research has been reopened. The current evidence says:

- majority conflict resolution regressed on the hidden leaderboard;
- 8B and Phi add limited but uncertain blend gains;
- 14B is weaker alone on the fixed development cohort but adds materially more blend diversity;
- the next score-focused milestone is leakage-safe multi-model OOF ensemble selection before spending another Kaggle submission.

No claim is made that the 14B blend has improved the 0.91425 private leaderboard score yet.

[Competition](https://www.kaggle.com/competitions/jigsaw-agile-community-rules) · [Data card](DATA_CARD.md) · [License](LICENSE) · [Historical closeout](docs/PROJECT_CLOSEOUT.md)
