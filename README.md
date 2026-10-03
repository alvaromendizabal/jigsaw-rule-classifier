# Jigsaw · Rule-conditioned NLP

**A reproducible rule-conditioned NLP portfolio: lexical baselines → support-adapted Qwen3-4B → transfer analysis → AWS multi-backbone ensembles → pseudo-supervision and external-data stress tests.**

Built by [Alvaro Mendizabal](https://github.com/alvaromendizabal).

[![Quality](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml/badge.svg)](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml)

**0.91425 private ROC AUC · 0.91808 public ROC AUC · +0.29469 private AUC over the lexical baseline**

[Latest supervision frontier](notebooks/30_pseudo_supervision_frontier_review.ipynb) · [Five-model frontier](notebooks/29_five_model_frontier_review.ipynb) · [14B frontier](notebooks/28_qwen14b_frontier_review.ipynb) · [Start here](START_HERE.md) · [Model card](MODEL_CARD.md)

## The problem

Community moderation is not a fixed toxicity task. A comment can be acceptable under one policy and violate another. This project ranks English-language comments by whether they violate a supplied community rule, using permitted/prohibited support examples while explicitly testing transfer beyond familiar policies.

## Verified competition results

| System | Public Kaggle AUC | Private Kaggle AUC | Decision |
| --- | ---: | ---: | --- |
| Original lexical reference | 0.59191 | 0.61956 | Baseline |
| **Support-adapted Qwen3-4B** | **0.91808** | **0.91425** | **Retained scored system** |
| Strict-majority 4B candidate | 0.91720 | 0.91288 | Rejected regression |

The retained 4B system improves private AUC by **0.29469** over the lexical baseline. These were successful **late submissions**; no original placement, medal, or leaderboard percentile is claimed. AUC is a ranking metric, not classification accuracy.

## Latest frontier result: Qwen2.5 adds useful diversity

The Qwen3-14B study showed that larger models were not automatically better, but that weaker standalone models could still add useful ensemble diversity. The next AWS-only extension therefore introduced a separate **Qwen2.5-14B** route and evaluated it through a held-out-policy screen, full two-policy OOF confirmation, compression controls, and a fixed global deployment gate.

On the fixed 881-comment / two-policy development cohort:

| Candidate | Policy-macro AUC | Change |
| --- | ---: | ---: |
| Prior four-model global deployment | 0.734595 | — |
| Qwen2.5-14B standalone | 0.730772 | −0.003823 |
| Fixed Qwen2.5 insertion | 0.738231 | +0.003636 vs prior global |
| **Promoted five-model global candidate** | **0.740351** | **+0.005757** |

The promoted candidate improves both observed policies (**+0.008060 Advertising, +0.003454 Legal Advice**) and its grouped-bootstrap interval for the gain is **[+0.000128, +0.011706]**, with **0.9783 probability of a positive gain**.

Two controls prevented overclaiming: an aggressively optimized leave-one-policy-out blend was rejected after a policy regression, and a 2/3-model compression study was preserved as a valid negative result because its strongest compact candidate failed the stability gate. The exact deployment weights remain private; GitHub publishes aggregate evidence, model-family identities, validation decisions, and tests.

This is **development evidence only**. The official retained Kaggle result remains **0.91808 public / 0.91425 private AUC** until the frozen candidate receives an official score. [Frontier report](docs/FIVE_MODEL_FRONTIER.md) · [Machine-readable checkpoint](reports/checkpoints/five_model_frontier.json).

## Follow-on frontier: pseudo-supervision and external data

The promoted five-model development candidate remained the fixed reference while two structurally different supervision routes were tested in AWS.

**Owned pseudo-supervision (E33).** Three uncertainty/disagreement variants completed six matched policy fits. The best point estimate reached **0.740544 policy-macro AUC**, only **+0.000193** over the incumbent, with **0.507** bootstrap probability of a positive gain. The route is retained as a valid negative rather than tuned further.

**External soft labels + DeBERTa (E36).** A deterministic **10,000-comment** public Reddit moderation corpus was labeled by fold-specific Qwen teachers and used to train a complementary DeBERTa-v3-base student. External soft labels improved the DeBERTa student by **+0.004259** over its labeled-only control, but the fixed candidate still trailed the incumbent by **-0.005671** and regressed Legal Advice by **-0.012798**. That route is also a valid negative.

These experiments narrow the next mechanism to **cross-rule hard-negative transfer** on the immutable Qwen representation cache. Deep Mutual Learning remains blocked before a valid scientific test. [Frontier report](docs/PSEUDO_SUPERVISION_FRONTIER.md) · [Aggregate checkpoint](reports/checkpoints/pseudo_supervision_frontier.json).

## What I built

**Task-adapted neural ranking.** The retained scored path uses Qwen3-4B-Instruct-2507, LoRA adaptation from original training labels plus legitimate supplied support labels, one-position decision loss, forward-only final-token scoring, length-sorted inference, restored row ordering, and within-rule rank normalization.

**Transfer-aware model research.** A separate fixed 881-comment study isolates support adaptation: policy-macro AUC rises from **0.61460 to 0.71989** on the same 4B backbone. Subsequent Phi, 8B, 14B, and Qwen2.5 studies preserve negative results, test complementary error structure, and use group-safe OOF evidence rather than private leaderboard scores.

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
| [30 · Supervision frontier](notebooks/30_pseudo_supervision_frontier_review.ipynb) | Owned pseudo-supervision and public external-data stress tests, both preserved as valid negatives |
| [29 · Five-model frontier](notebooks/29_five_model_frontier_review.ipynb) | Qwen2.5 diversity result, negative compression control, and promoted development ensemble |
| [28 · Qwen3-14B frontier](notebooks/28_qwen14b_frontier_review.ipynb) | Backbone scaling result that motivated diversity-first ensemble research |
| [27 · Project review](notebooks/27_latest_system_checkpoint.ipynb) | Retained Kaggle result, matched adaptation evidence, and feature-transfer lesson |
| [26 · Public-method map](notebooks/26_top_solution_integration.ipynb) | Leading-solution mechanisms and independent implementation plan |
| [03 · Results](notebooks/03_saved_results.ipynb) | Detailed model comparisons and decisions |
| [02 · Feature research](notebooks/02_baseline_and_review.ipynb) | Ablations, transfer failures, and negative results |
| [01 · Validation](notebooks/01_data_and_validation.ipynb) | Leakage controls and data boundaries |

## Current research state

The public portfolio remains complete and reviewable, while competitive frontier research is active. Current evidence says:

- majority conflict resolution regressed on the hidden leaderboard;
- standalone Qwen3-14B is weaker than 4B, but adds useful ranking diversity;
- Qwen2.5-14B is also weaker alone, yet materially improves a fixed ensemble;
- a leakage-aware five-model global candidate improves both observed policies and passes the grouped-bootstrap promotion gate;
- compact 2/3-model deployment variants were rejected on stability rather than promoted for convenience;
- small-cohort uncertainty/disagreement pseudo-supervision completed as a valid negative;
- public external soft labels improved a DeBERTa student relative to its matched control but did not beat the incumbent;
- Deep Mutual Learning remains blocked by historical source-parity uncertainty and has **not** produced a valid negative scientific result.

The frozen five-model candidate remains development-promoted but **not yet Kaggle-scored**. Active AWS research now prioritizes cross-rule hard-negative transfer and other structurally distinct representation objectives before the next scored submission; leaderboard feedback is not used for model selection.


[Competition](https://www.kaggle.com/competitions/jigsaw-agile-community-rules) · [Data card](DATA_CARD.md) · [License](LICENSE) · [Historical closeout](docs/PROJECT_CLOSEOUT.md)
