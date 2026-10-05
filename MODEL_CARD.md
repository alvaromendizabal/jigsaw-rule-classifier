# Model card · retained scored system and frontier context

**Retained scored model:** support-adapted Qwen3-4B-Instruct-2507.  
**Owner:** Alvaro Mendizabal.  
**Status:** retained leaderboard system; post-closeout frontier research active.

## Task and retained model

Rank English comments by whether they violate a supplied community rule. The retained backbone is `Qwen/Qwen3-4B-Instruct-2507`, pinned to revision `cdbee75f17c01a7cc42f958dc650907174af0554`.

The original training labels and legitimate supplied positive/negative examples supervise one LoRA epoch. Conflicting normalized rule/comment pairs are dropped in the retained scored path. Training applies loss at the decision position. Inference uses final-token answer scores, length-sorted batches, restored row order, and within-policy ranks.

These rank scores are not calibrated probabilities. The scored runtime is recorded in [reports/checkpoints/kaggle_adaptation.json](reports/checkpoints/kaggle_adaptation.json), [configs/kaggle_adaptation.json](configs/kaggle_adaptation.json), and [scripts/kaggle_adaptation.py](scripts/kaggle_adaptation.py).

## Verified leaderboard performance

| Evaluation | Result | Meaning |
| --- | ---: | --- |
| Kaggle public | 0.91808 ROC AUC | Successful late evaluation |
| Kaggle private | 0.91425 ROC AUC | Retained leaderboard result |
| Private gain over lexical baseline | +0.29469 AUC | End-to-end improvement |
| Strict-majority candidate | 0.91720 public / 0.91288 private | Rejected regression |

No original placement, medal, percentile, accuracy percentage, or state-of-the-art claim is implied.

## Controlled development evidence

At fixed 4B backbone, support adaptation raises policy-macro AUC from **0.61460 to 0.71989** on 881 novel comments. This cohort is repeatedly inspected and contains only two policies; it is diagnostic development evidence, not a substitute for hidden evaluation.

The latest Qwen3-14B AWS study reports:

- Qwen3-4B: **0.719893** policy-macro AUC
- Qwen3-14B: **0.708455**
- fixed 50/50 4B+14B blend: **0.730175**

Standalone 14B is therefore not promoted. The blend improves both observed policies and is retained as evidence of model complementarity. Its grouped-bootstrap interval crosses zero, so no leaderboard improvement is claimed. See [docs/QWEN14B_FRONTIER.md](docs/QWEN14B_FRONTIER.md).

The subsequent Qwen2.5-14B extension tested whether a different backbone family could add complementary ranking signal. On the same 881-row / two-policy development cohort, the final frozen five-model prior reaches **0.740351 policy-macro AUC**, up **0.005757** over the previous four-model global deployment reference (**0.734595**). Both observed policies improve, and the grouped-bootstrap 95% interval for the gain is **[+0.000128, +0.011706]** with **0.9783 probability of a positive gain**.

A compact 2/3-model deployment study was retained as a valid negative result after failing stability gates, and an aggressively optimized leave-one-policy-out blend was rejected after a policy regression. Exact ensemble weights remain private. See [docs/FIVE_MODEL_FRONTIER.md](docs/FIVE_MODEL_FRONTIER.md).

Two later supervision studies kept that five-model candidate fixed. E33 tested three owned uncertainty/disagreement pseudo-supervision variants and closed as a valid negative: the best candidate improved policy-macro AUC by only **+0.000193** with **0.507** bootstrap probability of a positive gain. E36 then used 10,000 deterministic public Reddit moderation comments, fold-specific Qwen soft labels, and a complementary DeBERTa-v3-base student. External soft labels improved the DeBERTa student over its matched control by **+0.004259**, but the fixed blend still trailed the incumbent by **-0.005671** and regressed Legal Advice by **-0.012798**. See [docs/PSEUDO_SUPERVISION_FRONTIER.md](docs/PSEUDO_SUPERVISION_FRONTIER.md).

A subsequent architecture/context frontier evaluated support-adapted Llama, pairwise continuation, semantic example retrieval, a ModernBERT/Ettin encoder, multi-view support context, and pretrained NLI transfer. The Llama challenger reached **0.743436 policy-macro AUC** versus **0.740351** for the accepted development incumbent, but its uncertainty evidence did not clear the registered promotion gate. Pairwise ranking and semantic retrieval did not improve on that preserved challenger; the Ettin/context and NLI families remained below the incumbent. These outcomes are retained as aggregate evidence in [docs/POST_CLOSEOUT_FRONTIER.md](docs/POST_CLOSEOUT_FRONTIER.md), with a machine-readable receipt in [reports/checkpoints/post_closeout_frontier_20261004.json](reports/checkpoints/post_closeout_frontier_20261004.json).

## Data and operational boundaries

The neural competition path uses original competition training labels and legitimate supplied support labels. Released hidden targets are excluded. Development query bodies are removed from adaptation sources across rules.

AWS is the canonical private workspace for raw data, row-level predictions, weights, optimizer state, environments, caches, and logs. GitHub contains public source, configuration, aggregate evidence, tests, and executed notebooks only.

## Intended use and limitations

This is a reproducible research classifier and potential decision-support component for human review—not a deployed autonomous moderation service.

No production threshold, automatic content deletion, account penalty, fairness certification, multilingual guarantee, adversarial robustness guarantee, or production-load capacity has been validated. Missing conversation context, contradictory examples, and unfamiliar policy intent remain important limitations.

## Frontier research status

Post-closeout research is explicitly separated from the retained scored system. Current evidence favors complementary multi-backbone ensembles and strict champion/challenger gates over standalone scaling or repeated context/objective tuning. Pseudo-supervision, hard-negative transfer, pairwise continuation, semantic retrieval, ModernBERT/Ettin context, and NLI variants have been preserved when they produced valid negative evidence rather than being tuned after the fact. The five-model candidate has passed the public development promotion gate but has **not** received an official Kaggle score. Active AWS research now prioritizes provenance-verified Qwen3-14B live replay as a prerequisite for controlled cross-model learning; further model selection does not use leaderboard scores.

The separate historical 0.6B embedding/routing artifact remains documented in [HISTORICAL_MODEL_CARD.md](HISTORICAL_MODEL_CARD.md) and must not be confused with the retained 4B system.


For a concise employer-facing technical narrative, see the [project case study](CASE_STUDY.md).
