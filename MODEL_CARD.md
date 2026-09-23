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
| Historical winner | 0.92930 private AUC | External benchmark |
| Strict-majority candidate | 0.91720 public / 0.91288 private | Rejected regression |

No original placement, medal, percentile, accuracy percentage, or state-of-the-art claim is implied.

## Controlled development evidence

At fixed 4B backbone, support adaptation raises policy-macro AUC from **0.61460 to 0.71989** on 881 novel comments. This cohort is repeatedly inspected and contains only two policies; it is diagnostic development evidence, not a substitute for hidden evaluation.

The latest Qwen3-14B AWS study reports:

- Qwen3-4B: **0.719893** policy-macro AUC
- Qwen3-14B: **0.708455**
- fixed 50/50 4B+14B blend: **0.730175**

Standalone 14B is therefore not promoted. The blend improves both observed policies and is retained as evidence of model complementarity. Its grouped-bootstrap interval crosses zero, so no leaderboard improvement is claimed. See [docs/QWEN14B_FRONTIER.md](docs/QWEN14B_FRONTIER.md).

## Data and operational boundaries

The neural competition path uses original competition training labels and legitimate supplied support labels. Released hidden targets are excluded. Development query bodies are removed from adaptation sources across rules.

AWS is the canonical private workspace for raw data, row-level predictions, weights, optimizer state, environments, caches, and logs. GitHub contains public source, configuration, aggregate evidence, tests, and executed notebooks only.

## Intended use and limitations

This is a reproducible research classifier and potential decision-support component for human review—not a deployed autonomous moderation service.

No production threshold, automatic content deletion, account penalty, fairness certification, multilingual guarantee, adversarial robustness guarantee, or production-load capacity has been validated. Missing conversation context, contradictory examples, and unfamiliar policy intent remain important limitations.

## Frontier research status

Post-closeout research is explicitly separated from the retained scored system. Current evidence favors diverse ensembles over standalone backbone scaling. The next score-focused step is group-safe multi-model OOF ensemble selection before another official submission.

The separate historical 0.6B embedding/routing artifact remains documented in [HISTORICAL_MODEL_CARD.md](HISTORICAL_MODEL_CARD.md) and must not be confused with the retained 4B system.
