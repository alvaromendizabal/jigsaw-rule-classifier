# Model card · retained competition system

**Model:** support-adapted Qwen3-4B-Instruct-2507. **Owner:** Alvaro Mendizabal. **Status:** final retained system for the completed research portfolio, September 21, 2026.

## Task and model

Rank English comments by whether they violate the supplied community rule. The backbone is `Qwen/Qwen3-4B-Instruct-2507`, pinned to revision `cdbee75f17c01a7cc42f958dc650907174af0554`. The original training labels and supplied positive/negative examples supervise one LoRA epoch. The scored system drops conflicting normalized rule/comment pairs; strict-majority conflict handling is a separate, unpromoted experiment.

Training applies loss at the decision position. Inference uses the final-token answer score, length-sorted batches, restored row order, and within-policy ranks. These rank scores are not calibrated probabilities. The canonical implementation and exact runtime are recorded in [the scored receipt](reports/checkpoints/kaggle_adaptation.json), [training settings](configs/kaggle_adaptation.json), and [inference source](scripts/kaggle_adaptation.py).

## Verified performance

| Evaluation | Result | Meaning |
| --- | ---: | --- |
| Kaggle public | 0.91808 ROC AUC | Successful late evaluation |
| Kaggle private | 0.91425 ROC AUC | Retained final leaderboard result |
| Private improvement over lexical baseline | +0.29469 AUC | End-to-end system improvement |
| Fixed-backbone development study | 0.61460 → 0.71989 macro AUC | 881 novel comments; isolates support adaptation |

The historical winner's recorded private score is 0.92930: the absolute gap is 0.01505. No original placement, medal, accuracy percentage, or state-of-the-art claim follows from this comparison. The matched development gain has a simultaneous 95% interval of [0.06124, 0.14935], conditional on fixed predictions and the observed policies; the repeatedly inspected cohort is not a fresh final holdout.

## Data and operational boundaries

Only original competition training labels and legitimate supplied support labels are used in the neural competition path. Released hidden targets are excluded. Development query bodies are removed from adaptation sources across rules. The historical post-competition route documented below uses a different data boundary and must not be substituted for this model or its score.

The recorded offline two-T4 preview completed 117 optimizer steps in 603.5 worker seconds, with about 8.26 GiB peak allocated GPU memory. Those are **preview measurements**, not hidden-evaluation timing or deployment benchmarks. Checkpoints preserve adapter, optimizer, scheduler, FP16 loss scaler, random state, and data order within the documented recovery contract.

## Intended use and limitations

This is a reproducible research classifier and a starting point for human-review support—not a deployed autonomous moderation service. No automatic content deletion, account penalty, operating threshold, demographic fairness guarantee, multilingual performance, adversarial robustness, or production-load capacity has been validated. Missing conversation context, contradictory examples, and unfamiliar policy intent remain important limitations. Model and data licenses retain their respective terms; pinned assets and attribution are preserved.

## Supplementary experiment

Strict-majority candidate submission 56444879 was last observed pending at 2026-09-21 23:08 UTC. No later score is verified for this closeout. Its completed supervision audit and preview are preserved; it is not the final retained model. [Snapshot](reports/majority_submission/summary.json) · [Project closeout](docs/PROJECT_CLOSEOUT.md).


## Historical research artifact

The separate 0.6B embedding/routing artifact, its 43,509-row protected confirmation, and its original operating limits are preserved in [HISTORICAL_MODEL_CARD.md](HISTORICAL_MODEL_CARD.md). That artifact is not the retained 4B competition model.
