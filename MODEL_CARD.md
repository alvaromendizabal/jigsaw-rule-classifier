# Model card · Retained support-adapted classifier

**Task:** score whether a comment violates its supplied community policy. **Backbone:** Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. **Adaptation:** LoRA on original training labels and supplied positive/negative examples. Hidden query targets are not eligible training data.

## Interfaces and output

[Support construction](scripts/competition_features.py) normalizes text and excludes overlap/conflicts. [Decision training](scripts/decision_training.py) optimizes native last-token answer logits. [Inference](scripts/kaggle_adaptation.py) converts logits to within-policy ranks, restores IDs and checks schema and finiteness. Ranks preserve ordering for AUC; they are not calibrated probabilities and do not specify an operational moderation threshold.

## Recorded evaluation

Historical external ROC AUC was **0.91808 public / 0.91425 private**. A same-backbone development comparison measured **0.61460 → 0.71989** equal-weight policy-macro AUC on 881 novel comments and two rules. The development population was previously examined. A ten-row notebook preview verifies execution only, not full hidden-set cardinality or model quality. [Original receipt](reports/checkpoints/kaggle_adaptation.json)

## Limitations and responsible interpretation

Two policies do not establish transfer to every policy, community or language. Support examples may be misleading or conflicting. Training overlap can inflate apparent transfer without purging. The model is a research artifact, not a production moderation system or a replacement for human review. Native miniature-model tests verify arithmetic and recovery, not the historical 4B model's performance.

The public CPU classifier and browser engine are explicit controls/demonstrations, not the submitted neural system. Private comments, labels, adapters, pretrained weights and full inference outputs are not distributed. [Data card](DATA_CARD.md) · [Source credits](docs/TOP_SOLUTION_INTEGRATION.md) · [Reproduction](docs/REPRODUCIBILITY.md)
