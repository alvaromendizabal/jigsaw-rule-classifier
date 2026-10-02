# Top-solution integration research track

This track converts publicly documented competition methods into independent, testable project implementations. Third-party notebooks are research references, not vendored production code.

## Current implementation matrix

| Public-method mechanism | Project status | Evidence |
| --- | --- | --- |
| Support-label online adaptation | Recreated + validated | Retained 4B scored system |
| Drop subreddit from neural path | Recreated + validated | Retained 4B + 14B frontier config |
| One decision-position loss | Recreated + validated | Retained 4B + 14B |
| Forward-only answer scoring | Recreated + validated | Retained 4B + 14B |
| Length-sorted inference / within-rule ranks | Recreated + validated | Retained scored system |
| Expanded Yes/No/Y/N verbalizers | Recreated + validated in 14B AWS study | `qwen3_14b_frontier.json` |
| Strict-majority conflict resolution | Recreated + hidden-scored | Rejected: 0.91288 private |
| Qwen3-14B adaptation | Recreated + AWS validated | Standalone not promoted; blend complementary |
| Multi-model weighted ensemble | Recreated + AWS validated | Five-model fixed prior promoted on two-policy development evidence |
| Qwen2.5-14B diversity route | Recreated + AWS validated | Standalone weaker; fixed ensemble contribution promoted |
| Uncertainty-selected pseudo-labeling | Not implemented | High-value capability gap |
| Deep Mutual Learning | Blocked before valid scientific test | Historical 14B source/prompt parity remains incomplete |
| Task-trained contrastive BGE route | Not implemented | Diversity route, lower priority than ensemble/DML |

## What the latest ensemble result changes

The project has now moved beyond a simple “larger backbone” question. Qwen3-14B was weaker
alone but complementary; Qwen2.5-14B repeated that pattern with a different error structure.

The latest AWS evidence shows:

- Qwen2.5-14B is not a standalone replacement for the incumbent;
- a fixed Qwen2.5 insertion improves both observed policies;
- a five-model global prior reaches **0.740351 policy-macro AUC**, **+0.005757** over the prior four-model global reference;
- the grouped-bootstrap 95% interval for that gain is **[+0.000128, +0.011706]**;
- compact 2/3-model alternatives were rejected after failing stability gates;
- an aggressively optimized LOPO blend was rejected after a policy regression;
- exact deployment weights remain private and no Kaggle score was used for model selection.

That shifts the remaining bottleneck from **ensemble discovery** to **hidden-score transfer and
structurally new training mechanisms**.

## Current priority

1. Preserve the frozen five-model development candidate; do not reopen nearby weight tuning.
2. Build the actual immutable competition submission outside Kaggle and spend one official score.
3. Keep Kaggle out of development/preflight work; AWS remains canonical for modeling.
4. If the official score transfers, preserve it and continue from the stronger base.
5. If it does not transfer, prioritize uncertainty-selected pseudo-supervision or unblock DML only after exact source parity is recovered.
6. Do not interpret DML implementation failures as evidence against the method.

## Validation and publication boundary

AWS remains canonical for model training, checkpoints, row-level predictions, and private logs. GitHub publishes aggregate metrics/configuration, implementation source, tests, attribution, and executed notebooks.

The [five-model frontier report](FIVE_MODEL_FRONTIER.md) and
[`reports/checkpoints/five_model_frontier.json`](../reports/checkpoints/five_model_frontier.json)
record the latest completed stage. The earlier [Qwen3-14B frontier report](QWEN14B_FRONTIER.md)
preserves the capacity experiment that motivated the diversity-first direction.
