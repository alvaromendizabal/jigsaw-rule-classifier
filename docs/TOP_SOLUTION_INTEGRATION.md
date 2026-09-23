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
| Multi-model weighted ensemble | In progress | Next frontier milestone |
| Uncertainty-selected pseudo-labeling | Not implemented | High-value capability gap |
| Deep Mutual Learning | Not implemented | High-value capability gap |
| Task-trained contrastive BGE route | Not implemented | Diversity route, lower priority than ensemble/DML |

## What the latest 14B result changes

The project no longer has a simple “larger backbone” implementation gap. Qwen3-14B has now been independently trained and evaluated in AWS under the fixed support-adaptation protocol.

The result is nuanced:

- standalone 14B is weaker than the retained 4B development reference;
- a fixed 50/50 4B+14B rank blend improves both observed policies;
- the blend gain is larger than prior 4B+8B and 4B+Phi development gains;
- grouped-bootstrap uncertainty still crosses zero.

That shifts the bottleneck from **backbone capacity** to **diversity-aware ensemble selection and stronger training mechanisms**.

## Current priority

1. Align and checksum the preserved OOF predictions for 4B, 8B, 14B, and Phi.
2. Select ensemble weights with normalized-body group cross-fitting and leave-one-policy-out checks.
3. Do not tune weights on Kaggle public/private scores.
4. If the meta-ensemble transfers without policy regression, freeze one candidate and spend one official submission.
5. If not, move to a structurally new capability—DML or uncertainty-selected pseudo-labeling—rather than another cosmetic prompt/feature tweak.

## Validation and publication boundary

AWS remains canonical for model training, checkpoints, row-level predictions, and private logs. GitHub publishes aggregate metrics/configuration, implementation source, tests, attribution, and executed notebooks.

The [Qwen3-14B frontier report](QWEN14B_FRONTIER.md) and
[`reports/checkpoints/qwen14b_frontier.json`](../reports/checkpoints/qwen14b_frontier.json)
record the latest completed stage.
