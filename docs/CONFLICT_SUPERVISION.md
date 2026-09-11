# Conflict-aware supervision audit

## Research question

The current support-adaptation protocol drops `(rule, body)` pairs whose available training/support labels conflict. Strong Jigsaw competition solutions repeatedly treated label noise, duplicate collapse, and soft targets as important. This milestone asks whether those conflicts are large and structured enough in our exact frozen training protocol to justify a bounded ablation before any new GPU training.

## Data boundary

- Original competition training snapshot only: 2,029 rows.
- Immutable training SHA256: `83948d06a1e4b16421b738add60ef489cf1d44a2349ca711958fbb41c6207a0a`.
- No query labels, saved prediction arrays, hidden targets, or model inference were used.
- The audit first had to reconstruct the previously published support-adaptation conflict counts exactly. Failure to reproduce them would have stopped the milestone.

## Verified audit result

The reconstruction passed for both new-rule folds.

| Policy | Conflicting pairs | Conflicting occurrences | Conflict occurrence fraction | Majority-recoverable pairs | Nontrivial soft targets |
| --- | ---: | ---: | ---: | ---: | ---: |
| Advertising | 2 | 79 | 0.598% | 2 | 1 pair in `[0.2, 0.8]` |
| Legal advice | 10 | 339 | 2.573% | 9 | 2 pairs in `[0.2, 0.8]`, including 1 near `[0.4, 0.6]` |

The legal-advice fold contains materially more conflicting supervision than advertising and includes one exact-tie conflict pair. This is relevant because legal-advice was also the policy that regressed in the earlier retrieved-example experiment.

## Decision

The audit **authorizes only a bounded CPU ablation** comparing three supervision treatments with the same representation and unseen-rule validation:

1. `drop_conflicts` — current behavior;
2. `majority` — collapse duplicate labels to hard majority targets, dropping exact ties;
3. `soft` — collapse duplicate labels to their empirical positive rate.

The CPU promotion gate is predeclared: a candidate must improve mean grouped unseen-rule AUC by at least **+0.002** over `drop_conflicts` and win on at least **3 folds** before any single-policy GPU adaptation ablation is considered.

**GPU training remains unauthorized at this stage.**

## Compute discipline

The real ablation uses a single `ml.m5.large` SageMaker Processing job with a 300-second hard runtime cap. It uses word+character TF-IDF plus the same Ridge ranker for all three regimes, so only supervision treatment changes. No model weights or embeddings are downloaded.
