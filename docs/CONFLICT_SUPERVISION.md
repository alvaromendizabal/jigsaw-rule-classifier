# Conflict-aware supervision audit and ablation

## Research question

The current support-adaptation protocol drops `(rule, body)` pairs whose available training/support labels conflict. Strong Jigsaw competition solutions repeatedly treated label noise, duplicate collapse, and soft targets as important. This milestone tested whether those conflicts are large and structured enough in our exact frozen training protocol to justify changing supervision before any new GPU training.

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

The legal-advice fold contains materially more conflicting supervision than advertising and includes one exact-tie conflict pair. That made conflict-aware supervision worth a bounded CPU ablation, but not GPU training by itself.

## CPU ablation

The ablation held the representation and ranker fixed across all supervision treatments:

- word TF-IDF: 1-2 grams;
- character TF-IDF: 3-5 grams;
- Ridge ranker;
- validation grouped by exact rule string so a rule never appears in both train and validation inside one fold.

Compared treatments:

1. `drop_conflicts` — current behavior;
2. `majority` — duplicate labels collapsed to hard majority targets, exact ties dropped;
3. `soft` — duplicate labels collapsed to empirical positive rates.

### Result

| Regime | Mean unseen-rule AUC | Delta vs current |
| --- | ---: | ---: |
| `drop_conflicts` | **0.569899** | baseline |
| `soft` | 0.569537 | **-0.000362** |
| `majority` | 0.568688 | **-0.001211** |

Policy-specific behavior matters more than the tiny aggregate difference:

- **Legal-advice holdout:** current `0.508565`, majority `0.506143`, soft `0.505825`. Both conflict-aware treatments are worse.
- **Advertising holdout:** current `0.631233`, majority `0.631233`, soft `0.633249`. Soft targets help this fold by roughly `+0.0020`, but that does not offset the legal-advice regression.

The SageMaker worker itself completed the statistical work in **4.75 seconds** on one `ml.m5.large`; the processing job stayed well inside its 300-second hard cap.

## Methodological caveat

The frozen competition training snapshot contains only **two unique rules**, so unseen-rule validation can produce only two folds. The original predeclared requirement of three fold wins was therefore unreachable. We do **not** relax that gate after seeing results. The stop decision is independent of it because both candidate mean deltas are negative and the legal-advice holdout regresses.

Future promotion gates should be expressed as a fraction of available independent rule folds when the number of rules is not known in advance.

## Decision

**Stop conflict-aware majority/soft supervision before GPU training.**

- No adapter retraining is authorized from this family.
- No parameter sweep is justified.
- No retry with the same label treatments is justified.
- The current conflict-drop supervision remains the retained protocol for now.

## Next research family

Feature/representation research remains open. The next higher-value family is **rule-polarity contrastive geometry and cross-rule hard-negative construction**, because leading solutions used explicit positive/negative rule representations and contrastive/triplet structure, while our latest two retrieval/supervision changes have failed to improve the target policy reliably.

The next experiment should again begin with a CPU-only audit or reuse existing cached embeddings before any GPU adaptation run.
