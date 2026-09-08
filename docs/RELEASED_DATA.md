# Released competition data: protected research boundary

## Decision recorded before reading released targets

The competition host [released the former private test and solution files](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/641107). [Dataset version 1](https://www.kaggle.com/datasets/sorenj/jigsaw-agile-community-rules-classification/data) is published as CC0. It contains 54,059 evaluation rows and six policies. Its training and ten-row preview files exactly match the project's original inputs. The release is post-competition data; using its labels cannot establish a competition-time result.

This boundary was chosen from source metadata and text-only overlap counts. Released target values and model predictions were not inspected to choose it. `configs/released.json` pins the archive, all six member hashes, rule mapping, split policy and organizer metric attachment. Original experiment artifacts remain historical evidence on two policies.

| Role | Assignment | Permitted use |
| --- | --- | --- |
| Historical training | Original 2,029 rows | Existing experiments; subsequent research fitting within saved validation boundaries |
| Additional research | Former **Public** rows for advertising, legal advice, medical advice and illegal-activity promotion | Feature generation, training-only screening and grouped/rule-held-out development experiments |
| Reserved confirmation | Every former **Private** row, plus every financial-advice and spoiler row | No feature selection, vocabulary fitting, target encoding, calibration fitting or model tuning |
| Historical exposure exclusions | Reserved bodies already present in original training/preview comments or support examples | Excluded from the strict confirmation estimand; counted separately |
| Boundary exclusions | Research rows whose body or any support example matches an eligible reserved body | Excluded from research; never silently reassigned |

Normalization is Unicode NFKC, whitespace collapse and case folding. The partition uses row identity, policy, organizer Usage metadata and normalized text only. It does not stratify or balance on target values. Research-side purging extends to **all four supplied examples**, not just the comment. Original training/preview exposure is accounted for before reserving an apparently new comment. Complete duplicate groups remain on one side of the research/confirmation boundary.

The confirmation set is a **protocol reserve**, not an access-control guarantee: the source is public, and its solution file is retained for reproducibility. The research loader exports only permitted targets. It has no flag to return confirmation labels. A later evaluator must require a committed final feature/model/calibration manifest and must record the first opening of the reserve. That evaluator has not been implemented or run in this milestone.

## Executed preparation

The boundary was committed as `9b4d66448998da818b71e4e697aa8e937c9222f0` before research target materialization. Preparation run `a615ecc75c04e6a62ee1` verifies the pinned archive and produces 9,106 research rows, 43,576 reserved rows, 1,323 research-side overlap exclusions and 54 historical-exposure exclusions. The reserve retains six policies; financial advice and spoilers contribute no research labels. A complete replay reuses the checksummed assignment/export stage without reinterpreting targets.

The 11,135 combined development rows contain 544 repeated normalized body/policy rows and 39 groups with conflicting labels: 25 within the same subreddit and 14 across subreddits. These are observed data-quality findings, not repaired labels. Promotion research rows include 2,087 positives and 124 negatives, so accuracy alone would be misleading. The next study must keep duplicates grouped and report conflict and class-imbalance sensitivity.

`scripts/prepare_released_data.py` and `jigsaw_rules.released.load_research` verify source, protocol, implementation, assignment/export checksums and permitted row IDs. Automated tests change protected targets to nonnumeric sentinels and verify that they are neither interpreted nor returned; corrupt exports, changed protocols, ambiguous IDs and any support-field overlap fail closed. The new Plotly/static boundary figure and aggregate reports are rendered in canonical notebook `02`. No new model is fitted during preparation.

Exact isolation does not establish paraphrase, author or shared-origin independence. The schema has no timestamps or conversation IDs. Before confirmation, the existing approximate-copy audit must be extended to this boundary, with any exclusions chosen without target values. The two reserved policy types supply a stronger unseen-policy test; they do not support an unrestricted claim about all moderation rules.

## Next research experiment

1. Combine retained research rows with the original development data, auditing repeated normalized body/policy pairs and contradictory labels. Preserve the provenance of every row.
2. Freeze grouped familiar-policy and four leave-one-policy-out folds. Fit vocabulary, IDF, scaling, screening, target encodings and calibration components inside their allowed training partitions.
3. Repeat the major-family screen and matched additions/removals on the expanded cohort; a family that failed in tiny two-policy folds is not automatically dismissed at the larger sample size. Recheck the historical lexical reference, full character features, compact semantic summaries and normalized support-centroid representation on the identical folds. Include the broad combined-family negative control and comment/rule/support ablations. Keep classifier settings fixed.
4. Attribute improvements using paired rule-macro AUC, per-policy behavior, group ablations, simultaneous uncertainty and support/example sensitivity. Preserve failed families and measured inference cost.
5. Investigate a remaining encoder or contextual hypothesis only when a documented development-set error pattern justifies it. Do not use reserved scores to choose the hypothesis.

The executable study specification and stopping rule must be committed before running these new comparisons. This document locks the **data boundary**, not a claim that the new feature study has executed. The present milestone creates no new performance result and does not close notebook `02`.

## Finishing contract

Feature research ends when the major plausible families have measured outcomes on the expanded development set, a compact representation survives removals and robustness checks, and one selected representation is frozen for confirmation. Confirmation can reject the representation; a negative result must not trigger an undocumented search on the same reserve.

After that gate: fit the frozen production pipeline, evaluate calibration and selective human-review behavior, verify that offline inference consumes the selected artifact hashes, and publish a small demonstrator plus model/data cards. An employer should reach the problem, measured feature contribution, limitations and reproducible inference in a five-minute README/notebook path. Large models and additional features are optional; defensible evidence and a usable result are required.
