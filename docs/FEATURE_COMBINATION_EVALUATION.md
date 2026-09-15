# Frozen feature combination evaluation

## Scope
New feature creation is paused. Summarize all 15 completed rounds and compare 49 fixed configurations (98 new classifier fits) in three bounded batches: 34, 34, and 30 fits. Existing reference readouts and feature banks are reused; missing/corrupt banks fail closed. No new encoder, embeddings, data source, model download, calibration, ensemble search, submission, cloud API or Git write is included.

## Feature union and honest counts
The common anchor has the adapted answer margin, nine frozen geometry features and 24 context-evidence features. Add 16 groups: behavior, scope, rule_alignment, relations, legacy_support, local_density, matched_pairs, conditioned_geometry, actor_support, crossmodal, prototypes, consistency, passages, bm25, windows and lexical. Reference banks from R6–15 are loaded from their completed checksum markers. Raw R1–2 measurements and three-group cross-fitted old support contrasts are recomputed on the same eligible support population, because the older models used a different pooled training population. Existing word/character TF-IDF is refitted only on this population. No new feature family is invented.

R3 separate-policy slopes are represented by the separate per-policy readouts; a constant extra policy indicator would only duplicate a representation. R4 diagnostic weighting and R5 scope encoding are alternative lexical configurations, not all counted together as independent columns. The latter has its corresponding collapsed-copy control. Full union, 16 add-one, 16 leave-one, six fixed pairs, compact combinations, C=0.1 sensitivity and frozen-margin sensitivity are reported. No column total is presented as a gain. Training-constant/exact-duplicate dense columns and unseen sparse columns are pruned using training values only. The retained counts are recorded per model and policy.

## First entry
The original scored lexical Version 2 is documented at reports/checkpoints/kaggle_submission.json and kaggle/reference.ipynb. At most 40,000 word unigram/bigram dimensions plus exactly 8 hand-built similarity features. The original-data vocabulary is counted separately with the same vectorizer settings; no classifier is trained for this count, and that full-data vocabulary is never reused by evaluation.

## Models and controls
All primary contrasts use logistic regression C=1, liblinear, max_iter=2000, random_state=2025 and existing repeat weights. C=0.1 is a named robustness comparison, not tuned from the query scores. Dense scaling, constant filtering and duplicate detection use training values only; sparse TF-IDF retains its family normalization. Cached context and Qwen references are validated against saved coefficient/feature predictions before new fitting. The common anchor is also refitted to expose preprocessing differences. Every coefficient, scale, mask and prediction stays private in per-candidate checkpoints.

## Validation boundary and limitations
This is a fixed exploratory screen on the exact historical 234+647 queries, NOT nested CV, independent confirmation or proof of the best subset. Labels are only joined after features and readouts are built. Query text is purged from training/reference sources. Inherited adapted training answer margins are in-sample; the frozen-margin arms test sensitivity but do not repair the earlier adaptive selection process. All two-policy comparisons remain conditional on the observed policies. Full raw-pooled, mean per-policy and within-policy-ranked pooled AUC are distinct reported metrics. Kaggle states column-averaged AUC with one output column; the aggregate per-policy diagnostic must not be casually substituted for a submission score. Brier/log-loss and policy-specific regressions are shown alongside AUC.

## Multiplicity
One thousand paired normalized-comment-group bootstrap resamples are shared across every available model. Centered max-deviation bands cover all completed contrasts jointly. Intermediate-batch bands are explicitly partial. The final set includes every model versus raw Qwen and context reference, full-minus-family effects, and the scope/copy contrast. These conditional intervals do not account for encoder training, repeated historical feature research or hyperparameter/feature selection. There is no automatic promotion or hidden choice of a secondary winner.

## Proper next validation gate
Before selecting a deployable winner, freeze a shortlist and rebuild every label-dependent reference bank, TF-IDF/evidence transform, screen and scaler inside each inner group split. Outer groups score only the inner-selected configuration. Do not run GridSearchCV on the already cross-fitted full training bank: rows can contain statistics computed from another row's outer validation label. Evaluate frozen base margins or genuinely out-of-fold adapted margins. Report support-only and historical query results separately, and do not call an adaptively reused historical cohort fresh. Confidence intervals after this screen must be labeled exploratory. Independent policies/time periods or a frozen new submission are needed for external confirmation.

## Research basis
- https://scikit-learn.org/stable/common_pitfalls.html — fit preprocessing and feature selection on training only.
- https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html — separate selection from evaluation.
- https://www.jmlr.org/papers/v11/cawley10a.html — selection bias from repeated model/feature search.
- https://www.kaggle.com/competitions/jigsaw-agile-community-rules/overview/abstract — competition scoring description.

## Runtime and provenance
One 240-second hard limit per input-preparation or model batch, 110 seconds per notebook rendering stage, a 540-second launcher budget and 600-second external limit. Candidate fits are individually atomic and checksum-protected. New files only; old reports/notebooks/checkpoints are never rewritten. The only public return data are code, manifests, aggregate metrics and new executed notebooks; no raw comments, row-level labels, arrays, vocabularies or secrets. No project accounts are modified by this package.
