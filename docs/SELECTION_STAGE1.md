# Selection Stage 1 — nested readout selection

Feature engineering is paused. This stage selects among the **already-saved feature/model readout channels** from the completed 49-configuration combination screen. It does not invent a feature family, download a model, or submit to Kaggle.

## Metric discipline

The task is binary classification and AUC is the primary development metric. Macro AUC is used consistently for ranking readouts and choosing readout count. Brier score, probability RMSE, log loss, pooled AUC, and per-policy AUC are secondary diagnostics. For binary probabilities, RMSE is the square root of Brier score, so they are not double-counted as independent objectives.

The count rule first applies the one-standard-error rule to macro AUC. Among those AUC-eligible counts, it applies a one-standard-error Brier guard and chooses the smallest surviving count. This favors simpler representations while refusing a material probability-quality deterioration.

## SHAP and permutation importance

The stage fits a linear logistic meta-model to saved readouts. For an interventional linear explanation with an independent background, exact SHAP values are `coef_j * (x_j - mean_j)`. SHAP is calculated inside inner folds only. Because model readouts can be strongly correlated, SHAP is not used alone: it is paired with held-out **macro-AUC** permutation importance, exact-duplicate removal, 0.98 Spearman-correlation clustering, and outer-fold selection-frequency stability.

## Validation protocol

- Five grouped outer folds estimate the complete selection procedure.
- Four grouped inner folds rank channels and choose the readout count.
- The global normalized-comment group IDs created by the combination screen are preserved, so the same normalized body cannot cross train/validation boundaries even if it occurs under both policy cohorts.
- Policy + class are stratified jointly.
- Standardization, correlation clustering, SHAP background means, permutation importance, and count selection are learned only from the outer-training partition.
- Outer-fold results are never used to choose that fold's readouts or readout count.
- Completed outer folds are checkpointed and reused after interruption.

This reduces meta-selection optimism, but it does **not** turn the repeatedly inspected 881-comment development cohort into a fresh holdout. Historical feature invention already adapted to these policies.

## Scope

This stage is readout/family-level selection. It is not a claim that raw TF-IDF token columns have been individually selected. Model-specific raw-column selection belongs inside Stage 2 model pipelines so it can be fit strictly inside their training folds.

## What comes next

Stage 2 will compare a fixed, bounded model set on the Stage-1 shortlist with nested grouped CV and model-specific feature selection. Stage 3 will compare simple averaging, rank averaging, constrained non-negative blending, and bounded greedy ensemble selection using out-of-fold predictions only. No model or ensemble is selected before Stage-1 evidence is reviewed.
