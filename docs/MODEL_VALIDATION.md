# From validated features to a final model

This protocol follows the closed [feature gate](FEATURE_COVERAGE.md). It is committed before new calibration fits or scores. The previous feature results motivate one fixed route, with no classifier, encoder or feature-family search.

## Candidate and boundaries

Use the existing `all_transfer` model when the normalized supplied policy is present in the fitted training partition. Use the original frozen centroid otherwise. The familiar model uses words, characters, structure, lexical comparisons, support-token interactions, semantic scalars and training-reference percentiles. **It contains no target encodings, community categories or raw embedding coordinates.** Earlier generic planning mentioned cross-fitting target features if present; the selected representation does not contain them.

Policy membership must be computed separately for every training fold. Unseen-policy predictions bypass the familiar feature pipeline. Transform accepts only target-free inference inputs and frozen vectors with the exact model/input contract. Rebuild the selected feature transformations from saved vocabulary, screens, scaling and rank distributions; verify parity with all three saved familiar-policy models before fitting a new one.

## Calibration needs its own validation layer

The only challenger is a two-parameter positive-slope sigmoid on clipped probability logits. Bounds, weak regularization, optimizer tolerance and selection thresholds are fixed in [the machine-readable protocol](../configs/model_validation.json). An identity mapping remains the fallback. No policy-specific unseen-policy calibrator is fitted.

For familiar policies, reuse the three original outer splits and their verified fitted models. Within each outer training partition, create three new stratified normalized-body-group splits and purge inner validation bodies from every training body/support field. Refit the complete selected feature pipeline inside each inner training partition. Fit the outer calibrator only on these inner OOF predictions and labels. Apply it to the saved outer model's predictions on the untouched outer validation partition. Nine inner classifier fits are required; the three completed outer fits are reused.

Simply splitting the pooled original OOF predictions into calibration folds is insufficient: models producing other folds' predictions may have learned from the current validation labels. The nested construction removes that path. For the label-free frozen centroid, calibration can directly use scores and labels from each purged outer training partition, with its held-out policy excluded.

Report raw versus calibrated AUC, log loss, Brier and per-policy results, together with paired normalized-body bootstrap intervals for log-loss improvement. The 97.5% two-sided intervals conservatively account for the two route-specific decisions. Retain calibration only if all predeclared probability-quality and ranking tolerances pass. These remain development decisions, conditional on four observed policies and earlier adaptive feature selection.

The use of held-out predictions for calibration and a final estimator fitted on all development data follows the general [cross-validated calibration design](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html). The nested outer evaluation is an additional boundary. [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html) motivates simple post-hoc calibration; it does not establish that calibration will transfer to new community policies.

## Final artifacts and subsequent confirmation

After the development decision, fit one familiar pipeline and one lexical reference on all 11,135 development rows. Fit any retained familiar calibrator on the original familiar-policy OOF scores; fit any retained centroid calibrator on its label-free development scores. Bind feature family order, selected columns, model coefficients, normalized policy set, encoder revision/input format, calibration and training IDs into the inference manifest. Full-development fitting can change selected columns; save the actual final catalog and widths rather than copying fold counts.

The 43,576 reserved targets stay unopened throughout this stage. A separate confirmation protocol must bind the completed artifacts, target-blind eligibility audit, candidate/reference, metrics, uncertainty and acceptance thresholds before target access. A rejected confirmation must be preserved. Product promotion, an offline package and a demo remain subsequent gates.
