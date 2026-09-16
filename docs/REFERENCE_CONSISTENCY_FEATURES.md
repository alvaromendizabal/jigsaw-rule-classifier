# Round 12 — Reference consistency and reciprocal evidence

## Actual previous evidence
Round 10 primary: macro AUC 0.716649 (context anchor 0.729257); Round 11 primary: 0.716294. Neither was promoted. These new hypotheses do not repeat the failed lexical/semantic alignment or multi-prototype mechanisms. This is not a claim that all possible versions of those ideas are exhausted.

## Hypothesis
Does consistency with other labeled references make support evidence more useful without deleting rare or disputed examples?

24 ordinary label-neighborhood consistency features and 24 reciprocal-neighbor consistency features.

## Precise construction
Inside each same-rule reference pool, exclude each reference from its own seven-nearest-neighbor graph. Ordinary agreement uses (1 + agreeing neighbor count)/(2 + neighbor count). Reciprocal agreement applies the same smoothing to mutual neighbors. Each becomes a bounded reference quality weight 0.25 + 0.75*agreement. Eight classwise summaries are computed for ordinary and reciprocal quality: affinity change, expected similarity change, active quality, quality dispersion, high-quality mass, effective sample fraction change, top-five quality, and quality/similarity covariance. Permitted, violating and signed differences give 24 per family, 48 total.

Low consistency may identify a correct rare behavior, an inadequate embedding, or a mislabeled example; it does not adjudicate which. No pruning or relabeling occurs. Training sample weights remain unchanged; quality weights only enter new predictor construction.

Controls: permute the complete two-quality vectors within each class (preserving both distributions and their correlation); use label-free reciprocal centrality; shuffle supplied reference labels and recompute quality. Both families are separately ablated. Missing mutual neighbors use the declared smoothing, not an invented extreme weight.

## Same experiment contract for both new rounds
Six candidates: agreement_features, reciprocity_features, consistency_all, quality_null_all, geometry_only_all, label_null_all. Each retains exactly the context_evidence anchor and fixed classifier from the verified prior chain. Ten old readouts are byte/hash checked and design-parity verified; twelve new classifier fits are saved individually. The six new settings are fixed now, not chosen from either companion result. 8 reference-only kNN quality graphs plus 8 shuffled-label statistics; no encoder or classifier fits here.

## Leakage and information boundaries
Reference statistics are learned only from eligible same-rule references. New training features use three normalized-text-group folds: the held group is excluded before any graph or vocabulary statistic. Supplied held-out query labels are used later only for metrics. Query rows do not become references to each other. Reuse the verified 234 advertising + 647 legal-advice identities and exact vector row order. Distinguish new-feature cross-fitting from the inherited in-sample adapted Qwen support-training answer score. No new neural pass, downloads, cloud calls, external data or hidden labels.

## Evidence and gate
Primary consistency_all. Require +0.003 mean policy AUC against each of qwen_raw, frozen_basic, context_evidence, uniform_all, quality_null_all, geometry_only_all, label_null_all, simultaneous lower bounds above zero, no policy regressions, and nondecreasing within-policy-ranked pooled AUC vs raw Qwen. The paired normalized-comment bootstrap is conditional on fixed predictions; it does not correct the entire adaptive research history, dependence from overlapping supports, uncertainty over future policies, or fitting variability. Selecting this follow-up after Round 11 is adaptive. A pass is further-validation eligibility only; no automatic GPU work or submission. Classifier coefficient charts are associations, not causal feature importance.

## Restartability and privacy
Pin completed Round 10/11 results and prior source maps; reuse their accepted cache chain rather than refit them. Verify current input, source and dependency hashes. A corrupt completed stage stops, never silently recomputes. Save new feature banks and each model independently; return aggregate reports, exact source and executed notebook, not raw text, labels by row, fitted vocabulary or arrays. The eight Plotly figures are embedded in the notebook and standalone dashboard.

## Costs and bounded scope
Scientific hard timer 240 seconds; parent watchdog 260; notebook/replay watchdog 110; helper budget 540; outer command 600. Heartbeats every 15 seconds. These do not stop the SageMaker application. No package installs, AWS APIs, Git commits or pushes from this helper. Prior files and user edits are preserved.

## Research attribution
https://research.google/pubs/confident-learning-estimating-uncertainty-in-dataset-labels/
Confident Learning motivates examining label quality; this local neighborhood heuristic is not its class-noise estimation algorithm and makes no label-error claims.
Rule-based exemplar motivation: https://aclanthology.org/2023.acl-long.22/ . No cited dataset is imported.
