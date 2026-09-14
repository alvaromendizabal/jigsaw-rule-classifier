# Round 11: Within-class support subtypes

## Evidence and hypothesis
Round 8 primary `dual` had mean policy AUC 0.715516 versus raw Qwen 0.719893 and basic geometry 0.723068. Round 9 primary `conditioned_all` reached 0.728802, but legal-advice AUC 0.754134 remained below Qwen's 0.760533. Round 9 `context_evidence` reached 0.729257 and `uniform_all` 0.728910. All these numbers are descriptive, repeatedly inspected development results; neither primary was promoted.

Do multiple bounded, reference-only subtypes within each supplied class help beyond a single mean and matched random class partitions?

The Round 9 `context_evidence` readout is the adaptively selected exploratory anchor, not the accepted Kaggle model. The same anchor design (saved adapted answer margin + frozen basic geometry + 24 context-evidence columns) is retained in all six new candidates. We reconstruct the saved anchor design and reproduce its predictions before fitting. Raw Qwen, answer-only, basic geometry, context evidence and uniform context weighting provide **ten cached readouts** across two policies; no old model is refitted.

## Fixed design
Primary: `prototype_all`. Candidate configurations: prototype_location, prototype_coverage, prototype_all, random_partition_all, single_center_all, label_null_all.
New features in the primary: 36 (18 in each family). Exactly 12 new logistic-regression fits: six variants times two policy cohorts. Classifier C=1, liblinear, maximum 2000 iterations, seed 2025 and prior repetition weights are unchanged. Predictive value is not inferred from column count.

Ablations measure full-minus-family contributions. Mechanism controls: random_partition_all, single_center_all, label_null_all. No hyperparameter tuning, new encoding, neural forward pass, downloads, external labels, or ensemble weight search. The two new rounds are specified before either is run and do not import or depend on one another's output.

## Availability and validation
Only original competition training labels and explicitly supplied support labels under the same rule are allowed. Existing verified plan and arrays map original training-list order to saved vectors; queries are 234 advertising and 647 legal-advice comments. This is support adaptation, not target-policy-label-free zero-shot evaluation. All private arrays remain local.

Three text-group folds construct the new training features. A held group's body and label cannot enter its reference vocabulary, prototype, or similarity statistics. Outer-query features use the eligible full reference pool. Reference ordering is canonicalized by normalized text; queries never become references for one another. Query label arrays are joined only when computing metrics, not passed to feature constructors. Normalized duplicates and query/reference overlap are rejected.

**Important inherited limitation:** the adapted answer scores for support-training rows were generated with an encoder trained on those support labels. Cross-fitting the new features does not turn the encoder or those scores into out-of-fold predictions. The smaller inner reference pools also differ from the full inference pool. Therefore a positive result here is hypothesis-screening evidence, not an unbiased generalization estimate or permission for deployment. An honest independent evaluation requires a separate cross-fitted encoder or untouched eligible data; this package does not invent either.

## Evidence threshold
Primary comparisons: raw Qwen, frozen basic geometry, Round 9 context anchor, uniform-weight context control, and all three new mechanism controls. Require +0.003 macro AUC against EACH comparator, positive simultaneous lower confidence bounds, neither policy regressing against any comparator, and no decline in within-policy-ranked pooled AUC versus raw Qwen. Secondary winners do not replace the registered primary. The 500 shared normalized-comment-group bootstrap draws produce a centered maximum-deviation 95% band over this round's planned contrasts. This does not correct the whole adaptive project history or estimate uncertainty across future rules. Raw pooled and ranked pooled AUC, Brier and log loss are shown separately. No new Kaggle score is reported.

## Execution and checkpoints
Use the existing Python 3.12 CPU environment; no package installation. Hash-pin both returned Round 8/9 reports, prior source, original raw-data hash and cached results. Check reference-design parity for all ten readouts before the first new fit. Preserve each feature bank and candidate prediction with atomic payload/marker writes. Completed replay checks markers and hashes and performs zero fits. Missing/corrupt prior data stops, never silently regenerates. A 240-second POSIX timer, 260-second parent watchdog, 110-second notebook stage and 600-second outer command prevent unlimited runs. Heartbeats are every 15 seconds. These limits do not stop the SageMaker app.

Public return allowlist: new code, tests, config, executed notebook, aggregate reports and receipts. Never include comments, row labels, vocabularies, credentials, prediction arrays or model weights. No automatic Git commits, pushes, S3 writes, or cloud jobs. This round is research, not a declaration that the feature space is exhausted.

## Research motivation, not reproduced results
https://proceedings.mlr.press/v97/allen19b.html

The mixture-prototype paper motivates representing more than one subtype per class. Our fixed-capacity spherical partition is NOT the paper's Bayesian nonparametric method, learned encoder or adaptive infinite mixture. Up to four modes per class (at least four examples per planned mode), farthest-first deterministic initialization and 12 spherical Lloyd iterations are fixed. Empty modes collapse. Reference cluster assignment affects both prototype directions and radii.

Location family: nearest and top-two prototype cosine, occupancy-weighted mean cosine, soft affinity at fixed temperature 0.1, nearest/second-nearest gap, and between-prototype variation, for each class plus signed margins (18). Coverage family: largest and mean radius-adjusted affinity, normalized mode entropy, effective mode count, mass of nearby modes and nearest-mode mass, for each class plus margins (18). Radii are floored at 0.05. Estimated clusters are statistical subtypes, not asserted human-interpretable behaviors.

Random-partition control shuffles the learned cluster assignments within each class, preserving the exact occupancy histogram. Single-center control uses one class prototype and the same 36-feature summary schema. Label-null control shuffles support labels and recomputes the bounded partitions. Width matches but norms/effective capacity need not. There are 32 small class-partition fits across two cohorts, three inner folds plus the outer pool, and the actual/null class partitions; these are feature transformations, not additional classifier fits. Diagnostics expose collapsed mode counts, radius floors and outside-all-mode query fractions.
