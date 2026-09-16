# Round 10: Lexical–semantic agreement

## Evidence and hypothesis
Round 8 primary `dual` had mean policy AUC 0.715516 versus raw Qwen 0.719893 and basic geometry 0.723068. Round 9 primary `conditioned_all` reached 0.728802, but legal-advice AUC 0.754134 remained below Qwen's 0.760533. Round 9 `context_evidence` reached 0.729257 and `uniform_all` 0.728910. All these numbers are descriptive, repeatedly inspected development results; neither primary was promoted.

Does agreement between exact-word/character overlap and frozen semantic neighbors add evidence beyond the existing context-conditioned anchor?

The Round 9 `context_evidence` readout is the adaptively selected exploratory anchor, not the accepted Kaggle model. The same anchor design (saved adapted answer margin + frozen basic geometry + 24 context-evidence columns) is retained in all six new candidates. We reconstruct the saved anchor design and reproduce its predictions before fitting. Raw Qwen, answer-only, basic geometry, context evidence and uniform context weighting provide **ten cached readouts** across two policies; no old model is refitted.

## Fixed design
Primary: `crossmodal_all`. Candidate configurations: word_agreement, character_agreement, crossmodal_all, alignment_null, label_null, lexical_only.
New features in the primary: 48 (24 in each family). Exactly 12 new logistic-regression fits: six variants times two policy cohorts. Classifier C=1, liblinear, maximum 2000 iterations, seed 2025 and prior repetition weights are unchanged. Predictive value is not inferred from column count.

Ablations measure full-minus-family contributions. Mechanism controls: alignment_null, label_null, lexical_only. No hyperparameter tuning, new encoding, neural forward pass, downloads, external labels, or ensemble weight search. The two new rounds are specified before either is run and do not import or depend on one another's output.

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
https://aclanthology.org/2021.acl-long.316/

The retrieval paper reports complementarity of sparse/dense methods in question answering. This package does not perform query generation or import its data. It uses reference-only word unigrams/bigrams and character 3–5-grams (6000-column cap each). Each modality supplies eight measures for permitted references, eight for violating references and eight signed margins: nearest lexical cosine, top-five lexical mean, semantic similarity of lexical neighbors, lexical similarity of semantic neighbors, top-five intersection fraction, top-five joint positive affinity, semantic-affinity-weighted lexical overlap, and within-class alignment covariance. Together the two families yield 48 columns.

The alignment-null control permutes semantic reference columns WITHIN each class; this preserves the separate lexical and semantic class distributions but breaks their example-by-example pairing. A label-null control shuffles only supplied reference labels, preserving class counts. Lexical-only ablation zeros cross-modal columns. Equal allocated width is not a claim of identical effective regularization or equal row norms. Ties use canonical text order. Empty vocabulary maps to zeros and is reported. Vocabularies/IDF are fitted only on inner references, so 16 small vectorizer fits occur across both cohorts; they are feature fitting, not additional classifier fits.
