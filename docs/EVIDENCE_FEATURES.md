# Round 4 — Rule-specific lexical evidence

## Observed result, not a promoted model

Round 3 (`5927fe5eb15c3cf2d8aa`) completed 12 new fits and reused six controls. The rule-conditioned lexical representation reached policy-macro AUC 0.69451846, versus 0.68609926 for the actor-action anchor and 0.68481469 for its shared-copy control. Its per-policy gains against the actor anchor were +0.01082090 advertising and +0.00601749 legal advice. Its simultaneous interval for the macro delta was [-0.01546136, +0.03229975]. The registered decision remains `DO_NOT_PROMOTE_PRIMARY`. That decision is preserved, not overturned by treating an exploratory point estimate as a win.

The policy-specific dense behavioral block did not improve the shared-copy control. We therefore investigate lexical weighting, rather than adding another generic indicator list. All scores in this document refer to a repeatedly examined local cohort, not to the accepted Qwen model or a Kaggle submission.

## Research basis and boundaries

- Wang and Manning (ACL 2012), **Baselines and Bigrams: Simple, Good Sentiment and Topic Classification**, motivates learning diagnostic lexical weights from class-conditional log-count ratios: https://aclanthology.org/P12-2018/ . This study keeps logistic regression, uses bounded magnitude weights and family-wise row normalization, and is not an exact NB-SVM reproduction.
- Daume III (ACL 2007), **Frustratingly Easy Domain Adaptation**, motivates separating shared and rule-specific feature effects: https://aclanthology.org/P07-1033/ . Those blocks already exist in Round 3; this round changes their relative lexical weights.
- Clarke et al. (ACL 2023), **Rule By Example**, motivates rule-grounded learned representations: https://aclanthology.org/2023.acl-long.22/ . This small sparse study does not train a contrastive encoder and does not exhaust that avenue.

Historical project documentation already mentions a generic training-only NB weighting sensitivity. Its existence is not ignored. The question here is different: bounded, rule-specific evidence weighting of an already verified support-adapted policy block, with shrinkage, exact prediction parity of cached controls, equal row norms, and a matched weight-permutation control. No claim is made that the NB concept itself is new or previously unexplored.

## Fixed hypothesis

TF-IDF measures corpus rarity. It does not directly represent whether a token distinguishes explicitly labeled violating and permitted examples under a particular rule. Can the latter information improve the existing policy-specific feature block, without more columns or a changed classifier?

Use only eligible deduplicated training pairs produced by the existing historical study builder. Query bodies are purged from every training source; query targets never enter feature construction or fitting. Query-policy support labels are permitted inputs, so this is support adaptation, not zero-shot or label-free target-policy validation.

## Exact feature transform

1. Rebuild the same word/character vocabularies and TF-IDF values. Verify lexical-block equality and all ten saved control predictions before new model fitting.
2. Count each feature's document incidence, not repeated token counts. Use the existing training-repeat weights when summing class counts, matching the classifier's permitted training distribution. Each normalized rule/body pair appears only once in the canonical table.
3. Within each family, compute add-one-smoothed class distributions and their log ratio. Fit both global and policy-specific ratios using training only.
4. Shrink policy ratios toward the global ratio: `lambda = min(unique positive pairs, unique negative pairs) / (min(...) + 64)`. Repetition weights do not falsely increase this reliability count. Missing-class policies fall back to global evidence.
5. Set coordinate weights to `1 + min(abs(log_ratio), 3)`, so weights stay in [1,4]. The sign itself is not a new feature advantage for an unconstrained linear classifier; it can be absorbed into its coefficients. This experiment tests relative magnitude weighting.
6. Reweight only the additional policy-specific lexical block. Keep the shared lexical block and the previously screened behavior/actor features unchanged.
7. Renormalize every word row and every character row to its original family norm. Width, sparsity and full design-row norms must match the Round 3 anchor. This controls a uniform scale increase, not the intended per-coordinate change in regularization/representation.

New feature dimensions: **zero**. Existing thousands of word/character coordinates acquire tested rule-specific evidence weights. A column count is not a quality metric.

## Six fixed candidates and controls

The anchor is the unpromoted Round 3 `condition_lexical` result. Reuse ten old fits: lexical, behavior, actor, shared lexical-copy and conditioned lexical anchors for both policy cohorts. No old model is refitted.

- `global_both`: global evidence weights in both lexical families.
- `rule_words`: shrunk policy-specific word weights; characters unchanged.
- `rule_chars`: shrunk policy-specific character weights; words unchanged.
- `rule_both`: both policy-specific families; **registered primary**.
- `permuted_both`: the exact same per-rule weight multiset, deterministically reassigned to feature identities within each family. Row norms are still preserved. This control tests whether feature-specific association, rather than arbitrary reweighting, explains gains. One fixed permutation is a diagnostic, not a full permutation significance test.
- `unshrunk_both`: local ratios without global shrinkage. This isolates the backoff choice; it is not a tuning sweep.

Twelve new CPU fits, ten reused reference fits, two original-data cohorts (234 advertising and 647 legal-advice comments). Alpha, weight cap, shrinkage and permutation seed are fixed in configuration before real-data execution.

## Evidence and decision

The primary must beat BOTH the conditioned lexical anchor and the permuted-weight control by at least +0.003 policy-macro AUC, have positive simultaneous lower bounds, and avoid either policy declining against either comparator. Local ranked-pooled AUC cannot decrease versus the anchor. No secondary candidate can replace the primary after inspection. Passing is eligibility for further validation only, not automatic GPU authorization.

Five hundred paired normalized-comment-group bootstrap draws produce a centered maximum-deviation 95% band across 12 planned contrasts. These intervals condition on fixed predictions. They do not include model-refitting uncertainty, all prior adaptive experiments or uncertainty over future rules. Both novel query identities are checked against the prior report. Raw pooled, within-rule-ranked pooled, per-policy AUC, Brier and log loss are reported separately. None is a new hidden Kaggle score.

The eight Plotly figures show AUC, paired intervals, global/local/permutation/shrinkage contrasts, word/character ablations, training-weight quartiles, training-support reliability, cross-fold highest-weight feature overlap and probability diagnostics. Weight stability is descriptive, not evidence that stable features predict well. Raw token names, individual comments, row labels and prediction arrays stay private.

## Recovery, runtime and publication

Use the existing environment and original train hash. Every prior source map, environment record and completed candidate marker is checked. Each new candidate saves predictions and coefficients atomically, followed by its hash marker. Interruption reuses completed candidates. Corruption stops execution; changed source/configuration must not silently replace a completed public result.

The worker has a 240-second POSIX hard timer, the parent a 300-second watchdog, and the helper a 540-second internal budget with 15-second heartbeats. The terminal wrapper has a 600-second limit. Time limits do not stop the SageMaker app. No package/model/data download, AWS API call, Git write, background cloud job or submission occurs.

New source, notebook and aggregate results initially remain local and uncommitted. This helper does not claim AWS/GitHub equality. Public export contains explicit allowlisted files and checksums only. Preserve the original raw CSVs, all completed rounds, the old checkpoint folder and Python environment.

## Research remains open

This is one bounded lexical-representation question. It does not establish that a sparse surrogate's improvement also improves the accepted support-adapted Qwen model. Before escalating to another expensive inference job, compare any candidate's incremental information against eligible accepted-model predictions on the exact same query cohort, with separate data availability and hash checks. Learned exemplar/contrastive representations and rule-grounded semantic interaction features remain distinct avenues. No leaderboard guarantee or claim of exhausted feature research is made.
