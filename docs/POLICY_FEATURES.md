# Round 3 — Policy-specific feature effects

## Evidence leading to this experiment

The verified Round 2 run `b951f663d66c68cb6bca` finished 18 new CPU fits and reused four controls. Its actor–action addition improved mean policy AUC from 0.6798616 to 0.6860993, with advertising unchanged and legal advice +0.0124753. The simultaneous interval for the macro change is [-0.0143594, 0.0268347], so the registered primary was not promoted. Removing legal-topic indicators from the behavior bank dropped macro AUC to 0.6570954. The complete relational set reached 0.6753493 and did not beat the simpler actor representation. These are local exploratory results, not Kaggle scores.

## Hypothesis, not a claim of improvement

A cue may carry a different association under different policies. Explicitly multiplying a feature by a training-known policy indicator permits shared and policy-specific coefficients under the same linear classifier. This tests a representation constraint in the earlier shared model, not whether the accepted Qwen model lacks rule conditioning. The existing text-only vectorizer may already encode a few rule/body boundary ngrams; it does not explicitly condition every feature on the entire rule.

The diagnostic anchor is the already-executed `add_act_roles` model. It is reused because of its measured behavior, not promoted as a replacement for the accepted model. The selection of this next family and anchor is adaptive and must not be called a fresh holdout experiment.

## Representation

For a feature vector x and training-known rule r, preserve the global vector and append a sparse rule block:

    [x_global, 1(r=r1) * x, ..., 1(r=rK) * x].

An unseen rule has an all-zero additional block; the shared model remains available. This is a tested fallback, not evidence of good zero-shot performance. No validation targets enter the rule map, vocabulary, transformations or model fitting. Same-policy support labels are supplied inputs permitted by the existing project protocol. Because the query policy has those support labels, this is new-rule/support adaptation, not target-policy-label-free validation.

Two families are tested:

1. Policy x lexical features: the existing training-only word/character TF-IDF columns. With the current two-policy design and vocabulary caps, at most 64,000 added sparse columns. The measured active width and nonzero count are reported; constant zero columns are not called predictive features.
2. Policy x behavioral/actor features: the existing training-only screened columns (up to 24 behavior and 12 actor columns, hence at most 72 extra columns with two policies). No new vocabulary of generic regex cues is added.

No feature count is a quality claim. A family is retained only if its measured evidence warrants further work.

## A necessary control for feature scaling

Appending a copy of x changes the effective L2 regularization even when the numeric C is unchanged. Each policy-conditioned candidate therefore gets a matched shared-copy arm `[x_global, x, 0, ...]`. Both have exactly the same allocated width, row squared norms and number of nonzero entries for known policies. The copy arm cannot give a feature different slopes under different rules. It controls the added feature scaling, but it does not remove the intended increase in policy-specific capacity.

Six configurations: copy_lexical, condition_lexical, copy_behavior, condition_behavior, copy_both, condition_both. Two policy cohorts yield exactly 12 new LR fits. Six completed anchor fits (lexical, behavior, actor across two cohorts) are verified and reused without retraining. Reconstructed design predictions must match all six before any new candidate fit.

The classifier, C=1, solver, training repetition weights, seed, training vocabulary caps, and earlier dense screening budgets remain fixed. Model hyperparameter tuning and ensembles are outside this round.

## Validation and gate

Reuse the exact original-data study builder, global exclusion of comments already known as support, and all-source query-body purging. Expected novel query counts: advertising 234 and legal advice 647. Verify the exact query identities from Round 2, not merely the row counts. There is no query-target use in fitting, and no fabricated negative labels for other rules.

The primary is fixed before real-data execution as `condition_lexical`. It must beat BOTH the actor anchor and `copy_lexical` by >=0.003 macro AUC, have a simultaneous lower bound >0 for each contrast, avoid a per-policy regression against either comparator, and not decrease ranked-pooled AUC relative to the actor anchor. No secondary winner can replace this primary after seeing results.

500 paired normalized-comment-group bootstrap draws form a centered maximum-deviation simultaneous 95% band over 13 planned contrasts. These conditional intervals do not correct all historical adaptive exploration, model refitting variability or uncertainty over future policies. Report raw pooled AUC and within-policy-ranked pooled AUC separately; neither is a new Kaggle score. Brier and log loss are diagnostics, not calibration claims.

## Reproducibility and failure handling

No downloads or package installs. Reuse the verified raw train hash, both earlier run receipts, both earlier source maps, environment versions and all earlier fit checksums. New candidate identity includes code, configuration, raw-data and dependency identity. Persist each candidate's private predictions and coefficients atomically, followed by its checksum marker. A valid completed candidate is reused. Corrupt completed files stop the run rather than silently regenerating it. Uncommitted new files are preserved by the installer.

The scientific process has a 240-second hard POSIX timer and 15-second heartbeat. A notebook-parent watchdog stops it at 300 seconds. The launcher has a 540-second internal total budget and the supplied terminal command a 600-second outer bound. Timeouts do not stop the SageMaker app itself.

Public export: exact new source/config/tests/notebook, aggregate profiles/results, installation and execution receipts, and earlier aggregate results. Never export raw comments, labels by row, token vocabulary, private prediction arrays, credentials or model files. The eight chart notebook and standalone dashboard contain aggregate information only. Token-level model coefficients stay private; known authored feature names may appear in dense coefficient charts.

## Research sources and their scope

- [Daumé III, ACL 2007, Frustratingly Easy Domain Adaptation](https://aclanthology.org/P07-1033/): shared/domain-specific feature augmentation motivates this implementation. This is not a reproduction of the paper's datasets or results.
- [Park et al., 2021, Detecting Community Sensitive Norm Violations](https://aclanthology.org/2021.findings-emnlp.288/): motivates policy context beyond generic toxicity. No NormVio data are imported.
- [Clarke et al., 2023, Rule By Example](https://aclanthology.org/2023.acl-long.22/): exemplar-based contrastive encoder research remains a distinct future representation avenue. The current sparse interaction study does not train an encoder or reproduce RBE.

Feature research remains open. A CPU pass is eligibility for a further comparison, not a guarantee of improving the 0.91425 recorded accepted private AUC or beating the recorded 0.92930 target. Before another model-inference expense, incremental feature value should be measured relative to an eligible cached accepted-model comparison under a fixed protocol.
