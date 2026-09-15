# Round 14 — Term saturation and reference-length normalization

## Basis and scope
The supplied Round 12 primary had mean policy AUC 0.726598 and Round 13's primary 0.724384, versus the existing context-evidence anchor 0.729257. Both failed their registered gates. They remain preserved. A secondary word-passage result does not replace a registered primary.

## Hypothesis and feature family
Term-frequency saturation and reference-length normalization change existing comparisons, not labels or neural weights. The bounded similarity is normalized for query length; it is a BM25-inspired feature, not raw search-engine BM25. The same count vocabularies are reused for TF-IDF and b=0 controls.

Two families: **word_bm25** and **character_bm25**, each 24 columns, 48 total before the unchanged readout. Six registered configurations: word_bm25, character_bm25, bm25_all, tfidf_all, no_length_all, label_null_all. No count alone proves utility. Reference labels belong to entire comments. The rule is already represented in the same-rule reference selection and inherited anchor.

## Controls, leakage and inference availability
All fitted vocabulary, term statistics, or reference summaries use eligible references only. Three normalized-text-group folds build training features; a training group cannot reference itself. Outer query texts are excluded from all training reference sources; query target labels are joined only for evaluation. Query values cannot change training features. These are inference-available input transformations. References are explicitly supplied labels, never invented cross-rule negative labels.

The frozen reference vector input is hash verified even where new lexical features only use text. The same reference and query ordering reproduces ten old controls before any new classifier fit. The new experiment adds 12 classifier fits; 16 reference-only count vocabularies and TF-IDF transforms. Classifier, regularization, training weights and seed remain fixed. Feature selection and hyperparameters are not tuned on query outcomes.

## Predeclared comparisons and decision
Primary **bm25_all**, not a post-hoc winner. It must improve mean policy AUC by >=0.003 against each of: Qwen, frozen basic geometry, context-evidence, uniform context reference, and all three new controls. Each simultaneous lower confidence bound must exceed zero with no policy regression. Within-policy-ranked pooled AUC must not decline versus Qwen. Include matched family removals. 500 paired comment-group bootstrap draws give conditional simultaneous bands for this round only. Neither a pass nor a secondary win means a Kaggle score or automatic GPU authorization.

## Independence and limitations
Round 15's result is not used. Both are fixed before either run. The 881-comment two-policy dataset is repeatedly inspected; support labels make this support adaptation, not zero-shot transfer. The inherited adapted training-answer margin is in-sample; cross-fitting the new features cannot remove that limitation. No independent confirmation or deployment-readiness claim is made.

## Execution and preservation
CPU-only. New scientific identity includes source, configuration, environment, data and prior result hashes. Atomic feature banks and per-candidate checkpoints are preserved and verified before reuse; corruption causes a stop instead of recomputation. Hard scientific budget 240 seconds, parent watchdog 260 seconds, heartbeat 15 seconds. No package installation, cloud API call, download, Git write or Kaggle submission. Public exports contain source, executed notebook and aggregate evidence; raw comments, vocabularies, per-row labels, predictions and credentials stay private.

## Research source
[Robertson and Zaragoza, The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019). Motivation only; no external dataset is imported and the cited paper's experimental results are not attributed to this implementation.
