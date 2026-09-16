# Round 5 — Occurrence-level lexical scope

## Measured basis and limits

Round 4 run `e5628cf8367060756148` completed 12 fits and reused ten controls. Its rule-weighted word-plus-character candidate achieved advertising AUC 0.68869403, legal-advice AUC 0.71579813 and mean policy AUC 0.70224608. Relative to `condition_lexical`, the mean gain was 0.00772762 with conditional simultaneous interval [-0.00193118, 0.01738642]. Relative to shuffled weights, the gain was 0.00713682 with interval [-0.00252198, 0.01679562]. The recorded decision is DO_NOT_PROMOTE_PRIMARY. These are exploratory CPU results, not a Kaggle score or an accepted-Qwen comparison.

The current word feature represents a term without preserving whether each occurrence is quoted, code, or near a negation cue. Earlier rounds tested coarse scope counters and hand-selected actor cues; this round attaches context to the actual training-derived word unigrams and bigrams. It is not another list of generic regex predictors. It tests a specific missing representation while preserving the classifier.

## Hypothesis

Keeping separate effects for the same lexical term in different observable contexts may improve ranking. Scope markers never assign a moderation label. `You should not sue` remains a possible recommendation; negation is not a label inversion. The registered primary is the combined attribution-plus-negation representation, not whichever secondary candidate looks best.

## Feature families and controls

1. Attribution: authored text, explicit double/curly/block quotation, inline/fenced code, and cross-context or sentence-boundary bigrams.
2. Approximate negation: unmarked occurrences, occurrences in a fixed four-token cue window, and boundary bigrams. Punctuation, adversatives, and changes of quotation/code region stop the window. `not only`, `not just`, and `no wonder` are excluded cues. Single-quoted prose, implicit reported speech, nesting and syntactic negation are not fully parsed.

Generate ngrams from original token adjacency before classifying context. Do not remove a quotation and create a fake bigram between words originally separated by it. Reconstruct body-only TF-IDF with the existing eligible-training word vocabulary and IDF. Reuse the Round 4 training-derived rule-evidence weights; preserve each body-vector norm. The inherited anchor still contains both word and character features and the existing dense features.

For original term value v and nonnegative occurrence counts c_k in K contexts, split the term as:

    feature_k = v * sqrt(c_k / sum(c)).

The sum of squared context values equals v^2 separately for every original term, not just for the whole row. The collapsed control is [v, 0, ..., 0], with the same allocated width and squared term magnitude. Apply the existing training-only policy map to both arms; unknown rules receive zero extra rule-specific columns. Different contexts intentionally add capacity, and the sparse nonzero counts need not be equal. No claim of equivalent hypothesis classes is made.

There are four attribution channels and three negation channels. With two policies and a 12,000-column inherited word budget, the combined representation adds at most 168,000 sparse columns, including inactive zeros. Actual allocated and active dimensions and norms are reported. This bound is not a count of useful retained features. There is no new vocabulary search or validation-led feature selection.

## Experiment

Six candidates: collapsed attribution, scoped attribution, collapsed negation, scoped negation, collapsed both and scoped both. Two policy cohorts yield 12 new fits. Reuse 12 saved fits: lexical, behavior, actor, shared lexical copy, rule-conditioned lexical and Round 4 rule-weighted lexical, each on two cohorts. Reconstruct and compare every control's predictions before allowing a new fit. Do not rerun a missing or corrupt reference.

Keep the logistic-regression specification, C, training weights, upstream screening budgets, query protocol and data unchanged. Data are original training labels and legally supplied support labels only. The protocol globally excludes support-known query bodies and purges query text from eligible training. Expect 234 advertising and 647 legal-advice query IDs matching the prior run exactly. Supplied labels for the query rule exist: this is support-adapted evaluation, not zero-shot transfer.

## Fixed exploratory gate

`scope_both` must beat BOTH `rule_both` and `copy_both` by at least 0.003 mean policy AUC, have simultaneous 95% lower bounds above zero for both comparisons, and avoid any policy regression relative to either reference. Within-policy-ranked pooled AUC must not decline versus the Round 4 anchor. The 500-draw paired normalized-comment-group bootstrap covers 12 planned comparisons. It does not account for all adaptive choices, training randomness, or generalization to new policies. A pass is eligibility for a further validation only, never automatic GPU permission or model promotion.

## Bounded execution and preservation

Scientific process: 240-second hard POSIX timer; notebook-parent watchdog: 300 seconds; launcher: 540-second internal budget and provided terminal wrapper: 600 seconds plus 10 seconds for forced termination. Fifteen-second heartbeats, fit counters and independent private candidate checkpoints. Preserve all previous sources/results/checkpoints. A byte-level source/config/environment/raw-data contract identifies the run. Corrupted or changed completed work causes a stop, not silent regeneration. Neither timeout stops the SageMaker application.

Public export includes explicit source/config/tests/docs, executed notebook, aggregate reports and checksums. It excludes raw comments, row targets, private predictions, model arrays, vocabulary, credentials and the Python environment. Source files are installed locally only; no GitHub or AWS API call is included. The prior uncommitted rounds still need a verified publication checkpoint.

## Research references and what is actually borrowed

- Councill, McDonald and Velikovich (2010), *What's great and what's not: learning to classify the scope of negation for improved sentiment analysis*, https://aclanthology.org/W10-3110/ . Motivates representing scope rather than ignoring it; this four-token heuristic is not their learned model and sentiment is not rule violation.
- Kiritchenko and Mohammad (2017), *The Effect of Negators, Modals, and Degree Adverbs on Sentiment Composition*, https://arxiv.org/abs/1712.01794 . Supports caution about fixed label-inversion rules; we learn coefficients instead of flipping a label.
- Ribeiro et al. (2020), *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*, https://aclanthology.org/2020.acl-main.442/ . Motivates authored tests of formatting, quotation, scope boundaries and invariance. Test cases check representation mechanics, not expert-adjudicated moderation outcomes.
- Clarke et al. (2023), *Rule By Example*, https://aclanthology.org/2023.acl-long.22/ . Exemplar-based contrastive representation learning remains a separate high-value avenue. This CPU representation study does not train an encoder or reproduce that work.

## Next decision

A single failed scoped representation does not exhaust linguistic or feature research. Do not immediately launch another undirected lexical sweep. Combine the accumulated evidence into a publication checkpoint and establish which validated feature signals add information relative to cached accepted-model outputs on the same eligible cohort before expensive inference. The recorded leaderboard goal is a target, not a guarantee.
