# Four-policy feature study

This protocol is committed before inspecting any new model score. It extends the
completed two-policy study; it does not turn that repeatedly inspected study into
independent confirmation. The host's released labels support post-competition
research, not a retrospective leaderboard submission.

## Data and validation

Use only the original 2,029 training rows and the 9,106 rows returned by the
verified research loader: 11,135 rows across advertising, legal advice, medical
advice and promotion of illegal activity. The 43,576 reserved rows, including all
financial-advice and spoiler targets, are excluded. No study function should open
`solution.csv`. The boundary, source provenance and historical text-exposure
exclusions remain those in [RELEASED_DATA.md](RELEASED_DATA.md).

Keep all annotations in the primary experiment. Repeated comments can have
different supplied contexts; 544 repeated body/policy rows and 39 contradictory
label groups are not silently reconciled. Use the existing Unicode/whitespace
normalization. Three familiar-policy folds group by comment body and stratify by
policy/label, seed 2025. Four transfer folds each hold out one complete policy.
In both protocols, remove a training row if its body or any supplied example
matches a validation body. Save row IDs and verify coverage and every boundary.

The pre-score audit gives familiar-policy training sizes 3,000 / 2,643 / 2,936
and validation sizes 3,712 / 3,712 / 3,711. The context purge removes 4,423 /
4,780 / 4,488 candidate training rows. Transfer-fold training sizes are 5,277 /
5,925 / 8,418 / 8,289, in sorted policy-text order. This is a demanding,
conservative protocol with appreciable training-data loss; results must disclose
that cost. Promotion has only 124 negative research examples, so per-policy
uncertainty and probability quality matter.

## Fixed comparisons

Run 34 trained configurations on the same seven saved splits (238 fits): the
21 existing broad-family additions/removals; comment-only and rule/example lexical
references; full word, character and joint representations; their training-only
naive-Bayes weights; and five fixed 128-dimensional SVD controls. Keep logistic
regression C=2 and the existing family budgets fixed. No hyperparameter search.
The existing pinned, rule-conditioned Qwen encoder supplies both coordinate and
example-comparison features. Reuse verified vectors and encode only missing inputs.

The candidate families retain their rationale and availability analysis from
[FEATURE_RESEARCH.md](FEATURE_RESEARCH.md): lexical/style observables, rule/support
comparisons, support-token products, semantic coordinates and scalar contrasts,
training-reference percentiles, community frequencies and cross-fitted target
context. Vocabularies, screening, redundancy detection, scaling, SVD and all
target-derived encodings are fitted inside each purged training partition.
Report generated, rejected and retained counts per fold; column counts are not
counts of independent scientific hypotheses. No timestamp, author history,
opponent strength or external rating exists in this schema.

Seven previously specified, untrained support scores remain descriptive controls.
They use supplied positive/negative examples, not validation target labels.
Their fixed sigmoid/temperature scales are not calibrated probabilities.

## Robustness and interpretation

Keep the primary validation rows fixed for two additional training sensitivities:
exclude conflicting-label groups identified *within training only*, and remove
approximate training body/support copies of validation bodies using character
cosine >=0.95, token Jaccard >=0.90 and minimum length 40. Refit the lexical
reference and full-character control for each sensitivity (28 additional fits).
If no training rows change, reuse the identical primary fit and record that fact.
Never infer a training exclusion from validation labels. Separately report
descriptive evaluation excluding conflicting body/policy groups and equal total
weight per distinct body/policy; these do not choose features.

For the fixed semantic centroid score, evaluate one example per class (average
over all four positive/negative choices), supplied contexts shuffled within
policy using seed 2025, and removal of exact self-support matches. These stress
tests distinguish reliance on a particular support set from general language
signal. They do not claim resilience to missing rule text or arbitrary paraphrases.

Report policy-macro AUC first, alongside per-policy AUC, pooled AUC, log loss and
Brier score. Save row-level out-of-fold predictions privately. Compare every
configuration with the matched rule/example reference; separately compare each
family addition with screened words, and removals with all-transfer features.
Use paired normalized-body cluster bootstrap with 1,000 draws and simultaneous
intervals across the declared contrasts. These intervals condition on fixed
out-of-fold models and the four observed policies; they do not account for all
feature selection or estimate variability over arbitrary future rules.

Report fold-selection overlap, coefficient summaries and within-policy feature
group permutation importance. Permutation is an interpretation diagnostic;
ablations provide the direct with/without comparison. Do not add SHAP merely as
decoration when a linear model already exposes additive contributions.

## Execution and decision

Commit the executable configuration before scores, bind artifacts to sources,
data, configuration and encoder contract, and verify resume markers. Use a
bounded CPU worker (maximum two hours initially), UTC heartbeat events and S3
checkpoints of completed encoding/fold stages. A retry reuses valid stages.

This milestone can nominate a representation, not close the feature gate by
itself. Investigate remaining hypotheses when effects vary materially by policy
or robust alternatives disagree. The earlier NLI/instruction failures apply to
the historical two-policy study; they are not evidence of failure on every
policy. Only after a documented stopping decision should a candidate/reference
be frozen for one confirmation evaluation. Production inference remains a later
milestone and must explicitly consume the accepted feature contract.
