# Four-policy feature study

The protocol below was committed before inspecting any new model score. It extends the
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

## Measured expanded-study results

Run `a57bb74350fcbcd66592` completed all 238 primary fits and 22 sensitivity
refits; six unchanged sensitivity comparisons reused their primary fit. The
same seven partitions generated **188,209–188,212 candidate columns per fold**
and retained **9,170–9,549** across ten banks before final family selection.
These are fold-specific counts, not unique hypotheses, and a fitted model uses
only its declared banks. The retrieval extension and resolution controls are
reported separately below; historical NLI/instruction counts are not added here.

| Representation | Held-out policy-macro AUC | Log loss | Brier |
| --- | ---: | ---: | ---: |
| Rule/example lexical reference | 0.4728 | 0.8126 | 0.2992 |
| Screened words | 0.4376 | 0.8823 | 0.3220 |
| Words + compact semantic comparisons | 0.5761 | 0.8156 | 0.2969 |
| Compact semantic comparisons alone | 0.6876 | 0.6891 | 0.2456 |
| Semantic-interaction SVD | 0.6637 | 0.7939 | 0.2883 |
| Frozen normalized semantic centroid | **0.7042** | **0.6237** | **0.2177** |
| All transferable feature families | 0.5515 | 1.4700 | 0.4503 |
| Full character vocabulary | 0.4464 | 0.9169 | 0.3345 |

All rows use the same development data and validation boundary. Comparing these
scores with historical two-policy scores would confound changed data with
changed features. No new classifier hyperparameter search produced this gain.

The clearest matched addition is compact semantic geometry: **+0.1385 AUC**
over screened words, with within-study simultaneous interval **[0.0959, 0.1810]**.
Lexical support comparisons add +0.0677 [0.0251, 0.1102]; embedding-coordinate
features add +0.0476 [0.0050, 0.0901] to that weak word control. The latter result
does not make coordinates a leading representation: words plus coordinates
reach only 0.4852, and adding coordinates to all-transfer features lowers AUC.
Community metadata and character additions have negative point estimates.
Structure, support-token, percentile and target-context additions do not survive
the simultaneous interval test. They should not be called established gains.

The frozen centroid improves over the matched lexical reference by **+0.2314**
with pointwise interval [0.1985, 0.2600] and simultaneous interval
**[0.1889, 0.2740]**. This is a measured representation gain on the exposed
development benchmark, conditional on fixed predictions and four policies.
It is not independent confirmation or a claim about arbitrary future policies.

The centroid improves on the lexical reference for each observed policy, but
the improvements are highly unequal: advertising **0.7255 vs 0.6400**, legal
advice **0.5702 vs 0.5545**, medical advice **0.5969 vs 0.5173**, and illegal-activity
promotion **0.9241 vs 0.1793**. Most of the average uplift comes from avoiding the
lexical model's reversed ranking on promotion. Legal and medical advice remain
weak; the result does not justify a broad moderation-performance claim.

## Robustness findings

The 39 contradictory body/policy groups cover 187 rows. Training-only exclusion
barely changes transfer AUC for the two refit lexical controls. Approximate-copy
purging removes 61 / 5 / 3 additional training rows in familiar folds and one in
the promotion transfer fold; other transfer folds reuse their primary fits.
These results extend the earlier exact isolation without claiming independence
of paraphrases or conversation origin.

Excluding all 18 exact self-support matches leaves centroid AUC **0.7034**.
Using one positive and one negative example, averaged over all four choices,
gives **0.6918**. Shuffling complete supplied contexts within the same policy
gives **0.7062**. Thus copied examples do not explain the gain, and the particular
row's support pair is not uniquely necessary. The shuffled comparison is an
exploratory diagnostic; it does not justify using validation supports as a
deployment retrieval bank. Missing rule text and absent support sets remain
separate product conditions.

The familiar/transfer split is itself a major finding: all-transfer AUC is
**0.7989 on familiar policies** but **0.5515 on held-out policies**. The lexical
reference changes from 0.7311 to 0.4728. A familiar-policy-only report would hide
the core generalization failure. The frozen centroid uses no fitted training
labels, so its identical 0.7042 under both protocols is one scoring result,
not independent replication.

All 31 screened semantic scalar columns survive every fold. Their selected-name
Jaccard is 1.000; raw-coordinate overlap averages 0.192, structural overlap
0.286 and word overlap 0.288 across held-out folds. Stability alone is not
utility: target-context overlap is also 1.000 despite its weak incremental gain.
Group permutation and direct family removals remain available alongside these
selection diagnostics in notebook `02`.
