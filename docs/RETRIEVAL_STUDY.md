# Policy-excluded semantic retrieval

This extension is specified while the four-policy embedding run is in progress,
before inspecting expanded-study scores. It fills a concrete gap: raw coordinates,
linear SVD controls and supplied-example comparisons do not test local geometry
around *labeled training examples*. The input embeddings are already available;
no larger model, new labels or hyperparameter search is needed.

[Snell, Swersky and Zemel (2017)](https://arxiv.org/abs/1703.05175) motivate
classification by distances to class prototypes. Their method learns an embedding
metric; this study instead tests a fixed pretrained geometry and trains only the
same linear classifier used elsewhere. It is an inspired feature hypothesis,
not a reproduction of their results or evidence of moderation performance.

## Hypothesis and availability

Represent each row by its normalized rule-conditioned body embedding, the
positive-minus-negative supplied-support direction, and the body's coordinate
product with that direction. Retrieve 3, 9 and 27 nearest reference rows in each
space. Class proportions, temperature-weighted proportions, neighborhood
similarity/spread, nearest positive/negative similarities, class margins and
missing-class flags test local support. Global positive/negative prototype
cosines, coherence, separation and reference priors provide complementary
summaries. Signed square/root maps and interactions between retrieval margins
and the row's supplied-centroid margin test nonlinear confidence relationships.

Every reference vector and label is available from permitted training data.
The comment's own supplied examples remain valid inference inputs. Labels from
one policy are never assigned to comments under another policy: their aggregate
geometry is a candidate predictor whose transfer must be measured.

## Target leakage prevention

**Exclude the query policy from the labeled reference bank, including when the
query policy is familiar.** For training features, leave out each complete policy
and purge any remaining reference row whose body or supplied example matches
a held-out query body. Fit features for those query rows without reading their
targets. This more demanding cross-fit matches the intended unseen-policy use
and prevents a policy's labels from predicting themselves through retrieval.

For outer validation, use only the already purged outer-training rows, again
excluding a matching query policy. Reject a target column at transformation.
Validate text isolation, input order, vector shape and finite values; record
reference/query row IDs and policy sets for every bank. A missing class in a
small local neighborhood gets an explicit indicator, not an undisclosed NaN.
Final candidate screening and scaling use only the cross-fitted training matrix.

Tests must change every label of a query policy and prove its training feature
rows are unchanged, check all support-field leakage routes, reject target-bearing
validation frames, and verify serialized transforms and completed-fit reuse.

## Fixed comparison and stopping interpretation

Use the exact expanded-study folds, data and frozen embeddings. Select at most
64 candidates with the existing training-only screen. Fit retrieval alone and
add it to screened words, compact semantic comparisons, their combination, and
the all-transfer bank: five fixed fits on seven folds. Compare with the exact
saved controls, not fresh splits or a tuned baseline. Report the candidate count,
screening decisions, policy-level results, matched AUC deltas, simultaneous paired
intervals, log loss, Brier and group permutation diagnostics. Preserve private
OOF predictions and stage checksums.

This family earns further work only if its incremental transfer evidence is
credible and its reference-data/inference cost is justified. Failure narrows the
remaining target-derived feature avenue; success still needs stability checks
and a frozen confirmation comparison. Neither outcome consumes reserved labels
or automatically changes the production representation.

## Outcome on the expanded development cohort

Run `f69e5061eef584e520e3` completed all 35 fixed fits. Every fold generated
309 candidates and retained 64. Together with the expanded banks, the search
contains **188,518–188,521 columns per fold**, with **9,234–9,613** retained
before choosing families. These counts do not imply that one model uses them all.

| Transfer representation | Policy-macro AUC | Log loss | Brier |
| --- | ---: | ---: | ---: |
| Retrieval alone | 0.5085 | 1.3773 | 0.3727 |
| Words + retrieval | 0.4869 | 1.1316 | 0.3613 |
| Semantic comparisons + retrieval | 0.5590 | 1.4763 | 0.3613 |
| Words + semantic comparisons + retrieval | 0.5802 | 1.1566 | 0.3565 |
| All-transfer + retrieval | 0.5669 | 1.5584 | 0.4297 |

Adding retrieval to screened words gives +0.0493 AUC (simultaneous interval
[0.0040, 0.0945]), but the resulting model remains weak and its probability
losses worsen. Adding it to compact semantic comparisons **loses 0.1287 AUC**
[-0.1739, -0.0835] and sharply worsens probability losses. Its +0.0041 gain on
words plus semantics is uncertain [-0.0412, 0.0493]. All-transfer gains +0.0155
[-0.0297, 0.0607], also uncertain. Intervals are conditional on the four observed
policies and fixed predictions, with multiplicity handled within this study.

This family is **rejected as a promotion candidate**. Leakage-safe engineering
does not guarantee useful transfer. The 0.8013 familiar-policy AUC of
all-transfer plus retrieval must not hide its 0.5669 transfer AUC. The leading
frozen centroid remains stronger without this target-derived bank.
