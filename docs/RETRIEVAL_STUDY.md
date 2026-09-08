# Semantic retrieval as a final feature hypothesis

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
