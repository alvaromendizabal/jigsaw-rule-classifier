# Rule-polarity geometry / hard-negative CPU screen

## Research question

Can a supervised polarity direction learned from cached frozen Qwen representations transfer across unseen rules better than the same cache's frozen rule+support score, without new encoding or GPU compute?

This is a bounded precursor to any triplet/contrastive representation training. It deliberately reuses the immutable 2,029-row Qwen3-4B cache and original training snapshot.

## Candidate family

For each whole-rule training fold, the study evaluates five cached-vector transforms (`rule`, `support`, `rule_support`, `joint_minus_rule`, `joint_minus_support`) under three training-only polarity estimators:

- class-centroid direction;
- diagonal variance-whitened direction;
- hard-negative-weighted class-centroid direction, where an example is weighted more heavily when its nearest opposite-label training example has high cosine similarity.

No validation labels enter the direction construction.

## Gate

The candidate is only eligible for a later bounded representation-training experiment if the best method:

1. improves mean whole-rule AUC over the same-cache frozen joint score by at least 0.005;
2. wins both whole-rule folds;
3. has no policy regression.

Passing this CPU screen still does not itself authorize GPU training; it establishes a representation hypothesis worth implementing under a separate bounded GPU contract.
