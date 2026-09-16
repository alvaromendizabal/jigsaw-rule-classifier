# Top-solution integration research track

This track converts publicly documented competition methods into an independent,
testable project implementation. It does not vendor any third-party notebook or source
file. If future work directly incorporates licensed source, the corresponding license
and notice must be preserved.

## What the current project already has

The scored Qwen3-4B path already includes several techniques reported by leading systems:
support-label online adaptation, no subreddit in the current Qwen prompt, one-position
decision training, forward-only last-token scoring, length-sorted inference, and per-rule
rank normalization. Reimplementing those is not the next bottleneck.

## Immediate gaps

The fastest same-backbone gap is supervision and prompting. The current support builder
drops conflicting rule/body labels; stronger public systems instead resolve them by
majority or empirical soft targets. We therefore register matched 4B arms for drop,
majority, and soft resolution, plus concise/numeric/compliance prompt families and an
expanded decision-token set.

No arm is promoted from a public leaderboard number alone. Every change must use the
project's existing group-safe 881-row benchmark, target-free query construction, paired
bootstrap uncertainty, per-policy regression checks, durable checkpoints, and bounded
runtime.

## Scale and diversity

After the 4B recipe is fixed, the next primary challenger is Qwen3-14B. Public competition
evidence consistently places 14B materially above 4B. The current 4B remains valuable as
an ensemble member because diversity matters.

Two higher-complexity routes are staged behind that baseline:

1. uncertainty-selected soft pseudo-labeling into the 14B stage;
2. Deep Mutual Learning among diverse peer models with supervised decision loss plus
   peer KL divergence.

A fast contrastive BGE route is retained for error diversity rather than expected
single-model dominance: normal/flipped-rule triplet training, body-minus-class-centroid
features, and group-safe classical readouts.

## Feature-selection rule

This research track does not replace Stage 2 raw-feature modeling. The full compatible
feature matrix remains a candidate. No fixed top-k raw feature count is introduced.
SHAP, permutation importance, stability, and controlled ablations inform model-specific
regularization inside training folds.

## Visualization standard

Plotly outputs must be responsive and sized for their content:

- horizontal bars for long labels or six-plus categories;
- chart height scales with category count and is capped to avoid giant empty canvases;
- `automargin=True` on both axes;
- crowded point labels move to hover rather than overlapping permanently;
- legends are placed away from the data;
- no fixed width unless a downstream renderer explicitly requires one.
