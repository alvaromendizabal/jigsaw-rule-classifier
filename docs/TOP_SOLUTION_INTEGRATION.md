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

Conflict supervision is no longer only a paper-design gap. Earlier CPU proxy work in
PR #28 rejected majority and soft conflict resolution under a fixed TF-IDF/Ridge
whole-rule transfer study. Because that proxy is not the LoRA system, a later explicitly
exploratory end-to-end 4B check changed only conflict handling: strict annotation-occurrence
majorities are retained and ties remain excluded.

The September 21 audit recovered 10 normalized rule/comment pairs, discarded one tie,
preserved uncontested-label parity and both policy-level query-purge checks, completed the
bounded Kaggle preview, and was confirmed as submission 56444879. Its score was still
pending at the captured snapshot. This closes the implementation question—majority
supervision can be wired into the strong neural path reproducibly—but does **not** establish
predictive improvement or authorize promotion. [Executed checkpoint](../notebooks/27_latest_system_checkpoint.ipynb).

Remaining same-backbone gaps include prompt/verbalizer choices that have not been cleanly
benchmarked end to end. The next larger capability gap is backbone capacity and complementary
model diversity. No arm is promoted from a public leaderboard number alone; changes remain
subject to target-free query construction, policy-level regression checks, durable
checkpoints, bounded runtime and honest separation of exploratory from scored evidence.

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
