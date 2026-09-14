# Feature-value audit against the cached Qwen development reference

## Why this milestone, rather than another expansion

Round 5 completed 12 new fits and reused 12 controls. The combined scoped lexical model reached
0.7080975206 mean policy AUC, versus 0.7022460776 for the prior lexical-evidence reference.
But its matched collapsed-copy control reached 0.7084760503. The scope-versus-copy change is
-0.0003785297; its simultaneous interval is [-0.0084068071, 0.0076497477]. The registered
primary did not pass. The apparent improvement over the smaller anchor is not evidence that
attribution/negation context itself improved prediction. It also does not prove that all
context-sensitive representation learning is useless.

After five adaptive rounds, another regex/sparse-feature sweep could improve a proxy while
adding nothing to the accepted-model development reference. This audit compares the existing
feature models with cached Qwen outputs on the exact same original-data queries. It closes
that missing evidential link before proposing the next feature family. It is feature research,
not a switch to algorithm tuning, ensembling, or final delivery.

## Inputs and historical identity

Read the five original aggregate reports, their immutable source maps, completion markers,
and fourteen selected private prediction checkpoints (seven variants x two folds). Rebuild the
same target-free query plan from hash-pinned original train.csv. Verify ordered IDs and reproduce
every selected historical AUC. Never refit or substitute a missing checkpoint.

The neural comparator is the development-fold support-adapted Qwen3-4B output from
`d13858b7407993fcc6e8`. It is related to the accepted competition approach, but is NOT the hidden
Kaggle predictions and is NOT an independent holdout. Its previously published per-policy AUCs
are 0.6792537313432836 (advertising) and 0.7605330619753037 (legal advice). The script must reproduce
those values with exact ordered query IDs before making the new comparisons. It uses the saved
answer log-odds (scores column 1) for ranking, not potentially saturated float32 probabilities.
Probability diagnostics use float64 sigmoid of those margins, never percentile ranks.

Two private NPZs total 42,853,014 bytes. Use already verified local copies first. With explicit
`--recover-reference`, retrieve missing files using at most one S3 GetObject per file, no retry,
from the existing private Jigsaw bucket after the expected-account check. Use streaming reads,
size/sha checks, exclusive publication, and durable individual file receipts. Corrupt existing
cache entries cause a stop; no silent replacement. No adapter, model weight, optimizer, or
pickle is loaded. Only query-row IDs and query score arrays are opened; cached embedding arrays
are not accessed. The NPZ containers nevertheless include those arrays and stay private.

## Fixed analyses

1. AUC, log loss and Brier by policy for Qwen and seven selected historical feature readouts.
2. Seven candidate-minus-Qwen contrasts. 500 common normalized-comment-group bootstrap draws
   yield a centered maximum-deviation simultaneous 95% band. Intervals are conditional on fixed
   predictions and these seven comparisons; they do not account for prior adaptive selection,
   model refitting variation, or unknown future rules.
3. Exact pairwise AUC-credit decomposition: positive-negative pairs improved and damaged by a
   feature readout relative to Qwen. Give ties half credit. Improved credit minus lost credit
   must reproduce the AUC change. Pairs are dependent; pair count is NOT statistical sample size.
4. Within-policy Spearman rank correlation. Low correlation alone is not useful signal.
5. Fixed, overlapping, approximate text flags: quotation, code markup, negation words, question
   mark, links, directive phrases and first-person requests. Construct flags with body/rule/ID
   inputs only. After construction, join original labels for descriptive subset scoring. Show
   both flag-present and flag-absent subsets, suppress AUC when either class has fewer than ten
   comments, and do not search alternative thresholds. Slices are neither adjudications nor
   automatically learned labels. No content examples or row IDs leave the private workspace.
6. A ledger of the five rounds' actual matched additions, controlling feature duplication where
   available, with historical decisions kept unchanged.

## What the audit must not claim

There are zero model fits, zero neural forward passes, zero blends, and zero tuned thresholds.
No performance promotion gate is being applied, and no weighted or routed predictor is produced.
Ranking disagreement is evidence for studying representation gaps, not a promised ensemble gain.
Every result remains exploratory because these 881 queries have repeatedly informed choices.
No AUC from this audit is a new Kaggle score. The 0.91425 accepted private score and 0.92930 target
are a different evaluation; subtracting this cohort's AUC from either would be invalid.

## Subsequent research choices, not launched here

- Persistent Qwen errors in request/directive/disclaimer contexts motivate rule-action or
  contrastive support representations, with fixed ablations against cached reference output.
- A feature family that repairs some Qwen ranking errors but damages many other pairs merits
  examining how its information is represented, not automatically mixing predictions.
- Gains explained by copy controls do not establish semantic scope value. A later fixed
  scaling/regularization control may be justified, but is separate from a new feature claim.
- No residual gain calls for revisiting the information/representation, not another unbounded
  token-count or regex sweep. Missing conversation history cannot be invented from IDs.

Any subsequent experiment must freeze its candidate and method before scoring, use eligible
training-only fitting, retain per-policy ablations, and account explicitly for the consumed
validation cohort. Feature research stays open; a leaderboard record cannot be guaranteed.

## Preservation, bounds and publication

The scientific audit has a 240-second POSIX timer. The manual launcher uses a 540-second total
budget and a 600-second terminal watchdog, with 15-second heartbeats. Input recovery is reusable;
audit results and notebook execution have separate verified completion markers. A failure cannot
erase prior rounds. Existing source and reports are read-only in this milestone. New source is
installed only at its canonical new paths. No packages are installed and no git refs are changed.

Notebook 11 has eight executed code cells and eight interactive Plotly charts. The standalone
HTML contains plotting JavaScript and public aggregates only. Export includes canonical source,
tests, notebook, reports, hashes and stage logs, not data/NPZ/model/credential files. It does NOT
push or commit anything to GitHub. The completed local work must be reviewed and published through
a separate verified commit/CI milestone; no synchronization claim is made by this audit.

## Research basis

- Clarke et al. (ACL 2023), Rule By Example, https://aclanthology.org/2023.acl-long.22/ :
  exemplar-based contrastive rule representations are a plausible next feature direction. This
  audit neither reproduces their method nor imports their data or claims their performance.
- Cawley and Talbot (JMLR 2010), On Over-fitting in Model Selection and Subsequent Selection Bias,
  https://www.jmlr.org/beta/papers/v11/cawley10a.html : repeated selection against finite validation
  data can bias reported improvements. Historical per-round intervals are not an across-project
  guarantee against this effect.
