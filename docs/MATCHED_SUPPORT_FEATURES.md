# Round 7 — Matched opposing-support contrasts

## Observed evidence, not a claim of progress in accuracy

Round 6 (`32a90cf7c6607257c2bf`) completed 12 CPU fits and reused two Qwen readouts.
The same-query Qwen macro AUC was 0.7198934. Basic frozen geometry reached 0.7230676
(+0.0031742), but legal-advice performance declined. Adding local density to basic
geometry reached 0.7175129 (-0.0055547 vs basic); the preregistered primary failed.
Adapted geometry reached 0.7199956, an uncertain +0.0001022 over Qwen. No method
was promoted. All historical artifacts remain immutable.

## One new hypothesis

Absolute closeness to a violating or permitted example can reflect topic rather
than the behavior relevant to the rule. Form close, opposite-label pairs inside
one rule and measure the query along their difference direction. The pair midpoint
provides a query-local relevance proxy. This is an engineered contrast feature,
not encoder training, verified semantic matching, a causal matched study, or an
exact reproduction of Rule By Example. Frozen answer-prompt vectors may still be
poor retrieval representations. A failed result would not exhaust contrastive
representation learning.

## Construction and information availability

1. Start from the same hash-pinned frozen Qwen reference vectors and original
   permitted support labels as Round 6. Neither a web dataset nor a new model is used.
2. Normalize vectors and sort by normalized reference text to fix tie ordering.
3. Within one policy, solve a rectangular maximum-total-cosine one-to-one
   assignment between violating and permitted references. Each selected endpoint
   participates once; the larger class can leave unused references. All counts
   are reported. This is not ordinary repeated nearest-neighbor retrieval.
4. Let p and n be the normalized violating/permitted endpoints. The contrast is
   (p-n)/||p-n||; the relevance direction is (p+n)/||p+n||. Exclude gaps or midpoint
   norms <=1e-6. Exact duplicate/conflicting reference texts are rejected upstream.
5. Extract four global statistics and eight query-local statistics using fixed
   top-1/top-5/top-20 summaries, temperature 0.1 and tanh scale 5. No parameter or
   feature is selected against query labels. The 12 columns are not a size target.
6. Three-fold normalized-text-group cross-fitting builds training features without
   a row or its text copies in its own reference pool. Evaluation uses all eligible
   references. No query is a reference for another query. Their targets enter only
   AUC and diagnostic scoring after predictions are produced.

Global features: signed mean, standard deviation, smooth signed vote, and positive
fraction. Local features: nearest-midpoint margin, top-five mean, top-twenty mean
and standard deviation, relevance-weighted mean/standard deviation/smooth vote,
and weighted-minus-global mean. New predictors describe the paired evidence;
raw support texts and identities do not enter public exports.

## Matched comparisons and ablations

Every new readout starts with the same adapted answer log-odds and nine frozen
basic geometry columns from Round 6. Reuse all six cached readouts for qwen_raw,
answer_only, and frozen_basic across the two policies, checking prediction parity
and previously reported AUC before any new fit.

Six configurations across two policies (12 new fits):
- paired_global: four global contrast features;
- paired_local: eight local contrast features;
- paired_all: all twelve, the fixed primary;
- repaired_all: randomly permute the negative endpoints among the same selected
  endpoints; retain width and endpoint coverage but remove optimized pair links;
- orientation_all: randomize each contrast direction's sign using a fixed seed;
  keep matching, midpoint relevance, width, and pair separation unchanged;
- unnormalized_all: use p-n rather than its unit direction, isolating pair-specific
  separation normalization.

Re-pairing is a negative control, not fabricated training labels: true endpoint
labels are retained. Orientation randomization is explicitly a null feature arm,
never new supervision. These controls match endpoints/dimensions, not every
feature covariance or separation distribution. Normalization can amplify narrow,
noisy contrasts; the unnormalized arm is a necessary sensitivity comparison.

The readout is exactly Round 6's train-scaled LogisticRegression(C=1, liblinear,
max_iter=2000, seed=2025), with the same eligible repeated-example weights. No
classifier search, additional calibration sweep, routing, or blend is performed.

## Gate and uncertainty

Primary: paired_all. Require >=+0.003 policy-macro AUC, positive simultaneous lower
bounds and no per-policy regression against EACH of raw Qwen, answer_only,
frozen_basic, re-paired control and orientation control. Within-policy-ranked
pooled AUC cannot fall relative to Qwen. A secondary winner cannot replace the
primary. Global/local removals and raw/unit difference comparisons are descriptive
planned contrasts. Five hundred paired normalized-comment-group bootstrap draws
form a centered maximum-deviation simultaneous 95% band over thirteen contrasts.
These are conditional fixed-prediction intervals, not intervals over refitting,
new rules, or all historical adaptive choices. The 881-query cohort has already
been examined repeatedly; eligibility is not independent confirmation or a Kaggle
score. Model promotion and paid GPU work are not authorized by a pass.

## Retained limitations

The primary representation is frozen, but the retained adapted *training answer
scores* were obtained on support examples seen by that encoder. Cross-fitting
pair features does not remove this in-sample-answer limitation. Fits can therefore
learn a mixture of in-sample answer and cross-fitted geometric signals. This
protocol is held fixed for attribution, not claimed to solve that train/query
mismatch. Inner reference pools are smaller than the final query reference pool.
A subsequent validated experiment may need disjoint encoder training; no new
encoder execution is silently performed here.

## Bounded and reproducible execution

No installs/downloads/cloud calls/Git writes. Verify the Round 6 public receipt,
its exact source map/environment, all its feature/model checkpoints, the audit and
its underlying pinned inputs. Read but never replay/retrain prior stages. Save new
feature banks and fits transactionally with identities and hashes. Reject missing
completed stages or corrupted data, rather than recomputing unverified work.
The scientific worker has a 240-second POSIX hard cap, its launcher watchdog 260
seconds, notebook/replay watchdog 110 seconds, whole-helper budget 540 seconds,
and user's command 600 seconds (+10-second forced stop). Log heartbeats every 15
seconds and completed-fit counters. The timeout does not stop the SageMaker app.

The new notebook reuses the existing named project kernel. No service, firewall,
or global kernel configuration is changed. It displays only completed aggregate
outputs: no fitting on notebook Run All.
Eight charts: per-policy AUC, uncertainty, matching/orientation/normalization
controls, global/local ablations, pair similarity, reference coverage, coefficient
associations, and Brier/log loss. Previous notebooks are not rewritten.

Public return allowlist: new source/config/tests/docs/notebook, aggregate results,
execution/replay/test/install receipts, prior aggregate result, and checksums.
Never export private vectors, per-row labels/predictions, credentials, or raw text.
Changes stay local, uncommitted, until a separate reviewed publication step. A saved
S3 or local receipt does not establish GitHub/SageMaker tree equality. Do not wipe,
reset, reinstall, or delete the preserved project to make it look synchronized.

## Research and distinction from implementation

- Clarke et al., ACL 2023, Rule By Example:
  https://aclanthology.org/2023.acl-long.22/
  motivates exemplar-based contrast for rule-grounded moderation. This experiment
  does not train its contrastive encoder or reproduce its hate-speech evaluations.
- Khosla et al., NeurIPS 2020, Supervised Contrastive Learning:
  https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html
  motivates explicit same/different-class structure, not evidence of this method's
  effectiveness in Jigsaw. The original paper's image-classification results do not
  support a numerical prediction here.
- SciPy linear_sum_assignment implements one-to-one rectangular assignment; our
  selection objective is cosine similarity, not a moderation-label oracle.

Feature research remains open. Do not claim the 0.91425 recorded accepted private
AUC improved or that the 0.92930 historical reference has been beaten.
