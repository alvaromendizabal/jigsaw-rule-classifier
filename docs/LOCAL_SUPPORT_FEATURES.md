# Round 6 — Local support-density evidence

## Completed evidence, not a claim about leaderboard performance

Audit a6103241df718b4b27a4 compares the same 234 advertising and 647 legal-advice
comments. Qwen's cached adapted answer has policy AUCs 0.6792537313 and
0.7605330620 (macro 0.7198933967). The Round 5 scope readout has 0.6954850746 and
0.7207099665 (macro 0.7080975206). Thus the lexical representation is slightly
better on advertising but weaker on legal advice. Its macro difference from
Qwen is -0.0117958761 with a conditional simultaneous interval spanning zero.
Pairwise ranking repairs are offset by other ranking mistakes; disagreement
is not proof of a useful ensemble. No calibrated blend or automatic routing was
fit. The cohort has repeatedly informed research and is not independent validation.

## One new hypothesis

Word cues and a decision margin may not encode the amount and consistency of
nearby labeled support evidence. Test locally scaled, class-balanced semantic
densities and evidence-concentration features relative to both an answer-only
readout and the already investigated centroid/nearest/top-five features.

This is NOT a repetition of the previous nearest-example prompt selector, CSLS
diversity audit, or a claim to train a contrastive encoder. No prompts are
changed and no new neural forward passes occur. The previously saved frozen and
adapted last-token vectors and answer margins are reused. Feature/readout
training uses only the explicitly labeled supports for the query's rule.
The historical plan still purges all query text from every adaptation source.

## Features and availability

- Basic control: nine existing types of prototype features — centroid cosine,
  nearest cosine, top-five mean cosine per supplied class, and their differences.
- New local-density family: twelve scalar features — class-normalized log
  affinities, their ratio, evidence share, uncertainty, nearest locally-scaled
  affinities, effective-support fractions, and relative neighborhood radius.
- Every learned readout also includes the cached adapted answer log-odds as an
  input. Six fixed logistic readouts are trained; this is not a raw-score blend.

Normalize vectors. Let d^2=2(1-cosine). Each reference radius is its tenth nearest
reference distance, excluding itself, with k reduced only when necessary. A
query radius is its tenth reference distance, not a query-to-query statistic.
The affinity exponent is -d^2/(query_radius * reference_radius), with radii
floored at 1e-6. Compute class mean affinities in log space. Equal class priors
in the density ratio avoid automatically favoring the more numerous class.
ESS/class_count measures whether evidence is supported broadly or dominated by
few examples. No arbitrary negation of another rule's labels is allowed.

No thousand-column claim is made: 12 new features are tested, with 9 historical
geometry controls and one existing decision-margin feature (at most 22 inputs).
Near-constant columns (training variance <=1e-12) are excluded within training;
StandardScaler is fitted only on training. There is no query-led screening.

## Cross-fitting and limitations

For each outer query policy, split its eligible support pool into three
GroupKFold partitions by normalized body. Each training example's geometry uses
only the other groups. No copy of its own body is in that reference pool. Query
geometry uses the full eligible support pool. The reference pool stays within
the same rule and contains only explicit labels. Radius selection does not read
labels. Shuffled-label controls permute only the current reference pool, after
excluding held groups, and preserve class counts. Evaluation targets are joined
only for input-reference AUC verification and final scoring, never for feature
construction or fitting.

The frozen geometry does not contain adaptation-label training. The primary
uses that frozen geometry plus the adapted answer margin. Adapted training
margins were produced by the adapter on its training supports. Their evaluation
is not a fully nested or out-of-fold encoder calibration experiment. The
adapted-geometry sensitivity arm has the same limitation. Cross-fitting support
statistics does not undo supervised learning already done in the encoder.
Report this distinction; no claim of independent confirmation is permitted.

Cached training vectors have no individual training IDs. Their row order is
therefore tied to the immutable train-file bytes, the original build_study and
support_pairs source hashes, the exact historical query identities, canonical
training counts and the hash-verified NPZ. Never sort that training list before
indexing the vectors. No new data recovery is necessary after the passed audit.

## Fixed comparison

For each of two outer policies, fit these six representations:

1. answer_only: one decision-margin feature (readout/calibration control).
2. frozen_basic: answer + historical nine frozen geometry features.
3. frozen_local: answer + twelve new frozen density features.
4. frozen_all: answer + all 21 frozen geometry features (registered primary).
5. shuffled_all: same 22 input positions; reference labels shuffled within
   each eligible same-rule reference pool (label-information control).
6. adapted_all: answer + all 21 adapted geometry features (sensitivity only).

12 CPU fits, two raw-Qwen score controls reused. C=1, liblinear, max_iter=2000,
seed=2025; retain original plan repetition weights. This round uses same-rule
supports only for the readout, not the earlier lexical readout's multi-rule
training mix. Claims of incremental value rely on this round's matched controls,
not numerical differences between unrelated training protocols.

## Fixed primary decision

frozen_all must improve macro AUC by >=0.003 against EACH of raw Qwen,
answer_only, frozen_basic and shuffled_all. Require positive simultaneous lower
bounds and no per-policy regression against each comparator; ranked-pooled AUC
cannot decline versus Qwen. Eleven registered contrasts receive 500 paired
normalized-comment bootstrap draws. The centered maximum-deviation band is
conditional on the fitted predictions; it does not correct all project-wide
adaptive choices or simulate future rules. Secondary candidates cannot replace
the primary after seeing results. Passing means eligibility for further
validation only. No GPU, new Kaggle score or automatic model promotion.

## Recovery and execution

Existing raw files, reports, code and NPZs are read-only. New semantic feature
matrices are privately checkpointed per bank and fold. New fits have separate
prediction/coefficient/scaler checkpoints. Complete checksums are verified on
replay; corruption stops rather than silently regenerating. Reconstruct cached
readout scores to verify prediction parity. Both raw-Qwen cohort AUCs must match
the preceding audit before fitting.

240-second POSIX scientific timer, per-subprocess watchdog, 540-second launcher
budget and 600-second shell cap. Heartbeats every 15 seconds. No package install,
cloud API, training job, model download, S3 write, Git write or submission.
Notebook 12 and HTML contain only aggregate diagnostics. The return ZIP excludes
raw data, vectors, private predictions, vocabularies and credentials. Keep all
previous study folders. This is not an AWS/GitHub synchronization operation.

## Research rationale versus implementation

- Zelnik-Manor & Perona, Self-Tuning Spectral Clustering (NIPS 2004):
  https://proceedings.neurips.cc/paper/2004/hash/40173ea48d9567f1f393b20c855bb40b-Abstract.html
  motivates local affinity scales. We do not cluster or reproduce their experiment.
- Snell et al., Prototypical Networks (NIPS 2017):
  https://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html
  motivates class prototype comparisons, used here as a control, not a new discovery.
- Clarke et al., Rule By Example (ACL 2023):
  https://aclanthology.org/2023.acl-long.22/
  motivates rule-grounded exemplar representations. No contrastive neural
  training is performed here and their reported results are not claimed.
