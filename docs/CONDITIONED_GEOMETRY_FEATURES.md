# Round 8 — Reference-only geometry conditioning

## Observed result and new hypothesis

The verified Round 7 primary `paired_all` reached 0.720881 mean policy AUC,
versus 0.719893 for raw Qwen and 0.723068 for raw basic geometry. The primary
failed its registered gate. Those recorded results do not justify assuming
that similarity-based opposing pairs represent the relevant behavioral difference.

Hypothesis: some of the common mean and dominant covariance directions in
frozen decision vectors may obscure the similarity dimensions useful for
same-rule examples. Reference-only centering, common-component removal and
regularized whitening may produce better nine-statistic class comparisons.
This is unproven for these cached Qwen decision vectors. The dominant directions
could be useful policy/label signal; removing them can make performance worse.

## Feature definitions and ablations

Begin with unit-normalized frozen reference vectors. Fit the mean and covariance
basis using the current inner reference pool only. A fixed randomized SVD uses
at most 32 components and three power iterations. Effective rank is clipped to
min(32, n_reference-2, dimension-1); its actual value is reported. No query or
label chooses rank, axes, removal count, regularization, or seed.

Family A removes the first four estimated directions (or the available smaller
rank) after mean subtraction, then L2-normalizes. Family B performs full-space
regularized low-rank whitening: with eigenvectors U and eigenvalues lambda,
replace the component along U by gain

    g_i = sqrt(a / (a + 0.8 * lambda_i)),
    a = max(0.2 * total_variance / original_dimension, 1e-6).

The residual complement remains present, rather than projecting the text onto
32 dimensions and silently discarding the rest. Each output vector is normalized.
This is deliberately not full covariance whitening and not a neural encoder fit.
If a transform produces a zero vector or a degenerate centroid, stop and report it.

Each family produces nine features: permitted and violating centroid cosine,
nearest cosine, top-five mean cosine, and their three class margins. Thus the
primary uses **18 new scalar features**, in addition to ten reference columns.
A centering-only control adds nine alternative features. Feature counts are
not an evidence threshold; constant training columns are removed by the fixed
readout and their actual retention is reported through fitted coefficients.

The six candidates are `centered`, `deflated`, `whitened`, **`dual`** (primary),
`random_dual`, and `label_null_dual`. Both controls for the primary have the same
18-column output width. `random_dual` keeps the rank, eigenvalue gains, centering
and removal count but substitutes seeded random orthogonal axes. It tests whether
estimated axes matter; it does not guarantee equal per-query similarity geometry.
`label_null_dual` keeps transforms and reference rows fixed but permutes supplied
reference labels within that one rule. The full-minus-deflated and full-minus-
whitened comparisons quantify the contribution of each feature family.

Primary comparators: raw Qwen, answer-only, raw basic geometry, random axes, and
permuted labels. Covariance fitting is label-free, while class geometry still
uses legal reference labels. Null controls remain null controls, not alternative
training labels to retain after the experiment.

## Diagnostics

Report mean off-diagonal reference cosine before/after transforms, mean-vector
norm, explained-variance fractions, effective rank and minimum whitening gain.
These are representation diagnostics, not tests of improved accuracy. Do not
select a configuration because its vectors look more isotropic. Eight plots show
policy AUC, simultaneous intervals, controls, ablations, reference anisotropy,
spectrum coverage, standardized coefficients, and probability quality.

## Research sources and limits

- Mu, Bhat and Viswanath, *All-but-the-Top* (2017/2018):
  https://arxiv.org/abs/1702.01417. Motivates common-mean and leading-direction
  removal for word representations; it does not demonstrate this Qwen setting.
- Su et al., *Whitening Sentence Representations* (2021):
  https://arxiv.org/abs/2103.15316. Motivates covariance normalization for
  semantic representations. Our low-rank ridge transform differs from theirs.
- Huang et al., *WhiteningBERT* (2021):
  https://aclanthology.org/2021.findings-emnlp.23/. Supports investigating geometry
  in pretrained representations, not assuming Qwen moderation gains.

This does not rerun the historical fixed-direction polarity experiment: it
changes reference-space geometry before same-rule class comparisons, with
inner-fold-fitted unsupervised transforms and matched random-axis controls.

## Evaluation, controls, and selection

This is a feature/readout experiment, not an encoder-training experiment. Use the
same fixed logistic regression as Round 6 (C=1, liblinear, 2,000 iterations, seed
2025); use the prior training repetition weights. Keep the same cached adapted
answer log-odds and nine raw frozen basic-geometry features as the reference
readout. The new features append to that reference. They do not blend predictions
from independently fitted models. All dense screening/scaling is fitted only on
eligible training observations through the existing readout function.

Use only original competition training and explicitly supplied support labels. The
outer query identities are exactly the historical 234 advertising and 647
legal-advice comments. Reconstruct the canonical plan, preserve its training list
order when selecting vector rows, and purge query bodies from every source.
Only same-rule eligible reference examples enter new feature construction. Do not
infer a label for a text under a different rule. Test/reference arrays are never
changed. New features never receive query outcomes.

Every training feature is built in three normalized-text group folds. Entire held
text groups are omitted from fitted reference statistics, learned transforms,
and descriptor reference pools. Full reference pools serve the outer queries.
Thus the inference pool is larger than each cross-fitting pool; this distribution
difference remains a limitation. Both rounds use frozen input vectors, avoiding
encoder-in-sample geometry. The retained **adapted training answer margin** is
still in-sample on support labels. Cross-fitting the new features does not make
that margin out-of-fold. No independent or causal validation is claimed.

Six candidate configurations across two policy cohorts produce exactly 12 new
CPU classifier fits per round. Six pre-existing readouts are reused: raw Qwen,
answer-only, and raw basic geometry, each for both policies. Two of these six are
cached neural answer scores, not separately fitted classifiers. Every control
must reproduce its prior predictions and AUC before the first candidate fit.

The primary is frozen in the configuration before execution. It must beat each
specified primary comparator by >=0.003 mean per-policy AUC, with a positive
simultaneous 95% lower bound and no regression on either policy. Within-policy-
ranked pooled AUC cannot decline against raw Qwen. Secondary candidates are
reported, not silently promoted. A failure rejects this registered candidate;
it does not establish that every related feature or encoder technique is exhausted.

500 paired normalized-comment-group bootstrap draws produce centered maximum-
deviation simultaneous intervals across the fixed contrast family of that round.
These intervals are conditional on fixed predictions, not model-refit uncertainty.
They do not correct all the preceding adaptive experiments, future-policy shift,
or a global family including both new rounds. Means, both individual policies,
ranked-pooled AUC, Brier score and log loss are reported separately. No number in
these notebooks is a newly measured Kaggle score.

## Independence and bounded execution

Rounds 8 and 9 both depend on the same verified Round 7/6 cache chain. Neither
reads the other new round's source, results, predictions, selection decision, or
hyperparameters. Their configurations are fixed together before either is run.
Run them sequentially for cost clarity; do not run two helpers concurrently.
A scientifically negative completed Round 8 does not prevent the independent
Round 9. A software error should be diagnosed before continuing.

No downloads, package installation, cloud API calls, encoder calls, GPU jobs,
Git operations that write, or competition submissions occur. The experiment
has a 240-second POSIX alarm and 260-second parent watchdog. The complete
helper has a 540-second budget; the supplied terminal command has a 600-second
outer timeout with ten seconds for forced shutdown. Heartbeats and candidate
counters show progress. These limits do not stop the SageMaker application.

Each feature bank is atomically saved with a source/config/input/environment
identity and checksummed completion marker. Each fitted candidate is separately
checkpointed, with its exact query ordering, scores, coefficients, and scaler.
Completed result replay verifies every prior and current scientific checksum and
performs zero fitting. Missing or corrupted completed artifacts stop instead of
being silently regenerated. Raw data, previous source, previous notebooks and
previous checkpoints remain unchanged. One installed source file cannot replace
an existing edited file just because its filename matches.

A separate executed notebook receipt verifies eight code cells and eight Plotly
figures. It uses a local IPC Jupyter kernel and explicit shutdown, not a new
remote service. Notebook replay checks source and output hashes and executes
zero cells. The dashboard contains its JavaScript and public aggregates only.
The return ZIP excludes comments, raw data, vocabulary, row-level outcomes,
private predictions, private matrices, model files and credentials.

## Publication state

Installation is local and uncommitted in the existing SageMaker checkout based
on 824e92bfae7414aa156f0bf3195ba677a1f55b21. There is no automatic push, merge,
or reset. This milestone does not claim AWS/GitHub equality. Preserve both result
ZIPs and the private same-volume checkpoints for the later explicit publication
checkpoint. Local outputs are not disaster-recovery backups.
