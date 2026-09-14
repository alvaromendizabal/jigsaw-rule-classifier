# Round 9 — Behavior-conditioned support evidence

## Observed result and new hypothesis

Seven completed feature rounds have not established a replacement for the accepted
model. Round 7's matched-pair primary failed, and the earlier actor-role addition
was exploratory rather than confirmed. This round tests a *conditional relation*
between observed behavior and same-rule semantic support, not another unconditional
cosine summary and not a repetition of direct regex indicators.

Hypothesis: the local meaning of a query is better represented by semantically
related examples with similar action/attribution roles, with fallback to the
whole eligible rule pool when that behavior is poorly covered. Asking for advice,
offering it, quoting it, and reporting an experience need not behave identically.
Those distinctions do not determine the moderation target by themselves: the
legal rule may prohibit both requests and offers. No moderation label is invented.

## Observable descriptors and 48 new relational features

Reuse twelve **existing**, fixed Round 2 descriptor columns. Six actor descriptors
are self-request, other-request, directive, offer, experience, and reported
statement, each linked to legal content by the earlier clause-local heuristic.
Six context descriptors are quoted legal directive, negated legal directive,
disclaimer followed by legal directive, URL with call-to-action, owned-business
URL, and informational-resource URL. The descriptors are imperfect heuristic
measurements and can miss implicit acts, nuanced quotations and long-distance
relations. No new context is fetched from URLs or missing conversation threads.

For query q and eligible reference i, use frozen cosine c_qi and fixed affinity

    k_qi = exp((c_qi - max_j(c_qj)) / 0.1).

For each descriptor a, use a soft equality gate

    gate_qi(a) = 0.2 + 0.8 * 1[a(q) == a(i)].

Matching absence counts as matching; it is not evidence that the behavior is
present. The positive floor preserves both classes when a stratum is rare/empty.
Normalize affinity separately within each explicit label class. For each of the
twelve descriptors, four output features are calculated:

1. Change in class-balanced log affinity ratio relative to ungated evidence.
2. Change in affinity-weighted class mean cosine margin.
3. Change in class-difference log effective reference fraction.
4. Minimum gated coverage fraction across the two classes.

The first three are centered on the ungated query/reference calculation; the
fourth measures support coverage rather than declaring confidence. Six actor
axes times four statistics yield **24 features**, and six context axes yield
**24 features**. The primary contains **48 new features**, not the twelve old
descriptor indicators relabeled as new. No threshold, temperature or backoff is
tuned on this development result.

## Six fixed candidates and three mechanism controls

Candidates: `role_evidence`, `context_evidence`, **`conditioned_all`** (primary),
`permuted_all`, `uniform_all`, and `descriptor_only`.

`permuted_all` shuffles whole reference descriptor rows within the rule, preserving
the observed descriptor joint distribution and the reference rows/labels but
breaking their alignment. Query descriptors remain unchanged. `uniform_all`
removes semantic affinities and similarity values; it retains gate-coverage and
effective-count information from the behavior strata. `descriptor_only` keeps
only the twelve observed query flags in a 48-column zero-padded allocation and
contains no label-derived conditional evidence. Its width is matched, not its
rank or information content. Full-minus-role and full-minus-context comparisons
are the family-removal ablations.

Primary comparators: raw Qwen, answer-only, raw basic geometry, reference-descriptor
permutation, uniform semantic evidence, and direct descriptors. We require evidence
against every comparator rather than choosing the easiest baseline after the run.

## Diagnostics

For each fixed descriptor and both inner/outer reference settings, record the
reference/query prevalence, zero-stratum frequency and minimum class coverage.
No query outcomes choose the descriptors or supply the reference features. Empty
strata do not fabricate zero-class labels: the fixed 0.2 backoff retains the full
class. Eight figures show policy AUC, intervals, mechanism controls, ablations,
query descriptor coverage, fallback frequency, coefficients, and probability quality.

## Research sources and limits

- Park et al., *Detecting Community Sensitive Norm Violations* (2021):
  https://aclanthology.org/2021.findings-emnlp.288/. Motivates rule/context-sensitive
  moderation beyond generic toxicity; no NormVio records are added.
- Clarke et al., *Rule By Example* (2023):
  https://aclanthology.org/2023.acl-long.22/. Motivates rule-grounded exemplar
  learning. This hand-engineered frozen-cache study is not contrastive training
  and does not reproduce the paper's encoder or datasets.

Round 8's new geometry is not used here. The original frozen vectors are retained,
so the two experiments remain independently interpretable regardless of whether
Round 8 passes its scientific gate. The present study does not establish that a
reference with matching heuristic descriptors is semantically equivalent.

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
