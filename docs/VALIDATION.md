# Verification record

## Notebook publication

The suite contains **77 tests**, including 20 notebook checkpoint/publication regressions added to the 57-test foundation. [Candidate CI run 34079272166](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/runs/34079272166) passed compilation, Ruff, formatting, tests, source consistency, five encrypted Jupyter executions on committed real aggregate evidence, checkpoint reuse, standalone synthetic Kaggle inference, and the real pinned encoder integration. The public executions completed in 9.239 seconds; the second pass reused all five in 0.121 seconds. These are notebook-rendering times, not model benchmark times.

The executed public notebooks are committed at their canonical paths. Final CI checks these committed sources before execution; it does not regenerate stale sources to make the check pass. [PR #3](https://github.com/alvaromendizabal/jigsaw-rule-classifier/pull/3) records the final publication commit and CI result. Public rendering verifies aggregate integrity and provenance; it does not replace private OOF metric recomputation.

Notebook execution records UTC timestamps, per-cell progress, 15-second heartbeats, and nested stage/total clocks. Atomic publication rejects synthetic/private runs, stale source, unexecuted cells, execution errors, and stderr. Reuse requires matching source, input, environment, and output hashes. Completed notebooks are reused; an active interrupted notebook restarts from its first cell. Source regeneration retains verified outputs only when the generated narrative and code remain unchanged.

## Tested failure modes

The tests distinguish macro per-rule AUC from pooled AUC; reject undefined AUC, invalid probabilities, schema violations, duplicate IDs, and wrong submission order; and verify held-out-rule isolation, duplicate grouping, example leakage purging, vocabulary isolation, and deterministic splits.

Runtime tests cover reuse without fitting, output corruption, interrupted stages, input/configuration changes, nested timing, and heartbeat events. Notebook tests additionally reject unapproved paths, synthetic provenance, mutated public artifacts, and concurrent canonical-source edits. An intentionally failed cell must leave its canonical notebook unchanged and must not create a completed checkpoint.

S3 tests cover interrupted transfers, manifest preservation, content reuse, unsafe paths, corruption, and local conflict protection. Download tests verify file-boundary resume and use of manually supplied files without network access.

## Recorded model evidence

The original baseline and semantic runs use the same 2,029 competition training rows and five saved validation assignments. Their public aggregate evidence is in `reports/baseline/` and `reports/semantic/`. Private OOF predictions, model states, and embedding shards remain outside Git. Original source/data hashes are retained rather than relabeled as new experiments after presentation changes.

The historical semantic release recomputed every reported metric from private OOF predictions and verified its 16 source-module hashes against the original run commit. The full real experiment finished at 2026-09-07T01:42:55Z in 1,004.708 seconds. These are historical training/review checks, not work repeated by the public notebook runner.

The real pinned Qwen3 encoder integration checks 1,024-dimensional unit vectors, single/batched consistency, and a second cache pass with encoding disabled. Its historical local four-example check measured a maximum absolute vector difference of 2.403e-7, took 16.217 seconds after downloading weights, and peaked at 3.831 GiB RSS. This is a software integration measurement, not a competition score or a throughput benchmark. [Recorded semantic CI](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/runs/34073870803).

Semantic tests cover both padding directions, invalid vectors, input order and deduplication, interrupted shards, corruption, model/prompt changes, feature ordering, fold-only scaling/fitting, reference-split identity, source-data identity, reuse without fitting, rejection of test encoders on real data, and paired grouped uncertainty intervals.

## Evidence boundaries

Five public notebooks now execute against real, reviewed aggregate evidence. The standalone Kaggle integration uses explicitly synthetic input and never establishes competition performance. The in-process engine tests cell logic but not Jupyter transport; the GitHub Actions workflow uses ordinary Jupyter kernels with CurveZMQ encryption required. No warning suppression is used.

Completed CPU folds and embedding shards resume; interrupted solver iterations do not. Frozen embeddings are cached in 64-input shards and can be reused without loading weights. Fine-tuning optimizer/RNG checkpointing remains a later phase. Snapshots assume one writer; a new workspace must restore its saved state before publishing a replacement manifest.

Kaggle-hosted execution, an authenticated late submission, a leaderboard score, joint cross-encoder/LoRA training, and nested calibration are not completed by this publication milestone. No AWS training job or instance resize was launched for it. A running Studio app and persistent storage can still incur charges.
