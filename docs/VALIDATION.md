# Verification record

## Current notebook and experiment release

The suite contains **139 tests**: the previous 108-test foundation plus 31 feature, inference, publication, and workspace-update cases. [PR #6](https://github.com/alvaromendizabal/jigsaw-rule-classifier/pull/6) records implementation review and the final exact-head checks. The locked quality gate checks compilation, Ruff, formatting, tests, and canonical notebook sources before executing all five public notebooks in encrypted Jupyter kernels. It then verifies checkpoint reuse, standalone inference on explicitly synthetic input, the real pinned encoder, and public evidence rendering.

The implementation's [verification run 34175540097](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/runs/34175540097) executed all five notebooks in actual Jupyter kernels, verified their reuse, and exercised the 11-cell standalone inference notebook including its download link. The committed implementation and notebook outputs were inspected from the checksummed CI artifact. Final review uses the original read-only workflow, with no development source-transfer step or branch-write permission. The final CI checks committed notebook sources; it does not regenerate stale sources to make the quality gate pass.

Public notebooks render checksummed recorded aggregate evidence. **That is not new model training or private OOF metric recomputation.** The optional experiment cell in `02_baseline_and_review.ipynb` remains disabled for public execution. No competition-data results from the new four-candidate feature experiment are claimed by this release. An absent feature report prints an explicit pending-results message instead of displaying synthetic scores.

## Tested failure modes

Metric tests distinguish rule macro AUC from pooled AUC and reject undefined AUC, invalid probabilities, duplicate IDs, schema violations, and wrong submission order. Validation tests cover held-out-rule isolation, duplicate grouping, example leakage purging, training-only vocabularies/scalers, deterministic splits, and exact saved reference assignments.

Feature tests check the 2/16/8/26 dense feature groups, positive/negative example permutation invariance, finite structural cues, validation-label independence, OOF coverage, and reuse of completed folds with fitting disabled. The complete synthetic experiment preserves the reference run byte-for-byte, creates no submission CSV, and cannot export synthetic results as public competition evidence. A separate authored aggregate fixture exercises the populated analysis cell and SVG chart; it is not a model result.

Submission tests exercise validated generation, checksum-verified download HTML, unchanged-input reuse without fitting or prediction, input changes, corrupt prediction batches, invalid contracts, and interruption after a completed prediction batch. Model and batch checkpoints survive a failed notebook independently. Download links are validated as generated HTML; browser-specific click behavior is not an automated browser test.

Publication tests use isolated local Git repositories, not live GitHub writes. They verify allowlisted commits, repeated publication without empty commits, rejection of unsafe branches/origins/staged changes/stale notebook evidence, preservation of private files, and a retained commit after failed push followed by successful retry. Workspace tests preserve both ordinary public notebooks and locally executed `kaggle/submission.ipynb` during a safe update. No force push or destructive reset is used.

Runtime tests cover output corruption, interrupted stages, input/configuration changes, nested elapsed-time clocks, and heartbeat events. Notebook publication rejects synthetic/private execution, unexecuted cells, execution errors, stderr, and concurrent source changes. Failed execution preserves the last-good canonical notebook.

S3 regression tests cover interrupted transfers, content reuse, unsafe paths, corruption, local conflicts, incomplete checkouts, concurrent writers, and artifacts that change during upload. Publishing the latest pointer requires the previous ETag or first-write absence. A failed or incomplete snapshot does not replace the previous latest manifest; a new workspace must restore missing saved state before publishing.

## Recorded model evidence

The preserved lexical and semantic runs use the same 2,029 competition training rows and five saved validation assignments. Public evidence remains in `reports/baseline/` and `reports/semantic/`; private OOF predictions, model states, and embedding shards remain outside Git. Their original source/data hashes are retained rather than relabeled after presentation changes.

The historical semantic experiment finished at 2026-09-07T01:42:55Z in 1,004.708 seconds. Its historical release recomputed reported metrics from private OOF predictions and checked the original source hashes. The current frozen semantic margin has held-out rule macro AUC 0.6351 versus 0.6156 for the lexical rule/example reference; its paired improvement interval crosses zero and probability losses worsen slightly. This does not establish a reliable replacement or state-of-the-art performance.

The real pinned Qwen3 integration checks 1,024-dimensional unit vectors, single/batched consistency, and a second cache pass with encoding disabled. It uses four authored inputs and is a software integration check, not competition evaluation. Semantic tests also cover padding, invalid vectors, deduplication/order, interrupted shards, model/prompt changes, and grouped uncertainty intervals.

## Evidence and recovery boundaries

The official overview names column-averaged AUC. The project reports rule macro ROC AUC and pooled ROC AUC separately; executable official scorer parity and an external Kaggle score remain unverified. Feature intervals condition on the two observed rules and fixed predictions, and the four exploratory comparisons are not multiplicity-adjusted. They do not establish performance across arbitrary unseen policies.

Completed notebooks, feature folds, inference models, prediction batches, and embedding shards are reusable when contracts and checksums match. An interrupted active CPU fit or batch restarts; neural optimizer/RNG-state recovery remains unimplemented. S3 backups protect completed artifacts only after snapshot publication succeeds. Top-level logs are local; committed run event logs are included in backups.

The canonical submission notebook is for the user to run, validate, and download their own file. It never uploads to Kaggle. The committed notebook contains no generated submission output. Kaggle-hosted execution, authenticated late-submission eligibility, a leaderboard score, joint cross-encoder/LoRA training, and nested calibration remain separate milestones. No AWS training job or instance resize was launched for this release; an existing running Studio app and storage can still incur charges.
