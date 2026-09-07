# Continue the Jigsaw project

Your Git checkout, Kaggle authentication, private configuration, and lexical baseline review are already complete. Continue in the existing **SageMaker Studio → JupyterLab → jigsaw-rules-dev** space in **Oregon (`us-west-2`)**.

## 1. Update the existing checkout and restore saved work

In the JupyterLab Terminal:

```bash
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/projects/jigsaw-rule-classifier"
git pull --ff-only
bash bootstrap.sh
uv run jigsaw restore
uv run jigsaw review
```

Bootstrap installs the locked environment, registers **Python (Jigsaw Rules)**, and runs the quality gate. Restore verifies hashes and reuses matching local files. Review selects the latest completed real experiment, verifies its saved artifacts, and recalculates metrics without fitting or loading a neural model. Expected markers are `BOOTSTRAP_COMPLETED`, `RESTORE_COMPLETED`, and `REVIEW_VERIFIED` with `data_kind: competition`.

The existing `configs/local.json` supplies private S3 settings. Keep working in the checkout under `projects/`; the earlier archive directory is historical. There is no new clone, Kaggle login, or baseline computation in this sequence.

## 2. Open the semantic comparison

Open **notebooks/04_semantic_benchmark.ipynb**, select **Python (Jigsaw Rules)**, and use **Run → Run All Cells**. It compares the saved Qwen3 semantic models with the lexical reference, showing rule macro AUC, per-rule results, probability quality, paired uncertainty intervals, runtime, and memory. Its plots and tables use saved evidence.

The current small CPU app can review the completed experiment without loading Qwen weights. `notebooks/03_saved_results.ipynb` remains available for general saved-run review. The exported `reports/private/results.json` and HTML identify real versus synthetic data explicitly.

## 3. New model work after review

Phase 2A implements a pinned frozen encoder, order-invariant example comparisons, and a fold-fitted classifier. Phase 2B will test a rule-conditioned cross-encoder and context ablations, using the same validation discipline. See [PHASE_2.md](docs/PHASE_2.md) and [ROADMAP.md](docs/ROADMAP.md).

To compute a fresh semantic experiment, use a CPU environment with at least 8 GB RAM and 4 GB free. The measured Qwen process exceeds the capacity of the current 4 GB app. Review does not require resizing. On suitable hardware:

```bash
uv run --extra semantic python scripts/verify_semantic.py
uv run --extra semantic jigsaw semantic --cloud
```

The integration check downloads pinned public model assets, verifies their hashes, and checks pooling, batching consistency, and cache reuse. Full computation records UTC events, elapsed time, throughput, peak memory, and 15-second heartbeats. Completed 64-input shards and folds are reused after interruption; an incomplete shard or fold restarts. Changed model, data, source, or execution settings create a new experiment identity.

## Continued use

- Inspect saved results: `uv run jigsaw review`.
- Save local data and experiment artifacts: `uv run jigsaw backup`.
- Verify source, tests, and notebook consistency: `uv run --extra semantic python scripts/verify.py`.
- Develop each new phase on a feature branch, document the pull request, and merge after Quality passes.

S3 backup is a snapshot of the local project. Restore the current snapshot before working in a new checkout and before publishing another backup; do not replace the latest snapshot from a partially populated directory. Original snapshots remain immutable. Git preserves source history. Do not share competition comments, private configuration, or row-level predictions in the public repository.

When finished using Studio, stop the JupyterLab app to stop compute billing. Space disk and S3 checkpoints persist; storage remains billable.

## Kaggle later

The existing standalone `kaggle/submission.ipynb` runs the lexical reference offline and creates an exact-format `submission.csv`. The semantic experiment also validates preview predictions locally, but its offline Kaggle model bundle is a later deliverable. A preview file is not a scored submission. Authenticated late-submission availability remains to be checked; the original competition ended in 2025.
