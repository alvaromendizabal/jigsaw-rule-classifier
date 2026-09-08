# Start with the evidence

**Employer review:** open [03 · Results and model decision](notebooks/03_saved_results.ipynb), then [04 · Semantic benchmark](notebooks/04_semantic_benchmark.ipynb). The other three notebooks explain the audit, validation, and lexical reference. No AWS account, private data, or model download is required to read the five executed public notebooks.

## Update an existing SageMaker checkout

Use the existing locked environment and restored artifacts. Close project notebook tabs before updating so an old browser tab cannot autosave over a newer file. This block preserves tracked notebook changes in a named stash, requires `main`, fast-forwards, and verifies the source and five public notebooks. It does not repeat bootstrap or original training.

<!-- workspace-update:start -->
```bash
cd "$HOME/projects/jigsaw-rule-classifier" &&
test "$(git branch --show-current)" = main &&
git stash push -m "notebook-session-$(date -u +%Y%m%dT%H%M%SZ)" -- notebooks kaggle/submission.ipynb &&
git pull --ff-only origin main &&
.venv/bin/python scripts/verify.py &&
.venv/bin/python scripts/execute_notebooks.py
```
<!-- workspace-update:end -->

The stash is retained for inspection with `git stash list`; do not automatically pop or drop it. No local notebook changes is a successful no-op. Unrelated edits, untracked files, private data, model weights, runs, and `configs/local.json` remain in place. An unrelated conflicting edit, divergent branch, failed pull, or failed quality gate stops the sequence. Do not use `git reset --hard`, `git clean`, or a force push. A stash is local Git storage, not an S3 backup.

A terminal prefix such as `^[[200~` is a paste-control sequence. Press **Ctrl+C**, manually type `bind 'set enable-bracketed-paste off'`, and press Enter before pasting again. This changes only the current Bash session; pasted newlines can execute immediately. Copy only code, without the prompt or a trailing `~`.

## Next: run the controlled feature experiment, then publish from AWS

Your reference runs and saved split registry remain unchanged. The new CPU experiment tests four candidates: rule-text similarity, positive/negative support contrasts, writing structure, and their combination. It does not reload Qwen, rerun the reference cross-validation, choose a model automatically, or generate a submission.

Run in the existing, updated SageMaker environment:

```bash
.venv/bin/python -m jigsaw_rules.cli features --cloud --export &&
.venv/bin/python scripts/build_notebooks.py &&
.venv/bin/python scripts/execute_notebooks.py --publish &&
.venv/bin/python -m jigsaw_rules.cli backup
```

Alternatively, open `02_baseline_and_review.ipynb`, select **Python (Jigsaw Rules)**, set `RUN_FEATURE_EXPERIMENT = True` in its final experiment cell, and run it. Keep `CHECKPOINT_TO_S3 = True`. Before publication, run the last three commands above: the builder restores the default no-training notebook source, then execution renders the measured aggregate results. Do not reapply an older stashed notebook over this version.

Expected events are `FEATURES_COMPLETED`, `FEATURE_AGGREGATES_EXPORTED`, `NOTEBOOKS_VERIFIED` with `published=true`, and `snapshot_committed`. The notebook shows candidate metrics, paired intervals, descriptive structure-by-label statistics, fold coefficient sensitivity, and recorded fit time. The intervals condition on the two observed rules and fixed predictions; four exploratory comparisons do not establish a universal winner.

To commit and push the verified public results **from SageMaker**, use:

```bash
.venv/bin/python scripts/execute_notebooks.py --publish --push-branch results/feature-ablation
```

The helper requires the expected Jigsaw origin, a clean Git index, current Jupyter-executed canonical sources, and evidence matching current source/data hashes. It stages only the five canonical notebooks and the five fixed files under `reports/features/`; it never stages data, private predictions, local configuration, submission CSVs, or checkpoints. Unrelated edits cause a refusal, not deletion. It creates a documented commit, pushes the results branch without force, and prints `GITHUB_PUSHED`. Repeating it on that same branch reuses intact notebook checkpoints and does not make an empty commit. A failed push retains the local commit for retry.

Git write authentication must be available inside SageMaker; linking GitHub to ChatGPT does not configure the terminal's credential helper. Never paste tokens into a notebook or chat. Configure a missing Git author locally with `git config user.name "Alvaro Mendizabal"` and `git config user.email "108156083+alvaromendizabal@users.noreply.github.com"`. Open the pushed branch's pull request on GitHub, inspect its allowlisted files and metrics, wait for Quality, then merge. This helper does not merge or claim independent human review. After an earlier results branch has been merged, start a new `results/...` branch name rather than resetting it.

## Generate and download your own submission in the notebook

Open the existing **`kaggle/submission.ipynb`** in SageMaker, select **Python (Jigsaw Rules)**, and run all cells with `GENERATE_SUBMISSION = True`. The notebook detects the project data directory, fits or reuses the retained lexical reference, predicts in resumable batches, validates row IDs/order and finite probabilities, writes a checksummed manifest, and displays **Download submission.csv**. Set the flag to `False` to disable generation. There is no automatic upload or Kaggle submission.

The fitted model and completed prediction batches persist under `runs/submission_cache/`, independently of a notebook kernel failure. Source, inputs, package versions, seed, and batch size are fingerprinted. Changed inputs or corrupt artifacts cannot silently reuse stale predictions. The final download link verifies the actual generated bytes; files larger than 10 MB use the file browser to avoid embedding an unbounded payload. The current model remains the lexical reference until further evidence justifies promotion.

Generated CSV/manifest convenience copies remain under ignored `kaggle_output/`. Run `.venv/bin/python -m jigsaw_rules.cli backup` after generation to preserve inference checkpoints in S3. Neither the file nor a successful local run is a leaderboard score. Private reference metric recomputation remains `.venv/bin/python -m jigsaw_rules.cli review --run-id c15c2c2318fc0ed619c6`; it does not train models.

## Durability and progress

The runners emit UTC timestamps, cell/batch progress, 15-second heartbeats, stage time, and total invocation time. Live logs are `logs/notebook_execution.jsonl`, `logs/features.jsonl`, `logs/commands.jsonl`, and `logs/cloud.jsonl`. Committed event logs under `runs/` are backed up; the top-level `logs/` directory is local. Run backup, training, and restore sequentially.

An incomplete checkout cannot publish a latest snapshot that omits previously saved paths. Restore missing work first; never force-delete local conflicts. Access errors are not treated as an empty bucket. Conditional publication compares the previous ETag (`If-Match`) or requires an absent first snapshot (`If-None-Match`), preventing a competing backup from silently replacing the latest pointer. Old content-addressed snapshots remain. Changed model/data artifacts during upload stop publication; JSONL logs are captured as byte snapshots while they may continue appending. An interrupted upload can reuse already-uploaded content objects on retry. [AWS conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html).

Completed notebooks, model folds, and embedding shards are reusable when their contracts and checksums match. An active interrupted notebook, CPU solver fit, or embedding shard restarts; completed inference models and prediction batches remain reusable even when the notebook itself restarts. Neural optimizer-state recovery is not implemented. No software can guarantee that external services never fail; failed work must remain visible without destroying valid checkpoints.

Routine public verification does not rewrite canonical notebooks. Deliberate publication is separate:

```bash
.venv/bin/python scripts/execute_notebooks.py --publish
```

Only public aggregate evidence can be published. Private or synthetic execution cannot overwrite the five public notebooks. Failed execution preserves the last-good canonical file. Publish intentional source/output changes through a feature branch and reviewed pull request. The aggregate reader checks committed checksums and provenance; private `jigsaw review` recomputes metrics from saved row-level predictions.

## Run and submit on Kaggle

Download the canonical `kaggle/submission.ipynb` from SageMaker and import it into a Kaggle notebook. Attach the official `jigsaw-agile-community-rules` competition data, use CPU/no accelerator, disable internet, and select **Save Version → Save & Run All**. The notebook must produce `/kaggle/working/submission.csv` and `SUBMISSION_VALIDATED`.

For this code competition, submit the successful saved notebook version through the late-submission flow when available to the signed-in account. The downloaded preview CSV is not a hidden-test score. The October 23, 2025 deadline has passed; a visible public late-submission entry does not establish eligibility for a specific account. Record a score only after Kaggle returns one. [Official competition](https://www.kaggle.com/competitions/jigsaw-agile-community-rules).

The explicitly synthetic software-only inference check is `scripts/execute_notebooks.py --synthetic`. It never establishes competition performance.

## Next model experiment, not another setup cycle

Run and analyze the new feature ablations before choosing the next model. The semantic benchmark is completed, not a proven top-performing model. Preserve the lexical reference while testing a joint rule/comment cross-encoder with support-context ablations and training-fold-only calibration; see [PHASE_2.md](docs/PHASE_2.md). Its implementation, measured comparison, and optimizer-state recovery remain outstanding.

Do not rerun Qwen on the recorded 4 GB Studio app: the process alone peaked at 3.854 GiB. Saved review needs no resize. Approve hardware, maximum runtime, and spending before paid model experiments. Stop the JupyterLab app when finished; do not delete the space or S3 snapshots. Persistent storage remains billable.
