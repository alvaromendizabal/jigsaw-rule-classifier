# Start with the evidence

**Employer review:** open [03 · Results and model decision](notebooks/03_saved_results.ipynb), then [04 · Semantic benchmark](notebooks/04_semantic_benchmark.ipynb). The other three notebooks explain the audit, validation, and lexical reference. No AWS account, private data, or model download is required to read the five executed public notebooks.

## Update an existing SageMaker checkout

Use the existing locked environment and restored artifacts. Close project notebook tabs before updating so an old browser tab cannot autosave over a newer file. This block preserves tracked notebook changes in a named stash, requires `main`, fast-forwards, and verifies the source and five public notebooks. It does not repeat bootstrap or original training.

<!-- workspace-update:start -->
```bash
cd "$HOME/projects/jigsaw-rule-classifier" &&
test "$(git branch --show-current)" = main &&
git stash push -m "notebook-session-$(date -u +%Y%m%dT%H%M%SZ)" -- notebooks &&
git pull --ff-only origin main &&
.venv/bin/python scripts/verify.py &&
.venv/bin/python scripts/execute_notebooks.py
```
<!-- workspace-update:end -->

The stash is retained for inspection with `git stash list`; do not automatically pop or drop it. No local notebook changes is a successful no-op. Unrelated edits, untracked files, private data, model weights, runs, and `configs/local.json` remain in place. An unrelated conflicting edit, divergent branch, failed pull, or failed quality gate stops the sequence. Do not use `git reset --hard`, `git clean`, or a force push. A stash is local Git storage, not an S3 backup.

A terminal prefix such as `^[[200~` is a paste-control sequence. Press **Ctrl+C**, manually type `bind 'set enable-bracketed-paste off'`, and press Enter before pasting again. This changes only the current Bash session; pasted newlines can execute immediately. Copy only code, without the prompt or a trailing `~`.

## After the notebooks pass: real-data submission handoff

Run this in the fully restored checkout after the quality gate passes:

```bash
cd "$HOME/projects/jigsaw-rule-classifier" &&
.venv/bin/python -m jigsaw_rules.cli review --run-id c15c2c2318fc0ed619c6 &&
.venv/bin/python scripts/execute_notebooks.py --kaggle &&
.venv/bin/python -m jigsaw_rules.cli backup
```

This recalculates the selected lexical reference's metrics from saved OOF predictions, executes the standalone notebook against real downloaded competition data, and snapshots completed work to the existing private S3 bucket. Expected events: `REVIEW_VERIFIED`, `NOTEBOOKS_VERIFIED` with `mode="kaggle"`, and `snapshot_committed`.

The first offline inference execution fits the inexpensive lexical reference once on all training rows. Matching completed notebook checkpoints are reused subsequently. It does not rerun original cross-validation, load Qwen, download weights, launch a training job, or resize compute. A CSV establishes inference validity, not new model quality.

Inspect `reports/private/report.html`, `kaggle_output/submission.csv`, and `kaggle_output/submission_manifest.json`. The executed inference notebook and original CSV/manifest live in checksummed `runs/notebook_execution/` checkpoints, included in backup. `kaggle_output/` contains ignored convenience copies; restoring the snapshot and rerunning inference recreates them from a matching cache. Do not add private reports, data, predictions, credentials, or local configuration to GitHub.

## Durability and progress

The runners emit UTC timestamps, cell/batch progress, 15-second heartbeats, stage time, and total invocation time. Live logs are `logs/notebook_execution.jsonl`, `logs/commands.jsonl`, and `logs/cloud.jsonl`. Committed event logs under `runs/` are backed up; the top-level `logs/` directory is local. Run backup, training, and restore sequentially.

An incomplete checkout cannot publish a latest snapshot that omits previously saved paths. Restore missing work first; never force-delete local conflicts. Access errors are not treated as an empty bucket. Conditional publication compares the previous ETag (`If-Match`) or requires an absent first snapshot (`If-None-Match`), preventing a competing backup from silently replacing the latest pointer. Old content-addressed snapshots remain. Changed model/data artifacts during upload stop publication; JSONL logs are captured as byte snapshots while they may continue appending. An interrupted upload can reuse already-uploaded content objects on retry. [AWS conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html).

Completed notebooks, model folds, and embedding shards are reusable when their contracts and checksums match. An active interrupted notebook, CPU solver fold, or embedding shard restarts. Neural optimizer-state recovery is not implemented. No software can guarantee that external services never fail; failed work must remain visible without destroying valid checkpoints.

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

The semantic benchmark is completed, not a proven top-performing model. Preserve the lexical reference while testing a joint rule/comment cross-encoder with support-context ablations and training-fold-only calibration; see [PHASE_2.md](docs/PHASE_2.md). Its implementation, measured comparison, and optimizer-state recovery remain outstanding.

Do not rerun Qwen on the recorded 4 GB Studio app: the process alone peaked at 3.854 GiB. Saved review needs no resize. Approve hardware, maximum runtime, and spending before paid model experiments. Stop the JupyterLab app when finished; do not delete the space or S3 snapshots. Persistent storage remains billable.
