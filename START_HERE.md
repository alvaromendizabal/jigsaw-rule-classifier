# Start with the evidence

**Employer review:** open [03 · Results and model decision](notebooks/03_saved_results.ipynb), then [04 · Semantic benchmark](notebooks/04_semantic_benchmark.ipynb). The supporting notebooks explain the data audit and lexical reference. No AWS account, private data, or model download is required to read them.

## Continue in your existing SageMaker checkout

Your supplied log already shows successful bootstrap, 57 passing tests, and a completed restore. Do not repeat training to display those results.

Close the project notebook tabs before updating so an old browser tab cannot autosave over an updated file. This continuation requires the existing `main` checkout and locked `.venv`; it stops on another branch rather than switching your work silently.

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

The named stash preserves tracked notebook changes, including local outputs and any source edits, before the update. It is retained for inspection with `git stash list`; **do not automatically pop or drop it**. The updated canonical notebooks come from `main`, while the pre-update local versions remain in the stash. A stash is local Git storage, not an S3 backup. No local changes to save is a successful no-op. Unrelated source changes, untracked files, private data, model weights, saved runs, and `configs/local.json` are left in place by this command. An unrelated conflicting edit, divergent branch, failed pull, or failed quality gate stops the sequence before notebook execution. Do not use `git reset --hard`, `git clean`, or a force push to bypass those protections.

Open the canonical notebooks in `notebooks/` after the command completes. It verifies all five public notebooks against the checked-in aggregate evidence, not model weights. It emits UTC start/finish events, cell progress, 15-second heartbeats, stage time, total invocation time, and `NOTEBOOKS_VERIFIED`. Matching completed notebook checkpoints are reused. The checkpoint path is printed in `logs/notebook_execution.jsonl`. Default verification does not rewrite the canonical notebook files, so verification itself does not dirty the next pull. No dependency reinstall, model download, retraining, or new AWS resource is requested.

A terminal prefix such as `^[[200~` is a paste-control sequence, not part of the command. Press **Ctrl+C**, manually type `bind 'set enable-bracketed-paste off'`, and press Enter before pasting again. This changes only the current Bash session; pasted newlines can execute commands immediately, so inspect the copied text first. Copy only the code, without the shell prompt or a trailing `~`.

To execute all five public notebooks and atomically refresh their canonical files deliberately:

```bash
.venv/bin/python scripts/execute_notebooks.py --publish
```

Publication is allowed only for public aggregate notebooks. Synthetic or private Kaggle execution cannot overwrite them. A failed notebook does not replace its last-good canonical file; completed earlier notebooks remain cached. An interrupted active notebook restarts from its first cell, while its completed predecessors are reused. Review and publish changes through a feature branch and pull request, not a force push. Routine verification does not require `--publish`.

## Recalculate private metrics without retraining

The public notebooks verify aggregate checksums and provenance; they do **not** recompute metrics from row-level predictions. The stricter private review remains:

```bash
.venv/bin/python -m jigsaw_rules.cli review
```

Your existing private `configs/local.json` and restored run files remain in place. In a new workspace, restore the current S3 snapshot before private review or any new backup. Never replace the latest snapshot from a partially populated checkout.

## Explicit model work

The semantic comparison is a completed embedding benchmark, not a verified top-performing competition model. Keep the lexical reference while testing a joint rule/comment cross-encoder with context ablations. See [PHASE_2.md](docs/PHASE_2.md).

Do not rerun Qwen on the current 4 GB Studio app: the recorded process alone peaked at 3.854 GiB. Review does not require resizing. Approve hardware, maximum runtime, and spending before any paid model experiment. Completed embedding shards and model folds resume; an active CPU solver does not resume within an iteration. GPU optimizer-state recovery is not implemented yet.

## Kaggle and verification

The standalone `kaggle/submission.ipynb` is the offline lexical reference. Its synthetic integration test is explicit:

```bash
.venv/bin/python scripts/execute_notebooks.py --synthetic
```

That test never establishes competition performance. To deliberately fit offline inference against existing real competition data, use `--kaggle`; its validated CSV and manifest are copied to `kaggle_output/`. A preview CSV is not a scored submission. The 2025 competition has ended; authenticated late-submission eligibility is still unverified.

The complete source gate is `.venv/bin/python scripts/verify.py`. `tests/test_workspace_update.py` exercises the documented update block against disposable local Git repositories; its notebook runner is a stub, so these tests verify update safety rather than notebook execution or model quality. Existing notebook tests and CI cover actual execution. Training is not a prerequisite for reading or publishing the five portfolio notebooks.

When finished using Studio, stop the JupyterLab app to stop its compute billing. Do not delete the space or S3 snapshots; persistent storage remains billable.
