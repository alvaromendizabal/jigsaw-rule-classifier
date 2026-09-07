# Start with the evidence

**Employer review:** open [03 · Results and model decision](notebooks/03_saved_results.ipynb), then [04 · Semantic benchmark](notebooks/04_semantic_benchmark.ipynb). The supporting notebooks explain the data audit and lexical reference. No AWS account, private data, or model download is required to read them.

## Continue in your existing SageMaker checkout

Your supplied log already shows successful bootstrap, 57 passing tests, and a completed restore. Do not repeat training to display those results.

```bash
cd "$HOME/projects/jigsaw-rule-classifier"
git pull --ff-only
.venv/bin/python scripts/execute_notebooks.py --notebook 03 --notebook 04
```

Open the canonical notebooks in `notebooks/`. The command runs their cells against the checked-in aggregate evidence, not model weights. It emits UTC start/finish events, cell progress, 15-second heartbeats, stage time, total invocation time, and `NOTEBOOKS_VERIFIED`. Matching completed notebook checkpoints are reused. The checkpoint path is printed in the log.

To execute all five public notebooks and atomically refresh their canonical files:

```bash
.venv/bin/python scripts/execute_notebooks.py --publish
```

Publication is allowed only for public aggregate notebooks. Synthetic or private Kaggle execution cannot overwrite them. A failed notebook does not replace its last-good canonical file; completed earlier notebooks remain cached. An interrupted active notebook restarts from its first cell, while its completed predecessors are reused. Review and publish changes through a feature branch and pull request, not a force push.

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

The complete source gate is `.venv/bin/python scripts/verify.py`. Training is not a prerequisite for reading or publishing the five portfolio notebooks.

When finished using Studio, stop the JupyterLab app to stop its compute billing. Do not delete the space or S3 snapshots; persistent storage remains billable.
