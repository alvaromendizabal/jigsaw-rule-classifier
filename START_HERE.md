# Start here

**Retained scored system: support-adapted Qwen3-4B, 0.91425 private / 0.91808 public Kaggle ROC AUC.**

The portfolio is reviewable without AWS, Kaggle, model weights, or private data. Frontier research has been reopened after the original closeout; the latest public evidence is the completed Qwen3-14B AWS study.

## Fastest review path

1. Open [28 · Qwen3-14B frontier](notebooks/28_qwen14b_frontier_review.ipynb) for the latest backbone/diversity experiment.
2. Open [27 · Project review](notebooks/27_latest_system_checkpoint.ipynb) for the retained scored system and overall research narrative.
3. Read the [model card](MODEL_CARD.md) for data/operational boundaries.
4. Read [QWEN14B_FRONTIER.md](docs/QWEN14B_FRONTIER.md) for the latest AWS execution and decision.
5. Continue to [03 · Detailed results](notebooks/03_saved_results.ipynb), [02 · Feature research](notebooks/02_baseline_and_review.ipynb), and [01 · Validation](notebooks/01_data_and_validation.ipynb).

## Reproduce the public notebooks

The aggregate notebooks require no model loading and no cloud access.

```bash
uv sync --locked --group dev
uv run python -c "from pathlib import Path; import nbformat; from nbclient import NotebookClient; p=Path('notebooks/28_qwen14b_frontier_review.ipynb'); n=nbformat.read(p, as_version=4); NotebookClient(n, timeout=90, kernel_name='python3', resources={'metadata': {'path': str(Path.cwd())}}).execute(); nbformat.write(n, '/tmp/jigsaw-qwen14b-frontier-review.ipynb'); print('Saved /tmp/jigsaw-qwen14b-frontier-review.ipynb')"
```

Saved Plotly and SVG outputs are already embedded.

## Reproduce the retained neural inference separately

The retained scored model remains [scripts/kaggle_adaptation.py](scripts/kaggle_adaptation.py), with [decision-position training](scripts/decision_training.py), [pinned settings](configs/kaggle_adaptation.json), and the self-contained [submission notebook](kaggle/submission.ipynb).

The Qwen3-14B frontier result is **development evidence only**. Its public configuration is [configs/qwen3_14b_frontier.json](configs/qwen3_14b_frontier.json), and its aggregate checkpoint is [reports/checkpoints/qwen14b_frontier.json](reports/checkpoints/qwen14b_frontier.json).

## AWS and GitHub serve different purposes

**AWS:** raw comments/labels, row-level predictions, model weights, optimizer checkpoints, caches, resumable training state, full logs.

**GitHub:** source, compact configs, tests, aggregate results, attribution, and executed review notebooks.

Do not treat GitHub as an AWS mirror. Do not publish raw competition rows, private predictions, model weights, caches, credentials, or full training logs.

## Current score-focused direction

The latest evidence favors **model diversity over standalone parameter count**. The 14B model is weaker alone on the fixed development cohort, while a fixed 4B+14B rank blend improves both observed policies. The next milestone is leakage-safe multi-model OOF ensemble selection across preserved 4B, 8B, 14B, and Phi predictions. Only a fixed candidate that survives those AWS gates should be sent to Kaggle for one official score.

<details>
<summary>Existing tested operator continuation</summary>

Close notebook tabs first. This continuation requires `main`, preserves tracked notebook edits in a named local stash, fast-forwards only, and runs the quality gate before notebook execution. Review local changes first; a stash is not a cloud backup and is not automatically popped or dropped.

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

Private data, untracked artifacts, model weights, and environments remain outside publication. Stop on conflicts or a failed gate rather than discarding local work.

</details>
