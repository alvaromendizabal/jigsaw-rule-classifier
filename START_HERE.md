# Start here

**Retained scored system: support-adapted Qwen3-4B, 0.91425 private / 0.91808 public Kaggle ROC AUC.**

The portfolio is reviewable without AWS, Kaggle, model weights, or private data. The latest public evidence extends the promoted five-model candidate with two completed AWS supervision stress tests: owned pseudo-supervision and public external soft labels.

## Fastest review path

1. Open [30 · Pseudo-supervision frontier](notebooks/30_pseudo_supervision_frontier_review.ipynb) for the latest completed AWS research and negative-result decisions.
2. Read [PSEUDO_SUPERVISION_FRONTIER.md](docs/PSEUDO_SUPERVISION_FRONTIER.md) for the E33/E36 supervision, external-data and L4 engineering evidence.
3. Open [29 · Five-model frontier](notebooks/29_five_model_frontier_review.ipynb) for the promoted Qwen2.5 diversity/ensemble candidate.
4. Open [28 · Qwen3-14B frontier](notebooks/28_qwen14b_frontier_review.ipynb) for the backbone-scaling result that motivated diversity-first research.
5. Open [27 · Project review](notebooks/27_latest_system_checkpoint.ipynb) for the retained scored system and overall research narrative.
6. Read the [model card](MODEL_CARD.md), then continue to [03 · Detailed results](notebooks/03_saved_results.ipynb), [02 · Feature research](notebooks/02_baseline_and_review.ipynb), and [01 · Validation](notebooks/01_data_and_validation.ipynb).

## Reproduce the public notebooks

The aggregate notebooks require no model loading and no cloud access.

```bash
uv sync --locked --group dev
uv run python -c "from pathlib import Path; import nbformat; from nbclient import NotebookClient; p=Path('notebooks/30_pseudo_supervision_frontier_review.ipynb'); n=nbformat.read(p, as_version=4); NotebookClient(n, timeout=90, kernel_name='python3', resources={'metadata': {'path': str(Path.cwd())}}).execute(); nbformat.write(n, '/tmp/jigsaw-pseudo-supervision-frontier-review.ipynb'); print('Saved /tmp/jigsaw-pseudo-supervision-frontier-review.ipynb')"
```

Saved Plotly and SVG outputs are already embedded.

## Reproduce the retained neural inference separately

The retained scored model remains [scripts/kaggle_adaptation.py](scripts/kaggle_adaptation.py), with [decision-position training](scripts/decision_training.py), [pinned settings](configs/kaggle_adaptation.json), and the self-contained [submission notebook](kaggle/submission.ipynb).

The Qwen3-14B, five-model, and supervision-frontier results are **development evidence only**. Their aggregate checkpoints are [reports/checkpoints/qwen14b_frontier.json](reports/checkpoints/qwen14b_frontier.json), [reports/checkpoints/five_model_frontier.json](reports/checkpoints/five_model_frontier.json), and [reports/checkpoints/pseudo_supervision_frontier.json](reports/checkpoints/pseudo_supervision_frontier.json). Exact ensemble weights, teacher-score arrays, and private row-level predictions remain outside GitHub.

## AWS and GitHub serve different purposes

**AWS:** raw comments/labels, row-level predictions, model weights, optimizer checkpoints, caches, resumable training state, full logs.

**GitHub:** source, compact configs, tests, aggregate results, attribution, and executed review notebooks.

Do not treat GitHub as an AWS mirror. Do not publish raw competition rows, private predictions, model weights, caches, credentials, or full training logs.

## Current score-focused direction

The strongest public development candidate remains the **five-model complementary ranking ensemble**. Two follow-on supervision mechanisms have now been tested without leaderboard selection: small-cohort Qwen pseudo-supervision was a valid negative, while public external soft labels improved a complementary DeBERTa student relative to its control but still trailed the incumbent and regressed one policy.

The five-model candidate is frozen at the development level and **has not yet received a Kaggle score**. The next AWS research step is a bounded cross-rule hard-negative transfer screen using immutable cached representations; Deep Mutual Learning remains blocked until exact 14B source/prompt parity is recovered. Kaggle remains reserved for real scored submissions rather than exploratory development.

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
