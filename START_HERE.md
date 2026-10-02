# Start here

**Retained scored system: support-adapted Qwen3-4B, 0.91425 private / 0.91808 public Kaggle ROC AUC.**

The portfolio is reviewable without AWS, Kaggle, model weights, or private data. Frontier research has been reopened after the original closeout; the latest public evidence is the completed five-model Qwen2.5 diversity/ensemble study.

## Fastest review path

1. Open [29 · Five-model frontier](notebooks/29_five_model_frontier_review.ipynb) for the latest Qwen2.5 diversity/ensemble result.
2. Read [FIVE_MODEL_FRONTIER.md](docs/FIVE_MODEL_FRONTIER.md) for the staged screen → full OOF → compression → global validation decision path.
3. Open [28 · Qwen3-14B frontier](notebooks/28_qwen14b_frontier_review.ipynb) for the backbone-scaling result that motivated diversity-first research.
4. Open [27 · Project review](notebooks/27_latest_system_checkpoint.ipynb) for the retained scored system and overall research narrative.
5. Read the [model card](MODEL_CARD.md) for data/operational boundaries, then continue to [03 · Detailed results](notebooks/03_saved_results.ipynb), [02 · Feature research](notebooks/02_baseline_and_review.ipynb), and [01 · Validation](notebooks/01_data_and_validation.ipynb).

## Reproduce the public notebooks

The aggregate notebooks require no model loading and no cloud access.

```bash
uv sync --locked --group dev
uv run python -c "from pathlib import Path; import nbformat; from nbclient import NotebookClient; p=Path('notebooks/29_five_model_frontier_review.ipynb'); n=nbformat.read(p, as_version=4); NotebookClient(n, timeout=90, kernel_name='python3', resources={'metadata': {'path': str(Path.cwd())}}).execute(); nbformat.write(n, '/tmp/jigsaw-five-model-frontier-review.ipynb'); print('Saved /tmp/jigsaw-five-model-frontier-review.ipynb')"
```

Saved Plotly and SVG outputs are already embedded.

## Reproduce the retained neural inference separately

The retained scored model remains [scripts/kaggle_adaptation.py](scripts/kaggle_adaptation.py), with [decision-position training](scripts/decision_training.py), [pinned settings](configs/kaggle_adaptation.json), and the self-contained [submission notebook](kaggle/submission.ipynb).

The Qwen3-14B and five-model frontier results are **development evidence only**. Their aggregate checkpoints are [reports/checkpoints/qwen14b_frontier.json](reports/checkpoints/qwen14b_frontier.json) and [reports/checkpoints/five_model_frontier.json](reports/checkpoints/five_model_frontier.json). The exact promoted ensemble weights and private row-level predictions remain outside GitHub.

## AWS and GitHub serve different purposes

**AWS:** raw comments/labels, row-level predictions, model weights, optimizer checkpoints, caches, resumable training state, full logs.

**GitHub:** source, compact configs, tests, aggregate results, attribution, and executed review notebooks.

Do not treat GitHub as an AWS mirror. Do not publish raw competition rows, private predictions, model weights, caches, credentials, or full training logs.

## Current score-focused direction

The strongest public development evidence now favors **complementary multi-backbone ranking** over standalone parameter count. Qwen2.5-14B is weaker alone than the incumbent ensemble, but a fixed five-model prior improves both observed policies and passes the grouped-bootstrap promotion gate. A compact deployment control was rejected on stability, and Deep Mutual Learning remains blocked before a valid scientific test because historical source parity is incomplete.

The five-model candidate is frozen at the development level but **has not yet received a Kaggle score**. The next competition action is one actual scored submission. Model development and validation remain in AWS; Kaggle is reserved for the real submission rather than exploratory preflight work.

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
