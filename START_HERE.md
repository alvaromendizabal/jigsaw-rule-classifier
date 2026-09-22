# Start with the finished project

**Final retained system: support-adapted Qwen3-4B, 0.91425 private / 0.91808 public Kaggle AUC.** The portfolio is complete. Review does not require more training, a Kaggle submission, AWS access, or private data.

## A short review path

Open [27 · Project review](notebooks/27_latest_system_checkpoint.ipynb) for the result, architecture, controlled adaptation evidence, and final decision. Continue to [03 · Detailed results](notebooks/03_saved_results.ipynb), [02 · Feature research](notebooks/02_baseline_and_review.ipynb), and [01 · Validation](notebooks/01_data_and_validation.ipynb) for deeper evidence.

The [model card](MODEL_CARD.md) distinguishes the retained 4B system from the separate post-competition research artifact. The [closeout](docs/PROJECT_CLOSEOUT.md) records exactly what is complete and what is outside this release.

## Reproduce the public review

From the repository root, install the locked project environment once, then execute the aggregate-only notebook. This does not load a model or call AWS or Kaggle. The first command installs dependencies; the second performs only local notebook rendering.

```bash
uv sync --locked --group dev
uv run python -c "from pathlib import Path; import nbformat; from nbclient import NotebookClient; p=Path('notebooks/27_latest_system_checkpoint.ipynb'); n=nbformat.read(p, as_version=4); NotebookClient(n, timeout=90, kernel_name='python3', resources={'metadata': {'path': str(Path.cwd())}}).execute(); nbformat.write(n, '/tmp/jigsaw-project-review.ipynb'); print('Saved /tmp/jigsaw-project-review.ipynb')"
```

Alternatively, open that notebook in Jupyter with the project environment and Run All. Both Plotly figures have SVG fallbacks and a subsequent-cell check. Saved outputs are already embedded for readers who do not rerun anything.

Software verification remains `uv run --extra semantic python scripts/verify.py`. The full Quality workflow additionally tests the existing historical CPU inference path and pinned encoder; it is more substantial than merely viewing the saved portfolio.

## Reproduce neural inference separately

The retained model's implementation is [scripts/kaggle_adaptation.py](scripts/kaggle_adaptation.py), with [decision-position training](scripts/decision_training.py) and [pinned settings](configs/kaggle_adaptation.json). The self-contained [Kaggle notebook](kaggle/submission.ipynb) is the neural inference entry point; [kaggle/reference.ipynb](kaggle/reference.ipynb) preserves the lexical control.

Reproduction of the neural path requires its model assets and suitable GPU resources. The original train/supplied-support boundary is mandatory. The ten-row preview is an execution check, not hidden-test performance. Do not resubmit the recorded baseline or the supplementary majority candidate simply to review this repository. [Exact scored-version receipt](reports/checkpoints/kaggle_adaptation.json) · [Delivery and restore details](docs/DELIVERY.md).

## AWS and GitHub serve different purposes

AWS preserves private working data, model state, caches, and recoverable runs. GitHub preserves public source, aggregate results, tests, and executed analysis. Publishing this closeout does not alter the AWS workspace or imply that its local checkout was fast-forwarded. Never use a broad upload, `git add .`, force push, hard reset, or `git clean` to synchronize them.

<details>
<summary>Existing tested operator continuation (optional; not part of project review)</summary>

Close notebook tabs first. This existing continuation requires `main`, retains tracked notebook edits in a named local stash, fast-forwards, and then runs its verification steps. Review any local changes first. A local stash is not a cloud backup; do not automatically pop or drop it. No continuation command was executed against AWS for this publication.

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
