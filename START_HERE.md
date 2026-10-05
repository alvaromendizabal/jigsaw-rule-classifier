# Start here

**Retained scored system: support-adapted Qwen3-4B · 0.91808 public / 0.91425 private ROC AUC · +0.29469 private AUC over the lexical baseline.**

This repository is designed for two audiences at once: a hiring manager who wants the story quickly, and an ML practitioner who wants to inspect the evidence, validation, and engineering.

## Pick a review path

### 30 seconds · recruiter / hiring manager
Read the [README](README.md).

It gives the problem, headline result, architecture, selected research evidence, stack, and the three artifacts worth opening.

### 3–5 minutes · technical hiring manager
Open [27 · Project overview](notebooks/27_latest_system_checkpoint.ipynb).

It shows the retained system, the project’s measured baseline-to-neural improvement, adaptation evidence, and the overall research narrative in one executed notebook.

### 10–15 minutes · ML engineer / data scientist
Open:

1. [29 · Five-model frontier](notebooks/29_five_model_frontier_review.ipynb) — multi-backbone diversity, OOF ensemble evidence, stability gates, and rejected controls.
2. [01 · Validation](notebooks/01_data_and_validation.ipynb) — whole-policy holdouts, text-purge boundaries, grouped evaluation, and leakage controls.
3. [Model card](MODEL_CARD.md) — retained system, limitations, and public/private artifact boundaries.

### Deep research review
Continue to:

- [30 · Pseudo-supervision frontier](notebooks/30_pseudo_supervision_frontier_review.ipynb) — owned pseudo-supervision and public external-data experiments.
- [Post-closeout frontier](docs/POST_CLOSEOUT_FRONTIER.md) — Llama adaptation, pairwise ranking, semantic retrieval, ModernBERT/Ettin context, NLI transfer, and the decisions they produced.
- [28 · Qwen3-14B frontier](notebooks/28_qwen14b_frontier_review.ipynb) — capacity vs diversity.
- [02 · Feature research](notebooks/02_baseline_and_review.ipynb) — the larger 323-fit feature/generalization campaign.
- [03 · Detailed results](notebooks/03_saved_results.ipynb) — broader experiment decisions and comparisons.

## What this project demonstrates

- **End-to-end model ownership:** data audit → adaptation → inference → validation → ensemble selection → publication.
- **Modern NLP:** Qwen3, Qwen2.5, Phi, DeBERTa, LoRA, teacher/student soft labels, decision-token scoring.
- **Transfer-aware data science:** whole-policy holdouts, grouped OOF evidence, ablations, uncertainty, promotion gates, negative-result discipline.
- **AWS/GPU engineering:** SageMaker, L4 benchmarking, VRAM debugging, checkpoint reuse, gradient checkpointing, resumability.
- **Reproducibility:** pinned revisions, checksums, immutable experiment contracts, executed notebooks, CI, machine-readable checkpoints.
- **Judgment:** larger models and more data are not automatically promoted; every candidate has to earn its place through the validation contract.

## Reproduce the public overview

The employer-facing aggregate notebooks require no model loading and no cloud account.

```bash
uv sync --locked --group dev
uv run python -c "from pathlib import Path; import nbformat; from nbclient import NotebookClient; p=Path('notebooks/27_latest_system_checkpoint.ipynb'); n=nbformat.read(p, as_version=4); NotebookClient(n, timeout=90, kernel_name='python3', resources={'metadata': {'path': str(Path.cwd())}}).execute(); nbformat.write(n, '/tmp/jigsaw-project-overview.ipynb'); print('Saved /tmp/jigsaw-project-overview.ipynb')"
```

Saved Plotly/SVG evidence is already embedded in the tracked notebook.

## AWS and GitHub serve different purposes

**AWS:** raw comments/labels, row-level predictions, model weights, optimizer checkpoints, teacher-score arrays, caches, resumable training state, and full operational logs.

**GitHub:** source, compact configs, aggregate results, attribution, tests, and executed review notebooks.

The public repository is intentionally semi-reproducible rather than an AWS mirror.

## Current research direction

The accepted development champion remains the fixed five-model complementary ranking ensemble. Post-closeout research has now tested complementary Llama adaptation, pairwise ranking, semantic support retrieval, a ModernBERT/Ettin encoder with support context, and pretrained NLI transfer. The Llama route produced the strongest new point estimate but did not clear the registered confidence gate; the other formulations were preserved as valid negatives rather than micro-tuned indefinitely.

Active AWS research is now focused on **Qwen3-14B provenance and live replay** so higher-upside cross-model training can be evaluated from a verified historical component. Cached historical rankings have been reconciled, but live model/prompt parity remains a prerequisite. The public aggregate evidence is summarized in [POST_CLOSEOUT_FRONTIER.md](docs/POST_CLOSEOUT_FRONTIER.md).

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
