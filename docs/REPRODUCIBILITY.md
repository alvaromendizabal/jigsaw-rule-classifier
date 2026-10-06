# Reproducibility guide

This repository is intentionally **semi-reproducible**. It publishes the code, validation logic, compact experiment contracts, aggregate evidence, executed notebooks, and regression tests needed to review the engineering and scientific decisions while keeping competition-sensitive state private.

## Public reproducibility tiers

### Tier 1 · Review the evidence

No model download is required.

```bash
uv sync --locked --group dev
uv run python scripts/verify.py
```

This validates Python compilation, lint/formatting, the public pytest suite, and notebook-build determinism.

For the shortest evidence path, inspect:

- `README.md`
- `CASE_STUDY.md`
- `START_HERE.md`
- `notebooks/27_latest_system_checkpoint.ipynb`
- `notebooks/29_five_model_frontier_review.ipynb`
- `docs/POST_CLOSEOUT_FRONTIER.md`

### Tier 2 · Re-execute public evidence notebooks

The GitHub Actions workflow executes the public evidence path from a clean checkout with the locked environment.

```bash
uv sync --locked --extra semantic --group dev
uv run --extra semantic python scripts/execute_notebooks.py --publish
```

The notebooks use aggregate/public receipts rather than raw private competition rows.

### Tier 3 · Exercise portable inference and model plumbing

CI also runs the synthetic/offline inference path and verifies the pinned public encoder:

```bash
uv run --extra semantic python scripts/execute_notebooks.py --synthetic
uv run --extra semantic python scripts/verify_submission.py
uv run --extra semantic python scripts/verify_semantic.py
```

These paths demonstrate packaging, schema validation, inference orchestration, and model identity checks without exposing the private AWS research state.

## Reproducibility boundary

### Published

- reusable Python modules
- validation and metric code
- compact configuration contracts
- aggregate experiment receipts
- executed review notebooks
- synthetic/offline test fixtures
- regression tests
- pinned dependency lockfile
- GitHub Actions quality workflow
- model/data cards and research decisions

### Retained privately in AWS

- raw competition comments and labels
- row-level predictions
- exact private ensemble construction
- model and optimizer checkpoints
- teacher-score arrays
- private caches
- full prompts and competition-specific operational state
- full cloud logs and internal run directories

This boundary is deliberate: it makes the project inspectable and testable without publishing the artifacts that would reconstruct the competitive system end to end.

## Scientific reproducibility

The public evidence follows three rules:

1. **Official scores and development scores stay separate.**
2. **Negative experiments are preserved when they answer a preregistered question.**
3. **A higher point estimate is not promoted when stability or uncertainty gates fail.**

The latest public cross-model receipt is:

- `reports/checkpoints/cross_model_frontier_20261006.json`

It records only aggregate metrics, matched-control comparisons, and decisions. It excludes row-level outputs, exact ensemble weights, private paths, and checkpoint state.

## Clean-room review contract

An external reviewer should be able to verify:

- environment locking and installability
- metric and validation behavior
- notebook execution
- aggregate experiment arithmetic
- public/private boundary tests
- offline/synthetic inference
- provenance and checksum handling

An external reviewer should **not** be able to recreate the private leaderboard system solely from this repository.

## CI as executable evidence

`.github/workflows/quality.yml` runs from a clean checkout and currently covers:

- compile
- Ruff lint and formatting
- pytest
- notebook build checks
- public notebook execution
- checkpoint-reuse behavior
- synthetic offline inference
- original-preview notebook verification
- pinned encoder verification
- evidence rendering

That workflow is the public reproducibility contract for the repository.
