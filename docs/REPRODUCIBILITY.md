# Reproducibility guide

The public review supports runnable components and aggregate evidence. It publishes the code, validation logic, compact experiment contracts, aggregate evidence, executed notebooks, and regression tests needed to review the engineering and scientific decisions while keeping competition-sensitive state private.

## Run Policy Lens in the browser

[Open the public demo](https://alvaro-policy-lens.tartmacaw2.chatgpt.site), or serve the checkout locally. The JavaScript modules require an HTTP origin.

```bash
python -m http.server 8000
```

Open `http://localhost:8000/public-demo/`. Edit a policy, comment and support examples; inspect the matching terms and compare the same comment across fictional policies. TF–IDF vocabulary and IDF are fitted on examples only. A fixed rule-term weight and up to two strongest matches per class produce a violation-versus-allowed margin. Exact query copies are purged, contradictory labels are rejected and unsupported wording can abstain.

```bash
node tools/test_public_demo.mjs
```

The browser has no backend, model download or telemetry. Its authored examples and lexical margins are separate from the historical Qwen system and official scores. The Node command exercises the public demo; the Python commands below replay aggregate evidence and component checks.


## Five-minute public review

Install the locked development environment, verify the committed outputs, then execute the current review in a real Jupyter kernel:

```bash
uv sync --locked --group dev
uv run python scripts/review_portfolio.py
uv run python scripts/review_portfolio.py --execute
uv run python scripts/review_portfolio.py --frontier
uv run python scripts/review_portfolio.py --execute --frontier
```

Execution saves notebooks 31 and 29 under `runs/portfolio_review/`, reopens them, and checks complete cell execution, input/source fingerprints, aggregate arithmetic, numeric summaries, and Plotly/SVG values. Re-running these commands renders existing evidence; it does not train a model or improve a score. No AWS account, GPU, private data, or model download is needed after dependency installation. CI retains the real Jupyter outputs as the `employer-review-notebooks` artifact.

`--inprocess` is an explicitly labeled local diagnostic for environments that cannot open Jupyter sockets. It does not establish kernel-transport compatibility; the standard command and CI use real Jupyter kernels.

Notebook 31 separates September 10 successful late scores from October 6 development findings. E44 is recorded as planned, not completed. The full historical notebook collection remains available; the short route selects the completed system and current research decisions.

## Public reproducibility tiers

### Tier 1 · Review the evidence

No model download is required.

```bash
uv sync --locked --group dev
uv run python scripts/verify.py
```

This validates Python compilation, lint/formatting, the public pytest suite, and notebook-build determinism. The full suite is broader than the short review and needs working Jupyter kernels; semantic tests additionally require the optional environment shown below.

For the shortest evidence path, inspect:

- `README.md`
- `CASE_STUDY.md`
- `START_HERE.md`
- `notebooks/31_complete_project_review.ipynb`
- `notebooks/27_latest_system_checkpoint.ipynb`
- `notebooks/29_five_model_frontier_review.ipynb`
- `docs/POST_CLOSEOUT_FRONTIER.md`

### Tier 2 · Re-execute public evidence notebooks

The GitHub Actions workflow executes the public evidence path from a clean checkout with the locked environment.

```bash
uv sync --locked --extra semantic --group dev
uv run --extra semantic python scripts/execute_notebooks.py --publish
```

This historical runner executes notebooks 00–04. Use the short review commands above for notebooks 29 and 31. Notebook 27 is also re-executed in the regression suite. These are aggregate/public receipts rather than raw private competition rows.

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

Existing public source and configuration are not retroactively concealed by this policy. This boundary is deliberate: it makes the project inspectable and testable without publishing the artifacts that would reconstruct the competitive system end to end.

## Scientific reproducibility

The public evidence follows three rules:

1. **Official scores and development scores stay separate.**
2. **Negative experiments are preserved when they answer a preregistered question.**
3. **A higher point estimate is not promoted when stability or uncertainty gates fail.**

The latest public cross-model receipt is:

- `reports/checkpoints/cross_model_frontier_20261006.json`

It records only aggregate metrics, matched-control comparisons, and decisions. It excludes row-level outputs, exact ensemble weights, private paths, and checkpoint state.

## Public review contract

An external reviewer should be able to verify:

- environment locking and installability
- metric and validation behavior
- notebook execution
- aggregate experiment arithmetic
- public/private boundary tests
- offline/synthetic inference
- provenance and checksum handling

The private scored-system artifacts are outside this public review contract.

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
