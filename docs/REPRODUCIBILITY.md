# Reproduce the maintained components

## Original aggregate evidence · no dependencies

```bash
python3 -S scripts/review_portfolio.py --json
python3 -S review/run.py --output /tmp/jigsaw-review --with-node
python3 -S review/run.py --output /tmp/jigsaw-review --with-node --check
```

Use Python 3.11+; omit `--with-node` when Node.js 22/24 is unavailable. The checker verifies original evidence hashes, policy-macro arithmetic, cohort accounting and the historical submission comparison. The kit records actual checks and source hashes; changed output is rejected rather than silently reused.

## Trainable lexical control · Python 3.12

```bash
uv sync --locked --group dev
uv run jigsaw demo --root /tmp/jigsaw-cpu-demo
uv run jigsaw demo --root /tmp/jigsaw-cpu-demo
uv run jigsaw review --root /tmp/jigsaw-cpu-demo --allow-synthetic
```

This executes TF-IDF/logistic-regression training, seen/held-out-rule evaluation, overlap purging, independent full fitting and validated synthetic outputs. The second invocation reuses checksum-verified completed stages. Open the reported `review/report.html`. On Windows, use a writable output path outside the checkout.

## Native neural contracts and source packaging

```bash
uv sync --locked --group dev --extra semantic
uv run --extra semantic pytest -q
uv run python scripts/build_adapted_notebook.py
```

Tests construct tiny native Qwen causal models with LoRA: no pretrained download is needed. They check query-target exclusion, final-token projection, optimizer/scaler/RNG recovery, overflow retries, cached-score corruption and original row order. The builder embeds exact canonical code and validates its hashes in the offline notebook. It does not execute the full 4B model.

To execute that notebook separately requires licensed competition inputs, the pinned pretrained model files and a compatible GPU/runtime. Exact historical adapters, private comments and full prediction rows are not published. Rebuilding source is not a claim to reproduce the submitted weights or score.

## Browser diagnosis and replay

```bash
python -m http.server 8000
node tools/evaluate_public_challenges.mjs --check
node tools/test_replay_public_demo.mjs
```

Open `http://localhost:8000/public-demo/`. The authored engine and visible challenges are separate from neural model quality. [Export/replay contract](DEMO_REPLAY.md) · [Failure analysis](PUBLIC_CHALLENGES.md)
