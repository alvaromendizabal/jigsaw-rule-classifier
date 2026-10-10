# Employer review guide

## Three-minute review

1. Read the [README](../README.md) for my ownership and the recorded **0.91808 public / 0.91425 private ROC AUC** result.
2. Try [Policy Lens](https://alvaro-policy-lens.tartmacaw2.chatgpt.site): edit a comment, switch the rule, inspect matching support examples and export the explanation. [Local launch](REPRODUCIBILITY.md)
3. Read the [case study](../CASE_STUDY.md) for the transfer problem, matched adaptation study and decisions to reject unsupported model changes.

## Hands-on review

```bash
python -m http.server 8000
```

Open `http://localhost:8000/public-demo/`. The browser fits a transparent lexical support matcher on authored examples. It does not run the retained Qwen system or report a competition score.

```bash
node tools/test_public_demo.mjs
uv sync --locked --group dev
uv run python scripts/review_portfolio.py --execute
```

The Node checks exercise the demo engine. The Python command executes and verifies the aggregate notebook review without fitting a neural model. [Complete scope](REPRODUCIBILITY.md)

## What to inspect

| Question | Evidence |
|---|---|
| How does policy context change a decision? | Policy Lens inputs, matching-feature contributions and support inspector |
| Did adaptation help at a fixed backbone? | [Case study](../CASE_STUDY.md): 0.61460 → 0.71989 development AUC |
| How is leakage controlled? | [Validation notebook](../notebooks/01_data_and_validation.ipynb), text-overlap purging and group-aware checks |
| Why retain a negative experiment? | [Cross-model receipt](../reports/checkpoints/cross_model_frontier_20261006.json) |
| What survives interruption? | Checkpoint/source contracts, runtime modules and regression tests |

The 881-comment/two-policy cohort was repeatedly inspected. The historical scores are late evaluations; original competition placement and production moderation readiness are not claimed. Upstream methods retain their source credits. [Release closeout](PROJECT_CLOSEOUT.md)
