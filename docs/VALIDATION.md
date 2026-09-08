# Verification record

This record separates software correctness, executed research, notebook rendering and independent performance evidence. The last category is not established by passing tests.

## Executed local checks

The original main branch passed **139 tests** before changes. The expanded feature-research code passes **186 tests**, compilation, Ruff, formatting and canonical notebook-source checks in the locked Python 3.12 environment. Tests cover training-only screening, schema/finite-value contracts, exact duplicates, support-order invariance, nested target encoding, unseen-group fallbacks, cached embedding identity, OOF alignment, model-stage recovery, report checksums, publication boundaries and the open feature gate.

`uv run python scripts/verify_research.py` separately recomputes private OOF metrics without fitting. It verifies the broad study and the sensitivity, NLI and near-copy study review markers, public/private byte identity, row coverage, targets, rule identities and metric values. Aggregate display verification is not substituted for this check.

All five canonical public notebooks execute end to end through the in-process rich-output engine with no error cells or stderr. Plotly MIME output and SVG fallback rendering are checked. Local Jupyter transport cannot start because this execution environment prohibits network-interface discovery (`Operation not permitted`). That infrastructure error was diagnosed; encryption requirements were not weakened. Actual Jupyter execution and reuse are required in GitHub Actions before merge, and its resulting notebook artifacts are used for canonical publication.

The standalone Kaggle notebook remains the original offline reference. Its software-only test uses explicitly synthetic data. No new real submission CSV, upload or leaderboard score is created by the feature research.

## Experiments and preserved evidence

| Study | Run | Evidence |
| --- | --- | --- |
| Original lexical reference | `c15c2c2318fc0ed619c6` | Original purged splits and reference OOF predictions preserved |
| Original Qwen benchmark | `4e7e6c00d269c451c0a3` | 1,875 unique input embeddings; model/revision/input hashes verified |
| Early feature controls | `e9091086b7bc7ed137ce` | Four candidates, 20 fits; aggregate reports now published |
| Broad feature search | `9ec008d506b0bc64a717` | 21 configurations, 105 fits; full private screening catalogs and group attribution |
| Representation sensitivity | `28fdc0063dd3085842da` | 55 fits including NB, full-vocabulary and five low-rank controls; frozen geometry diagnostics |
| Frozen NLI | `537cf2213c813b8ebd4b` | 11,835 unique pairs; 25 fits; 26 truncated pairs |
| Approximate-copy stress | `a09455ec14e9b3d6ab61` | 10 fits; zero additional copies under the fixed conservative policy |

The previous 30-fit sensitivity run `8b05ea28598bc58af3bd` remains a historical checkpoint; its extension has a new source fingerprint and run ID. The broad feature and original reference artifacts were not relabeled or overwritten.

The original restored snapshot has SHA-256 `8762046fe72017d8567c9c56e68f82db95ba93e1edf2c4d33ee3c4744fee724f`: 375 paths, 357 unique objects, all checked. The broad-research checkpoint archive has SHA-256 `7970504257e77152109a50ba667fa1467a1c7329176d66849c433561ab6af84d` and is preserved under the isolated feature-research experiment prefix.

SageMaker job `jigsaw-pair-features-20260908-1734` completed successfully on one `ml.m5.2xlarge`, with a one-hour maximum runtime. Its worker recorded **493.2 seconds** total invocation time, return code zero and no checkpoint errors. The reported **1,658.6 inference seconds** sum concurrent worker durations; they are not wall-clock time. The 132 fitted-study objects and inference shards are preserved in its isolated S3 prefix; private review files were restored and metrics recomputed locally.

The fixed instruction-likelihood feature probe uses a separate bounded `ml.m5.4xlarge` job, four CPU inference workers and immutable Qwen3 assets. Its source upload is verified by an S3 SHA-256 checksum with an expected-bucket-owner condition. Results remain separate from the completed NLI study and must pass the same report/OOF checks before a result is claimed.

## Reproduction and acceptance

```bash
uv run python scripts/verify.py
uv run python scripts/execute_notebooks.py --publish
uv run python scripts/execute_notebooks.py --publish
uv run python scripts/execute_notebooks.py --synthetic
uv run python scripts/verify_semantic.py
# Restored private artifacts required; no fitting:
uv run python scripts/verify_research.py
uv run jigsaw gate
```

CI runs the software gate, encrypted Jupyter execution, completed-notebook reuse, synthetic offline inference and the real pinned encoder check. Source and logs are retained as workflow artifacts. The results-push helper additionally rejects incompatible sources, stale contracts, non-Jupyter publication, private/synthetic execution and unrelated edits. Failed execution retains the last valid canonical notebook.

Research encoders remain frozen. No neural fine-tuning, optimizer-state recovery, final calibration, final representation selection or independent official score is claimed. The rule-macro AUC implementation is tested separately from pooled AUC; parity with executable Kaggle scoring remains unverified. Within-study simultaneous intervals do not resolve selection across repeated studies on two rules.
