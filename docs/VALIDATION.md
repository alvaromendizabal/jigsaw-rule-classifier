# Verification record

This record separates software correctness, executed research, notebook rendering and independent performance evidence. The last category is not established by passing tests.

## Executed local checks

The original main branch passed **139 tests** before changes. The broad feature-research milestone passed **188 tests**; the released-data boundary milestone passes **208 tests**, compilation, Ruff, formatting and canonical notebook-source checks in the locked Python 3.12 environment. Tests cover training-only screening, schema/finite-value contracts, exact duplicates, support-order invariance, nested target encoding, unseen-group fallbacks, cached embedding identity, OOF alignment, model-stage recovery, report checksums, publication boundaries and the open feature gate.

`uv run python scripts/verify_research.py` separately recomputes private OOF metrics without fitting. It verifies the broad study and the sensitivity, NLI, instruction and near-copy study review markers, public/private byte identity, row coverage, targets, rule identities and metric values. Aggregate display verification is not substituted for this check.

All five canonical public notebooks executed end to end in **actual encrypted Jupyter kernels**, followed by a successful completed-checkpoint reuse pass and synthetic offline-inference check. SageMaker job `jigsaw-notebooks-20260908-1856` recorded 68.3 seconds worker invocation time. Its notebook archive SHA-256 is `b541f38c30dda5fdac9271c8dc8525435ac13ae4c18847e3273833363cbf766c`. After restoration, every notebook passed source-hash, complete input-contract, nonempty execution-count, no-error and no-stderr checks. Plotly MIME output and SVG fallbacks are present. GitHub Actions independently runs the same publication requirements, plus the real encoder integration check. Local in-process execution is also tested, but it cannot satisfy the Jupyter publication gate.

The standalone Kaggle notebook remains the original offline reference. Its software-only test uses explicitly synthetic data. No new real submission CSV, upload or leaderboard score is created by the feature research.

## Experiments and preserved evidence

| Study | Run | Evidence |
| --- | --- | --- |
| Original lexical reference | `c15c2c2318fc0ed619c6` | Original purged splits and reference OOF predictions preserved |
| Original Qwen benchmark | `4e7e6c00d269c451c0a3` | 1,875 unique input embeddings; model/revision/input hashes verified |
| Early feature controls | `e9091086b7bc7ed137ce` | Four candidates, 20 fits; aggregate reports now published |
| Broad feature search | `9ec008d506b0bc64a717` | 21 configurations, 105 fits; full private screening catalogs and group attribution |
| Representation sensitivity | `bce07fd60bc7543fca49` | 55 fits including NB, full-vocabulary and five low-rank controls; frozen geometry diagnostics |
| Frozen NLI | `537cf2213c813b8ebd4b` | 11,835 unique pairs; 25 fits; 26 truncated pairs |
| Frozen instruction likelihoods | `96bf69f42f9063e01a12` | 5,933 unique prompts in 186 verified shards; 15 fits; full cache replay |
| Approximate-copy stress | `a09455ec14e9b3d6ab61` | 10 fits; zero additional copies under the fixed conservative policy |

The previous 30-fit sensitivity run `8b05ea28598bc58af3bd` remains a historical checkpoint; its extension has a new source fingerprint and run ID. The broad feature and original reference artifacts were not relabeled or overwritten.

The original restored snapshot has SHA-256 `8762046fe72017d8567c9c56e68f82db95ba93e1edf2c4d33ee3c4744fee724f`: 375 paths, 357 unique objects, all checked. The broad-research checkpoint archive has SHA-256 `7970504257e77152109a50ba667fa1467a1c7329176d66849c433561ab6af84d` and is preserved under the isolated feature-research experiment prefix.

The final checkpoint contains 2,103 files (573,761,714 bytes), covering the current and historical sensitivity runs, approximate-copy study, current instruction study, all 186 instruction shards and corresponding sources. Its SHA-256 is `78478013f5b3c8d62c2587876a85608373ea4e389a1d7b35e0fde8ab9a84b962`. The encrypted S3 object checksum was verified in the project owner’s account; it uses a separate experiment prefix and leaves immutable snapshots intact.

SageMaker job `jigsaw-pair-features-20260908-1734` completed successfully on one `ml.m5.2xlarge`, with a one-hour maximum runtime. Its worker recorded **493.2 seconds** total invocation time, return code zero and no checkpoint errors. The reported **1,658.6 inference seconds** sum concurrent worker durations; they are not wall-clock time. The 132 fitted-study objects and inference shards are preserved in its isolated S3 prefix; private review files were restored and metrics recomputed locally.

The fixed instruction-likelihood feature probe uses a separate bounded `ml.m5.4xlarge` job, four CPU inference workers and immutable Qwen3 assets. Its source upload is verified by an S3 SHA-256 checksum with an expected-bucket-owner condition. Job `jigsaw-instruction-features-20260908-1800` completed successfully with no checkpoint errors: 1,926.4 seconds worker invocation time. Its 7,304.1 inference seconds sum four concurrent workers. All 931 cache objects were restored, every shard file hash checked, and all 5,933 unique prompt keys verified. A 21.2-second local replay under the updated provenance contract reused all inference and produced the current 15-fit study. The cloud source and earlier cloud study remain separately preserved.

## Reproduction and acceptance

### Released-data boundary milestone

Protocol commit `9b4d66448998da818b71e4e697aa8e937c9222f0` preceded released research-target materialization. The pinned version-1 archive has 19,271,777 bytes and SHA-256 `34e01e093b96698d50344966364d7e9ccb3a938a8f9c936794427a448de30c28`. All six members are verified against `configs/released.json`; original training/preview bytes match. Preparation run `a615ecc75c04e6a62ee1` materializes 9,106 research rows and records 43,576 reserved rows without interpreting their targets. A replay reuses the committed stage. The adversarial tests cover every support field, normalized historical exposure, shuffled metadata alignment, nonnumeric protected-target sentinels, ambiguous identities, modified exports and changed protocol rejection.

The source archive and prepared boundary are checkpointed separately from historical experiments. The eight-file checkpoint is 24,133,624 bytes, SHA-256 `0b809ee83a0925d77be5f11e29dc60e5af46e6523366b24348bb5259d41ab967`, under `experiments/released-boundary-20260908T2230/checkpoints/released-boundary.tar.gz` in the existing project bucket. Account ownership, AES256 encryption and the S3 object checksum were verified. It includes the original public release archive, assignments, research-only CSV, stage contract/marker, protocol and preparation source; no model weights or credentials are included.

The organizer's 2,443-row per-rule score attachment has SHA-256 `43b2af04973e2de8af75d984fd8c798020b442b15d955eb4cfa82141ee9e7c4f`. Among 2,437 complete rows, 2,428 agree with the unweighted mean of six policy AUCs within 1e-6. Nine discrepancies and six incomplete rows are retained in the public audit. This is strong primary-source corroboration of the aggregation, not an executable-scoring or project leaderboard claim.

All five updated notebooks passed local in-process rendering, and the existing private OOF metrics were recomputed without fitting. The local environment blocks Jupyter network-interface discovery; it was not treated as a passing publication check. AWS job `jigsaw-notebooks-20260908-2230` executed all five in actual encrypted Jupyter kernels, verified a second publication pass, and passed synthetic offline inference. The worker invocation recorded 65.99 seconds. Its 61,716-byte notebook archive has SHA-256 `9be69dceae4e336c43a29942b66482dbf0c0a6b6058a7c74cbf9e281d3389f22`. Restored notebooks match every local source/input contract and contain no error or stderr output. CI independently verifies the same publication path.

```bash
uv run python scripts/verify.py
uv run python scripts/execute_notebooks.py --publish
uv run python scripts/execute_notebooks.py --publish
uv run python scripts/execute_notebooks.py --synthetic
uv run python scripts/verify_semantic.py
# Restored private artifacts required; no fitting:
uv run python scripts/verify_research.py
uv run jigsaw gate
# Pinned released-data preparation; no confirmation scoring:
uv run python scripts/prepare_released_data.py
uv run python scripts/build_release_report.py
```

CI runs the software gate, encrypted Jupyter execution, completed-notebook reuse, synthetic offline inference and the real pinned encoder check. Source and logs are retained as workflow artifacts. The results-push helper additionally rejects incompatible sources, stale contracts, non-Jupyter publication, private/synthetic execution and unrelated edits. Failed execution retains the last valid canonical notebook.

Research encoders remain frozen. No neural fine-tuning, optimizer-state recovery, final calibration, final representation selection or independent official score is claimed. The rule-macro AUC implementation is tested separately from pooled AUC; parity with executable Kaggle scoring remains unverified. Within-study simultaneous intervals do not resolve selection across repeated studies on two rules.
