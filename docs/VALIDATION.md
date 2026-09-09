# Verification record

This record separates software correctness, executed research, notebook rendering and independent performance evidence. The last category is not established by passing tests.

## Current milestone: feature research closure

The 2026-09-09 closeout passes **263 tests in 115.66 seconds**, compilation, Ruff, formatting and canonical notebook-source checks. The full quality invocation takes 120.06 seconds. Tests additionally cover policy-intent compilation, support/batch invariance, forbidden target access, independent cache identity, all five replacement tolerances, fixed fusion and stale stopping-rationale publication. The feature gate now permits final model development and explicitly denies production promotion.

Formatting protocol `915ce70792adff3b85c0eac5bb7e461a646284aa` preceded inference and scores. Replacement tolerances were committed at `ceaea9f512507606d12532be40b5a6378f04242a` before new scores were inspected. The later, adaptive fixed-average control was committed at `af33a31837ea7d3e98b58783fa07ba77b46ac521` before computing that average. These are separate development decisions, not a claim that all research was preregistered before earlier findings.

| Evidence | Verified record |
| --- | --- |
| Semantic formatting/intent | Run `634ffc0b387c85f11041`; 28 fits, three frozen scores, 77 candidates; same 11,135 development rows and seven purged splits |
| Source-checked qualitative audit | 48 rows, 47 unique normalized bodies; all original source checks pass; no relabeling or reserved-target access |
| Feature stopping decision | Run `f64dbed83f793a3f5cc8`; seven alternatives and one fixed average fail replacement; retain original centroid for unseen policies |
| Private replay | 203 model/feature-stage files, 20 metric records, 28 saved model predictions, 28 feature transforms and eight frozen-score/protocol combinations |
| Resume experiment | Completed formatting run reused with model fitting and encoder preparation patched to raise; zero new fits or inference |
| Candidate banks | 188,595–188,598 offered and 9,281–9,660 retained per fold across banks; 323 fits in the full four-policy campaign |

SageMaker job `jigsaw-formatting-20260909-0030` completed on one `ml.m5.4xlarge`, with a 90-minute hard ceiling. Actual worker computation was 644.51 seconds; the processing interval was 00:31:21–00:44:16 UTC. The original 190-shard comment/support cache is unchanged. A new 35-shard plain-document cache contains 2,217 unique supports; a new 332-shard NLI cache contains 21,214 unique pairs. Their zero/four truncations and 739.8/1,271.3 summed parallel encoder seconds are separate from wall time. Compatible historical NLI inputs were not present in this worker's restore, so historical NLI reuse is correctly recorded as zero.

Under `experiments/semantic-formatting-20260909T002556/`, the development-only input bundle is 377,979,765 bytes (SHA-256 `4386488611a50cb6f21ee4bf85e203ecd22b831d78a7409e8bbe73b78fe4018d`). The additive checkpoint `checkpoints/semantic-formatting.tar.gz` has 2,069 members and 36,270,983 bytes, SHA-256 `943c15df408efaa748bd4e80a71a622392469b327d500be9eb4bb1d50195804d`. It was restored and verified against the existing immutable inputs. The bundle excludes the released solution and reserved targets; raw comments and row predictions remain private.

The notebook publication worker `jigsaw-feature-closeout-20260909-0106` uses one `ml.m5.2xlarge`, a 20-GB volume and a 900-second ceiling. It completed all five canonical notebooks in actual encrypted Jupyter kernels, reused all five on its second pass, and passed synthetic offline inference. Worker invocation took **69.65 seconds**. Restored notebooks have current source/input contracts, complete execution counts, no error or stderr outputs, and seven Plotly figures with SVG fallbacks. The new contrast and per-policy figures were visually inspected. Notebook `02` has 29 cells and `03` remains a 10-cell evidence tour.

The durable publication prefix is `experiments/feature-closeout-publication-20260909T0106/` in the existing owned project bucket:

| Object | Bytes | SHA-256 |
| --- | ---: | --- |
| `code/source.tar.gz` | 652,589 | `3365d6186d22ee919c6648743aaccb0c39869a21e69d2fcab2c48a7f0ed323b5` |
| `public/executed-notebooks.tar.gz` | 77,251 | `2251a2970afe023e2cddc0e873857a371b6302811ec54158ce5a33645075b703` |
| `checkpoints/feature-closeout.tar.gz` | 352,685 | `a21ae3d8eb8dbaf6e8918f1cf5c08ca026cc0d42e8a780447fd5e8d5faa35d07` |

The last archive preserves the private audit, current/earlier stopping decisions, fusion predictions and logs: 26 payload files plus a manifest (manifest SHA-256 `7d9ee4800e51a79368f3cdfdef89219c962a57f34542993b2f61204fc60e02fd`). A fresh S3 download verifies its full archive and every member. Fresh STS, expected-owner ACL and public-access-block checks confirm the user-owned destination; uploads use AES256. The earlier unused public source package under the `0104` publication prefix remains preserved; no experiment consumed it.

These checks close the declared feature phase. They do not claim independent confirmation, fitted calibration, a routed-model result, offline promotion or a leaderboard score. The 43,576 reserved targets remain unopened. [Next model milestone](FINAL_MODEL_PLAN.md).

## Executed local checks — historical milestones

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

## Expanded feature milestone

The four-policy protocol was frozen at `5c16f91c04880d52280bf5a0a71ff688b327bb70` before model scores. Run `a57bb74350fcbcd66592` binds the 11,135-row research-only cohort, split design, implementation, environment and expanded embedding-cache hashes. SageMaker job `jigsaw-expanded-20260908-225455` completed on one `ml.m5.4xlarge` with a two-hour limit. AWS recorded 22:57:09–23:42:11 UTC on 2026-09-08; the worker recorded 2,609.6 seconds including setup, encoding, fitting and archiving.

All 10,098 missing inputs were encoded once, reusing 1,875 existing unique queries. The verified cache has 190 shards and 11,973 unique inputs. Statistics cover 11,977 occurrences because four original keys were repeated identically; they record two truncated inputs, maximum original length 337 tokens, 6,304.0 summed inference seconds across historical/concurrent workers, and peak per-worker RSS 3.85 GiB. Summed inference time is not wall time. The current study itself reuses these cached vectors.

The job completed 238 primary fits and 22 changed-training sensitivity refits; six identical sensitivity cases reuse their primary fit. Its 2,585-file archive has **1,636,592,121 bytes**, SHA-256 `71763123eb696206bcec8858d16d325a7b67eb7202319f9aa9dfcd034746cc8a`, at `experiments/expanded-features-20260908-225455/checkpoints/expanded-study.tar.gz` in the existing project bucket. Source and worker uploads are separately checksummed; completed stage markers are uploaded last. Restoration verified the archive before extraction, rejected differing immutable checkpoints and preserved a differing mutable event log separately. No source snapshot was overwritten.

The retrieval protocol `087cec0aa08d77f40abc2f498a99731fd3d6a3db` preceded score inspection. Local run `f69e5061eef584e520e3` reused the verified expanded banks and embeddings for 35 additional fixed fits. Resolution protocol `ccd81fb0a589dd55cc763503fc7434b0c3e873ea` also preceded score inspection; run `143fb05be41909328ef4` checked six frozen prefixes with no inference or fitting. Its 1,024-dimensional reference reproduces the expanded centroid exactly. Both studies preserve private scores and validate exported metrics against them.

`python scripts/verify_expanded.py` independently verified **1,639 committed stage files**, recomputed **90 expanded plus 10 retrieval metric records**, and replayed **14 saved model predictions** from their actual saved feature banks. It fitted zero models and accessed no confirmation targets. The direct script invocation was tested after correcting its sibling-module import. The complete local quality gate passed **244 tests**, formatting, lint and notebook source parity. All five revised notebooks passed an in-process logic check; that smoke test is explicitly not Jupyter transport evidence.

AWS job `jigsaw-notebooks-20260908-2352` subsequently executed all five canonical notebooks in actual encrypted Jupyter kernels, verified checkpoint reuse on a second publication pass, and passed standalone synthetic offline inference. Its worker invocation took 68.69 seconds. The 61,447-byte notebook archive has SHA-256 `b8f4ccab9df959af4781acfb46055a7203d2aa5514a098a982de54d8e89dba39`. Restored notebook sources and every complete execution-input contract match the local canonical project; all code cells have execution counts and no error/stderr output. The archive is at `experiments/expanded-publication-20260908-2352/public/executed-notebooks.tar.gz`.

The reviewed notebook source archive has SHA-256 `d9df0b53326930beee33a0f10f18f6681d7bdfa22d072187d3b5ca3a665ab626`; the worker has SHA-256 `5aeda9a0009e3dd8cd65fed885bde7a00f284da7ab921367b234b31520228876`. Narrative documentation may be updated after execution; notebook source, package/configuration/report inputs and figure-builder hashes must remain identical.


The 268-file retrieval/resolution supplement is checkpointed under `experiments/expanded-publication-20260908-2352/checkpoints/supplement/`. Its archive is 1,116,734,309 bytes, SHA-256 `b84d844cb44c17a2afe140a638e2db446a990d6bb25641f55f4d2a83c273724e`. A single large upload exceeded the socket timeout; the successful checkpoint uses nine independently checksummed parts and a manifest uploaded last. All ten remote objects passed byte-count, SHA-256 and AES256 checks. The manifest SHA-256 is `c8aef17c507cd7ce9ff2383013504a8e0dff92c54bf082ed5f9019ca7f0bb766`. Ordered reassembly was verified against the archive hash. It includes both new studies, their sources/configurations, private prediction files, verification logs and the explicitly exploratory eight-example error-review note. It depends on the preserved expanded archive and the earlier research boundary; it contains no confirmation scores.

To restore the supplement, verify the manifest against the recorded hash, retrieve its ordered parts, verify each part's byte count and hash, concatenate them in manifest order, verify the whole archive hash, and extract only safe relative regular-file paths. Never treat a partial upload as a completed checkpoint.

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

Research encoders remain frozen. No neural fine-tuning, optimizer-state recovery, final calibration, final representation selection or independent official score is claimed. The rule-macro AUC implementation is tested separately from pooled AUC; parity with executable Kaggle scoring remains unverified. Within-study simultaneous intervals do not resolve adaptive selection across the historical two-policy and current four-policy studies.
