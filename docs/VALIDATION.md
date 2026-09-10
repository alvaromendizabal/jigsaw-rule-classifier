# Verification record

**Latest milestone:** the competition rebuild has completed its frozen-feature and
support-adaptation comparisons. The 0.92 competition objective remains open.
The earlier research release and its protected cohort are historical evidence.
[Current results and next gate](COMPETITION_REBUILD.md).

This record separates software correctness, executed research, notebook rendering and independent performance evidence. The last category is not established by passing tests.

## Competition rebuild verification — September 9, 2026

The current local gate passes **360 tests in 56.32 seconds**, compilation, Ruff,
formatting and canonical notebook-source parity. All five public notebooks
execute and replay with the in-process engine; the two current comparison figures
have Plotly and verified SVG representations. No error output remains. The prior
code commit `ca793600b17cedc7ff78a45c5ab51ac970d042ef` also passed all 360 tests and
all five actual Jupyter executions/replays in
[CI run 34413323057](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/runs/34413323057).
That CI run preceded the final adaptation result publication.

AWS support study `d13858b7407993fcc6e8` completed both folds and deliberately
reloaded a fresh model at optimizer step 8 in each. Adapter, optimizer, scheduler,
random state and data-order recovery are recorded. The existing 96 frozen shards
replay without loading the encoder, with identical output hashes. The completed
study uses 899.7 worker seconds and 1,253 billable L4 seconds; maximum allocated GPU
memory is 8.31 GiB. Both fold artifacts are downloaded and SHA-256 verified before
evaluation `3836f7a69996714f76b0`; its second invocation reuses all seven completed
files. Query labels are absent from the GPU plan and excluded from all readout fits.

On the same 881 novel comments, adaptation improves direct policy-macro AUC from
0.6146 to 0.7199. The fixed geometry blend and screened representation do not beat
that direct score. These previously examined two-policy development results are
not a Kaggle submission result. [Cloud receipt](../reports/checkpoints/support_adaptation.json)
and [aggregate checksums](../reports/support_adaptation/metadata.json) preserve the
runtime, exact configuration, source identity, measurements and limitations.

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

Research encoders remain frozen. The model milestone below adds final representation selection and development calibration. Neural fine-tuning, interrupted-optimizer recovery and an independent protected score are not claimed. The rule-macro AUC implementation is tested separately from pooled AUC; parity with executable Kaggle scoring remains unverified. Within-study simultaneous intervals do not resolve adaptive selection across the historical two-policy and current four-policy studies.


## Fitted policy route and nested calibration

The pre-score protocol was committed at `47ba1336fc1d467fba518a00c37aba16fc30efbc`; implementation/results were pushed at `e55902cfa2df22c91fc0b743440fed5d97d8b8c1` in PR #12. Model run `a971cf3bc6add1c2d818` completed in **294.5 seconds** using the existing frozen vectors. It adds nine nested inner fits and two full-development fits, with no hyperparameter or encoder search. The fitted familiar pipeline screens 181,958 columns down to 9,263. Familiar calibration passes all six checks; unseen calibration fails four and is not retained. [Full protocol and results](MODEL_VALIDATION.md).

The local quality gate passed **288 tests**, compile, lint, formatting and canonical notebook-source verification. `scripts/verify_model_validation.py` replayed nine saved inner pipelines and seven outer routes with zero classifier refits; it recomputed calibration decisions and verified final batch/order parity on 115 development rows. Familiar parent predictions match the source research within 1.12e-16; unseen routing matches exactly. A restart with both classifier fit methods patched to raise reused all **19 completed stages** and preserved every completion-marker hash. That proves completed-stage reuse, not recovery of an interrupted optimizer.

The separate target-blind eligibility run `60ba8e0ede00928c0943` completed in 127.8 seconds. Its fixed near-copy rule excludes 67 reserved rows and leaves 43,509 eligible identities. No reserved target or prediction was read or produced. The prepared model/eligibility backup contains **186 verified files**, with only complete stages, provenance and the named validation logs. The archive is **115,145,689 bytes**, SHA-256 `e70ef2a70e032e6d832b8979e47e74c2bdd05215721205d92f7b52a044bc0ca3`; its internal manifest is `e6027fc014b5341dccc4d3a73247ffd2a5e699196c1461deb63d8add17eb5984`. Every archived member was read back and checked locally.

**Backup and independent recovery verified on 2026-09-09 at 02:40 UTC.** The approved archive is stored at `s3://sagemaker-jigsaw-rules-560403859723-us-west-2/experiments/model-validation-20260909T0148/checkpoints/model-validation.tar.gz`, version `WtASv0BboNNGl4q0iM08NVhJ8hwtB_1B`. S3 reports the expected owner, AES-256 encryption, exact byte count and matching full-object SHA-256. An independent download matched the archive hash; all 186 payload hashes and all 20 completion-marker contracts passed. No models were refitted and no reserved targets were accessed. [Machine-readable recovery record](../reports/checkpoints/model_validation.json). Earlier research checkpoints remain intact.

[GitHub Quality run 34300777462](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/runs/34300777462) passed on implementation head `e55902cfa2df22c91fc0b743440fed5d97d8b8c1`, including **all five actual encrypted Jupyter executions**, a second completed-cache reuse pass, standalone synthetic offline inference, real pinned-encoder integration and public figure rendering. The downloaded 1,128,160-byte CI artifact has SHA-256 `5bb6a2ac5d92208fef5440494e4bdc0523bf260acfd83535b84294adcb9fcd5c`. Its preserved source ZIP matches the model/config/report/runner inputs; each restored notebook matches canonical source, carries Jupyter/encryption metadata, and has all code cells executed without errors or stderr. The five notebooks contain 5, 6, 15, 6 and 8 code cells respectively. Plotly output and SVG fallbacks include the new calibration and reliability figures. Documentation-only publication changes do not alter their execution inputs.

Reproduction after restoring private artifacts: `python scripts/run_model_validation.py` resumes the frozen protocol; `python scripts/verify_model_validation.py` replays artifacts without classifier fitting; `python scripts/prepare_confirmation.py` audits only target-free reserve inputs. The canonical offline notebook is still the lexical reference. Protected scoring and candidate promotion remain the next milestone.


## Protected evaluation completed — September 9, 2026

The canonical stage recovery and checkpoint-walker changes passed the full local gate: **328 tests**, compile, Ruff lint/format and canonical notebook-source verification. Seven added regressions cover completed-stage integrity, replay/no-overwrite behavior and refusing to traverse unpublished staging directories. The test suite took 72.86 seconds; the full gate took 76.67 seconds. The old worker logged only `FileNotFoundError`, so the tested staging-rename race is not asserted as its proven historical failing path.

Prediction freeze `530ad79929012e807cb42a5253f7b92b090682ec` was published and fetched before the first eligible target access at **2026-09-09T17:12:07.771236+00:00**. The fixed scorer completed in **6.88 seconds**: **all 12 checks passed**, policy-macro AUC **0.7770 versus 0.6801**, paired 95% gain interval **[0.0898, 0.1051]**. All six policies and both routing groups improve. An independent aggregate audit reconciles the report within **1.12e-16**. Reuse with target reading and scoring patched to raise returns the same completion marker; no model fitting or encoding was repeated.

All three new Plotly figures were rendered to SVG and visually inspected as PNG previews. Local Jupyter transport failed before a cell ran because network-interface discovery/ZeroMQ returned `Operation not permitted` and the kernel exited; publication therefore requires actual GitHub CI notebook execution. This failure is preserved in the local publication log and is not presented as successful execution.

The original predictions, source and input bundle remain in the existing S3 project prefix. After explicit user approval, the **3,228,501-byte prediction/evaluation archive was uploaded and independently restored**. Its SHA-256 is `fa52f10f8a12c41bab6c7366c48a3e8dccec32bc1855ab8f8d3a1c36fa7009f9`; S3 version `9UYSEPlAVt1EdIzkZOU4kQfvYHq3OBNW` has verified AES256 encryption, expected owner, exact size and matching full-object checksum. All 23 payloads passed independent recovery verification. This resolves the earlier automatic approval block. [Exact archive and verification record](../reports/checkpoints/protected_evaluation.json). The public result and target-access lineage are published through Git; the row-level archive stays private. Full new embedding-cache recovery is not claimed.


**Actual protected-results notebook publication passed.** [Quality run 34382476761](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/runs/34382476761) on `bb038cd4a22f275221dcd5dad141a44997694c3e` passed 328 tests (74.65 seconds), the full gate (79.65 seconds), all five encrypted Jupyter executions (17.11 seconds), all five completed-cache reuses, synthetic offline inference, real pinned-encoder integration and public rendering. The local publication gate also passed the same 328 tests in 56.33 seconds, 60.02 seconds overall.

The CI ZIP is 1,203,411 bytes, SHA-256 `3f388d2ee9ca0268fdfd8e8543a69c64d62afe52a7d08f5bddf245cf371bb0a6`. All five restored canonical notebooks match the current narrative/code source, every input hash and their completed-stage payload hashes. They contain 5/6/15/7/9 executed code cells respectively, no error or stderr outputs, and the three new protected Plotly figures with SVG fallbacks. The recorded execution environment is Python 3.12.14; the local environment is 3.12.13, so cross-environment cache reuse is not claimed. [Publication proof](../reports/checkpoints/protected_publication.json).


## Offline delivery and project closeout — September 9, 2026

The full local quality gate passes **342 tests in 60.00 seconds**, compilation, Ruff lint/format and canonical notebook-source verification; the complete gate takes 63.44 seconds. Fourteen new tests verify the external manifest pin before model deserialization, source/version/payload corruption rejection, complete support requirements before model loading, target-bearing input rejection and interrupted batch reuse.

The exact accepted candidate and pinned Qwen encoder passed real offline parity on six target-free rows, one per protected policy, without refitting or opening targets. Maximum cloud/local probability difference is **1.2219e-6**, within the predeclared `1e-5`; identical vectors give zero difference. Order/batch invariance, embedding reuse and completed-prediction reuse pass. All seven frozen budgets/guards pass. Cold verification, loading and six predictions take **28.32 seconds**; the four-example warm batch averages **1.11 seconds per comment**; peak memory is **3.88 GiB**. The full acceptance run takes 37.95 seconds with zero network calls. [Exact measurements and host limits](DELIVERY.md#measured-delivery-checks).

A distinct virtual environment was created from the archive with `uv sync --offline --locked --extra semantic`, using cached locked dependencies. Its own interpreter ran the extracted CLI with a new encoder cache and network calls disabled: four authored probabilities agree within **1.12e-16**, in 17.10 seconds. This is a clean environment restore, not a claim that a wheelhouse is included. [Restore proof](../reports/checkpoints/offline_restore.json).

The notebook and 1.23-GB private model archive are stored under `releases/v1.0.0/` in the owned project bucket. Expected-owner S3 HEAD checks verify encryption, byte counts, object versions and full SHA-256 checksums. Independent GET downloads verify the exact notebook bytes and all 57 archive payload hashes. [Cloud recovery proof](../reports/checkpoints/delivery.json). The notebook remains the original-training-only lexical reference; the accepted package remains the post-competition benchmark.

Current CI executes all five public notebooks in actual Jupyter kernels, verifies reuse, tests synthetic inference and executes the standalone notebook twice on hash-verified original training (2,029 rows) and preview (10 rows). [Original-data notebook proof](../reports/checkpoints/submission.json) · [Executed public notebook publication](../reports/checkpoints/delivery_publication.json). Neither preview execution nor the accepted benchmark is a Kaggle leaderboard score.


**Actual closeout publication passed.** [Quality run 34387055130](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/runs/34387055130) on `3b15e80a6d8e3969b18723a66939cd4421307217` passed 342 tests in 74.48 seconds (79.69-second full gate), all five encrypted Jupyter executions in 14.42 seconds, five completed-stage reuses, synthetic inference, original-data preview inference/replay, real pinned-encoder integration and rendering. The 1,324,577-byte CI ZIP has SHA-256 `93d7d014a4c5990b511f3a6186dfe320809fdeb680414d9a5c17262f3ef0ef9f`. Its five public notebook sources, current input hashes, execution counts and completion-marker hashes were independently verified before restoring canonical outputs. They contain 5/6/15/9/9 executed code cells and no error or stderr outputs. The original-data preview CSV hash is `d289784c950864c2cdceff35c13613f8c54214685c4375f7e07c14b384e9ed41`; only its proof record is published to Git, not the CSV or its embedded download output.

The workspace disconnected after local verification, before the publication commit. Bounded worker `jigsaw-delivery-publication-20260909-1813` recovered the already verified CI artifact, checked all notebook/input/completion hashes again, and preserved the public notebook bytes in the owned S3 release prefix. No research, model fitting, encoder inference or protected scoring was repeated.

Closeout is recorded by the merged delivery PR and versioned S3 artifacts. A separate GitHub Release page/tag is not created by the connected GitHub toolset.

## Complementarity and backbone-capacity evidence — September 10, 2026

The Phi comparison and Qwen3-8B capacity study completed their registered training,
prediction, checkpoint recovery and evaluation replay. Both fixed blends fail
their simultaneous uncertainty gates; neither is promoted. The scored canonical
4B candidate remains at 0.91425 private Kaggle AUC. The two studies cost about
$0.99646 in measured SageMaker compute, plus storage. Full model, data, source,
runtime and private archive identities are in the canonical
[rebuild report](COMPETITION_REBUILD.md) and checkpoint receipts.

The capacity-results local quality gate passed **391 tests in 76.44 seconds**,
compilation, Ruff lint/format and canonical notebook-source verification. The
complete gate took **81.79 seconds**, ending at **2026-09-10 05:14:02 UTC**.
Tests cover aggregate-only exports for both studies, changed-receipt cache
invalidation and byte-identical generated probe sources.

Local Jupyter transport again failed before execution because interface discovery
and ZeroMQ returned `Operation not permitted`. That failed process exited; it is
not counted as a passing notebook execution. All five public notebooks instead
rendered locally using the explicitly marked in-process engine in **4.98 seconds**
at **05:16:55 UTC**, and a second pass reused all five completed checkpoints.
Actual encrypted Jupyter execution and replay remain required on the final
GitHub head before this results PR is merged.

The separate private Kaggle probe, saved Version 1 / **348683165**, passed real
two-T4 FP16 training and optimizer recovery using eight authored comments. Its
206.1-second run is a hardware/software check only. The Output tab independently
shows the runtime folder, checkpoint folder and manifest with run identity
`4514a088d4eb9293b7f8`, status `passed`, and receipt SHA-256
`67da985d51c3f20b775513a83b9e889f13c36912c8075bdc5a68bfb2e385a655`.

**Actual capacity-results publication passed.** [Quality run 34440534627](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/runs/34440534627)
on `901dc0a680b61a3472643e2d1511da9d9c8cfcaa` passed the complete gate in
93.49 seconds, all five encrypted Jupyter executions in 18.33 seconds, five
completed-checkpoint reuses, synthetic/original preview inference and the real
pinned encoder check. Its exact source ZIP matches every tracked local file.
The five restored canonical notebooks match the current sources and every input
hash, have 5/6/22/14/9 executed code cells and contain no errors or stderr. The
comparison notebooks retain Plotly outputs and SVG fallbacks. Their recorded
Python version is 3.12.14; the local 3.12.13 environment is not relabeled as CI.

The 2,118,692-byte CI archive has SHA-256
`d12c4512ada39b3075d606d623da74b012f12f36ef07f11aa3aa780601ca5d65`.
The direct archive URL returned a download error; authenticated file
materialization recovered the same byte-verified artifact. No notebook or model
execution was repeated to recover it. [Exact publication and archive receipt](../reports/checkpoints/backbone_capacity_publication.json).
