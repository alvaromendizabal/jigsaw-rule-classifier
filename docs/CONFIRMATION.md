# Protected confirmation of the fitted policy route

**Completed September 9, 2026: candidate accepted, all 12 fixed checks passed.** On 43,509 eligible rows, policy-macro AUC improves from **0.6801 to 0.7770**; the paired 95% interval for the **+0.0969** gain is **[0.0898, 0.1051]**. Familiar AUC improves from 0.7581 to 0.8276 and unseen AUC from 0.5241 to 0.6757. Every individual policy improves. Overall log loss falls from 0.6685 to 0.5121; Brier falls from 0.2360 to 0.1739. [Verified aggregates](../reports/confirmation/results.json) · [Arithmetic audit](../reports/confirmation/audit.json).

Prediction freeze [530ad79929012e807cb42a5253f7b92b090682ec](https://github.com/alvaromendizabal/jigsaw-rule-classifier/commit/530ad79929012e807cb42a5253f7b92b090682ec) was published and fetched before eligible targets were first interpreted at **2026-09-09 17:12:07.771236 UTC**. The fixed comparison took 6.88 seconds locally and reused the completed cloud predictions. No classifier fitting or encoding was repeated. A checkpoint replay with target access and scoring patched to raise returned the same completion marker without reopening labels.

The SageMaker wrapper ended `Failed` after the prediction subprocess completed all 43,509 rows. Individually checkpointed prediction objects passed independent SHA-256 and lineage verification and were restored canonically. The final full-cache archive was absent; full embedding-cache recovery is not claimed. The worker now prunes unpublished staging directories before walking completed markers, addressing a tested rename race. The old log omitted the failing path, so that race cannot be established as the historical failure's exact cause. [Recovery proof](../reports/checkpoints/protected_recovery.json).

This is a post-competition confirmation conditional on six observed policies, including two withheld from development. It is not a leaderboard score. Offline packaging and an original-training-only competition workflow remain distinct product work. The protocol below is retained as the pre-access specification.


The feature gate is closed and the development model is fitted. This milestone asks one question: does that fixed representation improve on the lexical reference on data withheld from development? The [machine-readable protocol](../configs/confirmation.json) is committed before generating reserved predictions or interpreting reserved targets.

## Freeze the experiment before access

The candidate is the serialized route from model run `a971cf3bc6add1c2d818`: the seven-family classifier and accepted development sigmoid for four familiar policies, and the original frozen Qwen centroid with identity calibration for financial advice and spoilers. Its SHA-256 begins `a38b20e1`; the full-development lexical reference begins `daca3348`. Full hashes, the final model-stage marker and training identities are pinned in the protocol. There is no new fitting, tuning or feature search.

Target-blind eligibility run `60ba8e0ede00928c0943` fixes **43,509 rows**: 25,485 familiar-policy rows and 18,024 unseen-policy rows. The prior approximate-copy exclusion removes 67 rows before inference. Exact development body/support exposure and eligible self-support matches are zero. Preserve this cohort even if the result is unfavorable. The target-free inference worker must not receive `solution.csv`.

Freeze the complete prediction file and its SHA-256 in Git before reading the eligible targets. Record the protocol commit, prediction commit, source/configuration/model/input hashes and access timestamp. Read each solution row's identifier before interpreting a label; select only the frozen cohort and verify its existing policy/Usage assignment. Hashing the pinned source bytes establishes integrity without inspecting their labels. A successful score is a reusable completed stage; a restart replays the fixed experiment, never reselects a candidate.

## One primary comparison and explicit guards

The primary metric is the equal-weight mean of the six policy ROC AUCs. Compare candidate minus lexical reference with a **2,000-draw paired normalized-body bootstrap**, seed 2025. A body shared across policies receives the same resampling weight. Report the two-sided 95% percentile interval; the confirmatory family contains one contrast. This interval conditions on these six policies and frozen predictions. It does not estimate uncertainty over future policies or remove all possible conversation/paraphrase dependence.

Promotion requires every predeclared check:

| Requirement | Threshold | Reason |
| --- | ---: | --- |
| Six-policy macro AUC improvement | At least 0.010 | Require a useful gain beyond merely detectable improvement |
| Paired 95% interval lower bound | Greater than 0 | Check sampling uncertainty for the primary comparison |
| Familiar and unseen route macro AUC improvement | Each greater than 0 | Prevent one route from concealing failure of the other |
| Largest individual-policy AUC deterioration | At most 0.030 | Bound harm masked by a macro average |
| Log-loss increase, overall and each route | At most 0.010 | Reject materially worse probability quality |
| Brier increase, overall and each route | At most 0.005 | Check probability quality on another proper score |
| Each policy's positive and negative class counts | At least 20 | Refuse an unsupported AUC-based decision |

These are engineering acceptance thresholds, not estimated optimal values or guarantees of moderation safety. They are fixed from the development objective before confirmation. Failure of any guard blocks promotion; record the failed result without weakening the guard or trying another candidate on the same reserve.

## Explain the result without another selection loop

Also report the uncalibrated fixed route and centroid everywhere as descriptive controls. They isolate calibration and routing effects; they cannot replace the candidate after looking at confirmation labels. Report per-policy and route AUC, log loss, Brier, pooled AUC separately, fixed-threshold confusion, reliability, and confidence coverage/error at the predeclared cutoffs. These cutoffs are diagnostics, not a learned deployment threshold. If self-support matches occur, report their removal as the predeclared sensitivity; the current eligible audit records zero.

The development feature gains remain distinct from calibration improvements and the protected candidate-versus-reference result. Do not describe an aggregate routed gain as the marginal contribution of a single feature family. The canonical submission notebook remains the lexical reference until confirmation and subsequent offline parity, runtime and restore gates pass. No Kaggle submission or real submission CSV is authorized by this protocol.

## Bounded execution and preservation

Use one CPU SageMaker processing job with four encoder workers, the existing pinned float32 model/input format and a separate copy of the verified development embedding cache. Keep the original cache immutable. Encode only missing content hashes; checkpoint completed shards to the existing private project bucket at least every 45 seconds. Cap the job at 7,200 seconds and retain completed shards if it reaches the limit. Inference uses 128-row batches and preserves row identity. Download and verify prediction artifacts before any target access. Runtime measurements belong to the named hardware and include a separate account of startup, encoding and prediction.

After the fixed result, publish its provenance, acceptance decision and limitations, then execute the canonical notebooks. Product packaging and the employer release remain separate milestones. A failed candidate can still support a rigorous research report; it does not earn a claim of validated production readiness.

## Implementation preflight

The target-free input bundle contains 43,509 rows and 780 explicitly allowlisted files: prepared inputs/assignments, the fixed final artifacts and verified seed embedding shards. All payload hashes passed an independent archive read. The pinned encoder needs 45,479 unique formatted inputs; 2,367 already exist and **43,112 require encoding**. Bundle size is 104,749,097 bytes, SHA-256 `d2e5f53a8a9ea7028eb8a31a5bf3ef141be50aa9f886d5b9d10ec49ac0530581`.

Local preflight passed 292 tests and the full software gate. All 43,509 prepared identities, normalized-body hashes and policy assignments were verified. A 16-row real-artifact smoke run used development rows only and reproduced the routed predictions exactly. The new tests reject changed preregistration, re-ordered inputs, target-bearing inference frames and corrupted completed stages. The inference package contains no solution CSV or row targets.

## Current execution boundary

Preregistration is public at commit `5ca6bd50d54ceacc172786e6df0640810aab96d8`. The checkpointed inference implementation was pushed at `8f3788217ac5a6dbbb8530d36bf72e5efc964a40`; [Quality run 34304648502](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/runs/34304648502) passed the full gate, actual execution of all five notebooks, cache reuse, standalone synthetic inference and real pinned-encoder integration.

The fixed scoring implementation separately checks prediction/assignment parity, requires an exact Git prediction-freeze commit descending from preregistration, and creates an exclusive access receipt before interpreting eligible labels. A completed comparison reuses its checkpoint without reopening targets; changing the experiment after access fails. Synthetic acceptance, deliberate rejection, probability-quality regression, insufficient class coverage and target identity/Usage failures are tested. Synthetic results are not published as project performance.

**Historical launch record: protected inference started on 2026-09-09 at 03:17 UTC.** SageMaker job `jigsaw-confirmation-20260909-0250` uses the exact approved input bundle and committed source. All three uploaded objects passed S3 full-object checksum, expected-owner and AES-256 checks. No reserved target was present in the worker inputs; eligible targets were opened later under the published freeze as recorded above. [Launch and object-version evidence](../reports/checkpoints/protected_inference.json). Job creation is not completion evidence. The prepared input archive remains reproducible with `scripts/package_confirmation.py`; the earlier fitted-model/eligibility backup has already passed independent S3 recovery. Planned processing uses one `ml.m5.4xlarge` instance (16 vCPU, 64 GiB) and a 7,200-second job limit. AWS Price List SKU `S5RY38EFJ367KH6V`, queried on 2026-09-09, quotes $0.922/hour for Oregon processing: at most $1.844 of instance runtime under that limit, plus storage/transfer and provisioning charges where applicable. This is a cost bound for the planned job, not a bill or a completed-run measurement.

Reproduction order after cloud inference and checkpoint recovery:

```bash
uv run python scripts/run_confirmation.py freeze --run-id 314494da886e11bcc1f6
# Commit and push reports/confirmation/prediction_freeze.json; fetch that exact Git commit.
uv run python scripts/run_confirmation.py score --run-id 314494da886e11bcc1f6 --prediction-commit <published-40-character-commit>
```

Only aggregate metrics and lineage belong in the public report. The fixed result, whether accepted or rejected, must be preserved before any product-promotion decision. The reserve cannot become another development split.

## Recover and publish the result

For a normally completed worker, require both SageMaker `Completed` and the worker's `state.json` event `completed`. The actual failed-wrapper recovery used the separately verified completed prediction stage described below. Download `checkpoints/protected-predictions.tar.gz` from the recorded prefix using the expected bucket owner. Take its byte count and SHA-256 from that completed state. The recovery command validates the full archive and every completed stage before restoring only the four prediction artifacts. It refuses to overwrite different existing predictions and leaves the original development embedding cache unchanged. The downloaded archive preserves the new cache for later product work.

```bash
uv run python scripts/recover_confirmation.py --archive <downloaded-archive> --run-id 314494da886e11bcc1f6 --sha256 <completed-state-sha256> --bytes <completed-state-bytes>
uv run python scripts/run_confirmation.py freeze --run-id 314494da886e11bcc1f6
# Commit and push the prediction freeze, then fetch its exact published Git commit.
uv run python scripts/run_confirmation.py score --run-id 314494da886e11bcc1f6 --prediction-commit <published-40-character-commit>
uv run python scripts/build_protected_report.py --run-id 314494da886e11bcc1f6
```

The report exporter verifies evaluation, prediction, target-access receipt and preregistration lineage. An independent aggregate audit reconciles all six per-policy AUCs with the policy macro, row-weighted log loss and Brier with each route, and all 12 fixed acceptance checks. It preserves a rejected candidate. Three Plotly figures with SVG fallbacks explain policy gains, probability quality and confidence/coverage. Their manifests bind the actual report and rendering source. Notebook execution fingerprints include the protected report and its figure builder, so adding real results invalidates prior display checkpoints.

Recovery tests deliberately corrupt an archive, alter its digest, remove a prediction artifact, add a traversal or target-bearing path, and try to replace an existing prediction. Reuse tests verify that identical recovery preserves both predictions and seed vectors. Report tests reject inconsistent macro AUC, weighted loss, per-policy AUC, class counts, interval identity, nonfinite bounds and acceptance decisions. All use synthetic data; no synthetic metric is published as research evidence.

If the job reaches its limit, use only completed S3 shard markers for continuation under the original model/input contract. A launch record or partial shard count never substitutes for a completed prediction freeze. Do not change source files named in inference provenance during recovery; later reporting modules are separate from those frozen inference files.

## Verified preparation publication

The final local gate passed **305 tests** in 116.7 seconds, plus compile, Ruff lint/format and canonical notebook-source checks. [Quality run 34305118975](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/runs/34305118975) passed on scoring commit `a846906efd38785dca33331a21be9858afd34d9d`, including all five actual encrypted Jupyter executions, completed-cache reuse, synthetic offline inference, real pinned-encoder integration and public rendering.

The downloaded 1,190,503-byte CI artifact has SHA-256 `87bfb022cfdf25c1d152a27bb02fcd9e6fc128dd4a2363134dc0cbbb580ae40a`. Each restored notebook matches canonical source and every current input hash, with verified stage members and no errors or stderr. Their recorded execution environment is Python 3.12.14. [Publication proof](../reports/checkpoints/confirmation_preflight.json). These historical checks validated the preflight software and evidence publication. The completed protected experiment and later recovery tests are recorded above.


## Completed-stage recovery used for this result

The four individually downloaded prediction files are accepted only with the independently verified complete-marker digest. All payload hashes, model/protocol/input identities, 43,509 unique row IDs and finite probabilities passed. Different existing files are never overwritten; the marker is written last.

```bash
uv run python scripts/recover_confirmation.py --stage-directory <downloaded-prediction-stage> --run-id 314494da886e11bcc1f6 --marker-sha256 68af5b10cfda60bf587f96dddb9cdb89101380734762735b09ada60a172587d6
uv run python scripts/run_confirmation.py freeze --run-id 314494da886e11bcc1f6
# The published freeze commit must be present in local Git history.
uv run python scripts/run_confirmation.py score --run-id 314494da886e11bcc1f6 --prediction-commit 530ad79929012e807cb42a5253f7b92b090682ec
uv run python scripts/build_protected_report.py --run-id 314494da886e11bcc1f6
```

The score command reuses an intact comparison checkpoint; it does not select another candidate. Recovery tests cover marker/payload corruption, missing files, unexpected target files, no-overwrite behavior, idempotent replay and pruning active staging paths before traversal.
