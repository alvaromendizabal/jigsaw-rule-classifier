# Protected confirmation of the fitted policy route

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

**Real protected predictions and evaluation have not run.** No SageMaker job has been launched for this protocol, and no reserved target has been opened. The prepared input archive remains reproducible with `scripts/package_confirmation.py`; the earlier fitted-model/eligibility backup has already passed independent S3 recovery. Planned processing uses one `ml.m5.4xlarge` instance (16 vCPU, 64 GiB) and a 7,200-second job limit. AWS Price List SKU `S5RY38EFJ367KH6V`, queried on 2026-09-09, quotes $0.922/hour for Oregon processing: at most $1.844 of instance runtime under that limit, plus storage/transfer and provisioning charges where applicable. This is a cost bound for the planned job, not a bill or a completed-run measurement.

Reproduction order after cloud inference and checkpoint recovery:

```bash
uv run python scripts/run_confirmation.py freeze --run-id 314494da886e11bcc1f6
# Commit and push reports/confirmation/prediction_freeze.json; fetch that exact Git commit.
uv run python scripts/run_confirmation.py score --run-id 314494da886e11bcc1f6 --prediction-commit <published-40-character-commit>
```

Only aggregate metrics and lineage belong in the public report. The fixed result, whether accepted or rejected, must be preserved before any product-promotion decision. The reserve cannot become another development split.

## Verified preparation publication

The final local gate passed **305 tests** in 116.7 seconds, plus compile, Ruff lint/format and canonical notebook-source checks. [Quality run 34305118975](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/runs/34305118975) passed on scoring commit `a846906efd38785dca33331a21be9858afd34d9d`, including all five actual encrypted Jupyter executions, completed-cache reuse, synthetic offline inference, real pinned-encoder integration and public rendering.

The downloaded 1,190,503-byte CI artifact has SHA-256 `87bfb022cfdf25c1d152a27bb02fcd9e6fc128dd4a2363134dc0cbbb580ae40a`. Each restored notebook matches canonical source and every current input hash, with verified stage members and no errors or stderr. Their recorded execution environment is Python 3.12.14. [Publication proof](../reports/checkpoints/confirmation_preflight.json). These checks validate software and existing evidence publication; they do not replace the pending protected experiment.
