# CSLS-style support-selection experiment — completed

## Decision

**Stop this selector. Do not run a blinded relevance review or GPU inference.** It did not satisfy the predeclared support-reuse gate, and it produced no AUC evidence. The accepted support-adapted Qwen3-4B remains unchanged at **0.91808 public / 0.91425 private Kaggle AUC**.

## Bounded execution

GitHub PR #27 passed the full quality workflow before the experiment: project verification, all five actual Jupyter notebook executions/replay, synthetic and original-preview submission checks, pinned encoder verification and rendering. The real audit then ran once on AWS as processing job `jigsaw-csls-support-audit-20260911-043145` using one `ml.m5.2xlarge`, a 600-second hard cap, and the previously verified source plan plus two cached representation matrices. The worker read no query targets or prediction arrays, made zero model calls, used no GPU, and performed no training.

The job started at 04:32:26 UTC and ended at 04:33:57 UTC. Public aggregates are in `reports/support_selection_csls/`; the 734,735-byte selected-pair file remains private in S3.

## Result

| Policy | Changed pairs | Raw positive max reuse | CSLS positive max reuse | Raw negative max reuse | CSLS negative max reuse | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Advertising | 114 / 234 (48.72%) | 9.83% | 5.98% | 9.83% | 9.83% | Fail |
| Legal advice | 268 / 647 (41.42%) | 14.06% | 11.44% | 18.70% | **19.63%** | Fail |

CSLS increased distinct supports and effective support counts, so the hubness hypothesis was partly directionally useful. But the registered gate required maximum reuse to strictly improve for both positive and negative examples in every policy. Advertising negative maximum reuse was unchanged, while legal negative maximum reuse became worse. The experiment therefore stops before qualitative review or model inference.

## What this teaches us

A generic local-density correction does not resolve the legal-advice demonstration-selection problem. More diverse retrieval is not sufficient evidence of better adjudicative examples. Do not tune `k` on these same two repeatedly inspected policies to rescue the result.

The next high-value family is **conflict-aware training supervision**. The current data builder drops conflicting normalized `(rule, body)` labels. The existing adaptation protocol already shows policy-skewed conflict mass: 2 conflicting pairs / 79 occurrences for advertising versus 10 / 339 for legal advice. Before paying for retraining, the next milestone must reconstruct the legitimate occurrence-level votes/provenance on CPU, compare drop/majority/soft-target formulations, and freeze exactly one candidate with leakage tests and a stop condition.
