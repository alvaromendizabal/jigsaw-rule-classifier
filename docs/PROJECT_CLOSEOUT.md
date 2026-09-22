# Project closeout · rule-conditioned NLP

**Decision: complete the research-and-engineering portfolio and retain the verified support-adapted Qwen3-4B system.** This closeout follows the owner's instruction to finish the project. It does not depend on reaching the original stretch target or starting another modeling round.

## Delivered result

The final retained system scored **0.91808 public / 0.91425 private ROC AUC** in a successful late Kaggle evaluation. Private AUC improved **0.29469** over the 0.61956 lexical reference. The documented historical winner scored 0.92930 private AUC, leaving a **0.01505** absolute gap. This is a strong completed result and a substantial improvement, not an original rank, medal, or claim to have won.

[Exact scored version and runtime](../reports/checkpoints/kaggle_adaptation.json) · [Historical benchmark attribution](../configs/top_solution_integration.json) · [Final model card](../MODEL_CARD.md).

## Technical contribution

The project connects policy-conditioned language modeling, data-quality analysis, transfer-aware validation, feature research, parameter-efficient adaptation, reproducible inference, and cloud recovery in one coherent workflow.

The matched 881-comment study demonstrates that legitimate support adaptation adds signal at a fixed backbone: **0.614600 → 0.719893** policy-macro AUC, with simultaneous 95% gain interval **[0.061237, 0.149350]**. The interval is conditional on the observed policies and fixed predictions, not an adaptive-search-wide guarantee. [Results](../reports/support_adaptation/results.json) · [Uncertainty](../reports/support_adaptation/uncertainty.json).

The separate four-policy feature campaign records **323 fixed fits** and shows why larger feature banks can fail under policy shift: the full transferable-feature model reached **0.798916 familiar-policy AUC and 0.551454 held-out-policy AUC**. The original frozen centroid performed better on the held-out-policy protocol. These are post-competition research results, not Kaggle scores. [Expanded study](EXPANDED_STUDY.md).

A subsequent frozen confirmation used **43,509 eligible rows**, passed all **12** declared checks, and improved macro AUC from **0.680103 to 0.776991**. This evaluates the separate historical research route, not the final 4B neural system. [Confirmation](CONFIRMATION.md).

## Scope completed and preserved

The delivered portfolio includes the retained neural inference path, source/configuration/model identities, documented validation boundaries, executed evidence notebooks, matched comparisons, negative results, model/data cards, automated quality gates, and private-artifact recovery documentation. The project-review notebook renders public aggregates without loading a model or accessing private accounts.

The accepted competition notebook and historical research outputs are preserved. This publication changes presentation, closeout metadata, and report verification—not the scored model or its predictions.

## Latest supplementary experiment

The strict-majority conflict-resolution arm recovered **10** pairs, dropped **one tie**, preserved uncontested-label parity, and passed both recorded query-purge checks. It completed a ten-row engineering preview and was confirmed as submission **56444879**. Its last verified status was pending at **2026-09-21 23:08 UTC**; no later score was available in the reviewed evidence. The history identifies it through the unique description and prior version-specific submission command, not an independently returned version field.

This is a documented **unpromoted supplementary experiment**. It is not a hidden project blocker and is not a claimed performance improvement. No new preview, training job, submission, or automatic promotion is authorized by this closeout. [Immutable experiment snapshot](../reports/majority_submission/summary.json).

## Publication and storage boundary

**GitHub:** source, small configurations, tests, executed notebooks, compact aggregate results, attribution, and documentation.

**AWS/private storage:** raw comments and labels by row, row-level predictions, model weights, optimizer checkpoints, environments, private caches, and full operational logs. Public publication is not a cloud backup or a synchronization of every workspace file. The AWS checkout is not claimed to have been updated by a GitHub merge.

## What is deliberately outside this release

Stronger-backbone and ensemble experiments remain optional research opportunities. The original 0.92–0.93 stretch objective was not achieved by the retained result; that fact does not make the completed portfolio unfinished. The source remains available for maintenance, but no new research round is required to call this project complete.

Production deployment would require an explicit operating threshold, human-review policy, privacy and fairness review, robustness tests, and load validation. No such deployment is claimed.

## Employer-facing description

> Built an end-to-end rule-conditioned NLP system using Qwen3-4B and LoRA, achieving 0.91425 private Kaggle ROC AUC—a 0.29469 improvement over the lexical baseline. Combined controlled feature and representation studies, transfer-aware validation, support-example adaptation, resumable cloud execution, and tested offline inference. Published a reproducible notebook-first portfolio with clear model decisions and preserved negative results.

## Verification

The machine-readable [closeout record](../reports/portfolio/closeout.json) pins the reviewed result files. The [project-review notebook](../notebooks/27_latest_system_checkpoint.ipynb) is executed from those aggregates and stores its Plotly and SVG outputs. The closeout tests verify source hashes, metric arithmetic, unscored-candidate handling, report execution, and output persistence. GitHub Quality must pass on the exact publication head before merge.
