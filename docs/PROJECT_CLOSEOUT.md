# Project closeout · historical release and frontier extension

The original portfolio closeout retained the support-adapted Qwen3-4B system at **0.91808 public / 0.91425 private ROC AUC**. That scored result remains the canonical leaderboard system.

Competitive research was subsequently reopened. This document therefore distinguishes the **historical completed release** from the newer frontier extension rather than pretending the later experiments did not happen.

## Retained scored result

The retained Qwen3-4B system improved private AUC by **0.29469** over the 0.61956 lexical reference. The documented historical winner scored 0.92930 private AUC, leaving a **0.01505** absolute gap.

The matched 881-comment study shows support adaptation at fixed backbone: **0.614600 → 0.719893** policy-macro AUC. The repeatedly inspected cohort is development evidence, not a fresh final holdout.

## Post-closeout frontier extension

Three later directions are now resolved enough to update the public record:

- **Strict-majority conflict handling:** scored **0.91720 public / 0.91288 private**, a regression versus the retained 4B system. Rejected.
- **Qwen3-8B / Phi diversity studies:** standalone models did not beat the retained development reference; fixed blends produced small positive but uncertain gains.
- **Qwen3-14B AWS study:** standalone 14B scored **0.708455** policy-macro AUC versus **0.719893** for 4B on the fixed development cohort. A fixed 50/50 4B+14B rank blend reached **0.730175**, improving both observed policies by a combined **+0.010282** policy-macro AUC.

The 14B blend interval crosses zero and has not been hidden-scored. The correct conclusion is **complementarity worth further ensemble study**, not a promotion claim.

[Latest frontier report](QWEN14B_FRONTIER.md) · [Executed notebook](../notebooks/28_qwen14b_frontier_review.ipynb).

## Publication boundary

**GitHub:** public source, compact configurations, tests, aggregate evidence, attribution, and executed review notebooks.

**AWS/private storage:** raw comments/labels, row-level predictions, model weights, optimizer/checkpoint state, environments, caches, and operational logs.

This boundary is intentional. The repository is an employer-facing research artifact, not a backup of the cloud workspace.

## Employer-facing description

> Built an end-to-end rule-conditioned NLP system using Qwen3-4B and LoRA, reaching 0.91425 private Kaggle ROC AUC (+0.29469 over the lexical baseline). Extended the system with transfer-aware feature research, resumable AWS experiments, Qwen3-8B/14B and Phi backbone studies, and model-diversity analysis. Preserved negative results and separated hidden leaderboard scoring from development selection.

## Current next step

The latest evidence points toward leakage-safe ensemble selection across preserved 4B, 8B, 14B, and Phi OOF predictions. Another Kaggle submission should be reserved for a fixed candidate after that AWS evidence is available.

The original closeout remains valid as a historical release boundary; this frontier extension is an additional research phase.
