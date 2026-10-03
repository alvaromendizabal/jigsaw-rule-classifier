# Pseudo-supervision frontier: what transferred and what did not

## Purpose

After the five-model AWS ensemble became the strongest development candidate, the next question was whether **structurally different supervision** could improve policy transfer rather than continuing to tune ensemble weights. Two bounded AWS studies tested that idea: an owned pseudo-supervision route on the existing Qwen3-4B adaptation system and an external-domain soft-label route using a complementary DeBERTa student.

The official retained Kaggle result remains **0.91808 public / 0.91425 private ROC AUC**. Neither study used leaderboard feedback for model selection, and neither replaces the retained scored system or the **0.740351** five-model development candidate.

GitHub publishes aggregate evidence only. Raw comments, row-level predictions, exact ensemble weights, adapters/model weights, teacher-score arrays, optimizer state, caches, credentials, and full AWS logs remain private.

## E33 · owned uncertainty/disagreement pseudo-supervision

E33 reused frozen OOF teacher rankings from the incumbent ensemble and retrained only the Qwen3-4B adapted component. It tested three matched variants across both observed policies:

| Variant | Policy-macro AUC | Δ vs incumbent | Advertising Δ | Legal Advice Δ | P(Δ>0) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Uncertainty 20% soft | 0.739486 | -0.000866 | +0.000784 | -0.002515 | 0.296 | Reject |
| Disagreement 20% soft | 0.740520 | +0.000169 | +0.001306 | -0.000969 | 0.526 | Reject |
| Uncertainty 20% soft + GCE | 0.740544 | +0.000193 | +0.000112 | +0.000274 | 0.507 | Reject |

The best point estimate gained only **+0.000193**, far below the preregistered +0.004 gate, with bootstrap probability near chance. E33 therefore closes **small-cohort, same-family pseudo-supervision** rather than spending more compute on nearby pseudo-label fractions.

### Engineering evidence

E33 ran directly in SageMaker Studio on one **NVIDIA L4 (`ml.g6.4xlarge`)**. The full six-unit study completed in about **32.3 minutes** with **8.27 GiB** peak framework allocation. A real-workload batch benchmark retained batch 8 because it preserved exact score parity.

## E36 · public external text, Qwen teachers, complementary DeBERTa student

E36 asked a different question: does a larger target-domain text pool help when the student architecture is also changed? It sampled **10,000 deterministic public Reddit moderation comments** from a pinned public corpus, excluded normalized overlaps with local development text, ignored the corpus's source topic labels, and generated soft targets using the two fold-specific Qwen3-4B teachers.

The complementary student was **`microsoft/deberta-v3-base`**, trained under a matched labeled-only control and two external-data variants:

| Variant | Standalone macro AUC | 20% blend AUC | Δ vs incumbent | Gain vs DeBERTa control | Advertising Δ | Legal Advice Δ | P(Δ>0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Labeled-only control | 0.587230 | 0.730421 | -0.009930 | — | -0.008843 | -0.011017 | 0.010 |
| External soft labels | 0.644264 | 0.734680 | -0.005671 | +0.004259 | +0.001455 | -0.012798 | 0.071 |
| External confidence extremes | 0.638456 | 0.733699 | -0.006653 | +0.003278 | -0.001828 | -0.011477 | 0.040 |

The external soft-label route **did improve DeBERTa over its matched labeled-only control by +0.004259 macro AUC**, which is useful evidence that the extra supervision carried signal. However, the resulting fixed 20% blend still trailed the incumbent by **-0.005671**, and Legal Advice regressed by **-0.012798**. The route therefore failed the incumbent, policy-stability and bootstrap gates.

E36 is a valid negative result for this formulation, not evidence that external data or DeBERTa are universally unhelpful.

### Engineering evidence

The first full-batch DeBERTa attempt exposed an L4 memory problem. The corrected run benchmarked the longest actual external sequences, enabled gradient checkpointing, preserved effective batch 32 through accumulation, and selected microbatch 16. The completed six-unit V4 study used only about **4.19 GiB** peak framework allocation and finished in about **7.6 minutes** after reusing the already-computed 20,000 teacher scores.

## What this changes

The two studies narrow the next research step:

- do **not** reopen small pseudo-label fraction tuning around E33;
- do **not** tune E36's external confidence threshold merely because external labels improved the DeBERTa control;
- keep Deep Mutual Learning blocked until exact Qwen3-14B source/prompt parity is recovered;
- test **cross-rule hard-negative transfer** next using the immutable cached Qwen representation space, then escalate to stronger contrastive/ranking training only if that bounded screen provides evidence.

The public machine-readable evidence is [`reports/checkpoints/pseudo_supervision_frontier.json`](../reports/checkpoints/pseudo_supervision_frontier.json). The executed review notebook is [`notebooks/30_pseudo_supervision_frontier_review.ipynb`](../notebooks/30_pseudo_supervision_frontier_review.ipynb).
