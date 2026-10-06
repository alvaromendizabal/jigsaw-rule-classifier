# Frontier research · architecture, context, transfer, and cross-model learning

This document summarizes the post-ensemble research program that followed the five-model development frontier. It is intentionally **aggregate and employer-facing**: raw comments, row-level predictions, exact ensemble weights, optimizer/model state, private caches, credentials, and full cloud logs remain outside GitHub.

The purpose of this phase was not to accumulate experiments. It was to test materially different hypotheses against a frozen incumbent, preserve useful negative results, and avoid promoting a model simply because it was newer or larger.

## Evaluation contract

| Item | Public aggregate |
| --- | --- |
| Primary development metric | Policy-macro ROC AUC, higher is better |
| Evaluation population | 881 comments across two policies |
| Incumbent | Five-backbone fixed development ensemble |
| Incumbent score | **0.740351** |
| Selection rule | Registered controls + policy-specific stability + uncertainty gates |
| Caveat | Repeatedly inspected development cohort; not an untouched holdout or leaderboard score |

The original architecture/context receipt is stored in [`reports/checkpoints/post_closeout_frontier_20261004.json`](../reports/checkpoints/post_closeout_frontier_20261004.json). The later cross-model receipt is stored in [`reports/checkpoints/cross_model_frontier_20261006.json`](../reports/checkpoints/cross_model_frontier_20261006.json).

## Research map

### 1. Complementary Llama backbone

A support-adapted Llama-family challenger produced the strongest point estimate of this phase:

- policy-macro AUC: **0.743436**
- delta versus the accepted incumbent: **+0.003085**
- Advertising improved materially
- Legal Advice improved only slightly
- bootstrap evidence did not clear the registered confidence threshold

**Decision:** preserve as a promising challenger, but do not promote.

This is an important model-selection result: a higher point estimate is not automatically enough to replace the incumbent.

### 2. Pairwise ranking continuation

The next experiment tested whether the useful Llama component could be improved with an explicit ranking objective. The matched ordinary-continuation control and pairwise objective finished close to one another, with the ranking route still below the preserved Llama challenger.

**Decision:** valid negative. Do not spend the next round sweeping ranking-loss weights.

### 3. Semantic support retrieval

Three inference-time context strategies were compared with the same saved Llama adapters:

- supplied examples
- symmetric semantic retrieval
- asymmetric retrieval

Asymmetric retrieval improved over both context controls, but it remained below the preserved Llama challenger and regressed Legal Advice.

**Decision:** valid negative under the registered design. Retrieval infrastructure remains reusable, but the fixed formulation is closed.

### 4. Bidirectional encoder + support context

A ModernBERT-family encoder study tested:

- rule-only training
- single-view support context
- four-view support-context averaging

All three variants finished below the incumbent. Four-view averaging only slightly improved over single-view context and remained well below the rule-only control.

**Decision:** valid negative. More context is not automatically better.

### 5. Entailment transfer

A pretrained NLI family tested three controlled variants:

- frozen rule-level entailment
- frozen rule-clause decomposition
- supervised entailment adaptation

Clause decomposition improved over the direct entailment control, but all three variants remained below the incumbent, with weakness on Legal Advice.

**Decision:** valid negative. The experiment was useful because it eliminated a materially different architecture/task formulation rather than another small hyperparameter tweak.

### 6. Hard-negative and pseudo-supervision routes

Earlier post-closeout screens tested frozen hard-negative transfer, owned pseudo-supervision, and public-text teacher/student transfer. These studies are also retained as valid negatives where their registered gates failed.

**Decision:** keep the evidence; do not repeatedly micro-tune dead directions.

### 7. Matched native continuation and fixed-teacher transfer

The strongest later Llama control came from ordinary continuation with the same training schedule used to evaluate teacher-guided alternatives.

- matched native continuation: **0.743973**
- fixed-teacher probability transfer: **0.743455**
- label-anchored margin transfer: **0.743644**

Both teacher-guided formulations finished below their matched native continuation.

**Decision:** preserve the native continuation as a strong diagnostic reference and close these fixed-teacher formulations. Do not sweep nearby temperatures, loss weights, or blend weights after the registered comparison failed.

### 8. Dynamic reciprocal learning

A separate two-round reciprocal pilot tested whether changing peers could outperform fixed-teacher transfer.

- matched native peer control: **0.742236**
- reciprocal primary: **0.742539**
- delta versus matched control: **+0.000303**
- grouped-bootstrap positive fraction: **76.9%**
- 95% interval: **[-0.000512, +0.001205]**

The point estimate moved slightly upward versus the matched control, but the effect was small, the uncertainty interval crossed zero, and the result remained below the stronger native-continuation reference.

**Decision:** valid negative. Close this fixed reciprocal pilot rather than selecting a favorable directional diagnostic or tuning adjacent temperatures/weights.

### 9. Next frontier: contextual training, not more inference-only context

Earlier retrieval/context studies mostly changed which examples appeared at inference. The next AWS experiment changes the **training distribution** instead: paired labeled demonstrations are introduced during supervised Llama continuation, while native/plain and native/context controls are reused.

This design asks a new question: can the model learn to use examples because it was trained under that context structure, rather than simply being shown more context at inference?

The public repository records the mechanism and controls, but the exact competitive prompt construction, row-level examples, checkpoints, and private predictions remain in AWS.

## What the research program demonstrates

### Controlled attribution

Each major mechanism was evaluated against a matched control wherever practical. The research distinguishes:

- architecture changes from context changes
- training-objective changes from ordinary continuation
- retrieval effects from merely adding examples
- teacher/student benefits from standalone student quality
- point-estimate gains from statistically credible promotion evidence

### Champion/challenger discipline

The accepted development champion remains frozen until a challenger clears the registered gates. That prevents adaptive experimentation from silently redefining success after results are visible.

### Negative-result discipline

Negative results are preserved because they reduce uncertainty and stop repeated spend on weak directions. A correctly executed negative experiment is treated as successful research, not an engineering failure.

### Provenance and resumability

AWS remains the canonical private workspace. Long-running experiments use:

- pinned model/data identities
- content-addressed artifacts
- checksum-verified checkpoints
- resumable training and inference shards
- structured logs and heartbeats
- explicit resource/cost telemetry
- atomic return bundles

### Public/private reproducibility boundary

GitHub contains enough aggregate evidence to review the methodology and decisions without publishing competitive private state.

Public:
- source modules and tests
- compact configs
- aggregate metrics and decisions
- executed review notebooks
- public research notes
- machine-readable aggregate receipts

Private on AWS:
- raw competition comments
- row-level labels/predictions
- exact ensemble weights
- model and optimizer state
- teacher-score arrays
- private caches
- full operational logs

## Current frontier

Qwen3-14B live replay and peer-training qualification are now resolved. Fixed-teacher and reciprocal cross-model experiments have completed and are retained as valid negatives under their registered controls.

The active AWS direction is **paired-demonstration supervised adaptation**. It introduces labeled demonstrations during continuation training and evaluates them in a fixed factorial design against reused native controls. This is materially different from the closed inference-only retrieval/context routes.

The repository remains intentionally semi-reproducible. Review the exact public contract in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Employer review takeaway

This phase is useful even though most candidates were rejected. It demonstrates the part of applied ML work that is often missing from portfolio projects:

- choosing experiments by expected information value
- building leakage-aware controls
- debugging model/data provenance
- measuring uncertainty before promotion
- preserving strong incumbents
- treating GPU/runtime reliability as part of model quality
- knowing when to stop a research direction

For the broader project story, return to the [README](../README.md), [START_HERE](../START_HERE.md), or [model card](../MODEL_CARD.md).
