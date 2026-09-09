# Semantic formatting and policy intent

This is a bounded development experiment, committed before its encoder inference or scores. It follows the four-policy study, retrieval ablation and embedding-resolution study. Earlier development findings informed this experiment; it is not an independent confirmation.

The current frozen Qwen centroid reaches 0.7042 policy-macro transfer AUC, but legal and medical advice remain weak. An earlier review of eight extreme errors suggested confusion between topic and prohibited behavior, short advice requests, support mismatch and ambiguous labels. That selected review cannot estimate error prevalence.

## Two remaining hypotheses

The [Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) demonstrates instruction-bearing retrieval queries and plain documents. Our current representation instructs both the comment and its supports. Test an asymmetric representation by reusing every original comment vector and encoding only the supplied support texts as plain documents. Keep the encoder revision, precision, token limit, vector width and score fixed. The 2,217 unique support strings are available at inference; no training or validation label is used to generate their vectors. An independent cache contract prevents changing the historical representation.

The [pinned NLI model](https://huggingface.co/cross-encoder/nli-deberta-v3-small) predicts contradiction, entailment and neutrality. The historical hypothesis asks whether a comment violates a negatively worded rule. Test one affirmative rendering of the same prohibited behavior. `Do not offer or request legal advice` becomes `The author uses this comment to offer or request legal advice.` A list ending in `are not allowed` becomes `This comment contains ...`. Unsupported syntax fails explicitly. No policy-specific label statistics or semantic expansion enter the compiler. Compare the original and affirmative entailment probabilities, then their fixed feature map alone and added to the existing semantic scalars. This tests behavior representation without a larger encoder or classifier search.

## Frozen design

`configs/formatting.json` is the pre-score specification. Use the existing 11,135 development rows, exact seven saved splits, training-only feature screens and fixed logistic classifier. Reuse the original word/semantic banks and OOF controls after checking their stage hashes and row order. Four learned representations across seven folds require 28 fits; three frozen scores require none. Report seven named paired contrasts, per-policy AUC, log loss and Brier score. Conditional simultaneous intervals cover these seven contrasts, not the entire adaptive project history.

Only 32 asymmetric similarity scalars and a small fixed NLI interaction map are screened; there is no second arbitrary coordinate search. The existing broad feature catalogue remains the candidate-space evidence. Rank and missingness checks, duplicate/correlation rejection, per-fold selection stability, coefficients and within-policy permutation importance accompany the new fitted families. Support-order invariance, batch independence, forbidden target access, cache integrity, split alignment and serialization are explicit test requirements.

Select 48 legal/medical examples using the original centroid, balanced across policy, label and score tertile. Within each stratum choose four distinct normalized bodies by seeded content hash. Review speech act, policy behavior, context dependence and label ambiguity without seeing new-model predictions. Keep raw text and row annotations private; publish the sampling contract, aggregate review and limitations. Do not change labels or infer legal/medical correctness beyond the moderation-policy question.

## Boundary and stopping decision

The research loader never opens the 43,576 reserved targets. No new external labeled corpus, leaderboard probing, rule-specific tuning, probability calibration or final-model promotion is authorized by this experiment. All experiment actions themselves are covered by the user's standing authorization.

Preserve new inference shards and completed fits every 45 seconds in the owned project bucket. Reuse existing compatible inference; pin all model files and software. One CPU worker instance has a hard 90-minute ceiling. Record actual encoder inputs, truncation, elapsed time, memory and checkpoint hashes. A failed or negative comparison remains visible.

After both hypotheses and the structured error audit are complete, write a feature stopping decision. Prefer the existing centroid unless a candidate improves transfer ranking without a material per-policy or probability-quality regression; do not pick a candidate from seen-policy accuracy alone. Any justified confirmation must have a separately committed candidate, reference, calibration policy and acceptance specification before reserve targets are read. Product delivery and a quick employer-facing evidence tour follow that decision.

### Decision thresholds frozen before new score inspection

`configs/feature_decision.json` operationalizes the earlier qualitative decision rule. It was added after encoder inference started but before any new fitted or frozen-score result was inspected. The pre-score 48-row audit is complete; all sampled rules, bodies and labels match their original sources. No new-model scores were shown during annotation.

Compare all seven candidates directly with the existing centroid. A candidate must improve policy-macro AUC by at least 0.005, have a positive simultaneous lower confidence bound, lose at most 0.02 AUC on every observed policy, and increase log loss/Brier by no more than 0.01/0.005. These are declared practical tolerances, not universal statistical thresholds. Among qualifying candidates within 0.002 AUC of the best, prefer the declared simpler representation. Retain the existing centroid if none qualifies. The seven new centroid contrasts have their own conditional simultaneous correction; they do not retrospectively correct the whole adaptive search.

This rule separates choosing a development candidate from independent confirmation and production acceptance. A rejected feature remains a documented negative result. Closing the research phase requires the independent artifact replay and a coverage/stopping rationale as well as these numerical checks.
