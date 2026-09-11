# Asymmetric support selection — registered CPU milestone

**Status:** implementation and software tests complete; real cached-vector audit pending.  
**Base:** `6b41e0626d7ebedbcdc09a220804b5518e6e265a`.  
**No AUC, model training, GPU inference, or query-target access is part of this milestone.**

## Why this is the next experiment

The completed nearest/nearest adapted-vector selector changed almost every selected pair but failed the private, label-blind relevance review. The failure is consistent with a retrieval-space problem rather than evidence that demonstration context is exhausted: nearest permitted examples can collapse onto topic hubs without clarifying the rule boundary.

A top-five Jigsaw solution reported a materially different retrieval rule: choose the most similar positive example but the *least similar negative* example before prompting the LLM. That candidate reportedly improved its few-shot configuration by about 0.006 AUC. This does not transfer automatically to our model, but it creates a specific, falsifiable, target-free hypothesis using representations we already paid to compute.

The broader winner review also exposes two additional open families that stay on the research queue:

- **Supervision quality / conflict handling.** Our current `support_pairs` drops conflicting normalized `(rule, body)` pairs. The published adaptation protocol records 2 conflicting pairs / 79 conflicting occurrences in advertising and 10 pairs / 339 conflicting occurrences in legal advice. Second place used majority-vote conflict cleaning; fourth place used duplicate conflict ratios as soft targets plus a robust generalized cross-entropy loss. The conflict rate is especially skewed toward the policy that has been harder for recent prompt changes, so this family merits a later bounded audit before another expensive model search.
- **Diversity rather than raw capacity.** First, third, fifth, and sixth place solutions all benefited from model-family or representation diversity and rank blending. We already tested Phi and Qwen3-8B under fixed protocols without clearing promotion gates, so any new diversity experiment needs a distinct mechanism rather than another unregistered size sweep.

## Frozen hypothesis

For each query, within the same rule and legitimate supplied-support pool:

1. choose the **nearest violating** support in the cached adapted-vector space;
2. choose the **farthest permitted** support in that same space;
3. retain deterministic normalized-text tie breaking and exact vector-to-plan index alignment;
4. compare against both the historical character-TF-IDF selector and the rejected nearest/nearest adapted-vector selector.

The interpretation is geometric: a relevant positive anchor plus a deliberately different permitted anchor may span the rule boundary more clearly than two same-topic nearest neighbors. A far negative can also become irrelevant, so **diversity is not promotion evidence**; qualitative relevance review remains mandatory.

## Isolation and leakage contract

- Original training labels and supplied positive/negative support labels only.
- No organizer-released targets or consumed protected cohort.
- Query plan schema remains exactly `row_id, body, rule`; no query targets enter selection.
- Only `train_adapted_vectors`, `query_adapted_vectors`, and `query_row_ids` are read from cached NPZ files.
- No saved prediction arrays, model weights, encoder calls, training, or external LLM/API calls.
- Matrix row `i` remains bound to `plan[fold].training[i]`; sorting text never independently reorders vectors.
- Existing nearest/nearest audit artifacts remain immutable.

## Bounded execution

The audit is CPU-only, single-threaded by default, batch size 64, with a 600-second hard wall-time limit and 30-second-or-faster heartbeat. Shards and fold summaries are saved atomically with manifest-last completion markers. A completed run is checksum-verified and reused without recomputation.

Stop before any GPU experiment if any policy produces the same pair selection as the lexical comparator, if vector/plan alignment fails, if any query target appears, or if the assistant's blinded relevance review shows that far negatives are mostly irrelevant. A positive review would authorize **one** retained-4B inference ablation under the existing promotion gates; it would not authorize a sweep or submission.

## Software verification

Local isolated tests cover:

- stable min/max selection and exact-score tie behavior;
- nearest-positive / farthest-negative semantics;
- vector-to-training-row alignment under joint reordering;
- zero-vector exclusion;
- target-free aggregate reporting;
- bounded audit execution and saved private pair inventory;
- completed-result replay without recomputation;
- corruption rejection.

The exact repository CI remains the publication gate.

## Research references

- Jigsaw 5th Place, *Diverse Ensemble*: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/5th-place-solution-diverse-ensemble
- Jigsaw 2nd Place Solution: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/2nd-place-solution
- Jigsaw 4th Place, *Instruct LLM is all you need*: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/4th-place-solution
- Jigsaw 1st Place Solution: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/1st-place-solution
- Jigsaw 3rd Place Solution: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/3rd-place-solution
- Jigsaw 6th Place, *Online Distillation via Deep Mutual Learning*: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/6th-place-solution
