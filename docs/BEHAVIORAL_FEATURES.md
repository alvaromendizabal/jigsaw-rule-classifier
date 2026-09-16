# Behavioral feature investigation — Round 1

## Status and narrow question

Prepared for manual execution after notebook 05 passed. No real-data result has been measured for this new family. The fixed question is: **do author-attributed behavior, rule–behavior interactions, and cross-fitted supplied-support comparisons add value beyond the same lexical readout?**

This is a representation experiment, not a claim that algorithms cannot be a bottleneck. It does not replace or retrain the accepted support-adapted Qwen model. It does not implement the unmerged polarity PR. It is not a repetition of neural intent/NLI inference.

## Grounding versus hypotheses

**Retrieved research:** Park et al. (2021), *Detecting Community Sensitive Norm Violations in Online Conversations*, motivates looking beyond generic toxicity and respecting policy context. Clarke et al. (2023), *Rule By Example*, studies exemplar-based contrastive rule representations. Ribeiro et al. (2020), *CheckList*, motivates explicit tests of linguistic behavior rather than relying on a single aggregate score. These sources motivate the experiment but do not establish that these hand-authored English cues will improve Jigsaw.

**Competition evidence:** Guanshuo Xu's first-place writeup reports fine-tuned instruction models and a multi-model ensemble. Thus a feature-only explanation for our remaining score gap has not been demonstrated. We hold the algorithm fixed in this round to measure feature contributions, while leaving later training and ensemble hypotheses open.

**Our testable hypothesis:** quote/code separation and rule-relevant speech acts distinguish topical similarity from behavior. Same-rule support examples may expose which combinations matter. Approximate cue counts are cheap enough to ablate before investing in a new neural representation.

## What is new versus previously explored

Historical generic structure, support cosine geometry and neural NLI/affirmative-intent families already exist in the repository. We preserve their results. This round specifically measures transparent **authored versus quoted versus code regions**, sentence-local negation/action proximity, explicit rule/action co-occurrence, and **inner-comment-group-cross-fitted support distances in that cue space**. It is not described as globally novel NLP research, and a negative result must not be buried under another renaming.

## Four feature families

| Family | Candidate columns | Meaning | Main limitations |
|---|---:|---|---|
| Behavior | 58 | Presence/log counts of 28 fixed cue groups plus text length | Lexical proxies, not semantic adjudication |
| Scope | 121 | Same cues in author, quotation and code regions; approximate negation window; region fractions | Incomplete quote syntax and nuanced negation; quotations are not automatically permitted |
| Rule alignment | 36 | Rule-mentioned behavior matched to author/quote cues; sentence co-occurrence | English literal rule matching is approximate; novel rule syntax may not match |
| Support contrast | 65 | Positive/negative max/mean/top-three cosine summaries and relative cue distances | Heuristic cue space; inner cross-fit references differ in size from inference references |

**Total: 280 new dense candidate columns.** Existing word/character TF-IDF control is capped at 12,000/20,000 columns; those are not advertised as new research features. Each family is screened using eligible training labels only, up to 24 nonconstant, nonduplicate columns per fold. A fixed standardized-mean-difference ranking is used; no validation-score-directed column selection occurs.

Both requests and offers can be prohibited by the legal-advice rule. Disclaimers, quotes, contact details, or legal words do not supply a violation label. No targets are synthesized.

## Protocol and the 882-versus-881 distinction

The returned notebook-05 summary counted 234 advertising and 648 legal-advice comments after excluding support examples within each policy. The existing `scripts.support_adaptation.build_study` checks supplied support text **across every original policy**, producing the historical expected cohort 234 + 647 = 881. This round calls that function unchanged and stops unless those counts match. The notebook shows both counts. It preserves the readiness report rather than silently rewriting its scoped statistic.

Training consists of the canonical original-data bodies and explicitly labeled supplied support pairs. Query bodies are removed from all training/support sources by the canonical normalization, conflicts are dropped, and duplicate rule/body pairs are collapsed. The canonical repeat weighting is identical for every variant. This is **new-rule supplied-support adaptation**, not zero-shot transfer: labeled support examples of the held-out rule are available.

`support_contrast` training features use three-way stratified group cross-fitting by normalized body. A held-out inner row cannot be its own labeled reference, including under another rule. Outer query features use the full eligible training reference; references must have both explicit classes for the same rule. Other rules are never automatically assigned negative labels.

Query targets are read only for metric evaluation after predictions exist. No organizer-released targets, Kaggle hidden test labels, previous protected holdout, target-derived identifiers, inferred timestamps, web-fetched comments or model weights are accessed.

## Fixed comparisons and attribution

One logistic regression specification (C=1, liblinear, fixed seed, canonical repeat weights) is used throughout. Ten configurations × two policies = **20 fits**:

- Lexical control.
- Control plus each of the four families separately.
- Full four-family addition — **predeclared primary candidate**.
- Full addition minus each family separately.

The screen is identical for each occurrence of a family within a fold. Report per-policy AUC, equal-policy macro AUC, pooled ROC AUC, label-free per-policy-ranked pooled ROC AUC, Brier score and log loss. Column-averaged competition AUC and our equal-policy macro are not asserted to be interchangeable; all local values use only this specific development cohort. Per-family additions and leave-family-out results are the attribution evidence; coefficients and selected-name overlap are diagnostics, not proof of usefulness.

Paired bootstrap: 500 draws over normalized-comment groups, shared across all outputs; a centered maximum-deviation band covers the 13 registered macro-AUC contrasts simultaneously. Intervals condition on these fitted predictions and two repeatedly inspected policies. They do not correct the entire project history or establish performance on a new rule distribution.

## Gate — unchanged until results are returned

The full primary candidate must gain at least **+0.003 macro AUC** over the identical lexical control, have a simultaneous 95% lower bound greater than zero, have no per-policy regression, and not decrease local per-policy-ranked pooled AUC. Passing yields `ELIGIBLE_FOR_NEXT_VALIDATION_ONLY`; otherwise `DO_NOT_PROMOTE_THIS_FEATURE_SET`. Neither result closes feature research or promotes a Kaggle model. Individual families require their own addition/removal evidence; we do not automatically retain all four because the combined candidate passes.

## Compute and reproducibility

No package installation, data download, model download, new neural inference, cloud API write, or submission. Two CPU threads are not required: numerical libraries are bounded to one thread for predictable resource use. A 240-second worker timer and notebook-facing 300-second process-tree cap protect execution. Candidate payloads precede completion markers; completed payload hashes and identities are verified on resume. A changed source/configuration/environment cannot silently reuse an old run. A corrupted completion stops rather than regenerating evidence.

Private checkpoints under `runs/behavioral_features/<run_id>/` include the query plan, labels in eligible training pairs, vocabulary/IDF transforms, scalers, selected feature names, model coefficients and query predictions. They are **not** put in the public return ZIP or Git. The shared joblib transform is serialized for later replay but is not automatically deserialized by this workflow. Public aggregates and the executed notebook contain no raw comment text or row-level labels.

The same-volume checkpoints survive a kernel failure or stopped app, not deletion of the SageMaker space. No off-volume backup is silently claimed.

## Subsequent rounds, selected only after evidence

| Round | Hypothesis | Resource gate |
|---|---|---|
| Current 1 | Transparent behavior/scope/support comparisons | 20 fixed CPU fits; inspect first |
| Next candidate 2 | Relation-aware matching distinguishes same-topic/opposite-behavior examples | Reuse legally available cached representations where compatible; no inference before provenance/cohort checks |
| Next candidate 3 | Rule-conditioned contrastive representation learning improves learned decision boundaries | Requires a bounded training contract, leakage-safe explicit pairs and an informative small-scale comparison |
| Later integration | Complementary retained families add value to accepted-model predictions | Predeclared matched additions/ablations, uncertainty and per-policy checks; not a model/ensemble sweep |

Unavailable conversation threads, user history and timestamps are not manufactured. Permitted external sources need provenance, historical availability, rules eligibility and overlap checks. Feature coverage remains open and evidence-driven; neither large feature counts nor a target score proves completion.

## Sources

1. Park et al. (2021): https://aclanthology.org/2021.findings-emnlp.288/
2. Clarke et al. (2023): https://aclanthology.org/2023.acl-long.22/
3. Ribeiro et al. (2020): https://aclanthology.org/2020.acl-main.442/
4. Guanshuo Xu (2025), first-place competition writeup: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/1st-place-solution
5. Official objective and evaluation: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/overview/abstract

Research reviewed September 11, 2026. Source findings, implementation hypotheses, synthetic test outcomes, and future real-data results are separate evidence categories.
