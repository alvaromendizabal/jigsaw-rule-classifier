# Round 13 — Localized passage evidence against labeled supports

## Actual previous evidence
Round 10 primary: macro AUC 0.716649 (context anchor 0.729257); Round 11 primary: 0.716294. Neither was promoted. These new hypotheses do not repeat the failed lexical/semantic alignment or multi-prototype mechanisms. This is not a claim that all possible versions of those ideas are exhausted.

## Hypothesis
Does a localized phrase provide evidence that is diluted when a whole comment is represented by one vector?

24 word-passage and 24 character-passage summaries, including first/last evidence and strongest-evidence location.

## Precise construction
A deterministic punctuation/newline segmentation generates at most twelve passages. Excess passages are merged contiguously, retaining all normalized content. Inputs over 50,000 characters stop rather than silently truncate. Two reference-only vectorizers use word 1–2 grams and character-within-word 3–5 grams, min_df=1, sublinear TF, at most 6,000 terms each. Queries never fit the vocabulary or IDF. Every query passage is compared to full reference comments with supplied document labels. No passage is assigned a moderation label.

For each class, average its five best reference similarities for each passage, then summarize maximum, mean, dispersion, top-two mean, first passage, last passage, ending-minus-starting shift, and strongest-passage position. Permitted, violating, and differences give 24 per lexical family, 48 total. Word and character families are separately ablated.

Controls: use a single full-comment passage; split the identical token sequence into the same number of equally sized contiguous chunks (boundary null); shuffle reference labels. These controls have the same output width, but are not claimed to be row-norm matched. Empty vocabulary or no recognized terms produces explicit zero similarity and a reported coverage flag, never target-based imputation.

## Same experiment contract for both new rounds
Six candidates: word_passages, character_passages, passage_all, whole_comment_all, boundary_null_all, label_null_all. Each retains exactly the context_evidence anchor and fixed classifier from the verified prior chain. Ten old readouts are byte/hash checked and design-parity verified; twelve new classifier fits are saved individually. The six new settings are fixed now, not chosen from either companion result. 16 reference-only TF-IDF vocabulary/IDF fits, shared across six candidates; no encoder fits.

## Leakage and information boundaries
Reference statistics are learned only from eligible same-rule references. New training features use three normalized-text-group folds: the held group is excluded before any graph or vocabulary statistic. Supplied held-out query labels are used later only for metrics. Query rows do not become references to each other. Reuse the verified 234 advertising + 647 legal-advice identities and exact vector row order. Distinguish new-feature cross-fitting from the inherited in-sample adapted Qwen support-training answer score. No new neural pass, downloads, cloud calls, external data or hidden labels.

## Evidence and gate
Primary passage_all. Require +0.003 mean policy AUC against each of qwen_raw, frozen_basic, context_evidence, uniform_all, whole_comment_all, boundary_null_all, label_null_all, simultaneous lower bounds above zero, no policy regressions, and nondecreasing within-policy-ranked pooled AUC vs raw Qwen. The paired normalized-comment bootstrap is conditional on fixed predictions; it does not correct the entire adaptive research history, dependence from overlapping supports, uncertainty over future policies, or fitting variability. Selecting this follow-up after Round 11 is adaptive. A pass is further-validation eligibility only; no automatic GPU work or submission. Classifier coefficient charts are associations, not causal feature importance.

## Restartability and privacy
Pin completed Round 10/11 results and prior source maps; reuse their accepted cache chain rather than refit them. Verify current input, source and dependency hashes. A corrupt completed stage stops, never silently recomputes. Save new feature banks and each model independently; return aggregate reports, exact source and executed notebook, not raw text, labels by row, fitted vocabulary or arrays. The eight Plotly figures are embedded in the notebook and standalone dashboard.

## Costs and bounded scope
Scientific hard timer 240 seconds; parent watchdog 260; notebook/replay watchdog 110; helper budget 540; outer command 600. Heartbeats every 15 seconds. These do not stop the SageMaker application. No package installs, AWS APIs, Git commits or pushes from this helper. Prior files and user edits are preserved.

## Research attribution
https://aclanthology.org/C16-1220/
Multi-level text representation research motivates examining local evidence; this TF-IDF passage method does not reproduce learned word/sentence/document embeddings or add pretrained models.
Rule-based exemplar motivation: https://aclanthology.org/2023.acl-long.22/ . No cited dataset is imported.
