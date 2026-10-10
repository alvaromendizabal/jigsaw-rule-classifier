# Project closeout · scored research and public demonstration

This release presents the rule-conditioned NLP system I built and its measured research record, alongside an interactive public demonstration and repeatable review commands. It does not change historical scores or launch new model research.

## Delivered work

| Layer | Evidence |
|---|---|
| Retained support-adapted Qwen3-4B system | **0.91808 public / 0.91425 private ROC AUC** in successful late evaluation |
| Controlled support-adaptation study | **0.614600 → 0.719893** policy-macro AUC on the matched 881-comment cohort |
| Transfer and multi-backbone research | Whole-policy comparisons, five-model development evidence and preserved negative findings |
| Reliable execution | Source/data/model identities, recoverable optimizer/inference state and validated artifacts |
| Public demonstration | [Policy Lens](https://alvaro-policy-lens.tartmacaw2.chatgpt.site): authored support matching, editable context and inspectable contributions |
| Review and verification | [Case study](../CASE_STUDY.md), [review guide](EMPLOYER_REVIEW_GUIDE.md), [reproduction commands](REPRODUCIBILITY.md) |

## Results retained without revision

The retained model improved private AUC by **0.29469** over the 0.61956 lexical baseline. That comparison changes both backbone and training method; the matched 4B study isolates adaptation more directly.

Strict-majority conflict handling scored **0.91720 public / 0.91288 private**, below the retained system, and was rejected. Later multi-model work reached **0.740351** policy-macro AUC on the fixed development cohort. Teacher-transfer and reciprocal-learning variants failed registered promotion gates. None of those development outcomes is an official score for a new system.

The repeatedly inspected two-policy cohort is not an untouched confirmation set. Historical notebook and frontier documents retain their dates and experimental context; their earlier next-step language is not a commitment to automatic additional research.

## Public implementation boundary

Policy Lens fits a TF–IDF support matcher locally on fictional examples. Its lexical margin is not a probability or the Qwen model's output. Public Python commands verify components and replay aggregate evidence; private weights, raw comments, row predictions, optimizer state and exact ensemble construction remain outside the release.

Completion refers to the delivered public engineering and review surface. The classifier is not presented as a deployed autonomous moderation service. [Model card](../MODEL_CARD.md) · [Method credits](TOP_SOLUTION_INTEGRATION.md)
