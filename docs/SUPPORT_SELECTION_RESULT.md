# Cached-vector support selection: completed diagnostic

## Decision

**Stop the current raw adapted-vector cosine selector before additional GPU inference.** Keep the accepted support-adapted Qwen3-4B and its 0.91808 public / 0.91425 private Kaggle AUC unchanged. This is a compute-allocation decision based on diagnostic evidence, not proof of AUC inferiority. It does not promote the previously rejected lexical-prompt candidate or close feature research.

## Measured audit and subsequent review

The target-free CPU audit covers all 881 novel development comments. It changes 231/234 advertising pairs and 647/647 legal-advice pairs. The most frequently selected permitted legal example is reused 121 times under semantic selection versus 21 under lexical selection. More pair changes or higher cosine similarity do not establish better adjudicative examples.

| Policy | Lexical preferred | Semantic preferred | Tie | Unclear |
| --- | ---: | ---: | ---: | ---: |
| Advertising | 11 | 6 | 1 | 6 |
| Legal advice | 13 | 7 | 0 | 4 |
| Total | 24 | 13 | 1 | 10 |

One AI reviewer completed the fixed 48-case diagnostic at the owner's request. Method names were withheld during case judgments, but earlier aggregate findings were visible. This is not independent human evidence or a double-blind experiment. Preferences are not violation labels and must not be used to train a per-query selector. No new AUC was calculated.

## Provenance and preservation

The report loader checks every public aggregate against `reports/support_selection/metadata.json`. The original audit's awaiting-review status is deliberately preserved; the later `review_decision.json` records the completed decision. `executed_source_manifest.json` preserves the exact portable CloudShell source identity. Reformatting source for repository integration does not imply rerunning the audit or that its historical receipt used the new publication tree.

The 53-test CloudShell result and matrix-free completed replay are historical software evidence. New repository tests, actual Jupyter rendering and its replay must pass in the owner-run publication step. CI and merge status are determined by the exact GitHub head, not this document.

## Storage boundary

Public Git contains only code, configuration, aggregates and executed portfolio notebooks. Private selected examples and method-key checkpoints remain under `~/jigsaw-audit/results/adf2ff53497b509795385392` in Oregon CloudShell. They have not been migrated by the SageMaker restore or this publication. Do not delete that directory. The small S3 workspace bundle is an evidence/source backup, not a complete private-checkpoint backup. Original plans, vectors and adapters retain their previously verified S3 locations.

## Next scientific hypothesis — not executed

Test whether decision-token vector geometry favors a small set of frequently reused supports rather than rule-relevant behavior. This is unproven. Any changed representation needs training-only transforms, a frozen hypothesis and configuration, leakage/index tests, matched ablations and stability checks. The same two repeatedly inspected policies remain exploratory development; do not reopen organizer-released targets or consumed holdouts. No new GPU experiment is authorized by this report.
