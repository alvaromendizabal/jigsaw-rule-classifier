# Manual feature campaign: publication checkpoint

## Status

All completed original-training/supplied-support manual milestones through Round 13 are preserved. The accepted recorded model remains Qwen3-4B, 0.91808 public / 0.91425 private Kaggle AUC. No new leaderboard result is reported by this publication. Feature research remains open.

The same 881 development comments (234 advertising, 647 legal advice) have been inspected repeatedly. The saved intervals are conditional within-round intervals, not global adaptive-search correction. Cross-fitting new support statistics does not make the inherited adapted training-answer margin out of fold.

## Latest results

Round 12 consistency primary: 0.726598 policy-macro AUC versus 0.729257 context anchor (delta -0.002659). Round 13 passage primary: 0.724384 (delta -0.004873). Both recorded DO_NOT_PROMOTE_PRIMARY. Word-only passage features reached 0.730255 descriptively; they do not replace the registered primary. This is not a promoted model.

## Notebook reading order

- [05_data_readiness.ipynb](../notebooks/05_data_readiness.ipynb)
- [06_behavioral_feature_investigation.ipynb](../notebooks/06_behavioral_feature_investigation.ipynb)
- [07_relational_feature_investigation.ipynb](../notebooks/07_relational_feature_investigation.ipynb)
- [08_policy_feature_investigation.ipynb](../notebooks/08_policy_feature_investigation.ipynb)
- [09_lexical_evidence_investigation.ipynb](../notebooks/09_lexical_evidence_investigation.ipynb)
- [10_scope_lexical_investigation.ipynb](../notebooks/10_scope_lexical_investigation.ipynb)
- [11_feature_value_audit.ipynb](../notebooks/11_feature_value_audit.ipynb)
- [12_local_support_feature_investigation.ipynb](../notebooks/12_local_support_feature_investigation.ipynb)
- [13_matched_support_feature_investigation.ipynb](../notebooks/13_matched_support_feature_investigation.ipynb)
- [14_conditioned_geometry_investigation.ipynb](../notebooks/14_conditioned_geometry_investigation.ipynb)
- [15_behavior_support_investigation.ipynb](../notebooks/15_behavior_support_investigation.ipynb)
- [16_crossmodal_support_investigation.ipynb](../notebooks/16_crossmodal_support_investigation.ipynb)
- [17_multiprototype_investigation.ipynb](../notebooks/17_multiprototype_investigation.ipynb)
- [18_reference_consistency_investigation.ipynb](../notebooks/18_reference_consistency_investigation.ipynb)
- [19_passage_support_investigation.ipynb](../notebooks/19_passage_support_investigation.ipynb)

## Reproduction without private artifacts

The tracked notebooks contain the saved executed outputs; GitHub can display the saved tables and Plotly JSON. Standalone HTML dashboards are generated only on demand and are deliberately not tracked because they duplicate notebook outputs and bundle megabytes of browser JavaScript. Do not Run All on a fresh clone without private input/checkpoint recovery. CI verifies notebook integrity and runs the software tests with authored synthetic inputs; it does not reconstruct private experiments.

## What stays private

Original Kaggle CSVs, model arrays, token vocabularies, row-level labels/predictions, private checkpoints, environment directories, credentials, and full logs remain in the existing SageMaker storage/private S3. They are never added with a broad git add command. This commit is not a backup of the private workspace.

## Publication contract

The allowlist and byte checks are in reports/manual_feature_campaign/publication_manifest.json. The manifest intentionally excludes generated standalone HTML dashboards. The existing Quality workflow must pass on the exact PR head before merge. The merge helper then fetches and fast-forwards the original checkout, verifies its commit/tree against GitHub, and rechecks the raw-data hashes. It does not reset, clean, stash, delete branches, or overwrite experiments. The unfinished historical PR #29 is separate and is not merged by this workflow.

Future manual helpers must accept a clean descendant of the reviewed base and verify source/checkpoint content, rather than require the obsolete base HEAD. Previously delivered helpers are historical entry points, not instructions to rerun old studies after publication.
