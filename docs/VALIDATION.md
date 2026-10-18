# Validation and evidence boundaries

The maintained neural comparison is **same backbone, matched support inputs, frozen versus adapted direct answer score**. It contains 881 novel comments across two previously examined policies. Policy-macro AUC weights each policy equally; pooled AUC is separately recorded. [Original protocol](../reports/support_adaptation/protocol.json) · [Original results](../reports/support_adaptation/results.json)

The retained external score is **0.91808 public / 0.91425 private** from historical post-competition evaluation. The end-to-end comparison against the lexical control changes the backbone and method; its larger gain must not be attributed solely to LoRA. A ten-row preview is not an external evaluation.

Query targets never enter adaptation. Supplied examples may carry eligible labels, but overlapping query text is purged and contradictory pairs are rejected. Tests flip query targets and assert that adaptation inputs stay unchanged. [Tests](../tests/test_support_adaptation.py)

The original study was repeatedly inspected and is not a fresh holdout. Within-policy ranks are not calibrated probabilities. The synthetic browser challenge is visible and authored, so it does not establish unbiased generalization or moderation safety. Its eight misses remain published. [Public challenges](PUBLIC_CHALLENGES.md)
