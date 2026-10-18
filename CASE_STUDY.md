# Case study · Adapt to the rule, not just the comment

A moderation label depends on the policy being applied. A fixed global classifier can confuse policy transfer with repeated exposure to the same text. I built a rule-conditioned classifier, a support-adaptation procedure, a leakage-aware comparison and recoverable offline inference.

## Selected approach and ownership

The retained system uses the credited Qwen3-4B-Instruct-2507 backbone with one LoRA adapter. My work covers eligible-label construction, support/query purging, decision prompts, last-token scoring, checkpoint recovery, exact-ID output contracts and the public diagnostic interface. The backbone and its tokenizer are upstream assets, not my invented architecture. [Credits](docs/TOP_SOLUTION_INTEGRATION.md)

The [actual trainer](scripts/decision_training.py) keeps optimizer, scheduler, loss-scaler and random states with completed checkpoints. [Native miniature-model tests](tests/test_decision_training.py) check exact recovery after interruption and retries after overflow. The [inference component](scripts/kaggle_adaptation.py) verifies model inputs, restores original row order and rejects corrupt cached predictions.

## Evidence and the final decision

The historical submitted system recorded **0.91808 public / 0.91425 private AUC**. A controlled same-backbone study measured **0.61460 frozen versus 0.71989 adapted policy-macro AUC** on 881 novel comments across two previously examined rules. That **+0.10529** effect is different from the end-to-end comparison with a lexical control; pooling them would conflate populations and model changes. [Original submitted record](reports/checkpoints/kaggle_adaptation.json) · [Matched protocol](reports/support_adaptation/protocol.json)

The selected direct decision path is preserved. Alternate feature readouts, unused backbones and later unpromoted experiments are not part of the maintained release. Original compact comparison records remain unchanged where needed to substantiate selection.

## What can be reproduced publicly

A clone can train the lexical control end to end on authored data, verify the original aggregate arithmetic, exercise the actual native Qwen/LoRA interfaces on tiny generated models, regenerate the offline source notebook and replay browser decisions. The browser's 24 visible authored challenges retain eight misses. Those diagnostics reveal failure modes; they do not establish production moderation accuracy.

Private datasets, adapters, model weights and prediction rows are excluded. No live service or full-score reproduction is claimed. [Commands and boundaries](docs/REPRODUCIBILITY.md)
