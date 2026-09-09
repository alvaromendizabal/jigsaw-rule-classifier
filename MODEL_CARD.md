# Model card

**Model:** accepted rule-conditioned route `a971cf3bc6add1c2d818`, packaged as offline artifact `4ce868936ca14fdad7df`. **Owner:** Alvaro Mendizabal. **Release scope:** research and local human-review support, September 2026.

Given a comment, supplied rule, community and two violating/two permitted examples, estimate rule-violation probability. A normalized policy seen in development uses the fitted seven-family classifier with development-validated calibration. An unseen policy uses the original frozen semantic centroid; calibration for unseen policies was rejected on development evidence. Membership depends on the supplied policy text, not targets or test-cohort statistics.

The encoder is `Qwen/Qwen3-Embedding-0.6B`, pinned to revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. Encoding uses the frozen instruction/pooling contract, 256-token maximum, CPU float32 and four threads. The encoder was not fine-tuned. The familiar pipeline retains 9,263 of 181,958 candidate columns. [Feature research](docs/FEATURE_RESEARCH.md) · [Model and calibration protocol](docs/MODEL_VALIDATION.md).

## Evaluation

Development contains 11,135 rows across four policies. One protected comparison used 43,509 eligible rows across six policies, including two policies excluded wholly from development. Predictions were committed before target access. All 12 predeclared acceptance checks passed; no post-confirmation selection or refitting occurred.

| Policy group | Rows | Lexical AUC | Accepted AUC |
| --- | ---: | ---: | ---: |
| All six | 43,509 | 0.6801 | 0.7770 |
| Four familiar | 25,485 | 0.7581 | 0.8276 |
| Two unseen | 18,024 | 0.5241 | 0.6757 |

The primary metric averages ROC AUC equally across policies. The gain is 0.0969, with paired 2,000-draw 95% interval [0.0898, 0.1051], conditional on the six observed policies and frozen predictions. Overall log loss is 0.5121 and Brier score 0.1739. Every policy improves over the reference. This is an organizer-released post-competition benchmark; no Kaggle score, medal, executable-scorer parity or state-of-the-art claim is made. [Protocol and full results](docs/CONFIRMATION.md).

## Appropriate use and limitations

- Use the score to support a person's review of English comments with the specified rule and complete support examples. No automatic deletion, account penalty or deployment threshold was validated.
- Performance on two unseen policies does not establish broad policy, language, demographic or temporal generalization. Legal/medical intent and missing conversational context remain difficult; a moderation score is not professional advice.
- Four support examples are mandatory. Contradictory or unrepresentative examples can change the prediction; missing text is rejected. Long inputs are truncated under the pinned encoder contract.
- The unseen score is uncalibrated. A number such as 0.8 is not established to mean 80% correctness for a new policy. Closest-example similarity is descriptive, not a causal explanation.
- Author/thread/timestamp context is absent. Pretraining contamination cannot be fully audited. No demographic fairness, adversarial-security or production-load claim is made.

## Artifact and operation

Candidate SHA-256: `a38b20e1f34ff6d508bc70ba360ceb5f1a646a4f37cac4ebdcb56feebb20d698`.

Bundle manifest SHA-256: `dbf1428343c2e349590c4ceec32a840891e796d67005948d8cf3c22e535da18a`.

The private bundle preserves the exact accepted candidate, encoder, source and dependency lock. It passed offline parity, batch/order invariance, corrupted-artifact rejection, missing-support validation, restart reuse and a fresh-environment restoration. Measured peak memory was 3.88 GiB; warm mean latency was 1.11 seconds per authored comment on the tested CPU. These are small-run measurements. [Download, exact budgets and reproducibility](docs/DELIVERY.md).

Project code is MIT; upstream model/data terms remain separate. Weights are preserved in the owner's private S3 bucket. The GitHub submission notebook is a separately identified original-training-only reference and does not contain this post-competition fitted model.
