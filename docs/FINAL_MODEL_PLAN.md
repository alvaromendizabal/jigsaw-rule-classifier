# Model and local inference milestones completed

Feature research is complete for the declared four-policy development scope. The generated [stopping decision](../reports/feature_decision/decision.json) and [coverage ledger](FEATURE_COVERAGE.md) support that boundary. The development implementation and nested calibration are now [executed and verified](MODEL_VALIDATION.md). The protected comparison is also complete: all 12 checks passed, with macro AUC 0.7770 versus 0.6801. [Frozen protocol and result](CONFIRMATION.md). This document retains the historical milestone rationale and the completed product requirements.

## Resolve familiar versus unseen policies

The fixed `all_transfer` feature model reaches **0.7989 familiar-policy AUC**, compared with **0.7311** for its same-fold lexical reference, improving all four observed policies. Its unseen-policy AUC is only **0.5515**. The selected frozen centroid reaches **0.7042** on unseen-policy validation, compared with **0.4728** for the lexical reference. These are separate validation protocols. The executed route now reproduces each parent under its appropriate protocol; no artificial blend of the two cohorts is reported.

Implement one candidate that routes on normalized policy membership in the fitted training data: the validated `all_transfer` model for familiar policies and the original frozen centroid for unseen policies. Membership is available at inference and must never be inferred from validation targets or test-cohort frequencies. This follows the observed failure mechanism without another encoder or hyperparameter search. Preserve both parent controls and the lexical reference.

The completed model milestone required:

1. Replay the fixed route on existing OOF records, proving fold-specific membership, prediction alignment, query-target exclusion and stable behavior for unknown policies.
2. Fit candidate feature banks and model artifacts only from development data, using the selected seven families. This selected model contains no target-derived encodings; cross-fitting them is unnecessary here. Verify that unseen-policy rows bypass those banks completely. Record source, schema, screen, encoder, policy-set and calibration identities in the inference manifest.
3. Develop a low-capacity monotone calibration rule on development predictions only. Any reported development calibration improvement needs another validation level; fitting a calibrator and scoring the same OOF labels is not unbiased evaluation. Avoid policy-specific calibration for unseen policies. Freeze the exact calibration method and whether it is retained before confirmation.
4. Commit a machine-readable candidate/reference, protected row identities, metric/uncertainty plan, per-policy tolerances and acceptance rules before reading any of the **43,576 reserved targets**. Evaluate familiar and unseen policies separately and report both frozen parents. Financial-advice and spoiler policies remain wholly outside development.
5. Run that comparison once. Preserve a rejection and its limitations; do not use the reserve to select new features, routes, thresholds or models. Check normalized body/support overlap against fitted inputs before scoring. If reserve eligibility changes because of leakage, record exclusions before target access.

The route, nested calibration and protected comparison are now verified. Offline parity, serving measurements and a fresh-environment restore also pass; [the delivery guide](DELIVERY.md) records the package, budgets and download locations.

## Deliver the accepted representation

After acceptance, package the exact selected artifact for offline benchmark inference. Keep the original-training-only `kaggle/submission.ipynb` workflow separately identified: the accepted artifact used additional post-competition development labels. Verify parity between research and packaged predictions, batch/order independence, corrupted-artifact rejection, missing-support behavior and resumable batches. Define latency and peak-memory budgets before benchmarking them on named hardware. Package the pinned encoder assets and dependencies for an internet-disabled run. A small synthetic end-to-end fixture is distinct from a real held-out evaluation.

Expose an example-driven interface with a comment, supplied rule and positive/negative examples. Explain the score and the nearest supplied examples, show the model's limitations, and provide a human-review use case. Similarity is evidence about the representation, not a causal explanation or a claim of legal/medical correctness. Missing required support should produce a clear validation message unless a fallback is separately tested.

## Finish the employer release

Keep `03` as the short evidence tour and `02` as the research detail. Add a model card, data card, measured serving results and a clean-environment restoration record. The README should answer the problem, result, why it works, where it fails and how to reproduce it within a few minutes. Tag a release only after current tests, actual notebooks, offline inference and artifact restoration pass.

A public demo, a successful notebook, and a Kaggle score are different deliverables. Any Kaggle submission remains the user's action. A numerical employer rating cannot be guaranteed; a finished release can have explicit, reviewable acceptance evidence.
