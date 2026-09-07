# Semantic benchmark review

The lexical reference reaches only 0.61556 held-out-rule macro AUC. The next experiment tests whether frozen semantic representations can improve rule transfer while preserving the exact original validation memberships and training-row purges.

This release pins Qwen3-Embedding-0.6B and every model asset, adds resumable embedding shards, compares an untuned similarity margin with a fold-fitted similarity classifier, and reports paired grouped uncertainty, probability quality, runtime, memory, and truncation. The sixth notebook compares saved results without loading weights, so the existing small Studio app remains useful for review.

Quality covers 57 tests, six explicitly synthetic notebooks, and a small real-model integration test. Cache interruption and corruption, padding, order invariance, source identity, fold isolation, and review without training are explicit checks. UTC events, 15-second heartbeats, complete/total counts, and stage durations remain visible. Source files keep canonical names; version history lives in Git.

The real benchmark and public aggregate evidence are separate from synthetic software tests. Only two labeled rules are available, so grouped bootstrap intervals are conditional on those rules and fixed OOF predictions. The preview test overlaps training and cannot establish generalization. This phase does not fine-tune Qwen, launch a GPU job, or claim a Kaggle score. The offline semantic Kaggle package and rule-conditioned cross-encoder follow in later phases.

The completed real run is `4e7e6c00d269c451c0a3`, produced by source commit `6b6e9b1d472958872670d97cef4910161025ecd7`. Frozen-margin held-out AUC is 0.63507; the lexical reference is 0.61556, and the paired delta interval includes zero. The similarity classifier scores 0.58577 and worsens probability quality, so it is not promoted. The full run took 1,004.7 seconds with 3.854 GiB peak memory. Notebook 04 contains executed real outputs and public aggregate evidence is hash-verified by the chart renderer.
