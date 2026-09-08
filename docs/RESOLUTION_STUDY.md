# Frozen embedding resolution sensitivity

The [Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) states that
the pinned embedding model supports Matryoshka output dimensions from 32 to 1,024.
The previous SVD experiments learned a projection; selecting raw coordinates
also differs from using the model's trained prefixes. Neither substitutes for
this inexpensive representation check.

Before inspecting expanded-study scores, specify centroid comparisons at 32,
64, 128, 256, 512 and 1,024 dimensions. For each smaller prefix, L2-normalize the
comment and support vectors again, then use the existing fixed positive-minus-
negative centroid score. The full-dimensional control uses the exact existing
vectors, preserving its numerical predictions. There is no new encoder call,
trained model, calibration, vocabulary or target-derived feature fit.

Evaluate on the same 11,135 permitted development rows. Save row-aligned scores,
per-policy and macro AUC, probability losses and paired normalized-body cluster
bootstrap contrasts against 1,024 dimensions, with simultaneous intervals over
the five comparisons. These are exploratory feature-resolution controls, not
five independent confirmation tests. A retained deployment dimension would still
need to be frozen before reserved evaluation.

This study tests the scoring geometry, not a claim that a shorter output makes
the transformer encoder itself faster. Shorter stored vectors can reduce
downstream reference-bank memory and similarity-computation cost. Actual product
latency remains a later measurement.
