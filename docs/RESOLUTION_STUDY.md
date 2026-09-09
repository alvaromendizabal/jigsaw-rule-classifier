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

## Outcome

Run `143fb05be41909328ef4` reproduced the expanded centroid control exactly from
the saved vectors and verified every published metric against saved scores.
There were no fitted models, new encoder calls or confirmation-target accesses.

| Dimensions | Policy-macro AUC | Log loss | Brier |
| --- | ---: | ---: | ---: |
| 32 | 0.6848 | 0.6538 | 0.2290 |
| 64 | 0.6925 | 0.6498 | 0.2285 |
| 128 | 0.6899 | 0.6360 | 0.2233 |
| 256 | 0.6972 | 0.6295 | 0.2203 |
| 512 | 0.7001 | 0.6272 | 0.2192 |
| 1,024 | **0.7042** | **0.6237** | **0.2177** |

No shorter prefix improves the primary score or either probability loss.
The 32/64/128-dimensional losses exclude zero under simultaneous intervals;
the 256/512-dimensional intervals include zero. That does not establish
equivalence. Retain 1,024 dimensions as the research reference; any deployment
compression needs an explicit accuracy/memory acceptance tolerance. This
plausible feature-resolution avenue is now measured rather than left as a
generic future suggestion.
