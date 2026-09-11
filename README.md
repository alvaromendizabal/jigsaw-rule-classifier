# Jigsaw · Rule-conditioned comment classification

**Current decision — September 10, 2026:** the cached-vector support-selection audit and single-AI relevance review are complete. Stop this raw adapted-vector cosine selector before another GPU comparison; keep support-adapted Qwen3-4B unchanged (**0.91808 public / 0.91425 private Kaggle AUC**). Feature research remains open. [Audit and review](docs/SUPPORT_SELECTION_RESULT.md). No new AUC was measured.


Predict whether a comment violates a supplied community rule, using the rule and examples of permitted and prohibited comments.

**Start with [03 · Results and examples](notebooks/03_saved_results.ipynb), then [02 · Feature research](notebooks/02_baseline_and_review.ipynb).** The competition rebuild improves private Kaggle AUC from **0.61956 to 0.91425**. The **0.92–0.93 objective remains open**. Historical post-competition research is preserved separately. [Measured rebuild and remaining experiment](docs/COMPETITION_REBUILD.md).
