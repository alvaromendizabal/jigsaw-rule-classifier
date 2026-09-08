# Feature research record and completion gate

**Status: open.** The implementation is substantially more mature than the original four-candidate study. The evidence does not justify final model optimization, final retraining, or a claim that arbitrary community rules are solved. Notebook `02` is the canonical research narrative.

## What was actually inspected

The starting main commit was `a43d6672c5006fc2a669d5b1a039c373b0d9a1b6` (PR #7). Its README described a completed four-candidate CPU study, but the public feature aggregates and refreshed notebook outputs were absent. A broader feature-research prototype was preserved in S3 but was not an executed benchmark. The original quality gate passed 139 tests before changes.

The restored snapshot contained 375 paths referencing 357 unique content-addressed objects; every object's SHA-256 was checked. Actual data contain 2,029 training rows, two labeled rules, 100 communities, no missing text fields, and 162 duplicate training bodies. All ten preview-test comments overlap training comments. Eighteen training comments equal one of their own supplied support examples.

## Candidate space and availability

Every input below is available with the comment at inference, or is a transform fitted exclusively on retained training-fold observations. `row_id` is an identifier, never a predictor. Public reports exclude raw comment text, token vocabularies and row-level outcomes.

| Family | Candidate columns per fold | Rationale and provenance | Leakage boundary and outcome |
| --- | ---: | --- | --- |
| Word TF-IDF | 16,558–28,523 | Training-corpus word unigrams/bigrams capture topic and phrasing | Training-only vocabulary/IDF; 489–2,988 survive the broad screen; unscreened controls retained |
| Character TF-IDF | 39,426–60,000 | Character 3–5 grams capture morphology, punctuation and spelling variation | Training-only vocabulary/IDF; 3,710–4,096 retained; useful relative to screened words |
| Structure and interactions | 1,512 | 21 observable text measures; rule/support means and spreads; differences; signed square/root; body × support/rule products | No labels in construction; 128 retained; broad expansion hurts transfer |
| Lexical support/rule geometry | 207 | Positive/negative order-invariant cosine summaries, margins, spreads and nonlinear interactions | Only training-fitted vocabularies; 64 retained; no established incremental gain |
| Support-conditioned tokens | 16,558–28,523 | Comment TF-IDF multiplied by positive-minus-negative support TF-IDF | Uses supplied examples, never row targets; 85–240 retained; no robust gain |
| Qwen coordinates/interactions | 6,144 | Frozen 1,024-dimensional comment, support direction, products, relative distances and support dispersion | Model/hash/prompt contract fixed; training-only selection of 256 axes; poor transfer |
| Compact semantic geometry | 32 | Support similarities, normalized centroids, coherence and fixed kernel temperatures | Frozen per-row inputs; 31 retained; strongest broad addition to words |
| Relative ranks | 207 | Empirical percentile position of lexical geometry against training reference distributions | CDF fitted only on training rows; 64 retained; no robust gain |
| Community indicators | 63–93 | Community-level differences in moderation context | Training-only categories; unseen communities map to zeros; 23–59 retained; weak transfer |
| Context, count and target encoding | 9 | Community/rule/joint-group posterior, relative frequency and novelty | Three inner comment-group folds with example purging and inner-only prior; 4–8 retained; no reliable gain |
| Joint-text NLI | 99 | Frozen DeBERTa contradiction/entailment/neutral probabilities for rule and support pairs, contrasts and signed transforms | No competition-label encoder fitting; outer-training screen keeps 32 in the combined bank; no gain |
| Low-rank controls | 128 per representation | Training-only SVD of word, character, joint lexical, Qwen body and Qwen support-product matrices | Outer-training projection and row L2 normalization; fixed classifier; tests representation compression rather than axis selection |

The broad screen covers **88,924–104,665 candidate columns per fold**, retaining **4,860–7,916** across its ten family banks. Including the separate 99-column NLI bank gives the following auditable counts. These are bank columns, not the size of one chosen final model, and they are not summed across repeated folds.

| Protocol | Fold | Candidates | Retained | Rejected |
| --- | ---: | ---: | ---: | ---: |
| Familiar rules | 0 | 89,023 | 4,892 | 84,131 |
| Familiar rules | 1 | 94,615 | 5,513 | 89,102 |
| Familiar rules | 2 | 96,235 | 5,396 | 90,839 |
| Held-out rule | 0 | 104,764 | 7,948 | 96,816 |
| Held-out rule | 1 | 103,501 | 6,453 | 97,048 |

The five low-rank controls each derive up to 128 components from already-counted input matrices. They are separate representation alternatives, not extra raw variables quietly added to those screened-bank totals. The final selected feature count is **undecided**.

### Families that the available data cannot support

There are no timestamps, observation histories, conversation threads, user identifiers, organizational records, seasons, opponents, coaches or external ratings in the supplied schema. Lagged/rolling/recency/trend/volatility, strength-of-schedule and historical entity aggregates would therefore invent information or reinterpret row order as time. They were ruled inapplicable, not silently omitted. Empty-text indicators exist among structural candidates, but the strict input contract rejects missing required text; no useful missingness signal exists in this data.

External moderation context remains plausible only with legitimate source access, provenance, availability-at-inference and overlap checks. Current subreddit pages are not historical versions of the provided rules. No external comment corpus, hidden labels, leaderboard probing or web-fetched current metadata entered these experiments.

## Experimental design

The broad study (`9ec008d506b0bc64a717`) executes 21 configurations on the same five preserved folds: **105 fits**. Nine family additions use the same screened-word control; family-only controls, full combinations and six leave-one-family-out comparisons are retained. Logistic regression stays at C=2, liblinear, 2,000 maximum iterations, seed 2025. There is no model-hyperparameter search. Dense and sparse scaling conventions are part of each tested representation; the historical reference is not presented as an otherwise-identical scaling control.

The screen validates schemas, finite values and row counts; rejects constants/near-constants, incidence below three training rows, exact duplicate columns and suspicious perfect separators; ranks by training-only effect score; and applies fixed family budgets. Dense redundancy screening uses absolute correlation 0.995 in a bounded top-score pool. Sparse duplicate detection uses exact column hashes rather than quadratic all-pairs correlation. Each decision and selected-name list is preserved privately with hashes. This is a tractable search, not a claim to compare every possible feature subset.

The sensitivity study additionally tests full lexical vocabularies, training-only NB token reweighting, five training-only low-rank representations, lexical support scoring and seven frozen Qwen geometry scores. The NLI study (`537cf2213c813b8ebd4b`) tests rule-only, support-only, combined, word-plus-NLI and broad-plus-NLI representations: **25 fits**, plus a label-free rule margin. It encodes **11,835 unique pairs** for 12,174 requested occurrences; 26 pairs reach the truncation limit. It uses revision `fa2804872c3b4bd748f38c0185cc85775361e735` of `cross-encoder/nli-deberta-v3-small`, with verified assets and 256-token input limits.

The near-copy study (`a09455ec14e9b3d6ab61`) adds **10 fits** under a fixed text-only isolation policy: hashed character-ngram cosine ≥0.95, token Jaccard ≥0.90, minimum 40 characters. It detects no additional cross-boundary copies after exact purging. This narrow stress test does not establish semantic-paraphrase or shared-origin isolation. A separate sensitivity score excludes the 18 self-support rows without refitting.

### Uncertainty and interpretation

Paired comparisons use 1,000 normalized-comment-group bootstrap draws with common weights across compared predictions. Both pointwise and centered maximum-deviation simultaneous 95% intervals are saved. The simultaneous family covers the contrasts in that study; it does not retroactively correct selection across all historical and adaptive studies. These intervals are conditional on fixed OOF predictions and the two observed rules. They do not include all training variability or estimate the distribution of future policies.

Whole-family permutation is repeated three times within validation rules. Mean absolute logit contributions summarize fitted linear models. Retained-name Jaccard shows instability across folds: semantic scalars have identical selected names, while many raw coordinates/tokens do not. Identical names alone do not imply predictive utility. Group removal tests and matched additions remain the primary attribution evidence. SHAP is not necessary to duplicate these transparent linear-model diagnostics; it is not claimed as executed.

## Measured feature contributions

| Representation | Held-out rule macro AUC | Log loss | Brier |
| --- | ---: | ---: | ---: |
| Historical comment-only reference | 0.604086 | 0.6731 | 0.2400 |
| Historical rule/example reference | 0.615563 | 0.673577 | 0.240487 |
| Screened words | 0.589348 | See saved report | See saved report |
| Screened words + compact semantic geometry | 0.621738 | 0.724841 | 0.260377 |
| All transferable broad families | 0.554313 | 1.293287 | 0.339895 |
| Full character vocabulary | 0.623532 | 0.664553 | 0.236146 |
| Frozen normalized Qwen centroid margin | 0.641594 | 0.669302 | 0.238519 |
| NLI rule features | 0.547382 | 0.725690 | 0.263259 |
| Screened words + NLI | 0.596379 | 0.696421 | 0.247532 |

The full machine-readable comparisons, including low-rank controls and per-rule metrics, are in [research](../reports/research/results.json), [sensitivity](../reports/sensitivity/results.json), and [NLI](../reports/pairs/results.json). Notebook `02` renders them directly rather than maintaining an independent score table.

Against the **same screened-word representation**, compact semantic geometry contributes +0.032390 AUC, with pointwise interval [0.013066, 0.051477] but simultaneous interval [−0.015586, 0.080367]. Characters contribute +0.023176 with a positive pointwise interval but a simultaneous interval crossing zero. Structure and raw-coordinate additions are negative even under the broad study's simultaneous intervals. Nested target encodings add only about 0.00155; that is not a stable gain.

The centroid's observed improvement over the original rule/example reference is **+0.026031 AUC**. Its pointwise interval is [−0.005592, 0.057836]; simultaneous intervals are in the updated sensitivity report. Its lower log loss and Brier are descriptive, not a fitted-calibration claim. Excluding self-support rows leaves centroid AUC 0.636120 versus reference 0.609503. There is **no independently confirmed positive feature-engineering gain** yet, and no numerical uplift should be presented as a proven improvement.

## Completion assessment

An approximate **75% overall completion** is a planning judgment, not a statistical quantity or an employer rating. The estimate weights data/validation 15/20, feature research 28/35, model confirmation/final evaluation 10/20, engineering 14/15 and presentation 8/10. It is lower than simply calling working notebooks “80% complete” because confirmation, final representation choice and actual offline promotion carry substantial weight. Notebook `02` is about **85% complete as an implementation and narrative**, but its scientific completion gate remains open.

The major implemented families now have measured additions/removals, leakage controls, retained counts, negative results and reproducibility artifacts. Diminishing returns are visible for broad handcrafted expansion, raw-coordinate selection, metadata and this frozen NLI model. **Global exhaustion is not established.** High-value unresolved avenues are:

1. Independent rule types or a separately locked confirmation design after representation selection. Reusing two already-inspected rules cannot provide this evidence.
2. Legitimate additional conversational context or independently labeled rule data, with time/provenance and overlap isolation. These fields are absent from the current competition files.
3. A task-aligned frozen instruction representation, with fixed prompts and a fresh confirmation plan. Failure of this general NLI encoder does not rule out all joint encoders. Large prompt/model searches against the same two rules would increase selection bias.
4. Nested calibration and a locked decision rule after a representation earns promotion. These are downstream work, not a substitute for the feature gate.

The pipeline therefore exposes `jigsaw gate` and `require_feature_completion`; the current state fails final-feature acceptance. Public notebooks consume the latest verified feature reports. The offline Kaggle path still names and fits the original lexical reference. There is no silently outdated “best model”: no feature candidate has been promoted. A 9.9/10 employer-facing claim is not defensible without stronger generalization evidence, credible external/official evaluation, final-model lineage, calibration and deployment-budget verification.

## Research sources and external-data feasibility

- [Park et al., 2021: community-sensitive norm violations](https://aclanthology.org/2021.findings-emnlp.288/) motivates explicit policy/community context rather than toxicity alone. [The authors' NormVio repository](https://github.com/chan0park/NormVio) releases redacted comment IDs and requires reconstruction with Reddit credentials; it is not a ready independent text benchmark in the connected environment. No reconstruction or redistribution was performed.
- [He, May and Lerman, 2024: CPL-NoViD](https://arxiv.org/html/2305.09846v3) supports policy-conditioned prompts and cross-rule/community testing. Its fine-tuned contextual approach is not equivalent to our frozen NLI probe, and its reported performance is not transferred to this project.
- [Wang and Manning, 2012](https://aclanthology.org/P12-2018/) motivates NB log-count reweighting as a lexical feature control. Our classifier and validation are documented separately.
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) motivates semantic similarity representations; [Qwen's model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and [the NLI model card](https://huggingface.co/cross-encoder/nli-deberta-v3-small) identify the actual frozen assets, both with Apache-2.0 model-card licensing.
- [Official Kaggle overview](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/overview) and [rules](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/rules) remain authoritative for scoring and eligibility. Their executable scorer and authenticated external-data permissions were not available in the retrieved text. The previously cited arXiv `2511.17592` is an unrelated GigaEvo paper and was removed; it cannot support metric equivalence. A fifth-place writeup link was inspected but its body was not retrievable, so no unverified method or score from it is claimed.
