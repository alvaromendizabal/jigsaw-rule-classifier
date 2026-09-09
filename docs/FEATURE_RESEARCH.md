# Feature research record and completion gate

**Historical two-policy experiments.** The current four-policy results and decisions are in [EXPANDED_STUDY.md](EXPANDED_STUDY.md), [RETRIEVAL_STUDY.md](RETRIEVAL_STUDY.md) and [RESOLUTION_STUDY.md](RESOLUTION_STUDY.md). Counts and scores below retain their original cohort.

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
| Fixed instruction likelihoods | 36 | Three immutable Qwen3 prompts: rule, support, both; Yes/No probability, margin, answer mass and context differences | No generation, prompt search or competition-label encoder fitting; 23–28 retained; no gain |
| Low-rank controls | 128 per representation | Training-only SVD of word, character, joint lexical, Qwen body and Qwen support-product matrices | Outer-training projection and row L2 normalization; fixed classifier; tests representation compression rather than axis selection |

The broad screen covers **88,924–104,665 candidate columns per fold**, retaining **4,860–7,916** across its ten family banks. Including the separate 99-column NLI and 36-column instruction banks gives the following auditable counts. These are bank columns, not the size of one chosen final model, and they are not summed across repeated folds.

| Protocol | Fold | Candidates | Retained | Rejected |
| --- | ---: | ---: | ---: | ---: |
| Familiar rules | 0 | 89,059 | 4,919 | 84,140 |
| Familiar rules | 1 | 94,651 | 5,541 | 89,110 |
| Familiar rules | 2 | 96,271 | 5,419 | 90,852 |
| Held-out rule | 0 | 104,800 | 7,975 | 96,825 |
| Held-out rule | 1 | 103,537 | 6,480 | 97,057 |

The five low-rank controls each derive up to 128 components from already-counted input matrices. They are separate representation alternatives, not extra raw variables quietly added to those screened-bank totals. The final selected feature count is **undecided**.

### Families that the available data cannot support

There are no timestamps, observation histories, conversation threads, user identifiers, organizational records, seasons, opponents, coaches or external ratings in the supplied schema. Lagged/rolling/recency/trend/volatility, strength-of-schedule and historical entity aggregates would therefore invent information or reinterpret row order as time. They were ruled inapplicable, not silently omitted. Empty-text indicators exist among structural candidates, but the strict input contract rejects missing required text; no useful missingness signal exists in this data.

External moderation context remains plausible only with legitimate source access, provenance, availability-at-inference and overlap checks. Current subreddit pages are not historical versions of the provided rules. No external comment corpus, hidden labels, leaderboard probing or web-fetched current metadata entered these experiments. The released encoder assets do not permit a complete audit of overlap with their pretraining corpora; no pretraining-contamination clearance is claimed.

The observed policies concern advertising and legal advice. URL counts can indicate promotion but also ordinary helpful links; requests, modals and second-person language can indicate advice but also permitted discussion. The rule/support contrasts test these mechanisms. Their negative ablations show why an intuitive cue alone is insufficient.

## Experimental design

The broad study (`9ec008d506b0bc64a717`) executes 21 configurations on the same five preserved folds: **105 fits**. Nine family additions use the same screened-word control; family-only controls, full combinations and six leave-one-family-out comparisons are retained. Logistic regression stays at C=2, liblinear, 2,000 maximum iterations, seed 2025. There is no model-hyperparameter search. Dense and sparse scaling conventions are part of each tested representation; the historical reference is not presented as an otherwise-identical scaling control.

The screen validates schemas, finite values and row counts; rejects constants/near-constants, incidence below three training rows, exact duplicate columns and suspicious perfect separators; ranks by training-only effect score; and applies fixed family budgets. Dense redundancy screening uses absolute correlation 0.995 in a bounded top-score pool. Sparse duplicate detection uses exact column hashes rather than quadratic all-pairs correlation. Each decision and selected-name list is preserved privately with hashes. This is a tractable search, not a claim to compare every possible feature subset.

The sensitivity study additionally tests full lexical vocabularies, training-only NB token reweighting, five training-only low-rank representations, lexical support scoring and seven frozen Qwen geometry scores. The NLI study (`537cf2213c813b8ebd4b`) tests rule-only, support-only, combined, word-plus-NLI and broad-plus-NLI representations: **25 fits**, plus a label-free rule margin. It encodes **11,835 unique pairs** for 12,174 requested occurrences; 26 pairs reach the truncation limit. It uses revision `fa2804872c3b4bd748f38c0185cc85775361e735` of `cross-encoder/nli-deberta-v3-small`, with verified assets and 256-token input limits.

The near-copy study (`a09455ec14e9b3d6ab61`) adds **10 fits** under a fixed text-only isolation policy: hashed character-ngram cosine ≥0.95, token Jaccard ≥0.90, minimum 40 characters. It detects no additional cross-boundary copies after exact purging. This narrow stress test does not establish semantic-paraphrase or shared-origin isolation. A separate sensitivity score excludes the 18 self-support rows without refitting.

The instruction study (`96bf69f42f9063e01a12`) adds **15 fixed-classifier fits** and three label-free scores. Revision `c1899de289a04d12100db370d81485cdf75e47ca` of Qwen3-0.6B scores 5,933 unique prompts for 6,087 occurrences. Field budgets truncate 1,759 comment/rule/support field occurrences while preserving the question. All 186 shards were restored and hash-verified; the current source replay reused every shard. The three fixed templates test rule versus support context without a prompt sweep. Together, these five studies comprise **210 controlled fits**, excluding earlier references, the 20 early-control fits and reproducibility reruns.

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
| Fixed instruction features | 0.480826 | 0.775989 | 0.285879 |
| Screened words + instruction features | 0.575486 | 0.695832 | 0.249457 |
| Semantic centroid + instruction features | 0.571602 | 0.781163 | 0.286934 |

The full machine-readable comparisons, including low-rank controls and per-rule metrics, are in [research](../reports/research/results.json), [sensitivity](../reports/sensitivity/results.json), [NLI](../reports/pairs/results.json), and [instruction likelihoods](../reports/instructions/results.json). Notebook `02` renders them directly rather than maintaining an independent score table.

Against the **same screened-word representation**, compact semantic geometry contributes +0.032390 AUC, with pointwise interval [0.013066, 0.051477] but simultaneous interval [−0.015586, 0.080367]. Characters contribute +0.023176 with a positive pointwise interval but a simultaneous interval crossing zero. Structure and raw-coordinate additions are negative even under the broad study's simultaneous intervals. Nested target encodings add only about 0.00155; that is not a stable gain.

The centroid's observed improvement over the original rule/example reference is **+0.026031 AUC**. Its pointwise interval is [−0.005592, 0.057836]; simultaneous intervals are in the updated sensitivity report. Its lower log loss and Brier are descriptive, not a fitted-calibration claim. Excluding self-support rows leaves centroid AUC 0.636120 versus reference 0.609503. The gain is concentrated in advertising: centroid AUC rises from 0.667263 to 0.728820, while legal-advice AUC falls from 0.563862 to 0.554369. Full characters and words-plus-semantic summaries also decline on legal advice. Thus the strongest average scores are **not improvements across both held-out rules**. There is **no independently confirmed positive feature-engineering gain** yet, and no numerical uplift should be presented as a proven improvement.

## Completion assessment

The four-policy feature campaign is complete: **323 fixed fits** (238 primary, 22 changed-training sensitivities, 35 retrieval and 28 semantic-formatting/intent fits), six frozen resolution scores, three new formatting scores and one fixed complementarity control. It uses 11,135 development rows; **43,576 reserved targets remain unopened**. [Expanded results](EXPANDED_STUDY.md) · [Last semantic study](SEMANTIC_FORMATTING.md) · [Coverage and exclusions](FEATURE_COVERAGE.md).

**About 84% overall completion** is an effort-weighted planning estimate. The explicit rubric credits data/validation 17/20, feature research 35/35, model/evaluation/inference 10/20, engineering 14/15 and presentation 8/10. Feature credit means the declared phase has passed its evidence gate, not that every possible NLP feature has been exhausted. Working references and diagnostic evaluation earn partial model credit; independent confirmation, final calibration and production promotion do not. Notebook `02` completes this research phase with current verified aggregates and executed narrative. Neither percentage is a measured employer rating.

The complete bank offers **188,595–188,598 candidate columns per fold**. Training-only screens retain **9,281–9,660** across separate banks. These are not the selected model's width: the retained unseen-policy representation uses the contrast of positive and negative support centroids from the original frozen 1,024-dimensional embeddings. Compact semantic geometry gives the strongest matched addition (+0.1385 AUC); lexical support comparisons add +0.0677. Raw coordinates, community/target context, retrieval, shorter prefixes, asymmetric supports and tested NLI/intent variants do not replace the centroid.

The centroid's matched development gain over the lexical reference is **+0.2314 AUC**, reaching 0.7042. The latest fixed average reaches 0.7086, but its +0.0044 gain is uncertain, advertising declines 0.0222, and log-loss/Brier regressions fail the frozen tolerances. None of the final seven candidates or the one average passes replacement criteria. The [generated decision](../reports/feature_decision/decision.json) therefore retains the simpler centroid and closes the declared feature phase. All 28 new model predictions and feature transforms were independently replayed from private artifacts without inference or fitting.

**Diminishing returns are now defensible within this scope.** Remaining legal/medical weaknesses concern fine distinctions in intent and missing context. The 48-row source-checked review is qualitative, not independent label adjudication. Further encoders, supervised representation training, new policies and thread context remain legitimate future model/data research; no temporal or entity history can be reconstructed safely from row IDs. The feature gate no longer blocks final model development. It does not authorize claiming independent performance or a production-ready model.

Next, implement and verify the fixed familiar/unseen-policy route, develop calibration without reserve targets, and commit the exact final comparison before one protected evaluation. Then promote accepted artifacts to offline inference and an example-driven demonstration, measure serving budgets, add model/data cards and tag a reproducible release. The current standalone inference still uses the lexical reference. [Concrete next milestone](FINAL_MODEL_PLAN.md) · [Release roadmap](ROADMAP.md).

## Research sources and external-data feasibility

- [Park et al., 2021: community-sensitive norm violations](https://aclanthology.org/2021.findings-emnlp.288/) motivates explicit policy/community context rather than toxicity alone. [The authors' NormVio repository](https://github.com/chan0park/NormVio) releases redacted comment IDs and requires reconstruction with Reddit credentials; it is not a ready independent text benchmark in the connected environment. No reconstruction or redistribution was performed.
- [He, May and Lerman, 2024: CPL-NoViD](https://arxiv.org/html/2305.09846v3) supports policy-conditioned prompts and cross-rule/community testing. Its fine-tuned contextual approach is not equivalent to our frozen NLI probe, and its reported performance is not transferred to this project.
- [Wang and Manning, 2012](https://aclanthology.org/P12-2018/) motivates NB log-count reweighting as a lexical feature control. Our classifier and validation are documented separately.
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) motivates semantic similarity representations; [Qwen's model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and [the NLI model card](https://huggingface.co/cross-encoder/nli-deberta-v3-small), and [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) identify the actual frozen assets, with Apache-2.0 model-card licensing.
- [Official Kaggle evaluation](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/overview/evaluation) specifies column-averaged AUC, offline notebook execution and permission for public external data/pretrained models. The [host's per-rule attachment](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/641121) numerically corroborates rule averaging for 2,428 of 2,437 complete rows within 1e-6; nine discrepancies and six incomplete rows prevent an exact-parity claim. The [host dataset release](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/641107) and [CC0 version 1 dataset](https://www.kaggle.com/datasets/sorenj/jigsaw-agile-community-rules-classification/data) are the provenance for the new cohort. Source hashes are pinned before research target access. The unrelated arXiv `2511.17592` cannot support scoring equivalence.
