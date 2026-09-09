# Competition performance rebuild

The original late entry scored **0.59191 public / 0.61956 private**. Its successful
execution did not meet the performance objective. The target is approximately
**0.92 private AUC**, with no guarantee that a given experiment will achieve it.
The feature and model gates are reopened. The earlier post-competition research
release remains a reproducible historical artifact, not the competition solution.

## What failed

The submitted version used TF-IDF and logistic regression fitted on 2,029 original
training rows. It used eight lexical context statistics and had no semantic
encoder or adaptation to the supplied examples of new policies. The held-out-rule
benchmark already showed weak transfer. A completed notebook and a wide feature
search were insufficient evidence for declaring the performance work complete.

The later accepted model was a separate study using organizer-released labels.
Its 0.7770 protected policy-macro AUC is neither a Kaggle score nor a valid estimate
of the submitted lexical model. Those labels cannot be added to a new entry while
claiming a comparable original-competition result. The consumed protected cohort
will not be used for another tuning cycle.

There is no evidence that a submission-format violation caused the low score:
Kaggle accepted and scored the notebook. The official requirements permit public
external models, require internet-disabled notebook inference, cap CPU/GPU runs at
12 hours, and require `submission.csv`. The site permits five submissions per day.
The competition closed October 23, 2025; new entries are late evaluations.
[Overview](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/overview)
· [Rules](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/rules).

## Methods the first campaign missed

The winner trained on labeled support examples supplied with test policies,
deduplicated without subreddit, adapted language models with a Yes/No-only loss,
and combined predictions using ranks within each policy. His reported 4B Qwen
private result was 0.9198 and the six-model ensemble reached 0.9293. These are the
author's results, not a reproduction by this project.
[First-place write-up](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/1st-place-solution).

The third-place solution also adapted to supplied examples. It constructed
features from adapted last-token representations and distances to positive and
negative prototypes, then fitted classical classifiers. This motivates testing
learned representations and their interactions, not merely replacing the final
classifier or inflating the count of surface statistics.
[Third-place write-up](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/3rd-place-solution).

The competition's supplied support labels are legitimate inputs. Hidden body
targets and post-competition released targets are different information. Our
`support_pairs` contract rejects a target column in the inference frame, deduplicates
normalized rule/text pairs, excludes conflicting labels, and purges validation
text from every candidate training source. No assumption turns a violation of one
policy into a negative example for another.

## Bounded experiment: contextual representation features

The first comparison freezes **Qwen3-4B-Instruct-2507**, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, with verified asset hashes. This model
revision predates the competition deadline. It receives only the 2,029 original
development inputs. No competition targets enter the GPU feature extractor.
[Model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

| Feature family | Candidates | Hypothesis and test |
|---|---:|---|
| Rule-only last-token representation | 2,560 | Encode the comment's meaning relative to the written rule |
| Example-only representation | 2,560 | Infer a decision boundary from supplied positive/negative examples |
| Joint rule/example representation | 2,560 | Resolve ambiguous wording using both sources of context |
| Joint-minus-rule residual | 2,560 | Isolate information introduced by supplied examples |
| Joint-minus-example residual | 2,560 | Isolate information introduced by the written rule |
| Rule-by-example coordinate products | 2,560 | Test agreement between the two contextual representations |
| Probability, log odds and answer mass | 9 | Separate ranking signal from confidence that the model follows the task |
| **Unique candidate columns** | **15,369** | Counts are not summed across repeated folds or combined banks |

Every learned bank uses the same logistic classifier, fixed regularization and a
maximum of 64 retained columns. Screens use outer-training labels and statistics
only: constants, rare/duplicate columns, perfect separators, effect ranking and
bounded correlation checks. Ablations compare raw scores, each representation,
context interactions and their combination. Results include policy-macro AUC,
pooled AUC, log loss, Brier score, per-policy scores and paired group bootstrap
intervals. Feature-specific contrasts hold the backbone fixed. Comparing this
experiment against the earlier 0.6B probe does **not** isolate feature engineering.

Both historical validation protocols are retained: three grouped familiar-policy
folds and two held-out-policy folds, with validation comments removed from fitted
training context. Familiar-policy purging leaves only 237–287 training rows per
fold, which must be considered when interpreting learned high-dimensional banks.
The study is exploratory development on previously examined policies, not a new
untouched holdout. A diagnostic excludes 18 rows with self-support overlap.

The cloud worker uses one L4 GPU, a 3,600-second job cap, immutable source/config
identities and hash-verified checkpoints uploaded after every shard. Valid shard
replay does not invoke the encoder. The launch receipt records the exact AWS
image, source version, rate and job. Runtime, completed artifacts and measured
results must be checked before calling the experiment complete.

### Measured result: reject the frozen coordinate bank

The GPU extraction completed 6,087 prompts with zero truncations. The CPU study
completed all 45 fixed feature-bank fits and five lexical-reference fits.

| Original-data held-out policy comparison | Policy-macro AUC |
|---|---:|
| Lexical reference | 0.6156 |
| Frozen 4B rule answer | 0.7081 |
| Frozen 4B joint rule/example answer | 0.7056 |
| Screened rule representation | 0.5508 |
| Screened joint representation | 0.4881 |
| Screened combined feature bank | 0.4516 |

The full bank retains 64 of 15,369 columns in each fold: 15,300 exceed the fixed
screen budget and five are constant or nearly constant. The rule-answer gain over
the lexical reference is +0.0926, with a within-study simultaneous 95% interval
of [0.0476, 0.1376]. Adding supplied examples to the prompt changes AUC by -0.0026
[-0.0476, 0.0424]. This provides no evidence of a useful prompt-context gain.
The learned coordinate bank fails transfer and is rejected.
[Verified aggregates](../reports/competition_features/metadata.json).

A numerical audit found that float32 sigmoid rounded 867 rule-only probabilities
to one, losing ranking information. Reconstructing float64 probabilities from the
saved log odds corrects AUC from 0.6883 to 0.7081 without any new inference. The
recomputed study includes that correction and leaves the full-bank result at
0.4516. Raw probability losses remain poor. These are original-data development
scores, not new Kaggle scores or evidence that the 0.92 objective has been reached.

### Next controlled comparison: learning from supplied supports

`configs/support_adaptation.json` fixes one epoch of rank-8 LoRA on the same 4B
backbone and the identical rule-only prompt. Only the final Yes/No token contributes
to the loss. The new rule's supplied support pairs occur twice; other legitimate
training pairs occur once. No subreddit is used for deduplication or prompting.

The target-blind novel-comment cohort contains **881 rows**: 234 for one policy
and 647 for the other. All 1,148 rows whose body occurs in any supplied support
pool are excluded from evaluation. Per fold, 1,640/1,215 unambiguous unique training
pairs remain after purging, including 629/366 pairs from the new rule's supplied
supports. Query targets are absent from the cloud input; every query body is
excluded from every fitted source. This tests **support adaptation to a new rule**,
which is a different task from zero-shot held-out-rule validation.

The comparisons hold the model and training inputs fixed: direct log-odds ranking,
screened 2,560-dimensional decision representations, positive/negative centroid
and nearest-example margins, and a fixed 50/25/25 rank blend. Top-five-neighbor
margins are secondary geometry diagnostics. Readout screens use supplied training
labels only. The paired group bootstrap covers five declared contrasts. Neither
the original two-policy cohort nor the consumed post-competition holdout is a new
untouched validation set.

Training checkpoints include adapter weights, optimizer, learning-rate scheduler,
Python/NumPy/Torch/CUDA random state, exact data order, step and source/configuration
identity. A CPU interruption test with nonzero dropout reproduces uninterrupted
adapter weights bit for bit. The bounded GPU job also deliberately reloads a
fresh model after its first durable optimizer checkpoint in each fold. Completion
and successful GPU recovery must be observed before claiming that gate passed.

## Subsequent acceptance gates

1. **Representation evidence.** Publish retained/rejected counts, matched-family
   AUC changes and failures from the bounded comparison. More coordinates alone
   are not an improvement.
2. **Support adaptation.** Implement supervised representation learning from
   supplied support labels, including a separate evaluation that mimics new-rule
   support availability. Compare the same backbone before and after adaptation.
   Keep query bodies out of fitting for the measured novel-comment cohort.
3. **Adapted geometry.** Compare direct answer scores, adapted decision embeddings,
   positive/negative prototype distances, hard-negative comparisons and a compact
   combined representation. Screen and ablate within training boundaries.
4. **Competition execution.** Fit from legitimate support inputs inside Kaggle's
   offline hidden run. Verify optimizer/scheduler/RNG/data-order recovery, time,
   memory, asset attachment, row order, prediction parity and complete output.
5. **Promotion.** A validated candidate replaces the canonical submission only
   after the measured gains and inference budget support it. Preserve every
   submission receipt. A late score cannot change the completed leaderboard rank.

Temporal, rolling, player/team and opponent features are inapplicable: this
dataset supplies no timestamps or longitudinal entities. Additional public data
requires a documented license, availability date and contamination check.
Research closure requires diminishing returns across plausible semantic feature
families and defensible end-to-end performance; a numerical portfolio rating is
not an acceptance test.
