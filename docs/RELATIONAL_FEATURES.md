# Round 2 — action relationships and behavior-signal attribution

## Evidence that motivates this round

The original round-1 return archive passed all 18 declared SHA256 checks. Its run
`32e706fe1e7dddb7de34` completed 20 CPU fits in 23.893 seconds (44.78 seconds for the
launcher including checks). The full candidate failed its preregistered decision.

| Representation | Advertising AUC | Legal-advice AUC | Mean policy AUC |
| --- | ---: | ---: | ---: |
| Lexical control | 0.673022 | 0.641993 | 0.657508 |
| + behavior | 0.675784 | 0.683940 | 0.679862 |
| + scope | 0.661082 | 0.679957 | 0.670520 |
| Full round-1 features | 0.629440 | 0.664468 | 0.646954 |

The behavior comparison has delta +0.0223538 and simultaneous interval
[-0.0164979, +0.0612054]. Legal-advice log loss worsens from 0.656063 to 0.710278,
even though AUC improves. The full candidate's macro delta is -0.0105535.
These are exploratory local CPU results; the accepted Qwen model has not changed.

## Research hypothesis, frozen before this round's real-data results

More columns alone failed. Test whether narrowly specified local relationships
add beyond the measured behavior anchor. Distinguish a request for advice, a
directive, an offer, a personal recollection, and reported material. A local
commercial call to action linked to a URL is a different observation from an
isolated URL count. Negation and quotation should be measured at the action,
not treated as permission labels.

The primary is `add_act_roles` versus the reused `add_behavior` reference.
It does not switch to whichever new candidate has the highest held-out score.

## Feature availability and mechanisms

All 72 candidates are deterministic functions of inference-visible comment text.
No query outcome, model prediction, external request or missing context enters
feature extraction. Each family contains 12 distinct relations represented as
presence and capped log-count: 24 columns per family.

- `act_roles`: local binding of actor/utterance pattern with legal topic/actions;
  self versus other request, directive, offer, experience, reported evidence,
  question, conditional direction, resource mention, and unmatched controls.
- `link_intent`: nearby URL plus a call to action, owned business, contact,
  resource, price or affiliate cue; explicit referral parameters; transactional
  contact; authored versus quoted promotion. Domains are not fetched or ranked.
- `action_scope`: negation between predicate/action, unnegated comparison,
  quoted action, reported direction, disclaimer followed by advice and a
  conditional call to action.

The parser is a small heuristic, not a dependency parser, factual adjudicator or
implementation of natural-language understanding. It handles straight/curly
double quotes, Markdown block quotes and code fences. Single-quoted speech,
complex scope, sarcasm, nested quotations and indirect advice are limitations.
A negated directive ("you should not sue") may still be legal advice. No cue
sets the target. Missing thread/user/time data are not invented.

Unlike round 1's broad co-occurrence bank, this round limits links to short
clause-local relationships, separates removed quote/code spans, and explicitly
measures negation inside a directive such as "you should not sue". The new
family is a hypothesis, not a literature-proven set of the most predictive columns.

## Fixed experiment

Reuse round 1's `lexical_control` and `add_behavior` for both policy folds:
**four existing fitted controls, zero repeated control fits**. Check original
raw/source/environment identities, saved stage checksums, and exact row order.
Reconstruct training-only TF-IDF and the behavior transform, then reproduce the
saved control probabilities from numeric coefficients within 1e-10 tolerance
before any new candidate fit. Do not load arbitrary pickle/model objects.

Use nine new configurations per policy (18 fits total): three family additions,
all families, three full-minus-one-family ablations, and two behavior-bank
ablations that remove (a) the two legal-topic columns and (b) the two length
columns. The last two separately re-screen the remaining original candidates;
they do not establish that every possible topic/style signal was removed.

The logistic-regression specification, C, seed, training repeat weights, word
and character limits and basic behavior screen remain those from round 1.
Each new family is training-screened to at most 12 columns. Keep all new test
configuration settings fixed. Do not tune feature windows or thresholds after
reading query outcomes in this run.

## Cohort and leakage controls

Reuse the canonical historical training/support construction, global support
exclusion and all-source query purge: 234 advertising + 647 legal-advice queries.
The supplied new-rule support labels are eligible training data in this protocol;
these are not "zero-shot rule holdouts without support adaptation". Exact novel
query identities must match round 1. Normalized query text cannot appear in the
eligible training rows. Query labels are used only for post-prediction metrics.
No organizer-released targets or consumed protected research cohort are loaded.

## Gate and uncertainty

The primary must gain >=0.003 mean policy AUC over the behavior reference, have a
positive simultaneous 95% lower bound, no negative policy delta and no reduction
in per-policy-ranked pooled AUC. Use 500 paired normalized-comment-group bootstrap
draws. The simultaneous interval covers 13 comparisons in this round, not the
adaptive sequence of all earlier experiments. Two previously inspected policies
cannot establish performance on arbitrary unseen rules. A pass is only eligibility
for further validation, not promotion, automatic GPU spend, or a Kaggle score.
The original round-1 full candidate remains rejected regardless of this result.

## Bounded execution and preservation

Use the existing CPU environment. The worker has a 240-second alarm and candidate
checkpoints; the notebook-facing process has a 300-second process-group timeout.
The launcher has a 540-second internal budget and the provided terminal command
has a 600-second outer limit. Heartbeats appear every 15 seconds. These are maximum
limits, not runtime predictions, and they do not stop the SageMaker application.

Round 1's data, code, notebooks, reference predictions and receipts are read-only.
Round 2 uses separate public aggregates and private checkpoints. Each completed
candidate is verified before reuse, corruption stops instead of silently refitting,
and successful result/notebook replays perform zero new fits/cells. All public
exports are allowlisted: no raw comments, row labels, prediction arrays, credentials
or environment variable dumps. The helper does not change AWS resources or GitHub.

## Research sources and what they actually support

- Park et al., 2021, *Detecting Community Sensitive Norm Violations in Online
  Conversations*: https://aclanthology.org/2021.findings-emnlp.288/ . Supports
  examining rule/context-sensitive norm violations beyond toxicity. We do not
  import the external dataset or fabricate missing conversational context.
- Ribeiro et al., 2020, *Beyond Accuracy: Behavioral Testing of NLP Models with
  CheckList*: https://aclanthology.org/2020.acl-main.442/ . Motivates authored
  behavioral tests. Our tests assert feature extraction, not new moderation labels.
- Clarke et al., 2023, *Rule By Example*: https://aclanthology.org/2023.acl-long.22/ .
  Motivates a future rule-grounded contrastive-representation study. This regex
  feature experiment is not an implementation or reproduction of that paper.
- Guanshuo Xu, 2025, competition first-place solution:
  https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/1st-place-solution .
  Reports support-based fine-tuning, deduplication, answer-token training/scoring,
  per-rule rankings and a multi-model ensemble. Historical private ensemble
  score: 0.9293. It does not prove that hand-engineered features alone close our gap.

## Next decision

Record the real matched additions/removals and per-policy uncertainty. Publish
verified evidence before another follow-up. A better behavior relation may merit
a support-grounded semantic representation test, while a failed one remains a
recorded negative result. Do not claim this round exhausts feature research or
that it produces a model guaranteed to beat the historical winning score.
