# Data card

This project studies whether a comment violates a supplied community rule. Each row contains comment text, a community, a rule and two positive/two negative support examples; the target is binary `rule_violation`. Positive means violating the supplied rule, not universal toxicity.

## Provenance and boundaries

The original source is the [Jigsaw Agile Community Rules competition](https://www.kaggle.com/competitions/jigsaw-agile-community-rules). Expanded research uses the [organizer's post-competition release](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/641107), version 1 of [the released dataset](https://www.kaggle.com/datasets/sorenj/jigsaw-agile-community-rules-classification/data). Archive and member hashes are pinned in [`configs/released.json`](configs/released.json). The released dataset is listed as CC0; competition participation and third-party assets retain their respective terms.

| Cohort | Rows | Use |
| --- | ---: | --- |
| Original training | 2,029 | Original reference and development |
| Retained released Public research | 9,106 | Additional development labels, four policies in total |
| Combined development | 11,135 | Feature selection, model fitting and nested calibration |
| Reserved before final copy audit | 43,576 | Initially protected targets |
| Eligible protected confirmation | 43,509 | Single frozen comparison after 67 target-blind near-copy exclusions |
| Original downloadable test preview | 10 | Notebook software verification only |

The released full source has 54,059 rows. The preparation boundary excludes 1,323 research rows crossing protected comment/support identities and 54 reserve rows with historical exposure. Those exclusions precede the later 67-row copy audit; they must not be subtracted again from the final eligible cohort. [Assignment and exclusion protocol](docs/RELEASED_DATA.md) · [Final copy audit](docs/CONFIRMATION.md).

Four development policies concern advertising, illegal-activity promotion, legal advice and medical advice. Financial-advice and spoiler policies are wholly outside development. Final confirmation contains 25,485 familiar-policy and 18,024 unseen-policy rows.

## Validation and access

Splits isolate normalized comment and support text. Vocabulary, scaling, screening and model fitting use training partitions only. Target-derived research features are cross-fitted where used; the final selected model contains no target-derived encodings. Development comparisons retain paired predictions and fold identities. The final route and acceptance criteria were frozen before eligible protected targets were interpreted on September 9, 2026; the prediction freeze is [commit 530ad799](https://github.com/alvaromendizabal/jigsaw-rule-classifier/commit/530ad79929012e807cb42a5253f7b92b090682ec).

Public GitHub artifacts contain aggregate reports, source/configuration hashes, executed evidence notebooks and four newly authored demo examples. Raw comments, row-level labels/predictions, fitted artifacts and private recovery archives remain outside Git. The offline model bundle does not include raw development or confirmation rows. The four examples in [`configs/demo.json`](configs/demo.json) illustrate behavior and are not an accuracy test or samples copied from the competition.

## Limits

Rules and available examples provide incomplete social and conversational context. Labels may reflect annotator interpretation; the 48-row qualitative review was not independent relabeling and did not change targets. Text-only duplicate controls do not prove shared-origin or paraphrase independence. Author identities, timestamps and threads are unavailable, so temporal/entity features and temporal-generalization claims are unsupported.

Six confirmed policies cannot establish performance across all communities or languages. Dataset labels and encoder pretraining may contain bias; population fairness and full pretraining-overlap audits were not possible. Published uncertainty conditions on observed policies and does not quantify variation across arbitrary future policies.

The original-training-only Kaggle notebook and the post-competition research model have distinct data eligibility. A successful 10-row preview run is not a hidden-test score. [Submission and restoration guide](docs/DELIVERY.md).
