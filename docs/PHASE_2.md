# Phase 2 · Semantic rule generalization

Status: Phase 2A is implemented and its real-data benchmark is complete: frozen Qwen3 embedding/example comparison and a fold-fitted classifier. The exact lexical reference is preserved. Phase 2B now includes an executed frozen DeBERTa NLI feature probe, explicit rule/support ablations and low-rank controls. These did not justify model promotion; the feature gate remains open. See [the current research record](FEATURE_RESEARCH.md).

## Question and reference

Can semantic representations distinguish violations of a supplied rule better than lexical overlap, especially when that rule has no labeled training rows?

The predeclared reference is `rule_examples` from run `c15c2c2318fc0ed619c6`: held-out-rule macro AUC **0.61556**, advertising AUC **0.66726**, legal-advice AUC **0.56386**, log loss **0.67358**, and Brier **0.24049**. Improvement in one number does not establish improvement in every dimension.

## Experiment order

| Experiment | Inputs and fitting | Purpose |
| --- | --- | --- |
| Frozen semantic example matcher | Embed comment and provided positive/negative examples; score similarity margin | Test semantic transfer without fitting the encoder |
| Embedding feature classifier | Fit a regularized classifier on semantic similarities using retained training-fold rows | Test supervised ranking and probability quality |
| Rule-conditioned cross-encoder | Jointly encode rule, comment, and support examples; train within each retained fold | Test interactions independent embeddings cannot represent |
| Context ablations | Remove examples, permute their order, remove rule text, compare maximum/mean similarity | Test reliance on the intended evidence |

Implemented frozen encoder: **Qwen/Qwen3-Embedding-0.6B**. Its model card lists 0.6B parameters, up to 1,024 embedding dimensions, Apache-2.0 licensing, and a Transformers compatibility floor of 4.51.0. Revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` is pinned in `configs/semantic.json`; Hub object hashes are in `configs/model.json`. Inference uses CPU float32, SDPA, last non-padding-token pooling, unit normalization, and a 256-token cap. The exact prompt and all execution settings are recorded. [Official model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

Published benchmark claims do not establish performance on this dataset. Community-sensitive moderation and policy-conditioned prompting motivate the feature families; the [current research record](FEATURE_RESEARCH.md#research-sources-and-external-data-feasibility) links verified primary sources. The body of the linked competition writeup was not retrievable, so its methods are not asserted here.

## Measurement contract

Reuse the baseline's exact split memberships and purged training rows. Record their checksums and train-file checksum in every new run. Frozen, label-free per-text encoding can be cached across folds; any learned transform, calibration, supervised example expansion, or classifier is fitted within the training fold only. No test-time adaptation is part of this initial comparison.

Report rule macro AUC, per-rule AUC, pooled AUC, average precision, log loss, Brier, calibration plots, and fixed-threshold diagnostics. Retain OOF predictions, paired per-rule comparisons, inference latency, total wall time, throughput, peak RAM/VRAM, and truncation counts. Estimate paired uncertainty using comment groups within rules, acknowledging that two rules cannot estimate broad policy diversity.

Raw cosine margins are ranking scores, not calibrated probabilities. AUC/AP can evaluate them directly; probability metrics require a predeclared probability transform or a calibrator fitted inside training folds. Do not apply log loss to unbounded similarities or fit calibration on the OOF labels being reported.

## Reliability contract

Embedding cache keys include text hashes, role/prompt, model/tokenizer commits, software, device/precision, pooling, and length settings. Commit each complete batch atomically with checksums. Resume must reuse valid batches and reproduce uninterrupted outputs within declared numerical tolerance. The active incomplete batch restarts.

Fine-tuned checkpoints additionally require model, optimizer, scheduler, scaler, random-generator states, data-order position, and consumed batch count. Distributed training needs rank-aware state and a consistent checkpoint barrier. Implement and test these before advertising neural training as resumable.

Long steps emit UTC start/heartbeat/completion events, completed/total batch counts, elapsed time, and throughput. Report total run time separately from training time. Exceptions remain visible; address warnings rather than suppressing them broadly.

## Hardware and release gate

The current small CPU workspace supports saved review and lexical baselines. Do not assume it can efficiently run the neural candidate. Lock compatible dependencies, estimate weight/activation memory, and run a bounded smoke batch on the intended hardware. Record measured memory, latency, disk usage, maximum runtime, and spend limit.

The implementation tests cached-batch resume, corruption, model/prompt invalidation, padding/pooling, inference-mode consistency, fold isolation, and row alignment. A small real-model integration check passed locally and in GitHub Actions. The full experiment completed on CPU; the original source and every committed artifact were verified before publication. This release installs CPU neural dependencies through the `semantic` extra and verifies the real pinned model. It does not launch a GPU job. The four-example integration passed locally in approximately 16 seconds with 3.83 GiB peak RSS. The full benchmark took 1,004.7 seconds with 3.854 GiB peak RSS. These timings exclude the prior model download.


## Running or reviewing

Run `bash bootstrap.sh` after pulling the release. It installs the locked semantic extra and checks the code. To review a completed semantic experiment on the existing small instance, open the already executed `notebooks/04_semantic_benchmark.ipynb`. Restore S3 if refreshing its displays; review does not load Qwen weights.

A fresh computation needs a CPU workspace with at least 8 GB RAM and 4 GB free. The current 4 GB Studio instance is suitable for review but is too small for the measured model footprint. On suitable hardware:

```bash
uv run --extra semantic python scripts/verify_semantic.py
uv run --extra semantic jigsaw semantic --cloud
```

Rerun the same command after interruption. Completed 64-input shards are checksummed and reused; the active incomplete shard restarts. With `--cloud`, S3 snapshots follow each committed shard and model stage. Model weights remain in the local persistent Hub download directory and can be redownloaded by their immutable revision. Cached embeddings can be reused without loading weights.

The margin model uses an untuned temperature of 0.1, so its probability metrics are diagnostics rather than evidence of calibration. Bootstrap intervals use 500 paired draws of normalized comment groups shared across rules and are conditional on fixed OOF predictions. Source revisions, data hashes, model assets, split files, and cache contracts remain traceable.

## Recorded decision

The frozen margin scored 0.63507 held-out-rule macro AUC, versus 0.61556 for the contextual lexical reference. The paired delta interval spans −0.01323 to +0.04914, and its probability losses do not improve. The learned similarity classifier scored 0.58577 and has substantially poorer held-out probability quality. Familiar-rule semantic results are also below the lexical reference. Preserve the lexical model as the reference. The subsequent joint NLI probe and additional feature studies are recorded in [FEATURE_RESEARCH.md](FEATURE_RESEARCH.md). Do not tune this frozen benchmark repeatedly against the same two rules and claim an untouched final test.

The public [experiment record](../reports/semantic/README.md) includes both successful and unsuccessful outcomes, per-rule differences, original source/data provenance, and observed hardware behavior.
