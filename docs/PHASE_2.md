# Phase 2 · Semantic rule generalization

Status: experiment design; neural implementation and hardware execution have not started. The completed lexical baseline is preserved as the comparison reference.

## Question and reference

Can semantic representations distinguish violations of a supplied rule better than lexical overlap, especially when that rule has no labeled training rows?

The predeclared reference is `rule_examples` from run `c15c2c2318fc0ed619c6`: held-out-rule macro AUC **0.61556**, advertising AUC **0.66726**, legal-advice AUC **0.56386**, log loss **0.67358**, and Brier **0.24049**. Improvement in one number does not establish improvement in every dimension.

## Experiment order

| Experiment | Inputs and fitting | Purpose |
| --- | --- | --- |
| Frozen semantic example matcher | Embed rule/comment and provided positive/negative examples; score similarity margin | Test semantic transfer without fitting the encoder |
| Embedding feature classifier | Fit a regularized classifier on semantic similarities using retained training-fold rows | Test supervised ranking and probability quality |
| Rule-conditioned cross-encoder | Jointly encode rule, comment, and support examples; train within each retained fold | Test interactions independent embeddings cannot represent |
| Context ablations | Remove examples, permute their order, remove rule text, compare maximum/mean similarity | Test reliance on the intended evidence |

Candidate first frozen encoder: **Qwen/Qwen3-Embedding-0.6B**. Its model card lists 0.6B parameters, up to 1,024 embedding dimensions, Apache-2.0 licensing, and a Transformers compatibility floor of 4.51.0. The implementation must record an immutable model commit, tokenizer revision, precision, pooling, prompt, and truncation settings; these are not yet locked. [Official model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

Published benchmark claims do not establish performance on this dataset. The fifth-place competition writeup describes a diverse ensemble involving per-rule specialists and embedding models, motivating candidate families rather than guaranteeing a result. [Team writeup](https://www.kaggle.com/c/jigsaw-agile-community-rules/writeups/5th-place-solution-diverse-ensemble).

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

The next pull request must test cached-batch resume, corruption, model/prompt invalidation, padding/pooling, inference-mode determinism, fold isolation, and row alignment. Include a small real-model integration test. Merge after CI and hardware smoke tests pass; then execute the full experiment with cloud checkpoints. This release does not install neural dependencies or launch a GPU job.
