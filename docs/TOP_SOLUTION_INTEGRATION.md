# Source credits and implementation boundary

The retained backbone is **Qwen3-4B-Instruct-2507** from the Qwen team. Its model revision and file checksums are pinned in [the model input contract](../configs/competition_features.json). The [upstream license](../assets/qwen3/LICENSE) and tokenizer configuration remain with the offline source package. The historical execution used Guanshuo Xu's `wowfattie/qwen3-4b-instruct-2507/transformers/default/1` model mirror; mirror availability is not guaranteed by this repository.

Publicly documented support-label adaptation, last-position decision loss and rank-based inference informed this work. The maintained implementation consists of reviewable project source, its leakage and recovery contracts, and the original evaluation evidence; it does not claim authorship of the pretrained architecture or publish third-party weights.

Alvaro Mendizabal's project contributions are eligible-label construction, duplicate/conflict purging, recoverable LoRA execution, decision scoring, row-preserving offline packaging, the controlled evaluation and the authored public diagnostic interface. Alternate-backbone and teacher campaigns that did not replace the retained scored system are no longer part of the maintained implementation.

[Model card](../MODEL_CARD.md) · [Exact source builder](../scripts/build_adapted_notebook.py) · [Historical result](../reports/checkpoints/kaggle_adaptation.json)
