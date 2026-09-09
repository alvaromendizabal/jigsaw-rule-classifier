# Start with the evidence

**Project status:** competition performance remains open after the **0.59191 public / 0.61956 private** lexical baseline. The current candidate learns from legitimate supplied examples with Qwen3-4B; its matched development AUC is **0.7199**, not a Kaggle score. [Measured research and execution gates](docs/COMPETITION_REBUILD.md).

**Current neural notebook:** [`kaggle/submission.ipynb`](kaggle/submission.ipynb). **Historical CPU control:** [`kaggle/reference.ipynb`](kaggle/reference.ipynb), which reproduces the original Version 2 approach. The separate 0.7770 research model uses post-competition development labels. Keep these three model/data boundaries distinct.

**Employer review:** open [03 · Results and model decision](notebooks/03_saved_results.ipynb), then [02 · Feature research](notebooks/02_baseline_and_review.ipynb). The remaining notebooks explain the audit, validation and semantic diagnostics. No AWS account, private data, or model download is required to read the five executed public notebooks.

## Update an existing SageMaker checkout

Use the existing locked environment and restored artifacts. Close project notebook tabs before updating so an old browser tab cannot autosave over a newer file. This block preserves tracked notebook changes in a named stash, requires `main`, fast-forwards, and verifies the source and five public notebooks. It does not repeat bootstrap or original training.

<!-- workspace-update:start -->
```bash
cd "$HOME/projects/jigsaw-rule-classifier" &&
test "$(git branch --show-current)" = main &&
git stash push -m "notebook-session-$(date -u +%Y%m%dT%H%M%SZ)" -- notebooks kaggle/submission.ipynb &&
git pull --ff-only origin main &&
.venv/bin/python scripts/verify.py &&
.venv/bin/python scripts/execute_notebooks.py
```
<!-- workspace-update:end -->

The stash is retained for inspection with `git stash list`; do not automatically pop or drop it. No local notebook changes is a successful no-op. Unrelated edits, untracked files, private data, model weights, runs, and `configs/local.json` remain in place. An unrelated conflicting edit, divergent branch, failed pull, or failed quality gate stops the sequence. Do not use `git reset --hard`, `git clean`, or a force push. A stash is local Git storage, not an S3 backup.

A terminal prefix such as `^[[200~` is a paste-control sequence. Press **Ctrl+C**, manually type `bind 'set enable-bracketed-paste off'`, and press Enter before pasting again. This changes only the current Bash session; pasted newlines can execute immediately. Copy only code, without the prompt or a trailing `~`.

## Review the completed feature research

The four-policy feature campaign is complete: 323 fixed fits, broad screened banks, matched ablations, retrieval, embedding resolution, semantic formatting/intent, a source-checked error audit and a predeclared stopping rule. Notebook `02` records why the original centroid survives the final alternatives. Protected confirmation and offline integration are complete. Scientific closure is scoped to these data and hypotheses. The notebook performs no fitting by default. Do not rerun historical baselines merely to read these results.

```bash
uv run jigsaw gate
uv run python scripts/execute_notebooks.py --publish
# Strict metric recomputation after restoring private runs:
uv run python scripts/verify_research.py
uv run python scripts/verify_formatting.py
```

Reproduction commands are `jigsaw research --export`, `jigsaw diagnostics`, `jigsaw robustness`, `jigsaw pairs` and `jigsaw instructions`. The latter two are frozen-encoder feature experiments, not final training. They require suitable CPU memory and their pinned model assets. The bounded `scripts/processing.py` worker supports these jobs with immutable-source verification, isolated S3 checkpoints and optional `JIGSAW_RESUME_PREFIX` recovery. The ordinary `--cloud` snapshot option is supported by the original/broad experiment runners; use the processing worker for periodic NLI/instruction checkpoints.

The publication helper remains `scripts/execute_notebooks.py --publish --push-branch results/feature-ablation`. It requires current public Jupyter execution, the expected origin, a clean index and no unrelated source edits. Its explicit allowlist includes the verified feature-study aggregates and research figures. Previously committed historical reports may be re-rendered byte-for-byte; new or altered historical evidence requires review. It never commits data, row-level predictions, credentials, model weights or submission files. Merge a results PR only after Quality passes.

A changed implementation/data/configuration contract deliberately creates a different experiment. The original runs remain preserved. [FEATURE_RESEARCH.md](docs/FEATURE_RESEARCH.md) records counts, rationale, failures and the completed feature gate. [MODEL_VALIDATION.md](docs/MODEL_VALIDATION.md) records the executed routing, nested calibration and fitted artifacts. The fixed candidate passed all 12 protected checks on 43,509 eligible rows. Prediction hashes were published before first eligible target access at 17:12:07 UTC on September 9, 2026. [Protected results and lineage](docs/CONFIRMATION.md).

## Generate and download your submission

The canonical `kaggle/submission.ipynb` is the support-adapted GPU candidate. Import it into Kaggle, attach the official competition data and `wowfattie/qwen3-4b-instruct-2507/transformers/default/1`, select **GPU T4 x2**, and turn **Internet off**. The notebook verifies all 11 pinned model files, restores the upstream tokenizer configuration and license, and trains one fixed epoch from original training labels and supplied support labels. It scores comments in length-sorted batches, restores their original order, validates the CSV, and displays a download link. It does not submit automatically.

Training recovery includes the adapter, optimizer, scheduler, FP16 loss scaler, random states and exact data order. Two completed optimizer checkpoints and completed prediction batches are retained under `/kaggle/working/checkpoints/`. Intact completed runs are reused after checksum verification. These files survive a kernel restart within the session; preserving Kaggle outputs or downloading them is required before ending a disposable session. Saved-version outputs provide durable completed artifacts. A changed source/data/runtime contract requires a new fit.

The original 10-row test preview verifies execution only. A scored Kaggle submission reruns the saved notebook against hidden inputs, where additional supplied examples teach the new rules. Hidden query targets and released solution labels are never loaded. Within-policy rank scores preserve AUC and are not calibrated probabilities.

The historical CPU workflow remains `kaggle/reference.ipynb`. Local `scripts/execute_notebooks.py --synthetic` and `--kaggle` deliberately verify that control; they do **not** certify the neural notebook. The neural runtime has its own actual Kaggle GPU verification record in `reports/checkpoints/kaggle_adaptation.json`.

## Durability and progress

The runners emit UTC timestamps, cell/batch progress, 15-second heartbeats, stage time, and total invocation time. Live logs are `logs/notebook_execution.jsonl`, `logs/features.jsonl`, `logs/commands.jsonl`, and `logs/cloud.jsonl`. Committed event logs under `runs/` are backed up; the top-level `logs/` directory is local. Run backup, training, and restore sequentially.

An incomplete checkout cannot publish a latest snapshot that omits previously saved paths. Restore missing work first; never force-delete local conflicts. Access errors are not treated as an empty bucket. Conditional publication compares the previous ETag (`If-Match`) or requires an absent first snapshot (`If-None-Match`), preventing a competing backup from silently replacing the latest pointer. Old content-addressed snapshots remain. Changed model/data artifacts during upload stop publication; JSONL logs are captured as byte snapshots while they may continue appending. An interrupted upload can reuse already-uploaded content objects on retry. [AWS conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html).

Completed notebooks, model folds, and embedding shards are reusable when their contracts and checksums match. An active interrupted notebook, CPU solver fit, or embedding shard restarts; completed inference models and prediction batches remain reusable even when the notebook itself restarts. The new support-adaptation runner additionally saves adapter, optimizer, scheduler and RNG state; see the current rebuild receipt for executed recovery evidence. No software can guarantee that external services never fail; failed work must remain visible without destroying valid checkpoints.

Routine public verification does not rewrite canonical notebooks. Deliberate publication is separate:

```bash
.venv/bin/python scripts/execute_notebooks.py --publish
```

Only public aggregate evidence can be published. Private or synthetic execution cannot overwrite the five public notebooks. Failed execution preserves the last-good canonical file. Publish intentional source/output changes through a feature branch and reviewed pull request. The aggregate reader checks committed checksums and provenance; private `jigsaw review` recomputes metrics from saved row-level predictions.

## Run and submit on Kaggle

**Version 3 is submitted; hidden evaluation is running.** Kaggle showed **Notebook Running (after deadline)** at 23:47 UTC on September 9, 2026. The fresh offline T4 run completed all 117 training steps and validated the ten-row preview; it used 603.5 worker seconds and 8.26 GiB peak allocated GPU memory. Its CSV matches the earlier preview and checksum-verified replay. [Version 3](https://www.kaggle.com/code/alvaromendizabal/jigsaw-support-adapted-rule-classifier?scriptVersionId=348640051) · [Runtime and submission receipt](reports/checkpoints/kaggle_adaptation.json). This entry learns from original training labels and supplied support labels. The new score is pending; do not submit a duplicate while it runs.

**Version 2 is submitted and scored.** Kaggle reported **Succeeded (after deadline)** on September 9, 2026, with **0.59191 public / 0.61956 private**. Open the [competition's Submissions page](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/submissions#) while signed in as `alvaromendizabal` to inspect the entry. [Version 2 output](https://www.kaggle.com/code/alvaromendizabal/jigsaw-original-training-reference/output?scriptVersionId=348605174) · [Machine-readable receipt](reports/checkpoints/kaggle_submission.json). No AWS download, new upload or resubmission is needed to complete this milestone.

This entry uses only the original 2,029 training rows and the lexical reference. The 0.7770 protected research AUC is a separate post-competition result.

To recreate the historical run, import `kaggle/reference.ipynb`, attach the official competition data, select CPU/no accelerator, disable Internet, and use **Save Version → Save & Run All**. The current `kaggle/submission.ipynb` requires a GPU. Both the current `/kaggle/input/competitions/` mount and the older mount are supported. The old S3 `v1.0.0` notebook snapshot predates this path fix. The notebook produces `/kaggle/working/submission.csv`; its original 10-row preview is not the hidden-test output or a leaderboard score.

The explicitly synthetic software-only inference check is `scripts/execute_notebooks.py --synthetic`. It never establishes competition performance.

## Historical post-competition research boundary

The feature-completion gate has passed for the declared four-policy scope. The semantic-formatting/intent and fixed-fusion comparisons are complete and retain the original centroid. The final route is now fitted on development: 9,263 screened features with accepted calibration for familiar policies, and the raw frozen centroid for unseen policies. Nested calibration rejects a harmful unseen-policy correction. The [confirmation protocol](docs/CONFIRMATION.md) is now committed, with tested target-free inference and scoring gates. The protected comparison accepted the fixed route: policy-macro AUC 0.7770 versus 0.6801, with improvements on all six policies. Offline integration, verified restore and the requested late Kaggle submission are complete. The historical lexical control and current support-adapted candidate use the original competition boundary; both remain separate from the post-competition benchmark. [Completed release milestones](docs/ROADMAP.md#completed-deliverables).

The tested preparation command is `uv run python scripts/prepare_released_data.py`; it verifies or downloads the pinned public archive and materializes **only** the 9,106 retained research targets. Together with the original data, this provides 11,135 development rows before duplicate handling. Downstream research should use `jigsaw_rules.released.load_research(root)` and the original training file. Do not replace `data/raw/test.csv` or point the submission notebook at the released solution.

The 43,576-row reserve includes the former Private partition and financial-advice/spoiler policies. After 67 target-blind near-copy exclusions, the 43,509 eligible targets were used once for the frozen comparison. They were not used for model selection and must not become another tuning split. The exact assignment, source hashes, exclusions and data-quality findings are in [RELEASED_DATA.md](docs/RELEASED_DATA.md). Public notebooks display its aggregate boundary without loading row-level data.

Saved public review needs no model loading or instance resize. Full frozen-encoder experiments require a bounded compute plan and verified checkpoints. The recorded small Studio app is suitable for review, not simultaneous encoder workers. Preserve the Studio space and project snapshots when stopping compute.
