# Continue the Jigsaw project

Your real-data baseline is complete. Its run ID is `c15c2c2318fc0ed619c6`. Use this Git repository as the source of truth, restore the saved experiment, and open notebook 03. Existing Kaggle authentication and accepted competition rules do not need to be repeated.

## 1. Open the existing AWS workspace

Use **SageMaker AI → Studio → JupyterLab → jigsaw-rules-dev**, in **Oregon (`us-west-2`)**. The space already has persistent disk and a working CPU app. Keep this CPU instance for reviewing results. GPU model work is a separate next phase.

## 2. Connect the source code to Git

The first setup used an archive at `$HOME/jigsaw-rule-classifier`. Clone the Git repository under `$HOME/projects/` so the original working directory stays intact. Both directories use the normal project name.

In the JupyterLab **Terminal**, run:

```bash
export PATH="$HOME/.local/bin:$PATH"
PROJECT_DIR="$HOME/projects/jigsaw-rule-classifier"
if [ ! -d "$PROJECT_DIR/.git" ]; then
  git clone https://github.com/alvaromendizabal/jigsaw-rule-classifier.git "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"
git pull --ff-only
if [ ! -f configs/local.json ]; then
  cp "$HOME/jigsaw-rule-classifier/configs/local.json" configs/local.json
fi
bash bootstrap.sh
uv run jigsaw restore
uv run jigsaw review --run-id c15c2c2318fc0ed619c6
```

The configuration copy reuses the private bucket settings from the original project. If you originally extracted the archive elsewhere, use that directory in the `cp` line. The configuration is ignored by Git. A fresh contributor instead creates it from `configs/local.example.json` using their own bucket.

Bootstrap installs the locked environment, registers **Python (Jigsaw Rules)**, and runs the quality checks. Restore verifies checksums and reuses matching local files. It refuses to overwrite different files. The final review checks the old run without fitting any model, even though the current repository source has changed.

Expected output:

```text
BOOTSTRAP_COMPLETED
... REVIEW_VERIFIED ... data_kind: competition
seen_rule    comment_only   rule_macro_auc=0.728140
seen_rule    rule_examples  rule_macro_auc=0.728672
heldout_rule comment_only   rule_macro_auc=0.604086
heldout_rule rule_examples  rule_macro_auc=0.615563
REPORT .../reports/private/report.html
RESULTS .../reports/private/results.json
```

## 3. View the real evidence

Open `projects/jigsaw-rule-classifier/notebooks/03_saved_results.ipynb` in JupyterLab. Select **Python (Jigsaw Rules)** and **Run → Run All Cells**. It shows the run identity, metrics, comparison chart, and links to verified exports.

The exported `reports/private/results.json` explicitly says `data_kind: competition`. If sharing a result in chat, use that file or its accompanying HTML. Synthetic smoke tests are labeled `synthetic` and are rejected by the review command unless explicitly enabled.

**Do not rerun notebook 02 merely to open your old results.** It runs an experiment under the current code fingerprint. Notebook 03 reviews completed work.

## 4. Next modeling phase

[PHASE_2.md](docs/PHASE_2.md) predeclares the semantic experiments: an embedding/example matcher, then a rule-conditioned cross-encoder, compared on the same split memberships. Held-out-rule AUC **0.6156** is the reference to improve, alongside per-rule ranking, probability quality, latency, and memory.

The next implementation must pin model revisions and neural dependencies, checkpoint embedding batches, and pass an interruption/resume test and a bounded hardware smoke test before a full run. No neural model or GPU job is included in this release. The next experiment will be developed through its own pull request.

## Continued use

- Review the latest completed real experiment: `uv run jigsaw review`.
- Save new data and experiment artifacts: `uv run jigsaw backup`.
- Run or resume a new baseline: `uv run jigsaw baseline --cloud`.
- Verify the source: `uv run python scripts/verify.py`.
- Review changes: use a feature branch and pull request; merge after Quality passes.

When finished using Studio, stop the JupyterLab app to stop compute billing. Space disk and S3 snapshots persist; storage remains billable. Git keeps source history. S3 preserves committed experiment checkpoints; the active incomplete fold restarts after an interruption.

## Kaggle later

The standalone `kaggle/submission.ipynb` creates `submission.csv` offline with the required columns. Import it into Kaggle, attach the competition data, turn internet off, and save a completed notebook version. Late submission availability must be checked in the signed-in account. The original competition ended in 2025; this release does not claim a leaderboard score or medal.
