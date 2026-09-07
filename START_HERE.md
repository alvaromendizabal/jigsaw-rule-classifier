# Start the Jigsaw project

Your next practical milestone is **one real-data baseline run with a saved report**. The advanced model phase begins after that evidence is available.

## 1. Open the space already created for you

In **AWS → SageMaker AI → Studio**, choose region **Oregon (`us-west-2`)**, the existing domain ending in `erusdc`, and its default user. Open **JupyterLab → Jigsaw Rule Classification (`jigsaw-rules-dev`)**.

The space has 30 GB of persistent disk. Start a small CPU instance, such as the smallest CPU option offered by the Studio selector. This dataset and baseline do not need a GPU. Starting the app incurs compute charges; stopping it preserves the space disk but storage remains billable. The domain's user defaults specify 60-minute idle shutdown; confirm the space's effective idle setting when starting it. No app or training job was launched by this setup.

The private, encrypted, versioned S3 bucket is configured in the bundled `configs/local.json`. This file is excluded from Git. Existing role policy simulation allowed S3 listing, reading, and writing; the first backup tests actual access from the running Studio role.

## 2. Place the project in this space

Download `jigsaw-rule-classifier.zip`, then upload it to the JupyterLab home folder. Open a **Terminal** and run this whole block:

```bash
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME"
python3 -m zipfile -e jigsaw-rule-classifier.zip "$HOME"
cd "$HOME/jigsaw-rule-classifier"
bash bootstrap.sh
```

Bootstrap creates an isolated environment and notebook kernel, then runs the quality gate. It emits timestamps and heartbeats during installation and verification. Expected final marker: **`BOOTSTRAP_COMPLETED`**. Invoke it with `bash`; do not paste its internal shell settings into your interactive terminal.

This is the initial project archive. After creating the GitHub repository, use Git branches and edit the original filenames for subsequent changes. Do not repeatedly unpack archives over modified work.

## 3. Authenticate Kaggle and download its files

You already accepted the rules in your Kaggle account. This new AWS space still needs that account's local authentication.

```bash
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/jigsaw-rule-classifier"
uv run kaggle auth login
uv run jigsaw download
uv run jigsaw backup
```

Follow the official CLI's browser login instructions. If you already installed a supported Kaggle credential in this space, skip `auth login` and run the download directly. Do not re-create credentials unless the CLI reports that authentication is missing or expired. Your email is not an API credential.

The downloader retrieves the three CSVs separately and records their hashes. Completed downloads are reused. If you prefer Kaggle's **Download All**, extract its three CSVs into `data/raw/`, then run `uv run jigsaw audit` and `uv run jigsaw backup`.

Expected result: `train.csv`, `test.csv`, and `sample_submission.csv` pass schema checks, followed by a `snapshot_committed` event from S3.

## 4. Run the explanatory notebooks in order

In JupyterLab, select **Python (Jigsaw Rules)** as the kernel. Use **Run → Run All Cells** on each notebook, in order:

1. `notebooks/00_environment_and_data.ipynb`
2. `notebooks/01_data_and_validation.ipynb`
3. `notebooks/02_baseline_and_review.ipynb`

Notebook 02 trains two CPU models across both validation protocols, generates predictions, and saves an interactive HTML review. Cloud snapshots are on by default in this notebook. You can instead run the whole experiment from the terminal:

```bash
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/jigsaw-rule-classifier"
uv run jigsaw baseline --cloud
```

A `heartbeat` means the process is alive; it is not proof a model is improving. Fold and stage completion events show concrete progress. `elapsed_seconds` reports time since the named stage began. Expected final marker: **`RUN_COMPLETED`**. The following `REPORT` line gives the exact report location.

If the run stops, rerun the same command. If you stop or restart the same space, its local data and completed stages remain on disk. To recover into a fresh project directory on a different machine, configure the same S3 bucket and use:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run jigsaw restore
uv run jigsaw baseline --cloud
```

Restore refuses to replace different local content. Use a fresh directory when restoring an older snapshot. Only files in the latest committed snapshot are recoverable; the active incomplete fold restarts. S3 data restoration does not restore source code—retain Git history and this archive.

## 5. Create the public GitHub repository

The linked GitHub connection here can work with existing repositories but does not expose a create-repository action. Create it on GitHub using:

**Name:** `jigsaw-rule-classifier`

**Description:** `Rule-conditioned NLP with unseen-rule validation, resumable AWS experiments, calibrated evaluation, and offline Kaggle inference.`

Choose **Public** and **Add a README**. Then send the repository link in this conversation. I can use that existing repository to publish the tested files on a feature branch, open a documented pull request, check CI, and merge once its required checks pass. Those remote actions have not happened yet. No terminal token setup is required for this handoff.

The archive includes `docs/PULL_REQUEST.md` with the change rationale and verification boundary. Git ignores `configs/local.json`, raw data, experiments, models, and tokens. Real-data notebook outputs require review before public commits.

## 6. What to return before Phase 2

Send the new GitHub link and either the generated `review/report.html` or its `review/results.json`, plus the last progress lines if anything stopped. These are the inputs needed to choose the next model and GPU budget from evidence.

Do not spend time on GPU fine-tuning yet. Phase 2 will compare semantic embeddings and a cross-encoder under the same validation protocol; Phase 3 will introduce LoRA and checkpointed neural training.

## Later: the Kaggle notebook

Import `kaggle/submission.ipynb` into Kaggle. Attach **Jigsaw – Agile Community Rules Classification**, select CPU, turn **Internet off**, and **Save Version → Save & Run All**. It uses the canonical baseline model code and predicts every row in the test file supplied at execution.

If your signed-in account enables **Late Submission**, choose the completed notebook version and its `submission.csv`. This setup has not verified that account-level permission or submitted anything. The original competition ended in 2025; new medals are not available for late work.
