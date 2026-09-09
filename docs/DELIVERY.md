# Download and run the completed project

The research model and the competition entry have different training-data boundaries. Research and offline delivery are complete. Kaggle Version 2 has run successfully and is ready for the final submission action; no Kaggle score exists yet.

| Deliverable | Location | Purpose |
| --- | --- | --- |
| Submission notebook | [`kaggle/submission.ipynb`](../kaggle/submission.ipynb) · [download](https://github.com/alvaromendizabal/jigsaw-rule-classifier/raw/refs/heads/main/kaggle/submission.ipynb) | Original-training-only lexical reference; generates `submission.csv` |
| Results and example walkthrough | [`03_saved_results.ipynb`](../notebooks/03_saved_results.ipynb) | Executed research results and four authored inference examples |
| Accepted offline model | Private S3 bundle described below | Exact accepted post-competition model; generates `predictions.csv` |

## Your Kaggle submission

The notebook is already uploaded and successfully executed in your account: **[Jigsaw - Original Training Reference · Version 2](https://www.kaggle.com/code/alvaromendizabal/jigsaw-original-training-reference/output?scriptVersionId=348605174)**. It ran in 36 seconds on CPU with Internet off and generated the 10-row preview `submission.csv`. The final submission action remains pending; this is not a hidden-test score. [Current submission record](../reports/checkpoints/kaggle_submission.json).

To finish from the saved version:

1. Open the Version 2 output link above while signed in as `alvaromendizabal`.
2. Click **Submit to Competition** beside `submission.csv`.
3. Check **Notebook Version: Kaggle input path fix - GitHub 7e98f91** and **Output File: submission.csv**. Leave the output at the root file, rather than a checkpoint batch.
4. Optionally enter `Original-training-only lexical reference, Version 2` as the description, then click **Submit**.
5. Open the competition's **Submissions** tab and wait for **Complete** and a score. Kaggle reruns the notebook on hidden test data; a saved preview run alone is not the submission.

No AWS download or new notebook upload is needed for these steps. To obtain the current notebook separately, download the canonical file from GitHub using the table above. The original S3 snapshot at `releases/v1.0.0/kaggle/submission.ipynb` in `sagemaker-jigsaw-rules-560403859723-us-west-2` predates the current Kaggle mount fix. Its historical checksum and version remain in the [delivery record](../reports/checkpoints/delivery.json); use GitHub or saved Kaggle Version 2 for the corrected notebook.

Version 1 failed because Kaggle mounted its official files under `/kaggle/input/competitions/jigsaw-agile-community-rules`, while the original code expected `/kaggle/input/jigsaw-agile-community-rules`. The current notebook recognizes both layouts, honors an explicit local input override, and tells you to attach the official competition when required files are absent. No model or prediction algorithm changed.

For local or SageMaker use, open the same notebook and run all cells with `GENERATE_SUBMISSION = True`; its final cell provides the CSV download. The original downloadable preview contains **10 rows**, while original training contains **2,029 rows**. A 10-row local preview verifies software and is not the hidden competition test or a score. Kaggle supplies its evaluation input when running a submitted notebook. CI executes this notebook on both synthetic and hash-verified original preview data, then checks unchanged output on replay. [Execution proof](../reports/checkpoints/submission.json).

The accepted model below uses additional organizer-released post-competition labels. Its 0.7770 benchmark AUC must not be presented as this reference notebook's score.

## Restore the accepted model

Download `s3://sagemaker-jigsaw-rules-560403859723-us-west-2/releases/v1.0.0/offline-model.tar` using the connected AWS account. The private archive is **1,227,560,960 bytes**; SHA-256 is `f67d6cddefc6a9869889451816f0b8a056b45a6e78a582c185affb35ca55bda1`. It contains the fitted candidate, exact encoder assets, runtime source, dependency lock and checksum manifest. It contains no raw training or confirmation rows. S3 encryption, object checksum, version and an independent full download were verified, including all 57 payload hashes.

From a project checkout with the downloaded archive in its root:

```bash
sha256sum offline-model.tar
mkdir -p runs/delivery/4ce868936ca14fdad7df
tar -xf offline-model.tar -C runs/delivery/4ce868936ca14fdad7df
uv sync --locked --extra semantic --group dev
```

Compare the archive checksum above before extraction. Initial dependency installation can use the network. Once the locked dependencies are cached, `uv sync --offline --locked --extra semantic` also works; a new virtual environment was restored this way and ran the extracted package with network connections disabled. The archive does not contain a universal dependency wheelhouse. [Fresh environment proof](../reports/checkpoints/offline_restore.json).

For your own inputs, supply `row_id`, `body`, `rule`, `subreddit`, `positive_example_1`, `positive_example_2`, `negative_example_1` and `negative_example_2`. All four support examples are required. Target-bearing inputs are rejected. Generate benchmark probabilities with:

```bash
uv run --offline --extra semantic python scripts/offline_inference.py predict \
  --bundle runs/delivery/4ce868936ca14fdad7df \
  --manifest-sha256 dbf1428343c2e349590c4ceec32a840891e796d67005948d8cf3c22e535da18a \
  --input your_comments.csv \
  --output runs/offline_output
```

The output is `runs/offline_output/predictions.csv` plus its manifest. Intact completed batches and embedding shards are reused. Changed inputs use a new identity; corrupt committed artifacts are rejected. Missing support produces a validation error before encoder loading. The external manifest checksum is required before deserializing the trusted private candidate.

To try the four authored examples without preparing a CSV, open notebook `03`, restore the bundle at the path above, set `RUN_LOCAL_DEMO = True`, and run its final demo cell. The executed default walkthrough already displays the four predictions and closest supplied examples; it requires no model download to read.

## Measured delivery checks

On an AMD EPYC 9V74 host with nine visible logical CPUs and four PyTorch threads, the real offline acceptance run passed all seven frozen budgets/guards. The same six target-free rows, one per confirmed policy, differed from saved cloud probabilities by at most **0.00000122**; identical vectors gave exactly identical probabilities. Batch/order parity and completed-work reuse passed. No model was fitted and no targets were opened.

| Measurement | Observed | Declared budget |
| --- | ---: | ---: |
| Cold verification, model loading and first six predictions | 28.32 s | 60 s |
| Warm authored batch, mean per comment | 1.11 s | 10 s |
| Peak process memory | 3.88 GiB | 8 GiB |
| Whole acceptance run | 37.95 s | 180 s |

The warm measurement is a mean over four authored examples, not a tail-latency or production-throughput estimate. A separate fresh-environment run produced the same four probabilities within `1.12e-16`, in 17.10 seconds with an empty encoder cache and zero network calls. The serving samples do not add accuracy evidence to the protected comparison. [Machine-readable measurements](../reports/delivery/verification.json).

This closes the scoped research and local inference product. Hosted serving and arbitrary-policy generalization remain outside this scope. The saved Kaggle entry is ready for its final submission action; a leaderboard result is still pending. [Model card](../MODEL_CARD.md) · [Data card](../DATA_CARD.md).
