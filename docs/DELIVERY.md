# Download and run the completed project

The research model and the competition entry have different training-data boundaries. Both are preserved; neither is submitted automatically.

| Deliverable | Location | Purpose |
| --- | --- | --- |
| Submission notebook | [`kaggle/submission.ipynb`](../kaggle/submission.ipynb) · [download](https://github.com/alvaromendizabal/jigsaw-rule-classifier/raw/refs/heads/main/kaggle/submission.ipynb) | Original-training-only lexical reference; generates `submission.csv` |
| Results and example walkthrough | [`03_saved_results.ipynb`](../notebooks/03_saved_results.ipynb) | Executed research results and four authored inference examples |
| Accepted offline model | Private S3 bundle described below | Exact accepted post-competition model; generates `predictions.csv` |

## Your Kaggle submission

Download the canonical notebook from GitHub, or download its identical S3 copy at `releases/v1.0.0/kaggle/submission.ipynb` in `sagemaker-jigsaw-rules-560403859723-us-west-2`. The [delivery record](../reports/checkpoints/delivery.json) records its verified checksum and S3 version. No AWS training archive needs to be opened to find this notebook.

For the competition workflow, import the notebook into Kaggle, attach the original competition data, keep Internet disabled, and save a successful run. The notebook writes `/kaggle/working/submission.csv`, with exactly `row_id` and `rule_violation`. The competition's Late Submission interface selects an eligible saved notebook. It does not accept a model ZIP as an entry. The authenticated account had **no submissions found** when inspected on September 9, 2026; no leaderboard score is claimed.

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

This closes the scoped research and local inference product. Hosted serving, arbitrary-policy generalization and a Kaggle leaderboard result are separate future work. [Model card](../MODEL_CARD.md) · [Data card](../DATA_CARD.md).
