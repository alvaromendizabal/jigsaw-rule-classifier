# Verification record

Verified September 7, 2026. These are software checks, not a Kaggle performance claim.

| Check | Observed result |
| --- | --- |
| Fresh project environment from `uv sync --locked --group dev` | Passed; Python 3.12.14, pinned dependencies installed |
| Full `bash bootstrap.sh` | Passed; `BOOTSTRAP_COMPLETED` |
| Python compilation | Passed |
| Ruff and formatting | Passed |
| Automated tests | 38 passed; warnings treated as errors |
| Generated notebook consistency | Passed |
| All five notebooks, sequential code-cell execution | Passed using the explicit in-process IPython engine; output notebooks retained |
| Standard Jupyter kernel launch in this workspace | Blocked by the workspace's socket restrictions; not claimed as passed |
| Actual Kaggle dataset download | Completed in Studio; original file hashes recovered from S3 |
| Actual Kaggle-hosted offline execution/scoring | Pending |
| GitHub Actions | Tests and ordinary Jupyter execution passed in [the initial run](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/runs/34071531044); [Quality](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml) records each revision |
| Pull-request review and merge | [PR #1](https://github.com/alvaromendizabal/jigsaw-rule-classifier/pull/1) records the release and its checks |
| SageMaker space | API verified private 30 GB space; CPU JupyterLab app running |
| S3 settings | API verified versioning enabled, AES-256 encryption, all public-access blocks enabled, HTTPS-only bucket policy |
| Studio role S3 permissions | Actual completed S3 snapshots recovered; download hashes verified |
| S3 backup/restore implementation | In-memory tests passed: interrupted transfer, manifest preservation, reuse, path traversal, corruption, and local conflict protection |

## Tests cover meaningful failure modes

- Macro per-rule AUC versus pooled AUC, equal rule weighting, undefined AUC, and invalid probabilities.
- Missing columns, invalid labels, blank text, duplicate IDs, and exact submission row order.
- Held-out-rule isolation, duplicate-comment grouping, training-example leakage purge, vocabulary isolation, and determinism.
- Completed-stage reuse without refitting, output corruption, interrupted stage recovery, heartbeat events, and data/config fingerprint changes.
- Interrupted file downloads resume at file boundaries; manually supplied files need no network download.
- S3 uploads commit a manifest only after successful object transfers; restore checks integrity and protects unrelated local files.

## Evidence boundaries

The repository publishes reviewed aggregate metrics from the completed competition-data run, with original source/data hashes. Private OOF predictions were recovered and used to recalculate those metrics; the saved results matched. Tests and CI notebooks use explicitly labeled synthetic data. Raw competition CSVs and per-row predictions are excluded from Git.

The CI workflow executes notebooks using ordinary Jupyter kernels, with CurveZMQ encryption required. The in-process engine verifies cell logic and captures rich outputs, but it does not validate kernel transport, browser integration, or SageMaker-specific networking. The initial CI run passed and exposed old Action runtime and plaintext-kernel warnings; the release updates Actions to pinned Node 24 versions and requires encrypted kernel connections. Warning suppression is not used.

The baseline resumes completed folds and stages, not interrupted solver iterations. Neural optimizer and RNG checkpointing is a later phase. Source code and data fingerprints prevent reusing old model artifacts as a new experiment. Historical completed results remain reviewable without retraining through `jigsaw review`.

The added review tests reject synthetic results by default, verify artifact checksums, recompute metrics from OOF predictions, reject inconsistent provenance and unsafe paths, and assert that viewing saved results neither fits a model nor modifies its run directory.
