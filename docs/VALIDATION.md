# Verification record

Verified September 7, 2026. These are software checks, not a Kaggle performance claim.

| Check | Observed result |
| --- | --- |
| Fresh project environment from `uv sync --locked --group dev` | Passed; Python 3.12.14, pinned dependencies installed |
| Full `bash bootstrap.sh` | Passed; `BOOTSTRAP_COMPLETED` |
| Python compilation | Passed |
| Ruff and formatting | Passed |
| Automated tests | 31 passed; warnings treated as errors |
| Generated notebook consistency | Passed |
| All four notebooks, sequential code-cell execution | Passed using the explicit in-process IPython engine; output notebooks retained |
| Standard Jupyter kernel launch in this workspace | Blocked by the workspace's socket restrictions; not claimed as passed |
| Actual Kaggle dataset download | Pending authentication in the user's new Studio space |
| Actual Kaggle-hosted offline execution/scoring | Pending |
| GitHub Actions and pull-request merge | Pending repository creation |
| SageMaker space | API verified `InService`, 30 GB disk, private, no apps running |
| S3 settings | API verified versioning enabled, AES-256 encryption, all public-access blocks enabled, HTTPS-only bucket policy |
| Studio role S3 permissions | IAM simulation allowed list/read/write; live runtime access must be exercised from Studio |
| S3 backup/restore implementation | In-memory tests passed: interrupted transfer, manifest preservation, reuse, path traversal, corruption, and local conflict protection |

## Tests cover meaningful failure modes

- Macro per-rule AUC versus pooled AUC, equal rule weighting, undefined AUC, and invalid probabilities.
- Missing columns, invalid labels, blank text, duplicate IDs, and exact submission row order.
- Held-out-rule isolation, duplicate-comment grouping, training-example leakage purge, vocabulary isolation, and determinism.
- Completed-stage reuse without refitting, output corruption, interrupted stage recovery, heartbeat events, and data/config fingerprint changes.
- Interrupted file downloads resume at file boundaries; manually supplied files need no network download.
- S3 uploads commit a manifest only after successful object transfers; restore checks integrity and protects unrelated local files.

## Evidence boundaries

The archive includes executed synthetic demonstration notebooks and an offline report. They are visibly labeled. No synthetic metric should be described as a real-data score. The raw competition CSVs are not present in this release.

The CI workflow retains ordinary Jupyter execution as a gate after publication. The in-process engine verifies real notebook cell logic and captures rich outputs, but it does not validate kernel transport, browser integration, or SageMaker-specific networking. Those facts must not be inferred from the passing local tests.

The baseline resumes completed folds and stages, not interrupted solver iterations. Neural optimizer and RNG checkpointing is a later phase. Source code and data fingerprints intentionally invalidate incompatible old results.
