## Why

This project needs a reproducible starting point for rule-conditioned comment classification. Random validation alone can overstate performance on new policies, and interrupted runs should not discard completed work.

## Changes

- Add strict competition data/submission contracts and purged grouped/held-out-rule validation.
- Compare a comment-only lexical model with an example-context lexical model.
- Report rule macro AUC, pooled AUC, average precision, log loss, Brier, calibration, and fixed-threshold metrics.
- Add timestamped progress, heartbeats, atomic fold commits, checksum verification, and S3 snapshot/restore.
- Add narrative notebooks, an offline review report, and a generated standalone Kaggle notebook.
- Add pinned dependencies, automated tests, and a GitHub Actions quality workflow.

## Validation

See `docs/VALIDATION.md` for executed checks and their exact boundary. Synthetic tests verify implementation; they do not establish real competition performance. A successful remote GitHub Actions run is still required after publication.

## Remaining work

Authenticate Kaggle inside the new Studio space, run the real baseline, and inspect its report. Advance to semantic and instruction models only after reviewing those results. Authenticated late-submission eligibility remains unverified.
