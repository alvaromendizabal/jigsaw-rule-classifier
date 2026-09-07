# Release review

The project needs a reproducible reference implementation and an accurate record of the completed experiment. The originally shared files were synthetic examples, while the real baseline had completed in AWS. Rerunning training to inspect that evidence would waste work and obscure provenance.

This release publishes the exact baseline source in its foundation commit, then adds verified saved-run review, explicit dataset labels, public aggregate results/charts, a review notebook, and instructions for continuing from Git and S3. The original completed run remains unchanged.

Quality checks cover data/submission contracts, leakage controls, metrics, stage recovery, download recovery, S3 snapshot integrity, and review without fitting. CI executes five notebooks on synthetic data in an ordinary Jupyter kernel and retains logs and executed notebooks as artifacts.

Real cross-validation is published separately from synthetic tests. Familiar-rule macro AUC is 0.72867 and held-out-rule macro AUC is 0.61556 for the contextual baseline. These are not leaderboard results. No neural model, GPU job, or Kaggle submission is claimed. See `docs/PHASE_2.md` for the next experiment and `docs/VALIDATION.md` for verification boundaries.
