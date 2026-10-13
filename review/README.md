# One comment. Different rules. Explicit evidence.

**Alvaro Mendizabal · NLP / policy transfer / evaluation engineering**

Rule-conditioned classification with support-adapted language models, leakage-aware evaluation, and checkpointed research. Review how the system distinguishes policy transfer from memorizing overlapping comments.

[Project overview](../README.md) · [Printable inspection brief](index.html) · [Review configuration](project.json)

## Review the engineering, not just the headline

| Focus | Decision to inspect | Follow the proof |
|---|---|---|
| Applied AI | **Condition on the policy, not only the comment.** Represent supplied rules and support examples explicitly rather than treating moderation as a fixed global label. | [Code](../src/jigsaw_rules/formatting.py) · [Tests](../tests/test_formatting.py) · [Evidence](../reports/checkpoints/support_adaptation.json) |
| Data science | **Make query/support leakage an explicit failure mode.** Use policy-aware evaluation and purge overlapping query text from fitted or support sources. | [Code](../src/jigsaw_rules/splits.py) · [Tests](../tests/test_splits_model.py) · [Evidence](../docs/REPRODUCIBILITY.md) |
| ML engineering | **Preserve resumable inference and training boundaries.** Keep run identities, checkpoints, and reporting contracts in the execution layer instead of relying on notebook state. | [Code](../src/jigsaw_rules/runtime.py) · [Tests](../tests/test_runtime_pipeline.py) · [Evidence](../docs/REPRODUCIBILITY.md) |
| Data science | **Distinguish a matched effect from a later aggregate score.** Reconcile saved submission comparisons and development evidence without merging incompatible populations. | [Code](../scripts/review_portfolio.py) · [Tests](../tests/test_complete_review.py) · [Evidence](../reports/checkpoints/kaggle_adaptation.json) |

The linked tests are review entry points, not a claim that this compact command executes the full research suite. The project’s canonical reproducibility guide describes the broader environment and tests.

## One command, an inspectable result

From a checkout with Python 3.11 or newer:

```bash
python3 -S review/run.py --output /tmp/ml-review
```

Open `/tmp/ml-review/index.html`. The directory contains an accessible HTML brief, `receipt.json`, `manifest.json`, and logs for each executed command. No Python package installation, model download, cloud credentials, or external dataset is required. On Windows, substitute a writable directory outside the checkout for `/tmp/ml-review`.

To execute the browser engine checks as well, install Node.js and use a separate output directory:

```bash
python3 -S review/run.py --output /tmp/ml-review-node --with-node
python3 -S review/run.py --output /tmp/ml-review-node --with-node --check
```

The default path does not execute Node checks and never labels them as passed. CI runs both the Python and browser checks on Python 3.11 and 3.13. It retains the exact review artifacts with the workflow run.

| Check | Runtime | Inspected entry point |
|---|---|---|
| `publication-contract` | python | [`tools/check_public_demo.py`](../tools/check_public_demo.py) |
| `browser-policy-engine` | node | [`tools/test_public_demo.mjs`](../tools/test_public_demo.mjs) |

## What is actually reproduced

The dependency-free check verifies the browser publication boundary. Optional Node checks execute the actual TF-IDF policy engine, duplicate purging, conflicting-support rejection, and controls. Historical neural results are reviewed separately through saved evidence.

A repeat invocation verifies the saved bundle rather than rerunning completed checks. `--check` performs verification only. A changed linked source file, missing evidence, changed artifact, failed command, invalid path, or mismatched check mode causes a nonzero exit. Use a fresh output directory after an intentional source change; stale evidence is never silently relabeled as current.

Each command has a default 90-second deadline, configurable with `--timeout`. The runner records UTC time, interpreter version, elapsed command times, command identities, and SHA-256 hashes for its configuration, template, invoked entry points, linked implementations, tests, and evidence. Failed runs keep diagnostic logs in a separately named failed-run directory and never produce a success receipt.

The fingerprint covers this curated review surface, **not every transitive dependency or every repository file**. Checksums establish consistency, not independent authentication of historical scores. HTML links follow the current default branch; the receipt is the record of the local bytes inspected. The static page is explicitly an inspection brief, not an execution receipt.

## Public/private boundary

Private comments, labels, model weights, embeddings, prediction rows, exact ensemble specifications, and active inference orchestration remain excluded. Synthetic margins are not calibrated probabilities.

This review adds no training result and changes no historical score, selection decision, notebook output, source attribution, or license. A synthetic demonstration, an aggregate evidence check, public reference training, and exact private-model replay are different reproducibility levels.

## Verify the review kit itself

```bash
python3 -B -S -m unittest discover -s review -p test_run.py -v
python3 -B -S -O -m unittest discover -s review -p test_run.py
python3 -S review/run.py --check-page
```

After an intentional configuration or template change, regenerate the static brief with `python3 -S review/run.py --render`, inspect it, then run the checks above. The kit needs no JavaScript or network access to render its HTML. It uses semantic headings, keyboard-visible links, a skip link, responsive layouts, and print styling.
