"""Build reproducible narrative notebooks from canonical package code."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python (Jigsaw Rules)",
        "language": "python",
        "name": "jigsaw-rules",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
    nb.cells = []
    for kind, source in cells:
        cell = nbf.v4.new_markdown_cell(source) if kind == "md" else nbf.v4.new_code_cell(source)
        cell["id"] = hashlib.sha256((kind + source).encode()).hexdigest()[:12]
        nb.cells.append(cell)
    return nb


SETUP = 'import os\nfrom pathlib import Path\nimport pandas as pd\nfrom IPython.display import display\nfrom jigsaw_rules.review import public_evidence\nfrom jigsaw_rules.runtime import environment\n\nroot = Path(os.environ.get("JIGSAW_ROOT", Path.cwd())).resolve()\nif root.name == "notebooks":\n    root = root.parent\nbaseline = public_evidence(root, "baseline")\nsemantic = public_evidence(root, "semantic")\nassert baseline["training_sha256"] == semantic["training_sha256"]\nprint("Competition data | 2,029 training rows | recorded local cross-validation")\nprint("Verification: aggregate file checksums and provenance; no model fitting.")\n\nnames = {"comment_only": "Comment-only TF-IDF", "rule_examples": "Rule/example TF-IDF",\n         "semantic_margin": "Frozen semantic margin", "semantic_classifier": "Semantic classifier"}\nprotocols = {"seen_rule": "Familiar rules", "heldout_rule": "Held-out rule"}\n\ndef metric_table(records):\n    return pd.DataFrame([{"Model": names[r["model"]], "Validation": protocols[r["protocol"]],\n        "Rule macro AUC": r["metrics"]["rule_macro_auc"],\n        "Log loss": r["metrics"]["log_loss"], "Brier": r["metrics"]["brier"],\n        "Average precision": r["metrics"]["average_precision"]} for r in records]).round(4)\n'


def notebooks():
    outputs = {}
    outputs["notebooks/00_environment_and_data.ipynb"] = notebook(
        [
            (
                "md",
                "# 00 · Environment and data\n\n**Question:** Which data and environment produced the recorded experiment?\n\nThis notebook renders recorded **competition-data** evidence. It verifies the committed artifact hashes, not private out-of-fold predictions. No credentials, raw comments, weight downloads, or training are needed. Full private metric recomputation remains `uv run jigsaw review`.",
            ),
            ("code", SETUP),
            (
                "md",
                "## Reproduction environment\nThe original experiment environment and this notebook-execution environment are reported separately. `uv.lock` pins the project environment; rendering a historical result does not rerun that experiment.",
            ),
            (
                "code",
                'recorded = baseline["provenance"]["environment"]\ncurrent = environment()\nprint("Recorded experiment Python:", recorded["python"])\nprint("Notebook execution Python:", current["python"])\ndisplay(pd.DataFrame({"Recorded experiment": recorded["packages"],\n                      "Notebook execution": current["packages"]}))',
            ),
            (
                "md",
                "## Recorded data contract\nOne row pairs a comment with a rule and examples. These are the saved audit counts, not a fresh read of raw CSV files. The preview test is not an independent evaluation set.",
            ),
            (
                "code",
                'audit = baseline["audit"]\ndisplay(pd.DataFrame({"Recorded file": ["train.csv", "test.csv (preview)"],\n                      "Rows": [audit["train_rows"], audit["preview_test_rows"]]}))\nprint("Training SHA-256:", baseline["training_sha256"])\nprint("Lexical run:", baseline["run_id"])\nprint("Semantic run:", semantic["run_id"])',
            ),
            (
                "md",
                "## Continue\n[01 · Data and validation](01_data_and_validation.ipynb) explains the leakage controls. For the employer overview, start with [03 · Results](03_saved_results.ipynb). To audit raw files locally, use `uv run jigsaw audit`; raw data remain private.",
            ),
        ]
    )
    outputs["notebooks/01_data_and_validation.ipynb"] = notebook(
        [
            (
                "md",
                "# 01 · Data and validation\n\n**Question:** What does the validation design actually test?\n\nThis notebook renders recorded **competition-data** evidence. It verifies the committed artifact hashes, not private out-of-fold predictions. No credentials, raw comments, weight downloads, or training are needed. Full private metric recomputation remains `uv run jigsaw review`.",
            ),
            ("code", SETUP),
            (
                "md",
                "## Two observed rules, not broad policy coverage\nThe recorded training audit exposes different violation prevalence across the two rules. Count and rate are shown together so sample size stays visible.",
            ),
            (
                "code",
                'audit = baseline["audit"]\nby_rule = pd.DataFrame(audit["by_rule"]).rename(columns={"rule": "Rule", "size": "Rows", "mean": "Violation rate"})\nby_rule["Rule"] = by_rule["Rule"].str.split(":").str[0]\ndisplay(by_rule.round(4))\ndisplay(pd.DataFrame({"Audit finding": ["Duplicate training bodies", "Train / preview-test overlap"],\n                      "Count": [audit["duplicate_training_bodies"], audit["train_test_body_overlap"]]}))',
            ),
            (
                "md",
                "## Leakage controls\n**Familiar-rule validation** stratifies by rule and target while grouping normalized duplicate comments. **Held-out-rule validation** excludes all training rows from the evaluated rule. Both purge training rows whose body or supplied examples contain a validation body. Vocabulary and learned similarity classifiers are fitted only on retained training-fold rows.\n\nProvided validation examples remain legitimate inputs; their labels are not added as training observations. The semantic experiment reused the original saved splits. This aggregate-only view does not re-run the purging audit or reconstruct row assignments.",
            ),
            (
                "code",
                'config = baseline["provenance"]["config"]\ndisplay(pd.DataFrame({"Predeclared setting": ["Seed", "Familiar-rule folds", "Observed labeled rules"],\n                      "Value": [config["seed"], config["folds"], len(audit["train_rules"])]}))\nprint("Protocol limitations: exact-text isolation does not prove near-duplicate or shared-origin isolation.")',
            ),
            (
                "md",
                "## Metric contract\nThe official overview names **column-averaged AUC**. The project reports **rule macro ROC AUC**, the equal-weight mean of the rule-specific AUCs, and pooled AUC separately. The official overview does not expose executable scoring code.\n\nLog loss, Brier score, average precision, and calibration error diagnose different properties; AUC alone does not establish probability calibration. Threshold metrics use a predeclared 0.5 threshold. Only two labeled rules means two transfer cases, not a representative sample of future policies.\n\nRaw comment-length distributions and row-level errors are not inferred from these aggregate files. Continue to [02 · Baseline](02_baseline_and_review.ipynb).",
            ),
        ]
    )
    outputs["notebooks/02_baseline_and_review.ipynb"] = notebook(
        [
            (
                "md",
                "# 02 · Baseline and review\n\n**Question:** Does adding rule/example context improve the lexical reference?\n\nThis notebook renders recorded **competition-data** evidence. It verifies the committed artifact hashes, not private out-of-fold predictions. No credentials, raw comments, weight downloads, or training are needed. Full private metric recomputation remains `uv run jigsaw review`.",
            ),
            ("code", SETUP),
            (
                "md",
                "## Controlled feature comparison\n`comment_only` uses comment TF-IDF with logistic regression. `rule_examples` adds comment-to-rule/example cosine similarities, positive/negative maximum similarity, and their margin. Both are evaluated under the recorded split design. This is a lexical reference, not a claim of deep rule understanding.",
            ),
            ("code", 'display(metric_table(baseline["results"]))'),
            (
                "md",
                "![Lexical validation comparison](../reports/baseline/comparison.svg)\n\n## Held-out-rule feature effect\nThe difference below is descriptive. It is not a significance claim or a score from Kaggle.",
            ),
            (
                "code",
                'held = {r["model"]: r["metrics"] for r in baseline["results"] if r["protocol"] == "heldout_rule"}\ndisplay(pd.DataFrame([{"Metric": metric, "Rule/example minus comment-only": held["rule_examples"][metric] - held["comment_only"][metric]}\n    for metric in ["rule_macro_auc", "log_loss", "brier"]]).round(4))',
            ),
            (
                "md",
                "## Interpretation and reproducibility\nRule/example features modestly improve held-out ranking, while probability losses do not improve. The lexical model remains the reference for later candidates. Coefficients describe association, not causal effects. Raw examples and row-level errors are intentionally absent from the public notebook.\n\nTraining is explicit: `uv run jigsaw baseline --cloud`. That command fits or resumes a source-fingerprinted experiment; it is not required to read these results. Valid completed folds are reused, but an interrupted CPU solver restarts its active fold. Source changes can create a new experiment identity.\n\n[03 · Results](03_saved_results.ipynb) is the consolidated employer overview.",
            ),
        ]
    )
    outputs["notebooks/03_saved_results.ipynb"] = notebook(
        [
            (
                "md",
                "# 03 · Results and model decision\n\n**Question:** What did the completed experiments establish?\n\nThis notebook renders recorded **competition-data** evidence. It verifies the committed artifact hashes, not private out-of-fold predictions. No credentials, raw comments, weight downloads, or training are needed. Full private metric recomputation remains `uv run jigsaw review`.\n\n**Current decision:** retain the lexical rule/example reference. No semantic candidate has established an overall replacement, and no leaderboard or medal result is claimed.",
            ),
            ("code", SETUP),
            (
                "md",
                "## Compare model quality\nHigher rule macro AUC is better. Lower log loss and Brier score are better. Familiar-rule and held-out-rule results answer different questions; do not pool their conclusions.",
            ),
            (
                "code",
                'records = baseline["results"] + semantic["results"]\ndisplay(metric_table(records))',
            ),
            (
                "md",
                "![Recorded semantic and lexical comparison](../reports/semantic/comparison.svg)\n\n## Does the semantic margin improve transfer?\nThese paired bootstrap intervals resample normalized comment groups and condition on the two observed rules and fixed predictions. They do not quantify transfer to arbitrary new policies.",
            ),
            (
                "code",
                'intervals = pd.DataFrame(semantic["uncertainty"])\ndisplay(intervals.loc[intervals.protocol == "heldout_rule", ["model", "observed_delta", "ci_lower", "ci_upper", "draws"]].round(4))\nprint("Evidence verified for", baseline["training_rows"], "competition training rows.")\nprint("New training performed by this notebook: none.")',
            ),
            (
                "md",
                "## Decision\nThe frozen semantic margin has a **+0.0195** held-out AUC difference, with a paired 95% interval of **−0.0132 to +0.0491**. That interval includes no improvement, and probability losses worsen slightly. The learned semantic classifier performs worse. A negative experiment is retained rather than hidden.\n\nThe margin makes identical predictions in both validation protocols because it fits no fold labels. Those columns are not independent replications. The 10-row preview test checks inference plumbing, not generalization.\n\nContinue to [04 · Semantic benchmark](04_semantic_benchmark.ipynb) for per-rule diagnostics, runtime, and the next controlled experiment. Full OOF verification is `uv run jigsaw review` after restoring private artifacts.",
            ),
        ]
    )
    outputs["notebooks/04_semantic_benchmark.ipynb"] = notebook(
        [
            (
                "md",
                "# 04 · Semantic rule generalization\n\n**Question:** Where does a frozen semantic encoder help or fail?\n\nThis notebook renders recorded **competition-data** evidence. It verifies the committed artifact hashes, not private out-of-fold predictions. No credentials, raw comments, weight downloads, or training are needed. Full private metric recomputation remains `uv run jigsaw review`.\n\nThe recorded encoder is **Qwen3-Embedding-0.6B**, pinned to an immutable Hub revision. Qwen weights are frozen; only the similarity classifier learns fold labels. The semantic margin uses a predeclared temperature of 0.1 and is not claimed to be calibrated.",
            ),
            ("code", SETUP),
            (
                "md",
                "## Held-out behavior by rule\nAggregate improvements can conceal deterioration on individual policies. The table keeps the two observed rules separate.",
            ),
            (
                "code",
                'records = baseline["results"] + semantic["results"]\nper_rule = pd.DataFrame([{"Model": names[r["model"]], "Rule": rule.split(":")[0], "ROC AUC": auc}\n    for r in records if r["protocol"] == "heldout_rule" for rule, auc in r["metrics"]["per_rule_auc"].items()])\ndisplay(per_rule.pivot(index="Model", columns="Rule", values="ROC AUC").round(4))',
            ),
            (
                "md",
                "## Probability diagnostics\nCalibration error uses 10 equal-width bins. Precision, recall, and F1 use the fixed 0.5 diagnostic threshold; this is not a tuned deployment decision. Threshold selection and calibration fitting require nested validation.",
            ),
            (
                "code",
                'diagnostics = pd.DataFrame([{"Model": names[r["model"]],\n    "Pooled AUC": r["metrics"]["pooled_auc"], "Calibration error": r["metrics"]["ece_10_equal_width_bins"],\n    "Precision@0.5": r["metrics"]["precision_at_0_5"], "Recall@0.5": r["metrics"]["recall_at_0_5"],\n    "F1@0.5": r["metrics"]["f1_at_0_5"]} for r in records if r["protocol"] == "heldout_rule"])\ndisplay(diagnostics.round(4))',
            ),
            (
                "md",
                "## Recorded runtime and memory\nThese are measurements from the first successful model invocation, not the time required to render this notebook. They do not claim that earlier failed attempts cost no time. Truncation counts refer to unique encoded inputs.",
            ),
            (
                "code",
                'timing = semantic["timing"]\nencoder = timing["encoder"]\ndisplay(pd.DataFrame({"Measurement": ["First successful invocation (seconds)", "Encoding (seconds)", "Peak process memory (GiB)", "Unique inputs", "Truncated inputs"],\n    "Value": [timing["first_completion_wall_seconds"], encoder["encode_seconds"], encoder["peak_rss_gib"], encoder["unique_texts"], encoder["truncated_rows"]]}).round(3))',
            ),
            (
                "md",
                "## What to test next\nThe next model experiment should jointly encode rule, comment, and support examples. Predeclare comment-only, rule-plus-comment, and positive/negative-example ablations; preserve the split registry; then compare cross-encoder or parameter-efficient fine-tuning candidates with the unchanged lexical reference. More complexity is accepted only when measured evidence supports it.\n\nBefore any paid run, approve hardware, maximum duration, and spending limits. Current embedding checkpoints resume at completed shards, and CPU training resumes at completed folds—not inside an interrupted solver. GPU optimizer-state resume is a later deliverable.\n\nThe standalone [Kaggle notebook](../kaggle/submission.ipynb) is an offline lexical reference. A preview CSV is not a scored submission, and authenticated late-scoring eligibility remains unverified.\n\nReturn to [03 · Results and decision](03_saved_results.ipynb).",
            ),
        ]
    )
    feature_sections = [
        (
            "md",
            "## Next experiment · task-informed feature ablations\n\n**Hypothesis:** a rule violation depends on a comment's relationship to the supplied policy and its positive/negative examples, not toxicity alone. Word overlap, character similarity, and contrastive support statistics probe different relationships. Structural cues can also be shortcuts, so they are tested separately.\n\nFour predeclared candidates share the same training-only word vocabulary and saved, purged reference splits. They add (1) rule text similarity, (2) order-invariant positive/negative support contrasts, (3) writing structure, or (4) all groups. The combined candidate has 26 dense features plus the sparse comment representation. No validation labels enter vocabulary, scaling, or feature construction.\n\nRun `.venv/bin/python -m jigsaw_rules.cli features --cloud --export` in your existing SageMaker checkout. This is a new CPU experiment, not a rerun of the lexical/Qwen baselines. It makes **no submission CSV**. Completed folds are resumed; S3 checkpoints follow each fold. The fixed reference remains unchanged.",
        ),
        (
            "code",
            'import subprocess\nimport sys\nfrom jigsaw_rules.features import feature_evidence\n\n# Opt in only in your own configured workspace. Publication keeps this False.\nRUN_FEATURE_EXPERIMENT = False\nCHECKPOINT_TO_S3 = True\nif RUN_FEATURE_EXPERIMENT:\n    command = [sys.executable, "-m", "jigsaw_rules.cli", "features", "--export"]\n    if CHECKPOINT_TO_S3:\n        command.append("--cloud")\n    subprocess.run(command, cwd=root, check=True)\nfeature_run = feature_evidence(root)\nif feature_run is None:\n    print("Feature ablation code is ready; no competition-data ablation results are published yet.")\nelse:\n    feature_table = pd.DataFrame([{"Candidate": r["model"], "Protocol": r["protocol"],\n        "Rule macro AUC": r["metrics"]["rule_macro_auc"], "Log loss": r["metrics"]["log_loss"],\n        "Brier": r["metrics"]["brier"], "Original fit seconds": r["fit_seconds"]}\n        for r in feature_run["results"]])\n    display(feature_table.round(4))\n    display(pd.DataFrame(feature_run["audit"]).round(4))\n    intervals = pd.DataFrame(feature_run["uncertainty"])\n    display(intervals.loc[:, ["model", "protocol", "observed_delta", "ci_lower", "ci_upper"]].round(4))\n    import matplotlib.pyplot as plt\n    held = intervals.loc[intervals.protocol == "heldout_rule"].sort_values("observed_delta")\n    fig, ax = plt.subplots(figsize=(9, 4.5))\n    positions = range(len(held))\n    ax.hlines(positions, held.ci_lower, held.ci_upper)\n    ax.scatter(held.observed_delta, positions)\n    ax.axvline(0, linestyle="--", linewidth=1)\n    ax.set_yticks(list(positions), held.model.str.replace("_", " "))\n    ax.set_xlabel("Held-out rule macro AUC difference from the fixed lexical reference")\n    ax.set_title("Task-informed feature ablations | paired 95% group-bootstrap intervals")\n    ax.spines[["top", "right"]].set_visible(False)\n    fig.tight_layout()\n    from io import StringIO\n    from IPython.display import SVG\n    buffer = StringIO()\n    fig.savefig(buffer, format="svg")\n    display(SVG(buffer.getvalue()))\n    plt.close(fig)\n    coefficients = pd.DataFrame(feature_run["coefficients"])\n    combined = coefficients.loc[coefficients.model == "combined"]\n    display(combined.groupby(["protocol", "feature"]).coefficient.agg(["mean", "min", "max"]).round(4))\n',
        ),
        (
            "md",
            "### How to read the next results\nCompare each candidate against the preserved rule/example reference on the **same rows**. Inspect held-out rule macro AUC first, then each rule, log loss, Brier score, and paired comment-group intervals. These four-candidate exploratory intervals are not multiplicity-adjusted; they do not certify a winner. No automatic promotion, calibration claim, or leaderboard claim is made.\n\nWriting-structure associations and standardized dense coefficients are descriptive, not causal, and only two rules are observed. Coefficient ranges show sensitivity across folds; different regularization/scaling means this comparison is not solely a feature-count experiment. Candidate improvement here should motivate a separately validated joint rule/comment encoder, not a claim that lexical features solve arbitrary policy understanding.",
        ),
    ]
    nb = outputs["notebooks/02_baseline_and_review.ipynb"]
    outputs["notebooks/02_baseline_and_review.ipynb"] = notebook(
        [(c.cell_type.replace("markdown", "md"), c.source) for c in nb.cells] + feature_sections
    )
    model = (ROOT / "src/jigsaw_rules/model.py").read_text()
    model = model.replace("from __future__ import annotations\n", "").replace(
        "from jigsaw_rules.data import EXAMPLES\n", ""
    )
    data = (
        (ROOT / "src/jigsaw_rules/data.py")
        .read_text()
        .replace("from __future__ import annotations\n", "")
    )
    runtime = (
        (ROOT / "src/jigsaw_rules/runtime.py")
        .read_text()
        .replace("from __future__ import annotations\n", "")
    )
    submission = (ROOT / "src/jigsaw_rules/submission.py").read_text()
    submission = "\n".join(
        line
        for line in submission.splitlines()
        if not line.startswith("from jigsaw_rules.")
        and line != "from __future__ import annotations"
    )
    source_sha = hashlib.sha256((data + model + runtime + submission).encode()).hexdigest()
    outputs["kaggle/submission.ipynb"] = notebook(
        [
            (
                "md",
                "# Jigsaw · Generate and download your submission\n\nRun this notebook yourself to fit the unchanged lexical reference, generate predictions, validate the CSV, and display a **Download submission.csv** link. Nothing is uploaded or submitted to Kaggle automatically.\n\nWorks in SageMaker/Jupyter and Kaggle. On Kaggle, attach the official competition data and disable internet. A preview CSV is not a leaderboard score. This completed competition's late-scoring eligibility is not assumed.\n\n**Recovery:** completed model fitting and prediction batches are checksummed and reusable. An interrupted active fit or batch restarts; correct earlier work is kept. Private output and download payloads must never be committed to the public repository.",
            ),
            (
                "code",
                "from __future__ import annotations\nimport os\nos.environ['OMP_NUM_THREADS'] = '2'\nos.environ['OPENBLAS_NUM_THREADS'] = '2'",
            ),
            (
                "md",
                "## Canonical schema, model, and resumable runtime\nThese cells are generated from the tested package modules. No downloads, external model calls, or Kaggle API calls are required.",
            ),
            ("code", data),
            ("code", model),
            ("code", runtime),
            ("code", submission),
            (
                "md",
                "## Generate locally\n`GENERATE_SUBMISSION` controls this action. Running with `True` creates or resumes your own output; `False` performs no inference. Paths are detected from the project or Kaggle environment. The source/data/environment fingerprint prevents stale checkpoint reuse.",
            ),
            (
                "code",
                f'''GENERATE_SUBMISSION = True
candidates = [Path.cwd(), *Path.cwd().parents]
project = next((p for p in candidates if (p / "src/jigsaw_rules").is_dir()), None)
on_kaggle = Path("/kaggle/input").is_dir()
default_input = Path("/kaggle/input/jigsaw-agile-community-rules") if on_kaggle else (project or Path.cwd()) / "data/raw"
default_output = Path("/kaggle/working") if on_kaggle else (project or Path.cwd()) / "kaggle_output"
input_root = Path(os.environ.get("JIGSAW_KAGGLE_INPUT", str(default_input)))
output_root = Path(os.environ.get("JIGSAW_KAGGLE_OUTPUT", str(default_output)))
default_cache = project / "runs/submission_cache" if project and not on_kaggle else output_root / "checkpoints"
cache_root = Path(os.environ.get("JIGSAW_SUBMISSION_CACHE", str(default_cache)))
submission_path = None
if GENERATE_SUBMISSION:
    submission_path = generate_submission(input_root, output_root, cache_root, source_sha256="{source_sha}")
else:
    print("Generation disabled. No CSV created, no model fitted, no upload performed.")''',
            ),
            (
                "md",
                "## Download your validated file\nThe link below is created only after validation and checksum verification. Click it to download your file. For files over 10 MB, use the output file browser instead of embedding a large payload. You decide whether and when to submit.",
            ),
            (
                "code",
                'from IPython.display import HTML, display\nif submission_path is not None:\n    display(HTML(download_link(submission_path)))\n    print("Local output:", submission_path)\n    print("No Kaggle submission or upload was made.")',
            ),
        ]
    )
    outputs["kaggle/submission.ipynb"].metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return outputs


def same_sources(actual, expected) -> bool:
    """Execution outputs may evolve; generated narrative and code must match exactly."""
    fields = ("cell_type", "id", "source")
    return (
        actual.metadata.get("kernelspec") == expected.metadata.get("kernelspec")
        and len(actual.cells) == len(expected.cells)
        and all(
            all(a.get(field) == b.get(field) for field in fields)
            for a, b in zip(actual.cells, expected.cells, strict=True)
        )
    )


def write_notebook(path: Path, expected) -> None:
    """Keep verified outputs only while their entire generated source is unchanged."""
    if path.exists() and same_sources(nbf.read(path, as_version=4), expected):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nbf.writes(expected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for relative, nb in notebooks().items():
        path = ROOT / relative
        if args.check:
            if not path.exists() or not same_sources(nbf.read(path, as_version=4), nb):
                raise ValueError(f"Notebook source is stale: {relative}")
        else:
            write_notebook(path, nb)
    print("NOTEBOOK_SOURCES_VERIFIED" if args.check else "NOTEBOOKS_CREATED")


if __name__ == "__main__":
    main()
