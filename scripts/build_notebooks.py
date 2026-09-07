"""Build reproducible narrative notebooks from canonical package code."""

from __future__ import annotations

import argparse
import ast
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
    model = (ROOT / "src/jigsaw_rules/model.py").read_text()
    model = model.replace("from __future__ import annotations\n", "").replace(
        "from jigsaw_rules.data import EXAMPLES\n", ""
    )
    data = (
        (ROOT / "src/jigsaw_rules/data.py")
        .read_text()
        .replace("from __future__ import annotations\n", "")
    )
    runtime = (ROOT / "src/jigsaw_rules/runtime.py").read_text()
    progress_node = next(
        n for n in ast.parse(runtime).body if isinstance(n, ast.ClassDef) and n.name == "Progress"
    )
    progress_code = ast.get_source_segment(runtime, progress_node)
    outputs["kaggle/submission.ipynb"] = notebook(
        [
            (
                "md",
                "# Jigsaw · Rule and example baseline\n\nSelf-contained offline CPU inference. Add the official competition dataset, disable internet, then **Save Version → Save & Run All**. The official runtime limit is 12 hours. The completed notebook writes `/kaggle/working/submission.csv`.\n\nThis is a reference baseline; no medal-level result is claimed. The competition ended October 23, 2025. A late-submission button was visible but disabled while signed out on September 7, 2026; authenticated eligibility is unverified.\n\nSource: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/overview",
            ),
            (
                "code",
                "from __future__ import annotations\nimport os\nos.environ['OMP_NUM_THREADS'] = '2'\nos.environ['OPENBLAS_NUM_THREADS'] = '2'\nimport hashlib\nimport json\nimport threading\nimport time\nfrom datetime import UTC, datetime\nfrom typing import Any\nfrom pathlib import Path\nimport importlib.metadata\n",
            ),
            (
                "md",
                "## Canonical validated schema and model\nThe following cells are generated directly from the repository's schema and model modules. CI checks that they match the package.",
            ),
            ("code", data),
            ("code", model),
            ("code", progress_code),
            (
                "code",
                """input_root = Path(os.environ.get("JIGSAW_KAGGLE_INPUT", "/kaggle/input/jigsaw-agile-community-rules"))
output_root = Path(os.environ.get("JIGSAW_KAGGLE_OUTPUT", "/kaggle/working"))
output_root.mkdir(parents=True, exist_ok=True)
with Progress(output_root / "events.jsonl", "offline_submission") as log:
    train, test, sample = load_data(input_root)
    log.emit("data_validated", train_rows=len(train), test_rows=len(test))
    model = LexicalClassifier(context=True).fit(train)
    log.emit("model_fitted")
    pieces = []
    for offset in range(0, len(test), 5000):
        pieces.append(model.predict(test.iloc[offset:offset + 5000]))
        log.emit("prediction_batch", completed_rows=min(offset + 5000, len(test)), total_rows=len(test))
    submission = pd.DataFrame({"row_id": test.row_id, "rule_violation": np.concatenate(pieces)})
    validate_submission(submission, sample)
    temporary = output_root / "submission.csv.partial"
    submission.to_csv(temporary, index=False)
    os.replace(temporary, output_root / "submission.csv")
    manifest = {
        "rows": len(submission),
        "synthetic": (input_root / "SYNTHETIC.txt").exists(),
        "input_hashes": {name: hashlib.sha256((input_root / name).read_bytes()).hexdigest() for name in FILES},
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn"]},
    }
    (output_root / "submission_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.emit("SUBMISSION_VALIDATED", rows=len(submission))
submission.head()""",
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
