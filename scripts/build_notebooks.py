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


SETUP = """import os
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
from IPython.display import display, HTML, FileLink
from jigsaw_rules.data import load_data, audit
from jigsaw_rules.runtime import Progress, environment

root = Path(os.environ.get("JIGSAW_ROOT", Path.cwd())).resolve()
if root.name == "notebooks":
    root = root.parent
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#087f8c", "#bd633b", "#334ea0", "#70923b"]
display(HTML("<div style='padding:18px;background:#edf6f5;border-left:5px solid #087f8c'>"
             "<b>Jigsaw research workspace</b><br>Every result must identify its data and validation protocol.</div>"))
print("Project:", root)
"""


def notebooks():
    outputs = {}
    outputs["notebooks/00_environment_and_data.ipynb"] = notebook(
        [
            (
                "md",
                "# 00 · Environment and data\n\n**Question:** Are we working with the intended files in a reproducible environment?\n\nRun `bash bootstrap.sh` once, then authenticate Kaggle and run `uv run jigsaw download`. This notebook checks the files; it does not ask you to paste credentials. Run notebooks **00 → 01 → 02** in order.",
            ),
            ("code", SETUP),
            (
                "md",
                "## Inspect the environment\nPackage versions are pinned in `uv.lock`. Long operations report UTC timestamps, elapsed time, and 15-second heartbeats.",
            ),
            (
                "code",
                "display(pd.DataFrame(environment()['packages'].items(), columns=['Package', 'Version']))",
            ),
            (
                "md",
                "## Validate the competition files\nA row asks whether one comment violates one supplied rule. Positive examples violate that rule; negative examples do not. The downloaded test file is only a preview of hidden evaluation inputs.",
            ),
            (
                "code",
                "with Progress(root / 'logs/notebooks.jsonl', '00_schema'):\n    train, test, sample = load_data(root / 'data/raw')\n    overview = audit(train, test)\ndisplay(pd.DataFrame({'File': ['train.csv', 'test.csv', 'sample_submission.csv'], 'Rows': [len(train), len(test), len(sample)]}))\nprint('Schema checks passed')\nif (root / 'data/raw/SYNTHETIC.txt').exists():\n    print('SYNTHETIC SOFTWARE TEST — not competition data')",
            ),
            (
                "md",
                "## Next\nOpen **01_data_and_validation.ipynb**. Raw comments, credentials, and trained weights are excluded from Git. Your private S3 snapshots preserve data and completed run artifacts.",
            ),
        ]
    )
    outputs["notebooks/01_data_and_validation.ipynb"] = notebook(
        [
            (
                "md",
                "# 01 · Data and validation\n\n**Question:** What would a convincing test of rule generalization look like?\n\nThe official training set contains only two rules. A random split alone mostly tests familiar policies. We therefore report seen-rule and held-out-rule performance separately.",
            ),
            ("code", SETUP),
            (
                "code",
                "train, test, sample = load_data(root / 'data/raw')\noverview = audit(train, test)\ndisplay(pd.DataFrame(overview['by_rule']))\nprint('Duplicate bodies:', overview['duplicate_training_bodies'])\nprint('Train/preview-test overlap:', overview['train_test_body_overlap'])",
            ),
            (
                "md",
                "## Class balance and comment length\nRule imbalance and long comments affect both measurement and future transformer token budgets. These plots use only supplied data.",
            ),
            (
                "code",
                "by_rule = pd.DataFrame(overview['by_rule'])\nfig = px.bar(by_rule, x='rule', y='mean', color='rule', title='Violation prevalence by rule', labels={'mean': 'Violation rate', 'rule': 'Rule'})\nfig.update_layout(showlegend=False, height=420)\nfig.update_yaxes(range=[0, 1])\nfig.show()\nlengths = train.assign(characters=train.body.str.len())\npx.histogram(lengths, x='characters', color='rule', nbins=35, title='Comment length distribution', barmode='overlay', opacity=.6).show()",
            ),
            (
                "md",
                "## Freeze the validation design\n1. **Seen rule:** stratify by rule and label; group normalized duplicate comments.\n2. **Held-out rule:** train on other rules and evaluate the omitted rule.\n3. Purge training rows if their body or support examples contain a validation body.\n4. Fit vocabulary only on the remaining training rows.\n\nProvided validation examples are legitimate model inputs. We do not convert validation examples into new supervised training rows. Exact-text checks do not detect every paraphrase or shared source. Two rules are too few to estimate transfer across all community policies.",
            ),
            (
                "code",
                "from jigsaw_rules.splits import make_splits\nrows = []\nfor protocol in ['seen_rule', 'heldout_rule']:\n    for fold, (ti, vi, purged) in enumerate(make_splits(train, protocol, folds=3, seed=2025)):\n        rows.append({'Protocol': protocol, 'Fold': fold, 'Training rows': len(ti), 'Validation rows': len(vi), 'Purged rows': purged})\ndisplay(pd.DataFrame(rows))",
            ),
            (
                "md",
                "## Metric contract\nThe official overview calls the metric **column-averaged AUC**. We report **rule macro AUC** (equal-weight mean of rule-specific ROC AUC), consistent with published descriptions of the challenge, and also **pooled AUC** so the distinction stays visible. The overview does not expose executable scorer code.\n\nSecondary diagnostics: average precision, log loss, Brier score, calibration error, and precision/recall/F1 at a predeclared 0.5 threshold. We never replace an undefined single-class AUC with a favorable number.\n\nNext: **02_baseline_and_review.ipynb**.",
            ),
        ]
    )
    outputs["notebooks/02_baseline_and_review.ipynb"] = notebook(
        [
            (
                "md",
                "# 02 · Baseline and review\n\n**Question:** Does rule/example context help, and what breaks when the rule changes?\n\nThese CPU models are reference baselines, not the final advanced models. `comment_only` learns TF-IDF lexical patterns. `rule_examples` adds comment-to-rule/example cosine similarities and positive-minus-negative similarity features. The latter is predeclared for the starter submission. No test labels influence model choice.",
            ),
            ("code", SETUP),
            (
                "md",
                "## Run or resume the experiment\nCompleted stages are reused only when data, source, configuration, package versions, and output checksums match. If a fold is interrupted, that fold restarts; earlier completed folds remain valid. With cloud backup enabled, every completed stage is copied to S3. GPU optimizer-state recovery belongs to the later neural training phase.",
            ),
            (
                "code",
                "from jigsaw_rules.pipeline import run_baseline\nuse_cloud = os.environ.get('JIGSAW_CLOUD', '1') == '1'\nconfig_path = root / 'configs/local.json'\nif use_cloud and not config_path.exists():\n    raise FileNotFoundError('Create configs/local.json from the example, or explicitly set JIGSAW_CLOUD=0 for local verification.')\ncloud = json.loads(config_path.read_text()) if use_cloud else None\nrun_dir = run_baseline(root, cloud=cloud)",
            ),
            (
                "md",
                "## Compare ranking and probability quality\nA strong seen-rule result with weak held-out-rule AUC indicates poor policy transfer. AUC does not establish calibration. Keep the held-out-rule analysis separate from leaderboard results.",
            ),
            (
                "code",
                "results = json.loads((run_dir / 'review/results.json').read_text())\nsummary = pd.DataFrame([{'Model': r['model'], 'Protocol': r['protocol'], **{k: v for k, v in r['metrics'].items() if isinstance(v, float)}} for r in results])\ndisplay(summary.round(4))\nfig = px.bar(summary, x='Protocol', y='rule_macro_auc', color='Model', barmode='group', title='Baseline validation comparison')\nfig.update_yaxes(range=[0, 1])\nfig.show()\ndisplay(FileLink(str(run_dir / 'review/report.html')))",
            ),
            (
                "md",
                "## Investigate failures\nReview errors by rule and subreddit locally. Row IDs below let you join to private comments when needed. Avoid publishing raw comments or model outputs that expose them without a deliberate review.",
            ),
            (
                "code",
                "errors = pd.read_csv(run_dir / 'review/error_review.csv')\ndisplay(errors.head(12))\ncoefficients = pd.read_csv(run_dir / 'full_training/coefficients.csv')\ncontext = coefficients[coefficients.feature.str.contains('similarity')]\ndisplay(context)\nprint('Preview submission:', run_dir / 'full_training/submission.csv')",
            ),
            (
                "md",
                "## Next phase\nCompare an embedding/example matcher and a cross-encoder, then an instruction model with LoRA. Add confidence intervals, stricter near-duplicate audits, rule/example ablations, and nested calibration before selecting a final ensemble. The baseline's lexical coefficients describe association, not a causal explanation.\n\nUse **kaggle/submission.ipynb** for portable offline inference. It regenerates the submission using whichever test rows Kaggle provides. Late scoring remains dependent on your signed-in account's eligibility.",
            ),
        ]
    )
    outputs["notebooks/03_saved_results.ipynb"] = notebook(
        [
            (
                "md",
                "# 03 · Review completed evidence\n\n**Question:** What did the completed experiment establish, and what should we test next?\n\nThis notebook checks saved artifacts and recalculates metrics from out-of-fold predictions. It never trains a model. Open this notebook after restoring a saved run; you do not need to repeat notebook 02 when reviewing historical results.",
            ),
            ("code", SETUP),
            (
                "code",
                "from jigsaw_rules.review import review_run\nis_demo = (root / 'data/raw/SYNTHETIC.txt').exists()\nevidence = review_run(root, os.environ.get('JIGSAW_RUN_ID'), allow_synthetic=is_demo)\nprint('Data kind:', evidence['data_kind'])\nprint('Run:', evidence['run_id'])\nprint('Training rows:', evidence['training_rows'])\nprint('Training SHA-256:', evidence['training_sha256'])\nif evidence['data_kind'] == 'synthetic':\n    display(HTML('<b>SYNTHETIC SOFTWARE TEST — not competition performance</b>'))",
            ),
            (
                "md",
                "## The generalization gap\nCompare the two validation protocols separately. A small difference between models is not evidence of a statistically reliable improvement. With only two rules, new-rule transfer remains weakly measured. AUC evaluates ranking; probability quality needs log loss, Brier score, and calibration diagnostics as well.",
            ),
            (
                "code",
                "records = evidence['results']\nsummary = pd.DataFrame([{'Model': r['model'], 'Protocol': r['protocol'], **{k: v for k, v in r['metrics'].items() if isinstance(v, float)}} for r in records])\ndisplay(summary.round(4))\nfig = px.bar(summary, x='Protocol', y='rule_macro_auc', color='Model', barmode='group', title='Saved out-of-fold evidence: familiar versus held-out rules')\nfig.update_yaxes(range=[0, 1])\nfig.add_hline(y=0.5, line_dash='dash', annotation_text='Chance ranking')\nfig.show()\ndisplay(FileLink(str(root / 'reports/private/report.html')))\ndisplay(FileLink(str(root / 'reports/private/results.json')))",
            ),
            (
                "md",
                "## Phase 2 decision\nThe next controlled experiment compares a frozen semantic embedding/example matcher with a rule-conditioned cross-encoder under the same saved splits. Start from a pinned model revision, preserve embedding batches, record latency and peak memory, and compare per-rule as well as aggregate performance.\n\nSee `docs/PHASE_2.md` for the predeclared experiment and hardware gate. This review does not download weights or launch a paid job. Historical results remain valid evidence for their recorded source version; a code update does not require retraining merely to view them.",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for relative, nb in notebooks().items():
        path = ROOT / relative
        content = nbf.writes(nb)
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise ValueError(f"Notebook source is stale: {relative}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print("NOTEBOOK_SOURCES_VERIFIED" if args.check else "NOTEBOOKS_CREATED")


if __name__ == "__main__":
    main()
