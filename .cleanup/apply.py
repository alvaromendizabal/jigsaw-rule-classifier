"""One-time final-project curation; this preparation file is removed before merge."""
from pathlib import Path
import ast
import hashlib
import json
import re
import shutil
import subprocess

ROOT = Path.cwd()
BASE = '9bf521456819edc563c07711b88b41b535ab8694'
if subprocess.check_output(['git', 'rev-parse', 'origin/main'], text=True).strip() != BASE:
    raise RuntimeError('Main advanced; reconcile before applying curation')

def write(name, text):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')

def delete(name):
    p = ROOT / name
    if p.is_dir(): shutil.rmtree(p)
    elif p.exists(): p.unlink()

def replace(name, old, new):
    p = ROOT / name
    text = p.read_text()
    if old not in text: raise ValueError('Unexpected source: ' + name)
    p.write_text(text.replace(old, new))

def trim(name, names):
    p = ROOT / name
    text = p.read_text()
    lines = text.splitlines(keepends=True)
    spans = []
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            begin = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            spans.append((begin, node.end_lineno))
    if len(spans) != len(names): raise ValueError('Unexpected definitions: ' + name)
    for begin, end in reversed(spans): del lines[begin:end]
    p.write_text(''.join(lines))

parts = ['import json\nimport numpy as np\nimport pandas as pd\nfrom jigsaw_rules.data import EXAMPLES\n']
for name, symbol in [('tests/test_instructions.py', 'Tokenizer'), ('tests/test_research.py', 'frame')]:
    text = (ROOT / name).read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == symbol)
    parts.append(ast.get_source_segment(text, node))
write('tests/fixtures.py', '\n\n'.join(parts) + '\n')
keep_src = {'__init__.py', 'data.py', 'metrics.py', 'model.py', 'splits.py', 'runtime.py', 'pipeline.py', 'report.py', 'review.py', 'cli.py'}
for p in (ROOT / 'src/jigsaw_rules').glob('*.py'):
    if p.name not in keep_src: p.unlink()
keep_scripts = {'__init__.py', 'competition_features.py', 'support_adaptation.py', 'decision_training.py', 'kaggle_adaptation.py', 'kaggle_paths.py', 'build_adapted_notebook.py'}
for p in (ROOT / 'scripts').iterdir():
    if p.name not in keep_scripts: delete(p.relative_to(ROOT))
keep_tests = {'__init__.py', 'conftest.py', 'fixtures.py', 'test_data_metrics.py', 'test_splits_model.py', 'test_runtime_pipeline.py', 'test_review.py', 'test_kaggle_paths.py', 'test_kaggle_adaptation.py', 'test_decision_training.py', 'test_support_adaptation.py', 'test_competition_features.py'}
for p in (ROOT / 'tests').iterdir():
    if p.name not in keep_tests: delete(p.relative_to(ROOT))
for name in ['notebooks', 'research', 'AGENTS.md', 'HISTORICAL_MODEL_CARD.md', 'bootstrap.sh']:
    delete(name)
for p in (ROOT / 'configs').iterdir():
    if p.name not in {'competition_features.json', 'kaggle_adaptation.json'}: delete(p.relative_to(ROOT))
for p in (ROOT / 'kaggle').iterdir():
    if p.name != 'submission.ipynb': delete(p.relative_to(ROOT))
for p in (ROOT / 'docs').iterdir():
    if p.name not in {'assets', 'DEMO_REPLAY.md', 'PUBLIC_CHALLENGES.md', 'TOP_SOLUTION_INTEGRATION.md'}: delete(p.relative_to(ROOT))
for p in (ROOT / 'reports').iterdir():
    if p.name not in {'baseline', 'support_adaptation', 'checkpoints', 'public_challenges.json', 'public_challenges.html'}: delete(p.relative_to(ROOT))
for p in (ROOT / 'reports/checkpoints').iterdir():
    if p.name not in {'kaggle_adaptation.json', 'support_adaptation.json'}: delete(p.relative_to(ROOT))
for section in ['baseline', 'support_adaptation']:
    p = ROOT / 'reports' / section
    meta = json.loads((p / 'metadata.json').read_text())
    allowed = set(meta['files']) | {'metadata.json'}
    for f in p.rglob('*'):
        if f.is_file() and f.relative_to(p).as_posix() not in allowed: f.unlink()
trim('scripts/competition_features.py', {'cached_batch', 'feature_banks'})
trim('scripts/support_adaptation.py', {'decision_loss', 'save_training_state', 'restore_training_state', 'train_adapter', 'prototype_features'})
trim('tests/test_competition_features.py', {'test_checkpoint_reuses_without_model_and_rejects_corruption', 'test_bank_counts_and_context_ablation', 'test_screen_is_independent_of_validation_features_and_labels'})
replace('tests/test_competition_features.py', '    cached_batch,\n    feature_banks,\n', '')
replace('tests/test_competition_features.py', 'from tests.test_instructions import Tokenizer\nfrom tests.test_research import frame', 'from tests.fixtures import Tokenizer, frame')
trim('tests/test_support_adaptation.py', {'test_prototype_geometry_uses_only_supplied_reference_labels', 'test_evaluation_pipeline_and_verified_stage_reuse'})
for old in ['    decision_loss,\n', '    train_adapter,\n', '    prototype_features,\n']:
    replace('tests/test_support_adaptation.py', old, '')
replace('tests/test_support_adaptation.py', 'from tests.test_competition_features import ChatTokenizer, spec', 'from tests.test_competition_features import ChatTokenizer, spec\nfrom scripts.decision_training import decision_loss, train_adapter')
trim('tests/test_decision_training.py', {'test_disabled_scaling_preserves_original_training'})
replace('src/jigsaw_rules/pipeline.py', 'from jigsaw_rules.cloud import backup\n', '')
replace('src/jigsaw_rules/pipeline.py', '    cloud: dict | None = None,\n', '')
replace('src/jigsaw_rules/pipeline.py', 'seed=seed, cloud=cloud)', 'seed=seed)')
replace('src/jigsaw_rules/pipeline.py', '*, data_dir, folds, seed, cloud)', '*, data_dir, folds, seed)')
replace('src/jigsaw_rules/pipeline.py', '    def sync():\n        if cloud:\n            backup(root, cloud["bucket"], cloud["region"])\n\n', '')
p = ROOT / 'src/jigsaw_rules/pipeline.py'
p.write_text(re.sub(r'^\s*sync\(\)\n', '', p.read_text(), flags=re.M))
write('src/jigsaw_rules/cli.py', '''"""Run the maintained CPU reference; synthetic data are explicitly labelled."""
from __future__ import annotations

import argparse
from pathlib import Path

from jigsaw_rules.data import synthetic
from jigsaw_rules.pipeline import run_baseline
from jigsaw_rules.review import review_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["demo", "baseline", "review"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "review":
        review_run(root, args.run_id, allow_synthetic=args.allow_synthetic)
    elif args.command == "demo":
        data = root / "data/synthetic"
        if data.exists() and any(data.iterdir()) and not (data / "SYNTHETIC.txt").is_file():
            raise ValueError("Refusing to overwrite an unmarked input directory")
        if not (data / "SYNTHETIC.txt").is_file():
            synthetic(data)
        run_baseline(root, data_dir=data)
    else:
        run_baseline(root)


if __name__ == "__main__":
    main()
''')
replace('scripts/build_adapted_notebook.py',
    '"The 0.92–0.93 target remains open. This publication adds result documentation; "\n            "the scored saved version is unchanged.\\n\\n"',
    '"The historical scored version is unchanged. This maintained source build "\n            "does not claim byte-identical weights or a newly reproduced score.\\n\\n"')
write('scripts/review_portfolio.py', '''"""Verify original aggregate receipts; this is not a hidden-label evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review(root: Path) -> dict:
    root = Path(root).resolve()
    hashes = {}
    for section in ("baseline", "support_adaptation"):
        folder = root / "reports" / section
        metadata = json.loads((folder / "metadata.json").read_text())
        for name, expected in metadata["files"].items():
            path = (folder / name).resolve()
            require(path.is_relative_to(folder), "Evidence path escapes its section")
            require(path.is_file(), "Missing evidence: " + name)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            require(digest == expected, "Evidence checksum mismatch: " + name)
            hashes[path.relative_to(root).as_posix()] = digest
    receipt_path = root / "reports/checkpoints/kaggle_adaptation.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt["status"] == "submitted_and_scored", "Unverified submission state")
    submission = receipt["submission"]
    require(submission["status"] == "succeeded", "Submission did not succeed")
    require(submission["public_score"] == 0.91808, "Public result changed")
    require(submission["private_score"] == 0.91425, "Private result changed")
    comparison = receipt["comparison"]
    require(math.isclose(submission["private_score"] - comparison["baseline_private_auc"], comparison["private_auc_gain"], abs_tol=1e-12), "Incorrect baseline comparison")
    folder = root / "reports/support_adaptation"
    protocol = json.loads((folder / "protocol.json").read_text())
    require(protocol["rows"] == 881 and protocol["policies"] == 2, "Wrong comparison population")
    require(sum(row["novel_query_rows"] for row in protocol["folds"]) == 881, "Query accounting mismatch")
    results = json.loads((folder / "results.json").read_text())
    models = {row["model"]: row for row in results}
    scores = {}
    for name in ("frozen_raw", "adapted_raw"):
        metrics = models[name]["metrics"]
        values = list(metrics["per_rule_auc"].values())
        require(len(values) == 2 and all(math.isfinite(v) and 0 <= v <= 1 for v in values), "Invalid policy AUC")
        require(math.isclose(sum(values) / 2, metrics["rule_macro_auc"], abs_tol=1e-12), "Macro AUC must weight policies equally")
        scores[name] = metrics["rule_macro_auc"]
    require(set(models["frozen_raw"]["metrics"]["per_rule_auc"]) == set(models["adapted_raw"]["metrics"]["per_rule_auc"]), "Mismatched policy cohorts")
    hashes[receipt_path.relative_to(root).as_posix()] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    return {"status": "passed", "scope": "Original aggregate arithmetic and integrity; no hidden labels or retraining", "historical_submission": {"public_auc": submission["public_score"], "private_auc": submission["private_score"]}, "matched_development": {"rows": 881, "policies": 2, **scores, "gain": scores["adapted_raw"] - scores["frozen_raw"]}, "source_sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    print(json.dumps(review(args.root), indent=2))


if __name__ == "__main__":
    main()
''')
write('tests/test_complete_review.py', '''"""Original final-result receipts stay independently inspectable."""
import json
import shutil
from pathlib import Path

import pytest

from scripts.review_portfolio import review

ROOT = Path(__file__).resolve().parents[1]


def test_final_arithmetic_and_original_evidence():
    result = review(ROOT)
    assert result["status"] == "passed"
    assert result["matched_development"]["gain"] == pytest.approx(0.1052933125268161)
    assert len(result["source_sha256"]) == 11


@pytest.mark.parametrize("section,name", [("support_adaptation", "results.json"), ("baseline", "results.json")])
def test_modified_scientific_evidence_is_rejected(tmp_path, section, name):
    shutil.copytree(ROOT / "reports", tmp_path / "reports")
    (tmp_path / "reports" / section / name).write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        review(tmp_path)


def test_changed_submitted_score_is_not_accepted(tmp_path):
    shutil.copytree(ROOT / "reports", tmp_path / "reports")
    path = tmp_path / "reports/checkpoints/kaggle_adaptation.json"
    value = json.loads(path.read_text())
    value["submission"]["private_score"] = 0.99
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Private result changed"):
        review(tmp_path)


def test_metadata_cannot_escape_the_evidence_directory(tmp_path):
    shutil.copytree(ROOT / "reports", tmp_path / "reports")
    path = tmp_path / "reports/baseline/metadata.json"
    value = json.loads(path.read_text())
    value["files"]["../../outside"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="escapes"):
        review(tmp_path)
''')
write('README.md', '''# Jigsaw · Rule-conditioned language models

**Alvaro Mendizabal · Applied NLP · Low-rank adaptation · Reliable ML systems**

[![Quality](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml/badge.svg)](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/quality.yml)
[![Public demo](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/public-demo.yml/badge.svg)](https://github.com/alvaromendizabal/jigsaw-rule-classifier/actions/workflows/public-demo.yml)

![Policy-conditioned classification and verifiable support evidence](docs/assets/public-demo-hero.svg)

I built a support-adapted language-model classifier for rules that change between communities, with explicit query/support leakage controls and recoverable training and inference. The retained system is **Qwen3-4B-Instruct-2507 + LoRA**, using original training labels and supplied positive/negative policy examples—not hidden query targets.

**Recorded result: 0.91808 public / 0.91425 private ROC AUC.** These are historical post-competition scores, not a new result from the public demo. [Original receipt](reports/checkpoints/kaggle_adaptation.json)

[**Explore Policy Lens ↗**](https://alvaro-policy-lens.tartmacaw2.chatgpt.site) · [Case study](CASE_STUDY.md) · [Reproduce components](docs/REPRODUCIBILITY.md) · [Model card](MODEL_CARD.md)

## The final system

```mermaid
flowchart LR
  A[Rule, comment and supplied examples] --> B[Normalize and purge overlaps]
  B --> C[One LoRA adapter on eligible labels]
  C --> D[Native last-token No/Yes logits]
  D --> E[Within-policy ranking and exact row restoration]
  E --> F[Schema-validated output]
```

| Engineering decision | Inspect the implementation | Verification |
|---|---|---|
| Keep query targets out of adaptation | [Support construction](scripts/competition_features.py) and [adaptation inputs](scripts/kaggle_adaptation.py) | [Label-flip and overlap tests](tests/test_kaggle_adaptation.py) |
| Adapt the actual causal language model | [LoRA and decision prompts](scripts/support_adaptation.py) | [Native model and last-token parity](tests/test_support_adaptation.py) |
| Recover training without silently restarting | [Optimizer, scheduler, scaler and RNG state](scripts/decision_training.py) | [Exact-resume and overflow tests](tests/test_decision_training.py) |
| Separate ranking from calibration | [Policy ranks and prediction cache](scripts/kaggle_adaptation.py) | Extreme logits, original IDs and corrupt-cache rejection |
| Package canonical code for offline use | [Notebook builder](scripts/build_adapted_notebook.py) | [Exact embedded-source contract](tests/test_kaggle_adaptation.py) |

## What the evidence supports

| Comparison | Recorded result | Population |
|---|---:|---|
| Retained support-adapted model | **0.91425 private AUC** | Historical external evaluation |
| Matched frozen → adapted backbone | **0.61460 → 0.71989** | 881 novel comments, two previously examined policies |
| Matched adaptation gain | **+0.10529 AUC** | Equal-weight policy-macro AUC, not pooled AUC |

The matched study isolates adaptation on the same backbone; the external comparison also changes the model and input method. Neither the repeatedly inspected development cohort nor the visible demo challenge is an untouched generalization test. [Validation scope](docs/VALIDATION.md)

## Run something real from a clone

The source-linked review needs only Python 3.11+; Node.js 22 or 24 enables the actual browser-engine checks:

```bash
python3 -S scripts/review_portfolio.py --json
python3 -S review/run.py --output /tmp/jigsaw-review --with-node
python3 -S review/run.py --output /tmp/jigsaw-review --with-node --check
```

For the **trainable CPU control**, use Python 3.12 and the locked environment:

```bash
uv sync --locked --group dev
uv run jigsaw demo --root /tmp/jigsaw-cpu-demo
uv run jigsaw demo --root /tmp/jigsaw-cpu-demo
uv run jigsaw review --root /tmp/jigsaw-cpu-demo --allow-synthetic
```

This fits and evaluates the TF-IDF/logistic-regression control on authored data; the second run verifies and reuses completed stages. It is not a substitute for the retained neural result. [Full setup and native miniature-model tests](docs/REPRODUCIBILITY.md)

## Interactive failure analysis

Policy Lens recomputes a fitted lexical reference as the rule, examples and comment change. It exposes signed feature contributions, overlapping support and conflicting labels. The **24 visible authored challenges include 16 expected outcomes and eight misses**; every failure remains in the published diagnostics. [Challenge report](docs/PUBLIC_CHALLENGES.md) · [Replay an exported decision](docs/DEMO_REPLAY.md)

## Publication boundary

The maintained tree contains the final adaptation/training/inference components, their control, tests, compact original evidence and public demonstration. Abandoned feature banks, alternate-backbone campaigns, teacher studies, redundant notebooks and operator launchers are retired. The historical metadata may describe discarded comparisons, but their implementations are not presented as current capabilities.

Private comments, weights, adapters, cached representations and full inference outputs are excluded. The [offline notebook](kaggle/submission.ipynb) is generated from the retained canonical source and requires separately licensed data, pretrained weights and a suitable GPU. Public checks exercise native miniature Qwen/LoRA behavior; they do not retrain the historical 4B system or reproduce its score.

[Data card](DATA_CARD.md) · [Source credits](docs/TOP_SOLUTION_INTEGRATION.md) · [License](LICENSE)
''')
write('CASE_STUDY.md', '''# Case study · Adapt to the rule, not just the comment

A moderation label depends on the policy being applied. A fixed global classifier can confuse policy transfer with repeated exposure to the same text. I built a rule-conditioned classifier, a support-adaptation procedure, a leakage-aware comparison and recoverable offline inference.

## Selected approach and ownership

The retained system uses the credited Qwen3-4B-Instruct-2507 backbone with one LoRA adapter. My work covers eligible-label construction, support/query purging, decision prompts, last-token scoring, checkpoint recovery, exact-ID output contracts and the public diagnostic interface. The backbone and its tokenizer are upstream assets, not my invented architecture. [Credits](docs/TOP_SOLUTION_INTEGRATION.md)

The [actual trainer](scripts/decision_training.py) keeps optimizer, scheduler, loss-scaler and random states with completed checkpoints. [Native miniature-model tests](tests/test_decision_training.py) check exact recovery after interruption and retries after overflow. The [inference component](scripts/kaggle_adaptation.py) verifies model inputs, restores original row order and rejects corrupt cached predictions.

## Evidence and the final decision

The historical submitted system recorded **0.91808 public / 0.91425 private AUC**. A controlled same-backbone study measured **0.61460 frozen versus 0.71989 adapted policy-macro AUC** on 881 novel comments across two previously examined rules. That **+0.10529** effect is different from the end-to-end comparison with a lexical control; pooling them would conflate populations and model changes. [Original submitted record](reports/checkpoints/kaggle_adaptation.json) · [Matched protocol](reports/support_adaptation/protocol.json)

The selected direct decision path is preserved. Alternate feature readouts, unused backbones and later unpromoted experiments are not part of the maintained release. Original compact comparison records remain unchanged where needed to substantiate selection.

## What can be reproduced publicly

A clone can train the lexical control end to end on authored data, verify the original aggregate arithmetic, exercise the actual native Qwen/LoRA interfaces on tiny generated models, regenerate the offline source notebook and replay browser decisions. The browser's 24 visible authored challenges retain eight misses. Those diagnostics reveal failure modes; they do not establish production moderation accuracy.

Private datasets, adapters, model weights and prediction rows are excluded. No live service or full-score reproduction is claimed. [Commands and boundaries](docs/REPRODUCIBILITY.md)
''')
write('MODEL_CARD.md', '''# Model card · Retained support-adapted classifier

**Task:** score whether a comment violates its supplied community policy. **Backbone:** Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. **Adaptation:** LoRA on original training labels and supplied positive/negative examples. Hidden query targets are not eligible training data.

## Interfaces and output

[Support construction](scripts/competition_features.py) normalizes text and excludes overlap/conflicts. [Decision training](scripts/decision_training.py) optimizes native last-token answer logits. [Inference](scripts/kaggle_adaptation.py) converts logits to within-policy ranks, restores IDs and checks schema and finiteness. Ranks preserve ordering for AUC; they are not calibrated probabilities and do not specify an operational moderation threshold.

## Recorded evaluation

Historical external ROC AUC was **0.91808 public / 0.91425 private**. A same-backbone development comparison measured **0.61460 → 0.71989** equal-weight policy-macro AUC on 881 novel comments and two rules. The development population was previously examined. A ten-row notebook preview verifies execution only, not full hidden-set cardinality or model quality. [Original receipt](reports/checkpoints/kaggle_adaptation.json)

## Limitations and responsible interpretation

Two policies do not establish transfer to every policy, community or language. Support examples may be misleading or conflicting. Training overlap can inflate apparent transfer without purging. The model is a research artifact, not a production moderation system or a replacement for human review. Native miniature-model tests verify arithmetic and recovery, not the historical 4B model's performance.

The public CPU classifier and browser engine are explicit controls/demonstrations, not the submitted neural system. Private comments, labels, adapters, pretrained weights and full inference outputs are not distributed. [Data card](DATA_CARD.md) · [Source credits](docs/TOP_SOLUTION_INTEGRATION.md) · [Reproduction](docs/REPRODUCIBILITY.md)
''')
write('docs/REPRODUCIBILITY.md', '''# Reproduce the maintained components

## Original aggregate evidence · no dependencies

```bash
python3 -S scripts/review_portfolio.py --json
python3 -S review/run.py --output /tmp/jigsaw-review --with-node
python3 -S review/run.py --output /tmp/jigsaw-review --with-node --check
```

Use Python 3.11+; omit `--with-node` when Node.js 22/24 is unavailable. The checker verifies original evidence hashes, policy-macro arithmetic, cohort accounting and the historical submission comparison. The kit records actual checks and source hashes; changed output is rejected rather than silently reused.

## Trainable lexical control · Python 3.12

```bash
uv sync --locked --group dev
uv run jigsaw demo --root /tmp/jigsaw-cpu-demo
uv run jigsaw demo --root /tmp/jigsaw-cpu-demo
uv run jigsaw review --root /tmp/jigsaw-cpu-demo --allow-synthetic
```

This executes TF-IDF/logistic-regression training, seen/held-out-rule evaluation, overlap purging, independent full fitting and validated synthetic outputs. The second invocation reuses checksum-verified completed stages. Open the reported `review/report.html`. On Windows, use a writable output path outside the checkout.

## Native neural contracts and source packaging

```bash
uv sync --locked --group dev --extra semantic
uv run --extra semantic pytest -q
uv run python scripts/build_adapted_notebook.py
```

Tests construct tiny native Qwen causal models with LoRA: no pretrained download is needed. They check query-target exclusion, final-token projection, optimizer/scaler/RNG recovery, overflow retries, cached-score corruption and original row order. The builder embeds exact canonical code and validates its hashes in the offline notebook. It does not execute the full 4B model.

To execute that notebook separately requires licensed competition inputs, the pinned pretrained model files and a compatible GPU/runtime. Exact historical adapters, private comments and full prediction rows are not published. Rebuilding source is not a claim to reproduce the submitted weights or score.

## Browser diagnosis and replay

```bash
python -m http.server 8000
node tools/evaluate_public_challenges.mjs --check
node tools/test_replay_public_demo.mjs
```

Open `http://localhost:8000/public-demo/`. The authored engine and visible challenges are separate from neural model quality. [Export/replay contract](DEMO_REPLAY.md) · [Failure analysis](PUBLIC_CHALLENGES.md)
''')
write('docs/VALIDATION.md', '''# Validation and evidence boundaries

The maintained neural comparison is **same backbone, matched support inputs, frozen versus adapted direct answer score**. It contains 881 novel comments across two previously examined policies. Policy-macro AUC weights each policy equally; pooled AUC is separately recorded. [Original protocol](../reports/support_adaptation/protocol.json) · [Original results](../reports/support_adaptation/results.json)

The retained external score is **0.91808 public / 0.91425 private** from historical post-competition evaluation. The end-to-end comparison against the lexical control changes the backbone and method; its larger gain must not be attributed solely to LoRA. A ten-row preview is not an external evaluation.

Query targets never enter adaptation. Supplied examples may carry eligible labels, but overlapping query text is purged and contradictory pairs are rejected. Tests flip query targets and assert that adaptation inputs stay unchanged. [Tests](../tests/test_support_adaptation.py)

The original study was repeatedly inspected and is not a fresh holdout. Within-policy ranks are not calibrated probabilities. The synthetic browser challenge is visible and authored, so it does not establish unbiased generalization or moderation safety. Its eight misses remain published. [Public challenges](PUBLIC_CHALLENGES.md)
''')
write('docs/EMPLOYER_REVIEW_GUIDE.md', '''# Technical review

Start with the [case study](../CASE_STUDY.md), then inspect the [actual decision trainer](../scripts/decision_training.py), [support/query contracts](../tests/test_support_adaptation.py), [offline inference](../scripts/kaggle_adaptation.py) and [original final-result receipt](../reports/checkpoints/kaggle_adaptation.json).

Run the [documented CPU control and native miniature-model tests](REPRODUCIBILITY.md). The public interface makes every lexical contribution, excluded overlap and conflicting support input inspectable. Full-data neural weights and private prediction rows are outside the release.

The public tree represents the selected support-adaptation project, not a catalog of abandoned feature campaigns. Historical receipt timestamps and implementation hashes identify completed experiments; they are not claims that those experiments were rerun during publication cleanup.
''')
write('docs/PROJECT_CLOSEOUT.md', '''# Maintained release scope

The retained approach is support-adapted Qwen3-4B-Instruct-2507 with a trainable lexical control, exact-resume execution, offline source packaging and explicit validation boundaries. Original measured results remain unchanged. Later unpromoted feature banks, alternate-backbone/teacher campaigns, duplicated research notebooks and AWS operator launchers have been retired from the maintained tree.

Public reproducibility is scoped to the documented control, native miniature-model contracts, evidence arithmetic and authored browser analysis. Private datasets, adapters, model weights and full prediction rows are not distributed. [Reproduction](REPRODUCIBILITY.md) · [Model card](../MODEL_CARD.md)
''')
write('START_HERE.md', '''# Start here

[Final project and results](README.md) · [Engineering case study](CASE_STUDY.md) · [Runnable examples and tests](docs/REPRODUCIBILITY.md) · [Model limitations](MODEL_CARD.md)

The maintained neural approach is support-adapted Qwen3-4B-Instruct-2507. The public lexical examples are explicitly separate controls, not a substitute for the recorded submitted model.
''')
write('docs/TOP_SOLUTION_INTEGRATION.md', '''# Source credits and implementation boundary

The retained backbone is **Qwen3-4B-Instruct-2507** from the Qwen team. Its model revision and file checksums are pinned in [the model input contract](../configs/competition_features.json). The [upstream license](../assets/qwen3/LICENSE) and tokenizer configuration remain with the offline source package. The historical execution used Guanshuo Xu's `wowfattie/qwen3-4b-instruct-2507/transformers/default/1` model mirror; mirror availability is not guaranteed by this repository.

Publicly documented support-label adaptation, last-position decision loss and rank-based inference informed this work. The maintained implementation consists of reviewable project source, its leakage and recovery contracts, and the original evaluation evidence; it does not claim authorship of the pretrained architecture or publish third-party weights.

Alvaro Mendizabal's project contributions are eligible-label construction, duplicate/conflict purging, recoverable LoRA execution, decision scoring, row-preserving offline packaging, the controlled evaluation and the authored public diagnostic interface. Alternate-backbone and teacher campaigns that did not replace the retained scored system are no longer part of the maintained implementation.

[Model card](../MODEL_CARD.md) · [Exact source builder](../scripts/build_adapted_notebook.py) · [Historical result](../reports/checkpoints/kaggle_adaptation.json)
''')
for name in ['DATA_CARD.md', 'docs/TOP_SOLUTION_INTEGRATION.md', 'docs/DEMO_REPLAY.md', 'docs/PUBLIC_CHALLENGES.md']:
    p = ROOT / name
    text = p.read_text()
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: m.group(0) if m.group(2).startswith(('http', '#', 'mailto:')) or (p.parent / m.group(2).split('#')[0]).exists() else m.group(1), text)
    p.write_text(text)
p = ROOT / 'pyproject.toml'
t = p.read_text().replace('"boto3==1.43.89", "kaggle==2.2.4", "psutil==7.2.2",', '')
a = t.index('[tool.ruff.lint.per-file-ignores]')
b = t.index('[project.optional-dependencies]', a)
t = t[:a] + '[tool.ruff.lint.per-file-ignores]\n"src/jigsaw_rules/report.py" = ["E501"]\n\n' + t[b:]
p.write_text(t)
project = json.loads((ROOT / 'review/project.json').read_text())
project['signals'][2] = {'value': '4B + LoRA', 'label': 'Retained support-adapted model'}
project['cards'][0]['implementation'] = 'scripts/support_adaptation.py'
project['cards'][0]['tests'] = 'tests/test_support_adaptation.py'
project['commands'].insert(0, {'id': 'original-final-evidence', 'runtime': 'python', 'entry': 'scripts/review_portfolio.py', 'args': ['--json']})
write('review/project.json', json.dumps(project, indent=2) + '\n')
write('review/README.md', '''# Source-linked review

```bash
python3 -S review/run.py --output /tmp/jigsaw-review --with-node
python3 -S review/run.py --output /tmp/jigsaw-review --with-node --check
```

Python 3.11+ verifies the original final evidence and public packaging. Optional Node.js 22/24 checks execute the actual browser policy engine, visible challenges and export replay. The kit records source hashes and command outcomes. It does not execute private 4B inference or produce a new competitive result.

[Run the CPU control and native miniature-model tests](../docs/REPRODUCIBILITY.md) · [Case study](../CASE_STUDY.md)
''')
all_text = '\n'.join(p.read_text(errors='ignore') for base in ['README.md', 'CASE_STUDY.md', 'MODEL_CARD.md', 'DATA_CARD.md', 'docs', 'public-demo', 'scripts', 'tests'] for p in ([ROOT/base] if (ROOT/base).is_file() else (ROOT/base).rglob('*')) if p.is_file() and p.suffix in {'.md','.py','.html','.css','.js','.json'})
for folder in ['docs/assets', 'assets']:
    for p in (ROOT/folder).rglob('*'):
        if p.is_file() and 'qwen3' not in p.parts and p.name not in all_text: p.unlink()
for p in sorted(ROOT.rglob('*'), reverse=True):
    if p.is_dir() and '.git' not in p.parts and not any(p.iterdir()): p.rmdir()
print('Final Qwen support-adaptation curation applied; original metric records preserved.')
