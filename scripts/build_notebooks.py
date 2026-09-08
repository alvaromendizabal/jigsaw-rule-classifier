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


SETUP = """import os
from pathlib import Path
import pandas as pd
from IPython.display import display
from jigsaw_rules.review import public_evidence
from jigsaw_rules.runtime import environment

root = Path(os.environ.get("JIGSAW_ROOT", Path.cwd())).resolve()
if root.name == "notebooks":
    root = root.parent
baseline = public_evidence(root, "baseline")
semantic = public_evidence(root, "semantic")
assert baseline["training_sha256"] == semantic["training_sha256"]
print("Competition evidence | 2,029 rows | two labeled rules")
print("Aggregate checksums verified. This notebook performs no model fitting.")
protocols = {"seen_rule": "Familiar rules", "heldout_rule": "Held-out rule"}

def metric_table(records, heldout=False):
    return pd.DataFrame([{"Representation": r["model"], "Validation": protocols[r["protocol"]],
        "Rule macro AUC": r["metrics"]["rule_macro_auc"],
        "Log loss": r["metrics"]["log_loss"], "Brier": r["metrics"]["brier"],
        "Average precision": r["metrics"]["average_precision"]}
        for r in records if not heldout or r["protocol"] == "heldout_rule"]).round(4)
"""

RESEARCH_SETUP = """import sys
sys.path.insert(0, str(root / "scripts"))
from build_research_report import display_figure
from jigsaw_rules.features import feature_evidence
from jigsaw_rules.research import research_evidence
from jigsaw_rules.diagnostics import diagnostic_evidence
from jigsaw_rules.pairs import pairs_evidence
from jigsaw_rules.robustness import robustness_evidence
from jigsaw_rules.gate import feature_gate
from jigsaw_rules.instructions import instruction_evidence

controls = feature_evidence(root)
research = research_evidence(root)
sensitivity = diagnostic_evidence(root)
pairs = pairs_evidence(root)
robustness = robustness_evidence(root)
assert all(item is not None for item in (controls, research, sensitivity, pairs, robustness))
gate = feature_gate(root)
instructions = instruction_evidence(root)
print("Verified research runs:", gate["studies"])
"""


def notebooks():
    outputs = {}
    outputs["notebooks/00_environment_and_data.ipynb"] = notebook(
        [
            (
                "md",
                "# 00 · Environment and evidence\n\n**Question:** Which data, code and environment produced the results?\n\nThe five public notebooks are executed views of checksummed competition-data aggregates. Private comments, labels by row, model weights and prediction files are excluded. Opening or executing these notebooks needs no AWS account or model downloads.",
            ),
            ("code", SETUP),
            (
                "md",
                "## Reproduction environment\nThe original reference environment and the current notebook environment are separate records. `uv.lock` fixes dependencies; a historical result is not relabeled as a fresh experiment.",
            ),
            (
                "code",
                'recorded = baseline["provenance"]["environment"]\ncurrent = environment()\nprint("Recorded Python:", recorded["python"])\nprint("Notebook Python:", current["python"])\ndisplay(pd.DataFrame({"Original reference": recorded["packages"], "Notebook execution": current["packages"]}))',
            ),
            (
                "md",
                "## Data identity and limits\nEach observation contains a comment, rule, community, four labeled support examples and a binary target. There are no timestamps or longitudinal entity records. The ten-row preview test is not independent evidence of performance.",
            ),
            (
                "code",
                'audit = baseline["audit"]\ndisplay(pd.DataFrame({"File": ["train.csv", "test.csv (preview)"], "Rows": [audit["train_rows"], audit["preview_test_rows"]]}))\nprint("Training SHA-256:", baseline["training_sha256"])\nprint("Reference run:", baseline["run_id"])',
            ),
            (
                "md",
                "## Research lineage\nEach study records its source fingerprint, input identity, configuration and complete-stage checksums. The notebook executor additionally binds its cache to source, reports, environment and executed output.",
            ),
            ("code", RESEARCH_SETUP),
            (
                "code",
                'display(pd.DataFrame([{"Study": name, "Run": run_id} for name, run_id in gate["studies"].items()]))',
            ),
            (
                "md",
                "## Review path\nContinue to [01 · Validation](01_data_and_validation.ipynb), then [02 · Feature research](02_baseline_and_review.ipynb). [03 · Results](03_saved_results.ipynb) provides the compact decision view. Reading aggregate evidence and recomputing private OOF metrics are distinct operations; the latter is documented in [VALIDATION.md](../docs/VALIDATION.md).",
            ),
        ]
    )
    outputs["notebooks/01_data_and_validation.ipynb"] = notebook(
        [
            (
                "md",
                "# 01 · Data and validation\n\n**Question:** What can these splits establish about unseen community rules?\n\nOnly two rule types have labels. We explicitly retain that limit throughout the research; thousands of candidate features cannot manufacture additional policy coverage.",
            ),
            ("code", SETUP),
            ("code", RESEARCH_SETUP),
            (
                "md",
                "## Rule and duplicate audit\nCount and violation prevalence are reported together. Repeated comments are grouped before splitting. Preview-test overlap makes that file useful for schema checks only.",
            ),
            (
                "code",
                'audit = baseline["audit"]\nby_rule = pd.DataFrame(audit["by_rule"])\nby_rule["rule"] = by_rule["rule"].str.split(":").str[0]\ndisplay(by_rule.round(4))\ndisplay(pd.DataFrame({"Finding": ["Duplicate training bodies", "Train / preview-test overlap", "Comment equals its own support example"], "Rows": [audit["duplicate_training_bodies"], audit["train_test_body_overlap"], sensitivity["audit"]["self_support_rows"]]}))',
            ),
            (
                "md",
                "## Two validation questions\n**Familiar-rule CV** stratifies by rule and target, grouping normalized duplicate comments. **Held-out-rule CV** excludes the evaluated rule from training. Both purge any training row whose comment or supplied examples contain a validation comment. The research reuses the preserved row assignments.\n\nThis is stringent: the familiar-rule folds retain only 237–287 training rows. Sparse fold estimates are a real limitation, not grounds for weakening the boundary.",
            ),
            (
                "code",
                'split_audit = pd.DataFrame(robustness["audit"]["folds"])\ndisplay(split_audit.rename(columns={"before": "Exact-purged train rows", "after": "Near-copy-purged train rows"}))',
            ),
            (
                "md",
                "## Learned features obey the same boundary\nVocabulary, IDF, scalers, empirical ranks, feature screening and NB token weights use outer training rows only. Target/context encodings additionally use three inner comment-group folds, purging inner validation comments from training comments and examples; the prior is fitted inside each inner fold. Unseen groups fall back to training priors.\n\nProvided positive/negative examples are legitimate per-row inputs. Their semantics are not the current row's unknown target. Frozen encoders use no competition-label fitting. The 18 self-support rows are excluded in a separate scoring sensitivity analysis.",
            ),
            (
                "md",
                "## Approximate-copy stress test\nA second pass requires character-ngram cosine ≥0.95, token-set Jaccard ≥0.90 and at least 40 characters, with a fixed hashing representation. It uses text only, removes training rows, and keeps validation rows fixed. Thresholds were not optimized on outcomes. No additional copies met both thresholds after exact purging; this does **not** prove paraphrase or shared-origin isolation.",
            ),
            (
                "code",
                'print(robustness["audit"]["interpretation"])\ndisplay(metric_table(robustness["results"], heldout=True))',
            ),
            (
                "md",
                "## Inference scope\nTemporal, rolling, lag, season, team, opponent and coaching variables are unavailable or inapplicable. Row order is not time. There is no legitimate external ranking system for these comments. External model weights are pinned and license documented; competition-specific permission for any future data augmentation must be verified before use.\n\nThe project computes **rule macro ROC AUC** as its local approximation to the competition's column-averaged AUC description. Pooled AUC is separate. No official scoring implementation or independent Kaggle score has confirmed equivalence.\n\nContinue to [02 · Feature research](02_baseline_and_review.ipynb).",
            ),
        ]
    )
    outputs["notebooks/02_baseline_and_review.ipynb"] = notebook(
        [
            (
                "md",
                "# 02 · Feature engineering as a research gate\n\n**Question:** Which representations improve rule-conditioned transfer, and is the evidence strong enough to finish feature engineering?\n\n**Gate: OPEN.** Broad generation, screening and controlled experiments have executed. Independent confirmation of the selected feature representation has not. The research keeps the original lexical reference and does not trigger final retraining.",
            ),
            ("code", SETUP),
            ("code", RESEARCH_SETUP),
            (
                "md",
                "## 1 · Establish the reference\nComment-only TF-IDF measures lexical transfer. The rule/example reference adds similarities to the rule and supplied examples. The early four-candidate study isolated rule text, support contrasts, structure and their combination. These are historical, preserved experiments.",
            ),
            (
                "code",
                'display(metric_table(baseline["results"], heldout=True))\ndisplay(metric_table(controls["results"], heldout=True))',
            ),
            (
                "md",
                "## 2 · Search broadly, with a rationale\nThe broad bank contains word and character ngrams; 1,512 structural/context interactions; 207 lexical similarity transformations; support-conditioned token products; 6,144 frozen embedding coordinates/interactions; 32 semantic geometry summaries; 207 training-relative ranks; community indicators; and nine nested target/frequency/novelty candidates.\n\nThe separate joint-text probe adds 99 entailment/contradiction/neutral transformations from frozen DeBERTa. Support summaries are invariant to swapping the two positive or two negative examples. Every family has provenance, availability and leakage notes in the [feature catalog](../docs/FEATURE_RESEARCH.md). Counts below are **per-fold candidate columns**, not a sum over repeated CV fits.",
            ),
            (
                "code",
                'screen = pd.DataFrame(research["screening"]["folds"])\ncounts = screen.groupby(["protocol", "fold"])[["candidates", "retained", "rejected"]].sum()\njoint_counts = pd.DataFrame(pairs["screening"])\ncombined_counts = counts + joint_counts[joint_counts.model == "all_nli"].set_index(["protocol", "fold"])[["candidates", "retained", "rejected"]]\nif instructions is not None:\n    instruction_counts = pd.DataFrame(instructions["screening"])\n    combined_counts += instruction_counts[instruction_counts.model == "instruction_features"].set_index(["protocol", "fold"])[["candidates", "retained", "rejected"]]\nprint("Per-fold screened bank totals, including each joint-text family once:")\ndisplay(combined_counts)\ndisplay(screen.groupby("family")[["candidates", "retained"]].agg(["min", "max"]))\ndisplay_figure(root, "screening")',
            ),
            (
                "md",
                "## 3 · Screen inside training boundaries\nThe screen rejects nonfinite/schema errors, constant or nearly constant columns, features seen in fewer than three training rows, exact duplicates and suspicious perfect separators. Training-label effect scores impose fixed budgets. Dense redundancy screening checks |correlation| ≥0.995 within the strongest budgeted pool; sparse duplicates use column hashes to avoid a quadratic comparison of vocabulary columns.\n\nNo outer validation statistic participates. Detailed per-feature decisions and selected column hashes are private reproducibility artifacts. Unscreened controls below test whether the screen itself discarded useful lexical signal.",
            ),
            (
                "code",
                'decisions = pd.DataFrame([{"Family": r["family"], "Reason": reason, "Columns": count} for r in research["screening"]["folds"] for reason, count in r["decisions"].items()])\ndisplay(decisions.groupby("Reason").Columns.sum().to_frame("Column decisions across five folds"))\nassert (screen.candidates == screen.retained + screen.rejected).all()\nassert screen.missing_or_nonfinite.sum() == 0',
            ),
            (
                "md",
                "## 4 · Separate feature effects from model tuning\nTwenty-one broad configurations run on the same five saved folds: 105 fits. A fixed logistic classifier (C=2, liblinear, seed 2025) measures additions to the same screened-word control, family-only controls and leave-one-family-out ablations. No hyperparameter sweep is used. Scaling and fixed per-family budgets are part of the tested representation.\n\nIntervals use 1,000 paired normalized-comment-group bootstrap draws. The wide intervals adjust simultaneously for the study's planned contrasts. They condition on fixed OOF predictions and two rules; they do not correct for repeated research across studies or establish performance on arbitrary unseen policies.",
            ),
            (
                "code",
                'display_figure(root, "ablation")\neffects = pd.DataFrame(research["uncertainty"])\nheld_effects = effects[effects.protocol == "heldout_rule"]\ndisplay(held_effects.loc[held_effects.contrast.str.startswith("add_"), ["contrast", "observed_delta", "ci_lower", "ci_upper", "simultaneous_lower", "simultaneous_upper"]].round(4))',
            ),
            (
                "md",
                "## 5 · Keep the negative findings visible\nCharacter ngrams and compact semantic geometry improve the matched screened-word control at pointwise confidence, but their simultaneous intervals include zero. Raw embedding coordinates and the structural expansion hurt transfer. Community and nested target encodings do not establish robust gains. The all-family representation is substantially worse than the historical reference. More features are not automatically better.",
            ),
            (
                "code",
                'display(metric_table(research["results"], heldout=True).sort_values("Rule macro AUC", ascending=False))\ndisplay(held_effects.loc[held_effects.contrast.str.startswith("remove_"), ["contrast", "observed_delta", "simultaneous_lower", "simultaneous_upper"]].round(4))',
            ),
            (
                "md",
                "## 6 · Attribute performance and inspect stability\nWhole-family permutation within each validation rule measures dependence while avoiding impossible cross-rule shuffles. Repeated permutations and coefficient summaries are descriptive; correlated families can substitute for one another. Retained-name Jaccard measures selection stability separately from predictive usefulness. Raw token identities are excluded from public output.\n\nSHAP is not added merely as decoration: these fixed linear probes already expose coefficients, matched ablations and group permutation. Those diagnostics directly answer the current feature questions without another correlated attribution summary.",
            ),
            (
                "code",
                'display_figure(root, "stability")\npermutation = pd.DataFrame(research["importance"])\npermutation["mean_permutation_drop"] = permutation.within_rule_permutation_auc_drops.map(lambda values: sum(values) / len(values))\nselected_importance = permutation[(permutation.protocol == "heldout_rule") & (permutation.model == "all_transfer")]\ndisplay(selected_importance.groupby("family").agg(mean_auc_drop=("mean_permutation_drop", "mean"), minimum_fold_drop=("mean_permutation_drop", "min"), maximum_fold_drop=("mean_permutation_drop", "max"), mean_absolute_logit=("mean_absolute_logit_contribution", "mean")).round(4))',
            ),
            (
                "md",
                "## 7 · Challenge the selection and representation choices\nFifty-five additional fits compare full word/character vocabularies, training-only NB log-count weighting, and five 128-component SVD representations fitted only on outer training rows. None of the low-rank alternatives improves the original reference. Label-free lexical support scores and seven frozen Qwen geometry scores isolate aggregation effects. The normalized positive/negative centroid margin is the strongest observed geometry alternative; its uncertainty still matters. Self-support exclusion preserves the main ranking advantage, but does not create new policies.",
            ),
            (
                "code",
                'display(metric_table(sensitivity["results"], heldout=True).sort_values("Rule macro AUC", ascending=False))\nintervals = pd.DataFrame(sensitivity["uncertainty"])\ndisplay(intervals.loc[(intervals.protocol == "heldout_rule") & intervals.contrast.isin(["qwen_centroid", "character_full", "reweight_word", "reweight_character"]), ["contrast", "observed_delta", "ci_lower", "ci_upper", "simultaneous_lower", "simultaneous_upper"]].round(4))\nselected = ["original_reference", "qwen_centroid", "qwen_maximum", "word_semantic_scalar"]\nrows = [r for r in sensitivity["sensitivity"] if r["protocol"] == "heldout_rule" and r["model"] in selected]\ndisplay(metric_table(rows, heldout=True))',
            ),
            (
                "md",
                "## 8 · Joint text encoding without model fine-tuning\nThe pinned `cross-encoder/nli-deberta-v3-small` model jointly reads the comment with (a) a fixed violation hypothesis, (b) a compliance hypothesis, or (c) each supplied example. The encoder stays frozen. Rule-only, support-only, combined, word-plus-NLI and broad-plus-NLI probes use the same fixed classifier and folds: 25 fits. A fixed zero-shot rule margin is reported separately.\n\nWeights are trained on general NLI, not moderation judgments. Entailment is a feature hypothesis, not an assertion that NLI understands every policy. Inference batches have immutable hashes, truncation diagnostics, verified completion markers and S3 checkpoints.",
            ),
            (
                "code",
                'display(metric_table(pairs["results"], heldout=True).sort_values("Rule macro AUC", ascending=False))\nni = pd.DataFrame(pairs["uncertainty"])\ndisplay(ni.loc[ni.protocol == "heldout_rule", ["contrast", "observed_delta", "ci_lower", "ci_upper", "simultaneous_lower", "simultaneous_upper"]].round(4))\ndisplay(pd.Series({k: v for k, v in pairs["inference"].items() if k != "contract"}, name="NLI inference"))\nns = pd.DataFrame(pairs["screening"])\ndisplay(ns.groupby("model")[["candidates", "retained"]].agg(["min", "max"]))',
            ),
            (
                "md",
                "## 9 · Fixed instruction likelihoods\nA separate frozen Qwen3-0.6B probe scores Yes/No token likelihoods under three fixed templates: rule only, examples only, and both. Thinking and generation are disabled. There is no competition-label fine-tuning or prompt sweep. Thirty-six probability, margin, answer-mass and context-difference candidates feed the same fixed classifier, with lexical and semantic additions. Support examples are sorted within their label groups. Field budgets preserve the question when text is long. The completed probe does not improve held-out-rule performance. These negative results apply to this fixed small model and its documented field budgets; they do not establish failure of every instruction model.",
            ),
            (
                "code",
                'if instructions is None:\n    print("The bounded instruction-feature experiment has no published results yet.")\nelse:\n    display(metric_table(instructions["results"], heldout=True).sort_values("Rule macro AUC", ascending=False))\n    ii = pd.DataFrame(instructions["uncertainty"])\n    display(ii.loc[ii.protocol == "heldout_rule", ["contrast", "observed_delta", "ci_lower", "ci_upper", "simultaneous_lower", "simultaneous_upper"]].round(4))\n    display(pd.Series({k: v for k, v in instructions["inference"].items() if k != "contract"}, name="Instruction inference"))',
            ),
            (
                "md",
                "## 10 · Completion is an evidence decision\nBroad exploration is implemented and executed, but the best observed score is selected from many experiments on the same two rule types. The honest next research requirement is independent policy coverage or a separately locked confirmation design. Additional random interactions or a larger classifier would not resolve that limit.\n\nThe original offline classifier deliberately remains the reference; it does not silently consume an unpromoted feature bank. Final feature dimensionality and model promotion are unresolved. The gate cannot be closed by a successful pipeline run or a high candidate count.",
            ),
            (
                "code",
                'display(pd.DataFrame(gate["criteria"]))\nprint("Final training justified:", gate["final_training_authorized_by_evidence"])\nprint(gate["decision"])',
            ),
            (
                "md",
                "## Reproduce or inspect\nThe default notebook reads evidence only. The tested commands `jigsaw research --export`, `jigsaw diagnostics`, `jigsaw robustness`, `jigsaw pairs` and `jigsaw instructions` reproduce the separate studies after restoring private artifacts. Completed folds and encoding batches resume; changed dependencies or source create distinct identities. [VALIDATION.md](../docs/VALIDATION.md) records executed checks, cloud lineage and limits.\n\nContinue to [03 · Results and decision](03_saved_results.ipynb).",
            ),
        ]
    )
    outputs["notebooks/03_saved_results.ipynb"] = notebook(
        [
            (
                "md",
                "# 03 · Results and model decision\n\n**Decision:** Keep feature engineering open and retain the lexical reference. A large executed search has exposed useful semantic signals and many negative findings. It has not established a final representation with independent confirmation.",
            ),
            ("code", SETUP),
            ("code", RESEARCH_SETUP),
            (
                "md",
                "## Compare representations on identical held-out rows\nThese are local rule-macro AUC results, not leaderboard scores. The same fixed classifier is used for learned feature probes. Frozen scores require no competition-label fitting. Probability losses remain visible alongside ranking.",
            ),
            (
                "code",
                'records = [r for r in baseline["results"] if r["model"] == "rule_examples"]\nrecords += [r for r in research["results"] if r["model"] in ["word_screened", "word_semantic_scalar", "all_transfer", "all_with_metadata"]]\nrecords += [r for r in sensitivity["results"] if r["model"] in ["qwen_centroid", "character_full", "word_character_nb"]]\nrecords += pairs["results"]\nif instructions is not None:\n    records += instructions["results"]\ndisplay(metric_table(records, heldout=True).sort_values("Rule macro AUC", ascending=False))',
            ),
            (
                "md",
                "## What the feature work established\nCompact semantic relationships and character patterns deserve more attention than high-dimensional structural expansion or raw coordinate selection. Target/context and community features have not justified their availability and transfer risks. Unscreened controls are essential: screening reduces size but can also remove useful signal.\n\nSee [02 · Feature research](02_baseline_and_review.ipynb) for candidate counts, matched additions/removals, simultaneous intervals, stability, negative findings and the feature catalog.",
            ),
            ("code", 'display_figure(root, "ablation")'),
            (
                "md",
                "## Promotion and artifact lineage\nThe notebooks consume the latest verified research aggregates. The offline submission path still fits the explicitly named original lexical reference. No final candidate has been selected, no novel feature artifact has been promoted, and no CSV has been submitted to Kaggle. This is a deliberate open gate, not stale-model reuse disguised as a new result.",
            ),
            (
                "code",
                'display(pd.DataFrame(gate["criteria"]))\nprint("Gate:", gate["status"])\nprint(gate["decision"])',
            ),
            (
                "md",
                "## Remaining uncertainty\nTwo labeled policies cannot establish broad unseen-rule generalization. Repeatedly inspecting the same held-out scores also creates selection bias; within-study simultaneous intervals only address part of that problem. Calibration, final representation selection and final retraining remain downstream of the feature gate.\n\n[04 · Semantic diagnostics](04_semantic_benchmark.ipynb) examines policy-level behavior and probability quality. The [research record](../docs/FEATURE_RESEARCH.md) describes which high-value avenues were explored, ruled inapplicable, or remain unresolved.",
            ),
        ]
    )
    outputs["notebooks/04_semantic_benchmark.ipynb"] = notebook(
        [
            (
                "md",
                "# 04 · Semantic diagnostics\n\n**Question:** Why do compact semantic relationships transfer more plausibly than raw embedding coordinates?\n\nFrozen Qwen embeddings summarize each comment and its supplied examples. The joint DeBERTa probe adds rule/support entailment relationships. Neither encoder is fine-tuned on competition labels.",
            ),
            ("code", SETUP),
            ("code", RESEARCH_SETUP),
            (
                "md",
                "## Examine each observed policy\nA mean over two policies can conceal deterioration on one. These tables preserve per-policy AUC and do not treat identical label-free predictions across validation protocols as independent replications.",
            ),
            (
                "code",
                'records = baseline["results"] + semantic["results"]\nrecords += [r for r in sensitivity["results"] if r["model"] in ["qwen_centroid", "qwen_mean", "qwen_maximum"]]\nrecords += pairs["results"]\nif instructions is not None:\n    records += instructions["results"]\nper_rule = pd.DataFrame([{"Representation": r["model"], "Rule": rule.split(":")[0], "ROC AUC": auc} for r in records if r["protocol"] == "heldout_rule" for rule, auc in r["metrics"]["per_rule_auc"].items()])\ndisplay(per_rule.pivot(index="Representation", columns="Rule", values="ROC AUC").round(4))',
            ),
            (
                "md",
                "## Probability quality and operating points\nCalibration error uses ten equal-width bins. Precision, recall and F1 use a fixed diagnostic threshold of 0.5; threshold selection and calibration fitting are not claimed as completed. The temperatures of label-free scores are fixed transformations, not fitted calibrators.",
            ),
            (
                "code",
                'diagnostics = pd.DataFrame([{"Representation": r["model"], "Pooled AUC": r["metrics"]["pooled_auc"], "Calibration error": r["metrics"]["ece_10_equal_width_bins"], "Precision@0.5": r["metrics"]["precision_at_0_5"], "Recall@0.5": r["metrics"]["recall_at_0_5"], "F1@0.5": r["metrics"]["f1_at_0_5"]} for r in records if r["protocol"] == "heldout_rule"])\ndisplay(diagnostics.round(4))',
            ),
            (
                "md",
                "## Compute and resumability\nThe original Qwen run encoded 1,875 unique texts; the broad study reused all verified embeddings. The NLI probe deduplicates comment–hypothesis pairs and records truncated pairs. Batches are hashed and published only after all files complete. The bounded AWS processing job checkpoints completed work to its own experiment prefix.",
            ),
            (
                "code",
                'timing = semantic["timing"]\nprint("Original Qwen encoder:", timing["encoder"])\ndisplay(pd.Series({k: v for k, v in pairs["inference"].items() if k != "contract"}, name="NLI inference"))',
            ),
            (
                "md",
                "## Interpretation and next research boundary\nA positive-versus-negative centroid comparison uses the supplied task context directly and does not need hundreds of learned coefficients from a tiny fold. This is a plausible explanation for its transfer behavior, not a causal proof. Raw coordinates and many structural interactions are unstable across policies. General NLI features must earn their place through rule/support ablations, rather than being assumed superior because they use a transformer.\n\nFurther prompt or encoder searches would reuse already-inspected policies. Independent confirmation and a documented, legitimate source of broader rule coverage are the major unresolved avenues. [02 · Feature gate](02_baseline_and_review.ipynb) remains open. The [Kaggle notebook](../kaggle/submission.ipynb) remains the user's offline lexical-reference workflow; no automatic upload is performed.",
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
