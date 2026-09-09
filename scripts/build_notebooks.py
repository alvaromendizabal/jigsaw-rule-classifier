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
print("Historical references: 2,029 rows, two policies; expanded research is a separate cohort.")
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
from build_release_report import display_boundary
from jigsaw_rules.features import feature_evidence
from jigsaw_rules.research import research_evidence
from jigsaw_rules.diagnostics import diagnostic_evidence
from jigsaw_rules.pairs import pairs_evidence
from jigsaw_rules.robustness import robustness_evidence
from jigsaw_rules.gate import feature_gate
from jigsaw_rules.instructions import instruction_evidence
from jigsaw_rules.released import released_evidence
from jigsaw_rules.expanded import expanded_evidence
from jigsaw_rules.retrieval import retrieval_evidence
from jigsaw_rules.resolution import resolution_evidence
from jigsaw_rules.formatting import formatting_evidence
from jigsaw_rules.feature_decision import decision_evidence
from build_expanded_report import display_figure as display_expanded
from build_formatting_report import display_figure as display_formatting

expanded = expanded_evidence(root)
retrieval = retrieval_evidence(root)
resolution = resolution_evidence(root)
formatting = formatting_evidence(root)
decision = decision_evidence(root)
assert expanded is not None and retrieval is not None and resolution is not None
assert formatting is not None and decision is not None
controls = feature_evidence(root)
research = research_evidence(root)
sensitivity = diagnostic_evidence(root)
pairs = pairs_evidence(root)
robustness = robustness_evidence(root)
assert all(item is not None for item in (controls, research, sensitivity, pairs, robustness))
gate = feature_gate(root)
instructions = instruction_evidence(root)
released = released_evidence(root)
assert released is not None
print("Verified expanded study:", expanded["metadata"]["run_id"])
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
                "# 01 · Data and validation\n\n**Question:** What can these splits establish about unseen community rules?\n\nThe expanded study develops on four policies and keeps two entire policy types reserved. The original two-policy experiments remain historical evidence. A protected boundary improves the design; it does not itself establish generalization.",
            ),
            ("code", SETUP),
            ("code", RESEARCH_SETUP),
            (
                "md",
                "## A public source, with a prospective research reserve\nThe host released six-policy evaluation data after the competition. Source version, archive/member hashes and original train/preview parity are verified. The partition uses policy, Public/Private metadata and normalized text before research targets are materialized. Historical exposure removes 54 reserved rows; research body/support overlap removes 1,323 research rows. No model-selection function opens the released solution file.",
            ),
            (
                "code",
                'display_boundary(root)\nprint("Protected rows:", released["boundary"]["role_counts"]["confirmation"])\nprint("Target access:", expanded["audit"]["confirmation_labels_accessed"])',
            ),
            (
                "md",
                "## Policy prevalence and repeated annotations\nClass balance differs sharply by policy, so pooled accuracy or pooled AUC can mislead. Equal-weight policy AUC keeps the policy-transfer question visible. Repeated bodies can have different supplied examples; they remain grouped and receive sensitivity analysis rather than silent reconciliation.",
            ),
            (
                "code",
                'audit = expanded["audit"]\ncounts = pd.DataFrame(audit["class_counts"])\ncounts["positive_rate"] = counts["sum"] / counts["size"]\ncounts["rule"] = counts.rule.str.split(":").str[0]\ndisplay(counts.rename(columns={"size": "Rows", "sum": "Violations"}).round(4))\ndisplay(pd.Series({name: audit[name] for name in ["development_rows", "repeated_body_policy_rows", "conflicting_groups", "conflicting_rows"]}, name="Development audit"))',
            ),
            (
                "md",
                "## Two validation questions, one strict text boundary\n**Familiar-policy CV** groups normalized comment bodies and stratifies by policy/target. **Held-out-policy CV** excludes each evaluated policy from training. Both remove training rows whose comment or any supplied example matches a validation comment. The assignments are frozen before model comparisons.\n\nThis purge substantially reduces available training data; the table makes that cost explicit. It cannot establish independence of paraphrases or common conversation origin, because conversation IDs, authors and timestamps are unavailable.",
            ),
            (
                "code",
                'display(pd.DataFrame(audit["folds"])[["protocol", "fold", "training_rows", "validation_rows", "purged_training_rows"]])',
            ),
            (
                "md",
                "## Learned and target-derived features stay inside training\nVocabulary, IDF, scaling, reference percentiles, screening, NB weights and SVD use only each purged training partition. Target/context encodings use three inner grouped folds, inner support purging and inner-only priors. Unknown groups fall back to those training priors. Provided positive/negative examples are legitimate inference inputs, not the current row's target. Frozen encoders fit no competition labels.\n\nThe conflict sensitivity identifies conflicting groups from training labels only. The approximate-copy sensitivity uses character cosine ≥0.95, token Jaccard ≥0.90 and at least 40 characters, without target access. Both keep validation rows fixed; unchanged training sets reuse their primary fit.",
            ),
            (
                "code",
                'filters = pd.DataFrame(audit["training_sensitivities"])\nfilters["removed"] = filters.before - filters.after\ndisplay(filters.groupby(["protocol", "sensitivity", "model"])[["removed", "reused_primary"]].sum())',
            ),
            (
                "md",
                "## What the metric and schema permit\nThe project reports **policy-macro ROC AUC** and pooled AUC separately. The official column-averaged AUC description and host per-policy attachment strongly corroborate equal rule weighting: 2,428 of 2,437 complete published rows agree within 1e-6. Nine discrepancies and six incomplete rows remain explicit; executable scorer parity and a project leaderboard score are not claimed.\n\nTemporal, rolling, lag, season, team, opponent, coaching and external-rating variables are unavailable or inapplicable. Row order is not time. Current subreddit pages are not historical snapshots of supplied policies. External model revisions and licensing are documented; pretraining-overlap clearance is not claimed.\n\n[02 · Feature research](02_baseline_and_review.ipynb) applies these boundaries to every compared representation.",
            ),
        ]
    )
    outputs["notebooks/02_baseline_and_review.ipynb"] = notebook(
        [
            (
                "md",
                "# 02 · Feature engineering as a research gate\n\n**Decision:** The declared feature-research phase is complete. Retain the original frozen centroid for unseen policies. The 323-fit four-policy campaign, targeted semantic comparisons and fixed complementarity control support diminishing returns within this scope. Independent confirmation and product delivery remain separate gates.\n\nThis notebook explains candidate generation, training-only screening, matched ablations, negative findings and the stopping decision. The retained centroid reaches 0.7042 transfer AUC versus 0.4728 for the matched lexical reference. Compact semantic comparisons give the strongest family addition. A higher feature count is not the selection criterion.",
            ),
            ("code", SETUP),
            ("code", RESEARCH_SETUP),
            (
                "md",
                "## 1 · The research question and protected boundary\nThe historical search covered two policies. The host's released corpus now permits 11,135 development rows across advertising, legal advice, medical advice and illegal-activity promotion. The 43,576 reserved rows include financial advice and spoilers, whose labels remain outside research. [Pre-score protocol](../docs/EXPANDED_STUDY.md) · [Data provenance](../docs/RELEASED_DATA.md).\n\nUse three familiar-policy grouped folds and four held-out-policy folds. Every training body and supplied example is checked against validation bodies. Retain repeated annotations in the primary experiment; handle their influence through grouped validation and declared sensitivity analyses.",
            ),
            (
                "code",
                'audit = expanded["audit"]\nprint("Protocol commit:", expanded["metadata"]["protocol_commit"])\nprint("Development rows / policies:", audit["development_rows"], audit["development_policies"])\nprint("Primary / total fitted models:", audit["primary_fitted_models"], audit["actual_fitted_models"])\nprint("Confirmation labels accessed:", audit["confirmation_labels_accessed"])\ndisplay(pd.DataFrame(audit["folds"])[["protocol", "fold", "training_rows", "validation_rows", "purged_training_rows"]])',
            ),
            (
                "md",
                "## 2 · Search broadly, with a reason for each family\nWords and character patterns capture phrasing and morphology. Style measures test requests, links and emphasis. Rule/support similarities, token products and semantic geometry test whether the comment resembles prohibited examples more than permitted ones. Training-reference ranks test relative position; community frequencies and cross-fitted target context test group effects and shortcut risk.\n\nThe encoder is already rule-conditioned. Its raw coordinates, support products and compact scalar comparisons are separate representations. Full-vocabulary, NB-weighted and five fixed SVD controls test whether screening or representation scale explains an apparent gain. [Detailed family rationale, availability and leakage analysis](../docs/FEATURE_RESEARCH.md#candidate-space-and-availability).\n\nNo timestamps, author histories, threads, opponents, coaches or ratings exist in the schema. Temporal and historical sports/customer-style features would invent unavailable information.",
            ),
            (
                "code",
                'screens = pd.DataFrame(expanded["screening"]["families"] + retrieval["screening"] + formatting["screening"])\nassert (screens.candidates == screens.retained + screens.rejected).all()\ntotals = screens.groupby(["protocol", "fold"])[["candidates", "retained", "rejected"]].sum()\ndisplay(totals.rename(columns={"candidates": "All candidates", "retained": "All retained", "rejected": "All rejected"}))\nprint("Retained widths describe separate banks, not the selected centroid.")\ndisplay_expanded(root, "screening")',
            ),
            (
                "md",
                "## 3 · Fit every learned transform inside training\nVocabulary/IDF, rarity and redundancy decisions, effect-score screening, scaling, ranks, NB weights and SVD use the purged outer-training rows. Target/context features use inner grouped cross-fitting and inner text purging. Unknown groups fall back to inner-training priors. No validation target selects a column.\n\nA retained bank is not a final feature set: each model consumes only its declared families. The full candidate catalogs, selected names and training IDs remain checksummed private artifacts. Logistic regression stays at C=2; no hyperparameter search compensates for weak representations.",
            ),
            (
                "code",
                'decisions = pd.DataFrame([{"protocol": row["protocol"], "fold": row["fold"], "family": row["family"], **row["decisions"]} for row in expanded["screening"]["families"]]).fillna(0)\ndisplay(decisions.drop(columns=["protocol", "fold"]).groupby("family").sum().astype(int))',
            ),
            (
                "md",
                "## 4 · Attribute improvement through matched additions and removals\nEach addition starts from the same screened-word model. Each removal starts from the same all-transfer representation. These contrasts test incremental utility, while comparisons with the historical-style reference also change representation/scaling conventions. Paired normalized-body bootstrap draws share weights across models; simultaneous intervals cover the declared contrasts. They condition on fixed OOF predictions and four observed policies, not arbitrary future rules or all adaptive research decisions.",
            ),
            (
                "code",
                'display_expanded(root, "ablation")\ncontrasts = pd.DataFrame(expanded["uncertainty"])\nremovals = contrasts[(contrasts.protocol == "heldout_rule") & contrasts.contrast.str.startswith("remove_")]\ndisplay(removals[["contrast", "observed_delta", "ci_lower", "ci_upper", "simultaneous_lower", "simultaneous_upper"]].round(4))',
            ),
            (
                "md",
                "## 5 · Check whether the selected signals are stable\nSelection overlap measures whether folds retain the same columns; it does not establish usefulness. Group permutation shuffles each group's linear contribution within policy, preserving policy prevalence. Correlated families can substitute for one another, so the direct ablations remain the primary contribution evidence. Transparent linear contributions make a second SHAP plot unnecessary here.",
            ),
            (
                "code",
                'stability = pd.DataFrame(expanded["screening"]["stability"])\ndisplay(stability[stability.protocol == "heldout_rule"].groupby("family").retained_jaccard.agg(["min", "mean", "max"]).round(3))\nimportance = pd.DataFrame([{**r, "mean_permutation_auc_drop": sum(r["within_rule_permutation_auc_drops"]) / len(r["within_rule_permutation_auc_drops"])} for r in expanded["importance"] if r["protocol"] == "heldout_rule" and r["model"] == "all_transfer"])\ndisplay(importance.groupby("family")[["mean_absolute_logit_contribution", "mean_permutation_auc_drop"]].mean().round(4))',
            ),
            (
                "md",
                "## 6 · Challenge duplicate, label and support dependence\nTraining-conflict exclusions use training labels only. Near-copy removal uses text only and fixed character-cosine/Jaccard thresholds. The validation rows stay fixed for both refits. Separately, descriptive metrics give equal total weight to each body/policy group or exclude conflicting groups from scoring. None of these diagnostic outcomes changes the primary folds.\n\nThe centroid stress tests average all four one-positive/one-negative example choices, shuffle supplied contexts within policy, and exclude exact self-support matches. They test dependence on the supplied examples; they do not establish paraphrase independence or resilience to absent rule text.",
            ),
            (
                "code",
                'filters = pd.DataFrame(audit["training_sensitivities"])\nfilters["removed"] = filters.before - filters.after\ndisplay(filters.groupby(["protocol", "sensitivity", "model"])[["removed", "reused_primary"]].sum())\nrobust_records = [r for r in expanded["results"] if r["model"] in ["rule_examples", "character_full", "qwen_centroid"] or "exclude_conflicts" in r["model"] or "purge_near_copies" in r["model"]]\ndisplay(metric_table(robust_records, heldout=True))\ndisplay(pd.DataFrame([{ "Model": r["model"], "Primary AUC": r["metrics"]["rule_macro_auc"], "Equal group weight": r["equal_body_policy_weight"]["rule_macro_auc"], "Exclude conflicting groups": r["excluding_conflicts"]["rule_macro_auc"]} for r in robust_records if r["protocol"] == "heldout_rule"]).round(4))',
            ),
            (
                "md",
                "## 7 · Keep negative findings and scope limits visible\nThe original two-policy study also tested a frozen DeBERTa NLI cross-encoder and three fixed Qwen instruction-likelihood templates. Neither improved its lexical reference. These are useful negative findings for those models, inputs and policies; they do not establish failure on every policy. The expanded study tests the existing broad families and representation controls, rather than quietly attributing new-policy results to unexecuted NLI/instruction experiments.",
            ),
            (
                "code",
                'historical = [r for r in pairs["results"] if r["model"] in ["rule_nli", "word_nli", "all_nli"]]\nif instructions is not None:\n    historical += instructions["results"]\nprint("Historical two-policy evidence only:")\ndisplay(metric_table(historical, heldout=True))',
            ),
            (
                "md",
                "## 8 · Test a target-derived geometry hypothesis\nThe raw-coordinate and linear projection controls leave one plausible avenue: local neighborhoods and prototypes of labeled training examples. The preregistered retrieval extension uses three frozen geometric spaces and 309 neighborhood, prototype, nonlinear and supplied-margin interaction candidates. Training-only screening retains at most 64. It adds the family to four exact saved controls and also tests retrieval alone: 35 additional fixed fits.\n\nEvery labeled bank excludes the query policy, including familiar policies. Training features leave out each entire policy and purge reference body/support matches; validation transformation rejects a target column. Tests flip all labels of a query policy and prove its own feature rows are unchanged. This tests transferable geometry without assigning labels from one policy to another. [Protocol and rationale](../docs/RETRIEVAL_STUDY.md).",
            ),
            (
                "code",
                'display(metric_table(retrieval["results"], heldout=True))\nrc = pd.DataFrame(retrieval["uncertainty"])\ndisplay(rc[(rc.protocol == "heldout_rule") & rc.contrast.str.startswith("add_retrieval")][["contrast", "observed_delta", "ci_lower", "ci_upper", "simultaneous_lower", "simultaneous_upper"]].round(4))\ndisplay(pd.DataFrame(retrieval["screening"])[["protocol", "fold", "candidates", "retained", "rejected"]])',
            ),
            (
                "md",
                "## 9 · Check the model's trained embedding resolutions\nMatryoshka prefixes are different from learned SVD projections or arbitrary coordinate selection. Six preregistered resolutions (32–1,024 dimensions) reuse the same frozen vectors; smaller prefixes are normalized again. The 1,024-dimensional score must reproduce the existing centroid reference exactly. These are five additional scalar scoring controls, with zero new encoder calls or classifier fits—not thousands of new candidate columns. [Protocol and model-card rationale](../docs/RESOLUTION_STUDY.md).",
            ),
            (
                "code",
                'display(pd.DataFrame([{ "Dimensions": r["dimension"], "Policy-macro AUC": r["metrics"]["rule_macro_auc"], "Log loss": r["metrics"]["log_loss"], "Brier": r["metrics"]["brier"]} for r in resolution["results"]]).round(4))\ndisplay(pd.DataFrame(resolution["uncertainty"])[["candidate", "observed_delta", "ci_lower", "ci_upper", "simultaneous_lower", "simultaneous_upper"]].round(4))',
            ),
            (
                "md",
                "## 10 · Test the remaining semantic hypotheses\nThe preregistered extension compares instructed comment queries with plain support documents, and generic rule entailment with affirmative policy behavior. It adds 32 asymmetric and 45 intent candidates, four fixed models across seven folds, and three frozen scores. The original banks and controls are reused unchanged.\n\nPlain supports lower centroid transfer AUC to 0.6865. Affirmative wording improves a weak generic NLI score from 0.4341 to 0.5174, still far below the centroid. Adding intent to semantic scalars gives only +0.0033 AUC with an interval spanning zero, while probability losses worsen. [Protocol and results](../docs/SEMANTIC_FORMATTING.md).",
            ),
            (
                "code",
                'display_formatting(root, "contrasts")\ndisplay(pd.DataFrame(formatting["screening"])[["protocol", "fold", "family", "candidates", "retained"]])\nstability = pd.DataFrame(formatting["audit"]["stability"])\ndisplay(stability.groupby("family").jaccard.agg(["min", "mean", "max"]).round(3))',
            ),
            (
                "md",
                "## 11 · Explain the errors without relabeling them\nBefore seeing the new scores, a seeded 48-row legal/medical review examined speech act, policy behavior, context dependence and ambiguity. Every sampled body, policy and target matched its pinned source. Requests, personal experience, discussion and implicit advice can share vocabulary while differing in prohibited behavior.\n\nThis is a stratified qualitative review by one assistant, not independent human adjudication or a population label-error estimate. Twenty-three rows were flagged as ambiguous; none was relabeled. Raw text and row annotations remain private. The review motivated a falsifiable representation check, whose negative outcome remains visible.",
            ),
            (
                "code",
                'review = formatting["error_audit"]\ndisplay(pd.Series(review["taxonomy_counts"]["speech_act"], name="Reviewed speech acts"))\ndisplay(pd.Series(decision["verification"], name="Private artifact replay"))',
            ),
            (
                "md",
                "## 12 · Apply the stopping rule, including complementarity\nThe replacement criteria were committed before the new semantic scores: at least +0.005 macro AUC, a positive simultaneous lower bound, no policy loss above 0.02 AUC, and log-loss/Brier increases no larger than 0.01/0.005. None of seven alternatives passes. One separately preregistered 50/50 average checks whether centroid and semantic-intent scores complement each other; no weights or policy routing are tuned.\n\nThe average reaches 0.7086 AUC, but its +0.0044 gain is uncertain, advertising loses 0.0222, and both probability losses worsen. It fails all five conditions. Retain the simpler centroid. The [coverage ledger](../docs/FEATURE_COVERAGE.md) documents applicable families, exclusions and deferred model research; this is bounded evidence of diminishing returns, not universal feature exhaustion.\n\nFinal model training and calibration may now proceed. One protected confirmation and verified offline promotion are still required. The current standalone inference notebook remains the explicitly named lexical reference.",
            ),
            (
                "code",
                'eligibility = decision["decision"]["candidate_eligibility"] + decision["fusion"]["decision"]["candidate_eligibility"]\ndisplay(pd.DataFrame([{ "Candidate": r["candidate"], "AUC": r["macro_auc"], "Gain": r["auc_gain"], "Eligible": r["eligible"], "Failed checks": ", ".join(r["failed_conditions"])} for r in eligibility]).round(4))',
            ),
            (
                "code",
                'display(pd.DataFrame(gate["criteria"]))\nprint("Final training justified:", gate["final_training_authorized_by_evidence"])\nprint(gate["decision"])',
            ),
            (
                "md",
                "## Reproduce the evidence\n`python scripts/run_expanded.py --encode` extends only missing pinned embeddings, then resumes the preregistered study after the private research artifacts are restored. Omit `--encode` to require a complete verified cache. The default public notebook performs no fitting, downloads or target access. Sources, input identities, stage markers, full catalogs and OOF predictions are preserved. [Execution and restoration record](../docs/VALIDATION.md).",
            ),
        ]
    )
    outputs["notebooks/03_saved_results.ipynb"] = notebook(
        [
            (
                "md",
                "# 03 · Results and the next decision\n\n**Decision:** Feature research is complete for the declared four-policy scope. Retain the original frozen semantic centroid for unseen policies; final model validation and product delivery come next. Independent confirmation and deployment of the selected representation are unfinished.\n\nThis short notebook is the employer review path: the measured improvement, the failed alternatives, the remaining limitation and the concrete release gates.",
            ),
            ("code", SETUP),
            ("code", RESEARCH_SETUP),
            (
                "md",
                "## What the expanded research measured\nThirty-four fixed trained configurations share seven purged splits. The feature study adds/removes major families, checks full versus screened representations, and tests training-only NB/SVD controls. Frozen semantic scores are untrained comparison baselines. The 43,576-row reserve remains outside model selection.\n\nThe frozen semantic centroid reaches **0.7042 transfer AUC**, versus **0.4728** for the matched lexical reference. Its log loss improves from 0.8126 to 0.6237. Most of the gain comes from avoiding reversed lexical ranking on illegal-activity promotion; legal and medical advice remain difficult. The all-transfer model reaches 0.7989 on familiar policies but only 0.5515 on transfer.\n\nPolicy-macro AUC is the primary competition-oriented metric. These are post-competition development results, not a Kaggle score.",
            ),
            (
                "code",
                'display_expanded(root, "comparison")\nselected = {"rule_examples", "word_semantic_scalar", "semantic_scalar_only", "qwen_centroid", "all_transfer"}\nrows = [r for r in expanded["results"] if r["model"] in selected]\nrows += [r for r in formatting["results"] if r["model"] in {"asymmetric_centroid", "semantic_intent"}]\nrows += [{"model": "centroid_intent_mean (rejected)", "protocol": "heldout_rule", "metrics": decision["fusion"]["metrics"]}]\ndisplay(metric_table(rows, heldout=True).sort_values("Rule macro AUC", ascending=False))',
            ),
            (
                "md",
                "## What would count as a feature improvement?\nThe family-ablation figure in notebook `02` measures additions to the same screened-word control. The table below compares each representation with the same-fold rule/example reference. A higher point estimate with an interval crossing zero remains an uncertain development observation. Probability losses and policy-level failures are part of the decision. Simultaneous intervals apply within each preregistered study; combining rows below does not create one joint correction over all research.",
            ),
            (
                "code",
                'contrasts = pd.DataFrame(expanded["uncertainty"] + retrieval["uncertainty"])\ncompared = contrasts[(contrasts.protocol == "heldout_rule") & (contrasts.reference == "rule_examples")]\ndisplay(compared.sort_values("observed_delta", ascending=False).head(10)[["candidate", "observed_delta", "ci_lower", "ci_upper", "simultaneous_lower", "simultaneous_upper"]].round(4))',
            ),
            (
                "md",
                "## Why the feature phase can close, and the product cannot\nThe campaign covers 188,595–188,598 candidate columns per fold and 323 fixed fits. Training screens retain 9,281–9,660 columns across separate banks. The strongest transfer representation is simpler than those banks: a frozen support-centroid comparison. The last semantic variants fail the declared acceptance rule. A fixed average has the highest point AUC (0.7086), but its gain is uncertain and probability/policy regressions fail all five checks. The stopping decision retains 0.7042, rather than selecting the largest reported number.\n\nNext, test a fixed training-policy-membership route using the validated familiar-policy model and unseen-policy centroid, then develop calibration without the reserve. Commit the final candidate/reference and acceptance protocol before one protected comparison. This proposed route has not yet been executed as a final model.\n\nThe standalone inference notebook still uses the named lexical reference. Accepted artifacts must be connected to offline inference with parity, latency/memory and missing-support tests. A small example-driven demo, model/data cards and clean-environment release complete the product. [Next milestone specification](../docs/FINAL_MODEL_PLAN.md).",
            ),
            (
                "code",
                'print("Expanded study:", gate["expanded_development"])\nprint("Feature gate:", gate["status"])\nprint("Final training justified:", gate["final_training_authorized_by_evidence"])',
            ),
            (
                "md",
                "## Explore the reasoning\n[02 · Research methods, screening and ablations](02_baseline_and_review.ipynb) · [04 · Per-policy and probability diagnostics](04_semantic_benchmark.ipynb) · [01 · Leakage boundaries](01_data_and_validation.ipynb) · [Remaining release milestones](../docs/ROADMAP.md).\n\nThe historical two-policy search is preserved in the [research record](../docs/FEATURE_RESEARCH.md); its scores are not compared numerically with this larger cohort as if only the features had changed.",
            ),
        ]
    )
    outputs["notebooks/04_semantic_benchmark.ipynb"] = notebook(
        [
            (
                "md",
                "# 04 · Policy and semantic diagnostics\n\n**Question:** Where does the representation succeed, where does it fail, and how much does it depend on its supplied examples?\n\nThis notebook examines the expanded development study. It keeps ranking, probability quality and support dependence separate. No result here consumes the confirmation reserve.",
            ),
            ("code", SETUP),
            ("code", RESEARCH_SETUP),
            (
                "md",
                "## Policy-level performance can contradict the average\nThe held-out-policy models train on the other three policies. Fixed semantic scores do not fit labels; their identical predictions across validation protocols are not independent replications. A policy-level failure remains meaningful even if the overall mean improves.",
            ),
            (
                "code",
                'display_expanded(root, "policies")\nselected = ["rule_examples", "character_full", "word_semantic_scalar", "all_transfer", "qwen_centroid"]\nrecords = [r for r in expanded["results"] if r["protocol"] == "heldout_rule" and r["model"] in selected]\nper_rule = pd.DataFrame([{"Representation": r["model"], "Policy": rule.split(":")[0], "ROC AUC": auc} for r in records for rule, auc in r["metrics"]["per_rule_auc"].items()])\ndisplay(per_rule.pivot(index="Representation", columns="Policy", values="ROC AUC").round(4))',
            ),
            (
                "md",
                "## Ranking is not probability calibration\nLog loss and Brier assess probability quality; lower is better. Calibration error uses ten equal-width bins. Precision/recall/F1 use the fixed diagnostic threshold 0.5. There is no fitted calibrator or selected moderation threshold. Frozen centroid temperatures define scoring scales, not validated probability estimates.",
            ),
            (
                "code",
                'display(metric_table(records))\ndiagnostics = pd.DataFrame([{"Representation": r["model"], "Pooled AUC": r["metrics"]["pooled_auc"], "Calibration error": r["metrics"]["ece_10_equal_width_bins"], "Precision@0.5": r["metrics"]["precision_at_0_5"], "Recall@0.5": r["metrics"]["recall_at_0_5"], "F1@0.5": r["metrics"]["f1_at_0_5"]} for r in records])\ndisplay(diagnostics.round(4))',
            ),
            (
                "md",
                "## How much does the supplied support set matter?\nA centroid comparison asks whether a comment is closer to positive than negative examples under a rule-conditioned frozen embedding. One-example-per-class scoring averages every positive/negative choice. Shuffling complete support contexts within the same policy tests dependence on the particular supplied context. Exact self-support exclusion tests whether copied examples explain performance. These are fixed diagnostics, not new tuned models.",
            ),
            (
                "code",
                'stress = expanded["audit"]["support_stress"]\ncentroid = next(r["metrics"] for r in records if r["model"] == "qwen_centroid")\ncomparison = {"Original supplied context": centroid, **{name: value for name, value in stress.items() if isinstance(value, dict)}}\ndisplay(pd.DataFrame([{ "Condition": name, "Policy-macro AUC": value["rule_macro_auc"], "Log loss": value["log_loss"], "Brier": value["brier"]} for name, value in comparison.items()]).round(4))\nprint("Self-support rows excluded:", stress["self_match_rows"])',
            ),
            (
                "md",
                "## Compute is part of the representation decision\nThe expanded study reuses 1,875 verified original inputs and encodes 10,098 missing inputs, for 11,973 unique rule/comment or rule/example queries. Four CPU workers use the pinned model and immutable input contract. Completed shards and folds are checksummed and checkpointed to S3; interrupted active stages restart while completed work is preserved.\n\nAll 190 cache shards are verified. Their statistics record 11,977 encoded occurrences (11,973 unique inputs), two truncations, a maximum original length of 337 tokens and peak per-worker RSS of 3.85 GiB. The six-resolution comparison keeps 1,024 dimensions as the strongest reference; shorter outputs do not accelerate the transformer itself.\n\nEmbedding coordinates and compact comparisons share the same encoder cost, but their fitted feature dimensions differ. The final product still needs measured end-to-end latency, memory, offline parity and a support-availability policy after a candidate earns promotion.",
            ),
            (
                "code",
                'display(pd.Series(expanded["audit"]["embedding_cache"], name="Verified expanded cache"))\nprint("Research run:", expanded["metadata"]["run_id"])\nprint("Final training justified:", gate["final_training_authorized_by_evidence"])',
            ),
            (
                "md",
                "## Do intent features resolve the weak policies?\nSemantic-plus-intent slightly improves legal and medical ranking, but damages advertising and probability quality. The fixed average retains part of that tradeoff; it fails the predeclared acceptance rule. The qualitative audit contains 48 deliberately balanced examples, not a prevalence sample. Context and speech-act ambiguity are useful error mechanisms, not proof that the source labels are wrong.\n\nThe new job encoded 2,217 unique plain supports and 21,214 unique NLI pairs, with zero and four truncated inputs respectively. Original comment embeddings were reused. Compatible historical NLI predictions were not present in this job's restored inputs; the new 332-shard NLI cache is now checkpointed and reusable. Worker computation took 644.5 seconds on one bounded CPU instance; summed encoder time across parallel workers is not wall time.",
            ),
            (
                "code",
                'display_formatting(root, "policies")\ndisplay(pd.DataFrame(formatting["error_audit"]["taxonomy_counts"]).fillna(0).astype(int))',
            ),
            (
                "md",
                "## Interpret the limits\nThe four observed policies are broader than the original pair, but remain a finite benchmark selected by the competition's data construction. Fixed-model bootstrap intervals do not estimate the full distribution of arbitrary future policies. The protected reserve is intended for one later frozen comparison. If it rejects the candidate, that result must be retained rather than used for another search.\n\n[02 · Feature evidence](02_baseline_and_review.ipynb) · [03 · Current decision](03_saved_results.ipynb) · [Full historical research record](../docs/FEATURE_RESEARCH.md).",
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
