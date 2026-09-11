"""Render saved, hash-verified support selection evidence without accessing query data."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

RUN_ID = "adf2ff53497b509795385392"
EXPECTED = {
    "audit.json",
    "review_summary.json",
    "review_decision.json",
    "review_provenance.json",
    "replay_check.json",
    "executed_source_manifest.json",
}
POLICIES = {0: "Advertising", 1: "Legal advice"}


def evidence(root: Path) -> dict:
    """Read public aggregates only; fail closed on corruption or a different decision."""
    folder = Path(root) / "reports/support_selection"
    if folder.is_symlink():
        raise ValueError("Evidence directory must not be a symlink")
    manifest = json.loads((folder / "metadata.json").read_text())
    if manifest.get("run_id") != RUN_ID or set(manifest.get("files", {})) != EXPECTED:
        raise ValueError("Unexpected support-selection evidence manifest")
    records = {}
    for name, sha in manifest["files"].items():
        p = folder / name
        if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            raise ValueError(f"Support-selection evidence checksum differs: {name}")
        records[Path(name).stem] = json.loads(p.read_text())
    audit = records["audit"]
    review = records["review_summary"]
    decision = records["review_decision"]
    if any(records[k].get("run_id") != RUN_ID for k in records if k != "executed_source_manifest"):
        raise ValueError("Evidence run identity differs")
    if audit["total_queries"] != 881 or [f["queries"] for f in audit["folds"]] != [234, 647]:
        raise ValueError("Unexpected audit cohort")
    if any(x.get("query_targets_read") is not False for x in (audit, review, decision)):
        raise ValueError("Target-free boundary is not established")
    if audit["official_metric"] is not None or decision["new_auc"] is not None:
        raise ValueError("This diagnostic must not introduce AUC evidence")
    if review["reviewer_type"] != "single_ai_reviewer" or review["cases"] != 48:
        raise ValueError("Review identity or size differs")
    if decision["decision"] != "STOP_CURRENT_SEMANTIC_SELECTOR":
        raise ValueError("Unexpected research decision")
    totals = {k: 0 for k in ("lexical", "semantic", "tie", "unclear")}
    for fold in ("0", "1"):
        counts = review["counts_by_fold"][fold]
        if set(counts) != set(totals) or sum(counts.values()) != 24:
            raise ValueError("Incomplete review counts")
        for key in totals:
            value = counts[key]
            if type(value) is not int or value < 0:
                raise ValueError("Invalid preference count")
            totals[key] += value
    if totals != {"lexical": 24, "semantic": 13, "tie": 1, "unclear": 10}:
        raise ValueError("Review totals differ")
    if not records["replay_check"]["completed_result_reused"]:
        raise ValueError("Original audit replay was not verified")
    return records


def tables(records: dict) -> tuple[list[dict], list[dict]]:
    """Rows use only the saved audit counts and the saved review summary."""
    selections = []
    preferences = []
    for f in records["audit"]["folds"]:
        n = f["queries"]
        row = {
            "Policy": POLICIES[f["fold"]],
            "Queries": n,
            "Changed pairs": f["changed_pairs"],
            "Changed pairs (%)": round(100 * f["changed_pairs"] / n, 2),
        }
        for method in ("lexical", "semantic"):
            value = f["methods"][method]["classes"]["negative"]["maximum_reuse_fraction"]
            row[f"{method.title()} max permitted-example reuse"] = round(value * n)
        selections.append(row)
        counts = records["review_summary"]["counts_by_fold"][str(f["fold"])]
        preferences.append({"Policy": POLICIES[f["fold"]], **counts})
    return selections, preferences


def display_review(records: dict) -> None:
    """Display interactive counts with a portable SVG fallback, not an accuracy plot."""
    import matplotlib
    import plotly.graph_objects as go
    from IPython.display import display

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, rows = tables(records)
    names = [r["Policy"] for r in rows]
    methods = ("lexical", "semantic", "tie", "unclear")
    labels = ("Lexical preferred", "Semantic preferred", "Tie", "Unclear")
    fig = go.Figure()
    for key, label in zip(methods, labels, strict=True):
        values = [r[key] for r in rows]
        fig.add_bar(name=label, x=names, y=values, text=values, textposition="inside")
    fig.update_layout(
        title="Example usefulness: one AI review, not model accuracy",
        barmode="stack",
        height=440,
        yaxis={"title": "Cases out of 24 per policy", "range": [0, 27]},
        legend={"orientation": "h", "y": -0.2},
        margin={"l": 60, "r": 30, "t": 80, "b": 110},
    )
    with plt.rc_context({"svg.hashsalt": "jigsaw-selection-review"}):
        fallback, ax = plt.subplots(figsize=(10, 4.6))
        bottom = [0, 0]
        for key, label in zip(methods, labels, strict=True):
            values = [r[key] for r in rows]
            bars = ax.bar(names, values, bottom=bottom, label=label)
            ax.bar_label(bars, labels=[str(v) if v else "" for v in values], label_type="center")
            bottom = [a + b for a, b in zip(bottom, values, strict=True)]
        ax.set_ylim(0, 27)
        ax.set_ylabel("Cases out of 24 per policy")
        ax.set_title("Example usefulness: one AI review, not model accuracy", pad=16)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
        fallback.tight_layout()
        output = io.StringIO()
        fallback.savefig(output, format="svg", metadata={"Date": None})
        plt.close(fallback)
    display(
        {
            "application/vnd.plotly.v1+json": json.loads(fig.to_json()),
            "image/svg+xml": output.getvalue(),
        },
        raw=True,
    )


def notebook_cells() -> list[tuple[str, str]]:
    return [
        (
            "md",
            "## Latest diagnostic: cached-vector support selection\n\n"
            "**Decision: stop this raw adapted-vector cosine selector before another GPU run.** "
            "The accepted support-adapted Qwen3-4B is unchanged: **0.91808 public / 0.91425 "
            "private Kaggle AUC**. No new AUC was measured. Feature/representation research "
            "remains open.\n\n"
            "This test reused saved representations for 881 comments and compared one "
            "violating plus one permitted demonstration under each selector. It did not "
            "generate embeddings, train a model, or read query targets. The subsequent "
            "48-case preference review was completed by one AI reviewer, not independent "
            "human raters. The two repeatedly examined policies remain development evidence.",
        ),
        (
            "code",
            "from scripts.build_support_selection_report import evidence as selection_evidence\n"
            "from scripts.build_support_selection_report import tables as selection_tables\n"
            "from scripts.build_support_selection_report import display_review\n"
            "selection = selection_evidence(root)\n"
            "selection_rows, review_rows = selection_tables(selection)\n"
            "display(pd.DataFrame(selection_rows))\n"
            "display(pd.DataFrame(review_rows).rename(columns=str.title))\n"
            "display_review(selection)\n"
            "print('Decision:', selection['review_decision']['decision'])\n"
            "print('Original audit replay verified:', "
            "selection['replay_check']['completed_result_reused'])\n"
            "print('No new model run, query targets, or competition score in this diagnostic.')",
        ),
        (
            "md",
            "### Interpretation and next research question\n\n"
            "**878 of 881 pairs changed**, but change and cosine similarity are not measures "
            "of example usefulness. In the fixed review, lexical pairs were preferred in "
            "24 cases and semantic pairs in 13; one tied and ten were unclear. The most-used "
            "permitted legal example appeared 121 times under semantic selection, versus "
            "21 under lexical selection. These observations do not establish causal failure "
            "or AUC inferiority.\n\n"
            "The original awaiting-review audit is preserved unchanged; its later review "
            "decision takes precedence. Method names were hidden during case judgments, "
            "but prior aggregate findings were already visible. Do not treat this as "
            "double-blind human validation or use the preferences as training labels.\n\n"
            "The next hypothesis concerns concentrated decision-vector geometry and "
            "rule-relevant behavior; it is **not yet tested**. The earlier lexical-prompt "
            "candidate also remains rejected. "
            "[Full scope and preservation record](../docs/SUPPORT_SELECTION_RESULT.md).",
        ),
    ]
