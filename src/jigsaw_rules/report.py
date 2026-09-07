"""Offline HTML review: exact metrics, calibration, and per-rule comparisons."""

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jigsaw_rules.metrics import calibration_table
from jigsaw_rules.runtime import atomic_bytes


def build_report(
    directory: Path, results: list[dict], oof: pd.DataFrame, *, synthetic: bool
) -> None:
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Rule generalization", "Probability calibration")
    )
    for result in results:
        label = f"{result['model']} / {result['protocol']}"
        scores = result["metrics"]["per_rule_auc"]
        fig.add_trace(go.Bar(name=label, x=list(scores), y=list(scores.values())), row=1, col=1)
        subset = oof[(oof.model == result["model"]) & (oof.protocol == result["protocol"])]
        table = calibration_table(subset.rule_violation, subset.probability)
        fig.add_trace(
            go.Scatter(
                name=label,
                x=table.mean_probability,
                y=table.observed_rate,
                mode="lines+markers",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], name="Ideal calibration", line={"dash": "dash"}, showlegend=False
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        template="plotly_white",
        height=560,
        font={"family": "Arial", "size": 13},
        colorway=["#067f8c", "#b95b37", "#334ea0", "#76953c"],
        legend={"orientation": "h", "y": -0.35},
        margin={"t": 60, "b": 140},
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, 1], row=1, col=2)
    rows = []
    for result in results:
        m = result["metrics"]
        rows.append(
            {
                "Model": result["model"],
                "Validation": result["protocol"],
                "Rule macro AUC": m["rule_macro_auc"],
                "Pooled AUC": m["pooled_auc"],
                "Average precision": m["average_precision"],
                "Log loss": m["log_loss"],
                "Brier": m["brier"],
                "F1 @ 0.5": m["f1_at_0_5"],
            }
        )
    phase = (
        "Semantic reference experiments"
        if any(r["model"].startswith("semantic_") for r in results)
        else "Lexical reference experiments"
    )
    banner = (
        "SYNTHETIC SOFTWARE DEMONSTRATION — NOT COMPETITION PERFORMANCE"
        if synthetic
        else ("LOCAL CROSS-VALIDATION — NOT A KAGGLE LEADERBOARD SCORE")
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Jigsaw | Experiment review</title>
<style>body{{margin:0;background:#f3f5f6;color:#172b37;font:16px/1.65 Arial,sans-serif}}
main{{max-width:1160px;margin:auto;padding:42px 24px}}h1{{font-size:42px;line-height:1.1}}
.eyebrow{{color:#087c83;font-weight:bold;letter-spacing:2px}}.banner{{padding:14px 20px;background:#fff0cc}}
section{{background:white;padding:26px;margin:24px 0;border-radius:14px;overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dfe5e8}}
small{{color:#526570}}code{{background:#edf2f4;padding:2px 5px}}</style></head><body><main>
<div class="eyebrow">JIGSAW / RULE-CONDITIONED NLP</div><h1>Can the model handle a new rule?</h1>
<p>Reference models, honest validation, and inspectable probability estimates.</p>
<div class="banner">{banner}</div><section><h2>What this run measures</h2>
<p><b>Seen rule:</b> grouped comment folds. <b>Held-out rule:</b> train without the evaluated rule.
Validation comment text is purged from training bodies and support examples. Similar comments can still
share semantics; exact-text purging is not a guarantee against all leakage.</p>
<p>Two labeled rules provide only two rule-transfer experiments. These results cannot establish broad
unseen-policy generalization or support a medal-performance claim.</p></section>
<section><h2>Model comparison</h2>{pd.DataFrame(rows).to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)}</section>
<section>{fig.to_html(full_html=False, include_plotlyjs=True)}</section>
<section><h2>How to read the evidence</h2><p>AUC measures ranking; Brier and log loss measure probability
quality. Average precision summarizes the precision–recall curve. The fixed 0.5 threshold is diagnostic.
No decision threshold or calibration model is fitted to these evaluation labels.</p>
<p>Training and inference durations are recorded in <code>events.jsonl</code> and each fold's metrics.
Inspect <code>oof.csv</code> for row-level errors and each fold's <code>split.json</code> for provenance.</p>
<details><summary>Full metric definitions and values</summary><pre>{html.escape(json.dumps(results, indent=2))}</pre></details>
</section><small>Alvaro Mendizabal · {phase}</small></main></body></html>"""
    atomic_bytes(directory / "report.html", page.encode())
