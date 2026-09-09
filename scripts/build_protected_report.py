"""Plot only verified aggregates from the single protected comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jigsaw_rules.confirmation_report import confirmation_evidence, export_report
from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "all": "All six policies",
    "familiar": "Four familiar policies",
    "unseen": "Two unseen policies",
}
FIGURES = {"policy_gains", "probability_quality", "coverage"}
FILES = {f"{name}.{ext}" for name in FIGURES for ext in ["svg", "plotly.json"]}
COLORS = {"candidate": "#16837b", "reference": "#426b9a"}


def build(root: Path = ROOT) -> None:
    evidence = confirmation_evidence(root)
    if evidence is None:
        raise ValueError("Complete the protected comparison before plotting")
    render(root, evidence["results"])


def render(root: Path, result: dict) -> None:
    """Render audited aggregates; a rejected candidate is never silently replaced."""
    if "scopes" not in result:
        raise ValueError("Protected AUC is unavailable: retain the explicit rejection report")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    folder = root / "reports/confirmation"
    policies = list(result["per_policy"])
    labels = [p.title() for p in policies] + ["Policy macro · primary"]
    pairs = [result["per_policy"][p] for p in policies] + [result["scopes"]["all"]]
    delta = [r["candidate"]["rule_macro_auc"] - r["reference"]["rule_macro_auc"] for r in pairs]
    colors = ["#16837b" if value >= 0 else "#b86554" for value in delta]
    plot = go.Figure(
        go.Bar(
            x=delta,
            y=labels,
            orientation="h",
            marker_color=colors,
            showlegend=False,
            customdata=[
                [r["candidate"]["rule_macro_auc"], r["reference"]["rule_macro_auc"]] for r in pairs
            ],
            hovertemplate=(
                "%{y}<br>AUC gain %{x:.4f}<br>Candidate %{customdata[0]:.4f}"
                "<br>Reference %{customdata[1]:.4f}<extra></extra>"
            ),
        )
    )
    interval = result["primary_interval"]
    limits = [interval["ci_lower"], interval["ci_upper"]]
    plot.add_trace(
        go.Scatter(
            x=limits,
            y=[labels[-1]] * 2,
            mode="lines+markers",
            name="Primary 95% interval",
            line={"color": "#18324d", "width": 3},
            marker={"symbol": "line-ns", "size": 12},
        )
    )
    plot.add_vline(x=0, line_color="#18324d")
    plot.update_xaxes(title="Candidate minus lexical reference · policy-macro ROC AUC")
    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.barh(labels, delta, color=colors)
    ax.plot(limits, [labels[-1]] * 2, "|-", color="#18324d", linewidth=2, markersize=10)
    ax.axvline(0, color="#18324d", linewidth=1)
    ax.set_xlabel("Candidate minus lexical reference · policy-macro ROC AUC")
    ax.grid(axis="x", alpha=0.2)
    save(
        root,
        "policy_gains",
        plot,
        fig,
        "Does the frozen candidate survive independent confirmation?",
        f"Decision: {result['status'].upper()} · 95% paired body-group interval for primary only",
    )

    plot = make_subplots(rows=1, cols=2, subplot_titles=["Log loss", "Brier score"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for column, metric in enumerate(["log_loss", "brier"], 1):
        ax = axes[column - 1]
        for offset, (model, color) in enumerate(COLORS.items()):
            values = [result["scopes"][scope][model][metric] for scope in NAMES]
            plot.add_trace(
                go.Bar(
                    x=list(NAMES.values()),
                    y=values,
                    name=model.title(),
                    marker_color=color,
                    legendgroup=model,
                    offsetgroup=model,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
            ax.bar(
                [i + (offset - 0.5) * 0.36 for i in range(3)],
                values,
                width=0.36,
                color=color,
                label=model.title(),
            )
        plot.update_yaxes(title="Lower is better", row=1, col=column, rangemode="tozero")
        ax.set_xticks(range(3), ["All six", "Familiar", "Unseen"])
        ax.set(title=metric.replace("_", " ").title(), ylabel="Lower is better")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.12))
    save(
        root,
        "probability_quality",
        plot,
        fig,
        "Does stronger ranking preserve probability quality?",
        "Row-weighted proper scores · the predeclared guards apply overall and to each route",
    )

    plot = go.Figure()
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    for (scope, title), color in zip(NAMES.items(), ["#18324d", "#16837b", "#d18148"], strict=True):
        rows = [r for r in result["confidence_coverage"] if r["scope"] == scope and r["rows"]]
        x, y = [r["coverage"] for r in rows], [r["error_rate"] for r in rows]
        plot.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=title,
                line_color=color,
                customdata=[[r["minimum_confidence"], r["rows"]] for r in rows],
                hovertemplate=(
                    "Coverage %{x:.1%}<br>Error %{y:.1%}"
                    "<br>Minimum confidence %{customdata[0]:.2f}"
                    "<br>Rows %{customdata[1]}<extra></extra>"
                ),
            )
        )
        ax.plot(x, y, "o-", color=color, label=title, markersize=4)
    plot.update_xaxes(title="Retained fraction of eligible rows", tickformat=".0%", range=[0, 1.02])
    plot.update_yaxes(title="Error among retained rows", tickformat=".0%", rangemode="tozero")
    ax.set(
        xlabel="Retained fraction of eligible rows",
        ylabel="Error among retained rows",
        xlim=(0, 1.02),
    )
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    save(
        root,
        "coverage",
        plot,
        fig,
        "How much coverage remains as confidence increases?",
        "Six fixed confidence cutoffs · descriptive evidence, not a selected serving threshold",
    )
    atomic_json(
        folder / "figures.json",
        {
            "source_sha256": digest(root / "scripts/build_protected_report.py"),
            "metadata_sha256": digest(folder / "metadata.json"),
            "files": {name: digest(folder / name) for name in sorted(FILES)},
        },
    )


def save(root: Path, name: str, plot: go.Figure, fig, title: str, subtitle: str) -> None:
    plot.update_layout(
        template="plotly_white",
        height=510,
        font={"family": "Arial", "size": 12},
        title=title + "<br><sup>" + subtitle + "</sup>",
        margin={"l": 180 if name == "policy_gains" else 70, "r": 35, "t": 100, "b": 95},
        legend={"orientation": "h", "y": -0.22},
    )
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold", color="#18324d")
    fig.text(0.01, 0.91, subtitle, fontsize=9, color="#52677d")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    folder = root / "reports/confirmation"
    atomic_bytes(folder / f"{name}.plotly.json", plot.to_json().encode())
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight", facecolor="white")
    (root / "logs").mkdir(exist_ok=True)
    fig.savefig(
        root / f"logs/protected-{name}.png", dpi=140, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def display_figure(root: Path, name: str) -> None:
    from IPython.display import SVG, display
    from plotly.io import from_json

    if confirmation_evidence(root) is None:
        raise ValueError("No protected confirmation evidence")
    folder = root / "reports/confirmation"
    manifest = json.loads((folder / "figures.json").read_text())
    if (
        name not in FIGURES
        or set(manifest["files"]) != FILES
        or manifest["source_sha256"] != digest(root / "scripts/build_protected_report.py")
        or manifest["metadata_sha256"] != digest(folder / "metadata.json")
        or any(digest(folder / n) != sha for n, sha in manifest["files"].items())
    ):
        raise ValueError("Protected confirmation figure contract differs")
    from_json((folder / f"{name}.plotly.json").read_text()).show()
    display(SVG(filename=str(folder / f"{name}.svg")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Export an already completed private evaluation first")
    args = parser.parse_args()
    if args.run_id:
        export_report(ROOT, ROOT / "runs/confirmation" / args.run_id)
    build()
