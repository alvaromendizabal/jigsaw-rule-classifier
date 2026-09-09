"""Decision-focused calibration figures from verified development aggregates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jigsaw_rules.model_validation import validation_evidence
from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"seen_rule": "Familiar policies", "heldout_rule": "Unseen policies"}
FILES = {
    f"{name}.{ext}" for name in ["calibration", "reliability"] for ext in ["svg", "plotly.json"]
}


def build(root: Path = ROOT) -> None:
    evidence = validation_evidence(root)
    if evidence is None:
        raise ValueError("Complete nested model validation before plotting")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    folder = root / "reports/model_validation"
    decisions = evidence["calibration"]["decisions"]
    labels, values, lower, upper, colors = [], [], [], [], []
    for name, title in NAMES.items():
        decision = decisions[name]
        interval = decision["loss_interval"]
        labels.append(
            title + (" · retain sigmoid" if decision["retain_calibration"] else " · retain raw")
        )
        values.append(interval["log_loss_improvement"])
        lower.append(interval["log_loss_improvement"] - interval["lower"])
        upper.append(interval["upper"] - interval["log_loss_improvement"])
        colors.append("#16837b" if decision["retain_calibration"] else "#b86554")
    plot = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            error_x={"type": "data", "array": upper, "arrayminus": lower},
        )
    )
    plot.add_vline(x=0, line_color="#18324d")
    plot.add_vline(x=0.002, line_dash="dot", line_color="#7b8b9a")
    plot.update_xaxes(title="Log-loss improvement after calibration →")
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.barh(labels, values, color=colors)
    ax.errorbar(values, labels, xerr=[lower, upper], fmt="none", color="#18324d", capsize=5)
    ax.axvline(0, color="#18324d", linewidth=1)
    ax.axvline(0.002, color="#7b8b9a", linestyle=":")
    ax.set_xlabel("Log-loss improvement after calibration →")
    ax.grid(axis="x", alpha=0.2)
    save(
        root,
        "calibration",
        plot,
        fig,
        "Does calibration survive separate validation?",
        "97.5% paired body-group intervals · dotted line: minimum gain · all six gates apply",
    )

    plot = make_subplots(rows=1, cols=2, subplot_titles=list(NAMES.values()))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharex=True, sharey=True)
    for column, (protocol, title) in enumerate(NAMES.items(), 1):
        ax = axes[column - 1]
        for model, color in [("raw", "#426b9a"), ("calibrated", "#d18148")]:
            record = next(
                r
                for r in evidence["calibration"]["reliability"]
                if r["protocol"] == protocol and r["model"] == model
            )
            x = [b["mean_probability"] for b in record["bins"]]
            y = [b["observed_rate"] for b in record["bins"]]
            count = [b["count"] for b in record["bins"]]
            plot.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=model.title(),
                    legendgroup=model,
                    showlegend=column == 1,
                    line_color=color,
                    customdata=count,
                    hovertemplate=(
                        "Mean score %{x:.3f}<br>Observed rate %{y:.3f}"
                        "<br>Rows %{customdata}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            ax.plot(x, y, "o-", color=color, label=model.title(), markersize=4)
        plot.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"color": "#a0acb8", "dash": "dash"},
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        plot.update_xaxes(title="Mean predicted probability", range=[0, 1], row=1, col=column)
        plot.update_yaxes(title="Observed violation rate", range=[0, 1], row=1, col=column)
        ax.plot([0, 1], [0, 1], "--", color="#a0acb8", linewidth=1)
        ax.set(
            xlabel="Mean predicted probability",
            ylabel="Observed violation rate",
            title=title,
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.grid(alpha=0.15)
    axes[0].legend(frameon=False)
    save(
        root,
        "reliability",
        plot,
        fig,
        "Where do probability errors remain?",
        "Outer validation predictions · 10 fixed bins · descriptive curves",
    )
    atomic_json(
        folder / "figures.json",
        {
            "source_sha256": digest(root / "scripts/build_model_report.py"),
            "metadata_sha256": digest(folder / "metadata.json"),
            "files": {name: digest(folder / name) for name in sorted(FILES)},
        },
    )


def save(root, name, plot, fig, title, subtitle):
    plot.update_layout(
        template="plotly_white",
        height=450,
        font={"family": "Arial", "size": 12},
        title=title + "<br><sup>" + subtitle + "</sup>",
        margin={"l": 220 if name == "calibration" else 70, "r": 35, "t": 100, "b": 70},
    )
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold", color="#18324d")
    fig.text(0.01, 0.90, subtitle, fontsize=9, color="#52677d")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    folder = root / "reports/model_validation"
    atomic_bytes(folder / f"{name}.plotly.json", plot.to_json().encode())
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(root / f"logs/model-{name}.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def display_figure(root: Path, name: str) -> None:
    from IPython.display import SVG, display
    from plotly.io import from_json

    validation_evidence(root)
    folder = root / "reports/model_validation"
    manifest = json.loads((folder / "figures.json").read_text())
    if (
        name not in {"calibration", "reliability"}
        or set(manifest["files"]) != FILES
        or manifest["source_sha256"] != digest(root / "scripts/build_model_report.py")
        or manifest["metadata_sha256"] != digest(folder / "metadata.json")
        or any(digest(folder / n) != sha for n, sha in manifest["files"].items())
    ):
        raise ValueError("Model validation figure contract differs")
    from_json((folder / f"{name}.plotly.json").read_text()).show()
    display(SVG(filename=str(folder / f"{name}.svg")))


if __name__ == "__main__":
    build()
