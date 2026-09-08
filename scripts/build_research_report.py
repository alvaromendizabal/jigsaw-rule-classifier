"""Plotly research figures with equivalent SVG fallbacks for static notebook readers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jigsaw_rules.diagnostics import diagnostic_evidence
from jigsaw_rules.pairs import pairs_evidence
from jigsaw_rules.research import research_evidence
from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"ink": "#172b4d", "blue": "#277da8", "teal": "#16837b", "red": "#bd5d54"}


def style_plotly(fig, title, *, height=580):
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=height,
        font={"family": "Arial", "size": 13, "color": COLORS["ink"]},
        margin={"l": 210, "r": 45, "t": 90, "b": 65},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def style_static(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=36, color=COLORS["ink"])
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=10, color="#52677d")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=9)


def save_pair(root, name, interactive, static):
    folder = root / "reports/research"
    atomic_bytes(folder / f"{name}.plotly.json", interactive.to_json().encode())
    static.savefig(folder / f"{name}.svg", bbox_inches="tight", facecolor="white")
    static.savefig(root / "logs" / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(static)


def build(root: Path = ROOT):
    evidence = research_evidence(root)
    if evidence is None:
        raise ValueError("Research figures require verified aggregates")
    (root / "logs").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    effects = pd.DataFrame(evidence["uncertainty"])
    effects = effects[
        (effects.protocol == "heldout_rule") & effects.contrast.str.startswith("add_")
    ]
    effects = effects.sort_values("observed_delta")
    labels = effects.contrast.str.removeprefix("add_").str.replace("_", " ")
    title = "Which features improve transfer?"
    subtitle = "Same screened-word control and fixed classifier · 1,000 paired comment-group draws"
    interactive = go.Figure(
        go.Scatter(
            x=effects.observed_delta,
            y=labels,
            mode="markers",
            error_x={
                "type": "data",
                "symmetric": False,
                "array": effects.simultaneous_upper - effects.observed_delta,
                "arrayminus": effects.observed_delta - effects.simultaneous_lower,
            },
            marker={"size": 10, "color": COLORS["blue"]},
            name="Simultaneous 95% interval",
        )
    )
    interactive.add_vline(x=0, line_dash="dash", line_color="#6e7f91")
    interactive.update_xaxes(title="Held-out rule macro AUC change")
    style_plotly(interactive, title + "<br><sup>" + subtitle + "</sup>")
    fig, ax = plt.subplots(figsize=(11.5, 6))
    y = np.arange(len(effects))
    ax.hlines(
        y,
        effects.simultaneous_lower,
        effects.simultaneous_upper,
        color="#a9b9c9",
        linewidth=4,
        label="Simultaneous 95%",
    )
    ax.hlines(
        y,
        effects.ci_lower,
        effects.ci_upper,
        color=COLORS["blue"],
        linewidth=2,
        label="Pointwise 95%",
    )
    ax.scatter(effects.observed_delta, y, color=COLORS["blue"], s=45, zorder=3)
    ax.axvline(0, color="#6e7f91", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Change in held-out rule macro AUC")
    style_static(ax, title, subtitle)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save_pair(root, "ablation", interactive, fig)

    counts = pd.DataFrame(evidence["screening"]["folds"])
    grouped = counts.groupby("family")[["candidates", "retained"]].agg(["min", "max"])
    families = grouped.index.str.replace("_", " ")
    interactive = go.Figure()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for column, color, ax in zip(
        ("candidates", "retained"), (COLORS["blue"], COLORS["teal"]), axes, strict=True
    ):
        low, high = grouped[(column, "min")], grouped[(column, "max")]
        mid = np.sqrt(low.clip(lower=1) * high.clip(lower=1))
        interactive.add_trace(
            go.Scatter(
                x=mid,
                y=families,
                mode="markers",
                name=column.title(),
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": high - mid,
                    "arrayminus": mid - low,
                },
                marker={"color": color, "size": 9},
            )
        )
        y = np.arange(len(families))
        ax.hlines(y, low.clip(lower=1), high.clip(lower=1), color=color, linewidth=4)
        ax.scatter(mid, y, color=color, s=35)
        ax.set_xscale("log")
        ax.set_yticks(y, families)
        ax.set_xlabel("Feature columns · logarithmic scale")
        style_static(ax, column.title(), "Range across the five preserved folds")
    interactive.update_xaxes(type="log", title="Feature columns · range across folds")
    style_plotly(interactive, "Broad generation, training-only screening")
    fig.tight_layout()
    save_pair(root, "screening", interactive, fig)

    stability = pd.DataFrame(evidence["screening"]["stability"])
    summary = stability.groupby(["family", "protocol"]).retained_jaccard.mean().unstack()
    interactive = go.Figure()
    fig, ax = plt.subplots(figsize=(11.5, 6))
    y = np.arange(len(summary))
    for protocol, offset, color, label in (
        ("seen_rule", -0.13, COLORS["blue"], "Familiar-rule folds"),
        ("heldout_rule", 0.13, COLORS["teal"], "Held-out-rule folds"),
    ):
        interactive.add_trace(
            go.Scatter(
                x=summary[protocol],
                y=summary.index.str.replace("_", " "),
                mode="markers",
                name=label,
                marker={"size": 9, "color": color},
            )
        )
        ax.scatter(summary[protocol], y + offset, label=label, color=color, s=40)
    ax.set_yticks(y, summary.index.str.replace("_", " "))
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Jaccard overlap of retained feature names")
    style_static(
        ax,
        "Does screening retain the same features?",
        "Selection stability is separate from predictive usefulness",
    )
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    interactive.update_xaxes(range=[0, 1], title="Jaccard overlap of retained features")
    style_plotly(interactive, "Selection stability across folds")
    save_pair(root, "stability", interactive, fig)

    kinds = {
        "research": evidence,
        "sensitivity": diagnostic_evidence(root),
        "pairs": pairs_evidence(root),
    }
    manifest = {
        "schema": 1,
        "source_sha256": digest(Path(__file__)),
        "inputs": {name: item["metadata"]["run_id"] for name, item in kinds.items() if item},
        "files": {
            path.name: digest(path)
            for name in ("ablation", "screening", "stability")
            for suffix in ("svg", "plotly.json")
            for path in [root / "reports/research" / f"{name}.{suffix}"]
        },
    }
    atomic_json(root / "reports/research/figures.json", manifest)


def display_figure(root: Path, name: str):
    from IPython.display import display

    manifest = json.loads((root / "reports/research/figures.json").read_text())
    evidence = research_evidence(root)
    if evidence is None or manifest["inputs"]["research"] != evidence["metadata"]["run_id"]:
        raise ValueError("Research figures refer to a different experiment")
    if name not in {"ablation", "screening", "stability"}:
        raise ValueError("Unknown research figure")
    for suffix in ("svg", "plotly.json"):
        path = root / "reports/research" / f"{name}.{suffix}"
        if digest(path) != manifest["files"][path.name]:
            raise ValueError("Research figure checksum differs")
    display(
        {
            "application/vnd.plotly.v1+json": json.loads(
                (root / f"reports/research/{name}.plotly.json").read_text()
            ),
            "image/svg+xml": (root / f"reports/research/{name}.svg").read_text(),
        },
        raw=True,
    )


if __name__ == "__main__":
    build()
