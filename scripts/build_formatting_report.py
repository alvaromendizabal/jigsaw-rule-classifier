"""Plotly comparisons with static fallbacks and immutable semantic-study lineage."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jigsaw_rules.formatting import formatting_evidence
from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    f"{name}.{suffix}" for name in ("contrasts", "policies") for suffix in ("svg", "plotly.json")
}
LABELS = {
    "asymmetric_geometry": "Plain supports · centroid",
    "asymmetric_scalar": "Plain supports · scalar model",
    "asymmetric_with_words": "Plain supports · words + scalars",
    "affirmative_behavior": "Behavior wording vs generic wording",
    "behavior_vs_centroid": "Behavior entailment vs centroid",
    "intent_only": "Intent features vs semantic scalars",
    "intent_addition": "Intent added to semantic scalars",
}
MODELS = {
    "qwen_centroid": "Original centroid",
    "asymmetric_centroid": "Plain-support centroid",
    "generic_entailment": "Generic entailment",
    "behavior_entailment": "Behavior entailment",
    "semantic_scalar_only": "Semantic scalars",
    "asymmetric_scalar": "Plain-support scalars",
    "intent_scalar": "Intent scalars",
    "semantic_intent": "Semantic + intent scalars",
}


def save(root, name, plot, fig, title, subtitle):
    plot.update_layout(
        template="plotly_white",
        height=540,
        font={"family": "Arial", "size": 12},
        title=title + "<br><sup>" + subtitle + "</sup>",
        margin={"l": 280, "r": 40, "t": 90, "b": 65},
    )
    fig.suptitle(
        title,
        x=0.015,
        y=1.04,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        color="#18324d",
    )
    fig.text(0.015, 1.02, subtitle, va="top", color="#52677d", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    folder = root / "reports/formatting"
    atomic_bytes(folder / f"{name}.plotly.json", plot.to_json().encode())
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight", facecolor="white")
    (root / "logs").mkdir(exist_ok=True)
    fig.savefig(
        root / f"logs/formatting-{name}.png", dpi=130, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def build(root: Path = ROOT, *, synthetic_fixture: bool = False):
    evidence = formatting_evidence(root)
    if evidence is None:
        raise ValueError("Complete semantic formatting evidence before plotting")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    records = pd.DataFrame(evidence["uncertainty"]).sort_values("observed_delta")
    labels = [LABELS[n] for n in records.contrast]
    delta = records.observed_delta.to_numpy()
    low = delta - records.simultaneous_lower.to_numpy()
    high = records.simultaneous_upper.to_numpy() - delta
    colors = ["#16837b" if value > 0 else "#b86554" for value in delta]
    plot = go.Figure(
        go.Bar(
            x=delta,
            y=labels,
            orientation="h",
            marker_color=colors,
            error_x={"type": "data", "array": high, "arrayminus": low},
            customdata=records.reference.to_numpy(),
            hovertemplate="%{y}<br>AUC change %{x:+.4f}<br>Reference: %{customdata}<extra></extra>",
        )
    )
    plot.add_vline(x=0, line_color="#18324d")
    plot.update_xaxes(title="Change in policy-macro AUC")
    fig, ax = plt.subplots(figsize=(11.8, 5.1))
    ax.barh(labels, delta, color=colors)
    ax.errorbar(delta, labels, xerr=np.stack([low, high]), fmt="none", ecolor="#18324d", capsize=3)
    ax.axvline(0, color="#18324d", linewidth=1)
    ax.set_xlabel("Change in policy-macro AUC")
    ax.grid(axis="x", alpha=0.2)
    title_prefix = "SYNTHETIC LAYOUT TEST · " if synthetic_fixture else ""
    caption = (
        "Authored software-test scores; not development performance"
        if synthetic_fixture
        else "Seven matched comparisons · simultaneous 95% intervals · four fixed policies"
    )
    save(
        root,
        "contrasts",
        plot,
        fig,
        title_prefix + "Do the remaining semantic hypotheses improve transfer?",
        caption,
    )
    transfer = {
        r["model"]: r["metrics"] for r in evidence["results"] if r["protocol"] == "heldout_rule"
    }
    rules = sorted(transfer["qwen_centroid"]["per_rule_auc"])
    labels = [r.split(":")[0].removeprefix("No ").capitalize() for r in rules]
    values = np.array([[transfer[m]["per_rule_auc"][r] for r in rules] for m in MODELS])
    plot = go.Figure(
        go.Heatmap(
            z=values,
            x=labels,
            y=list(MODELS.values()),
            zmin=0,
            zmax=1,
            colorscale="RdBu",
            text=values,
            texttemplate="%{text:.3f}",
            colorbar={"title": "AUC"},
            hovertemplate="%{y}<br>%{x}<br>AUC %{z:.4f}<extra></extra>",
        )
    )
    plot.update_yaxes(autorange="reversed")
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    plotted = ax.imshow(values, cmap="RdBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(rules)), labels=labels)
    ax.set_yticks(range(len(MODELS)), labels=list(MODELS.values()))
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < 0.25 or value > 0.75 else "#18324d",
            )
    fig.colorbar(plotted, ax=ax, label="AUC")
    save(
        root,
        "policies",
        plot,
        fig,
        title_prefix + "An average gain must survive policy-level inspection",
        (
            "Authored software-test scores; not development performance"
            if synthetic_fixture
            else "Held-out-policy AUC · 0.5 is chance · reserved policies are not shown"
        ),
    )
    folder = root / "reports/formatting"
    atomic_json(
        folder / "figures.json",
        {
            "synthetic_fixture": synthetic_fixture,
            "source_sha256": digest(root / "scripts/build_formatting_report.py"),
            "metadata_sha256": digest(folder / "metadata.json"),
            "files": {name: digest(folder / name) for name in sorted(FILES)},
        },
    )


def verify_figures(root: Path = ROOT) -> dict:
    formatting_evidence(root)
    folder = root / "reports/formatting"
    manifest = json.loads((folder / "figures.json").read_text())
    if (
        manifest["source_sha256"] != digest(root / "scripts/build_formatting_report.py")
        or manifest["metadata_sha256"] != digest(folder / "metadata.json")
        or set(manifest["files"]) != FILES
    ):
        raise ValueError("Semantic formatting figure contract differs")
    for name, sha in manifest["files"].items():
        if digest(folder / name) != sha:
            raise ValueError("Semantic formatting figure checksum differs")
    return manifest


def display_figure(root: Path, name: str):
    from IPython.display import SVG, display
    from plotly.io import from_json

    if name not in {"contrasts", "policies"}:
        raise ValueError("Unknown semantic formatting figure")
    if verify_figures(root).get("synthetic_fixture"):
        raise ValueError("Synthetic layout fixtures cannot be published as research figures")
    folder = root / "reports/formatting"
    from_json((folder / f"{name}.plotly.json").read_text()).show()
    display(SVG(filename=str(folder / f"{name}.svg")))


if __name__ == "__main__":
    build()
