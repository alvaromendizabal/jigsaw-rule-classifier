"""Verified support-adaptation comparisons in Plotly with portable SVG fallbacks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import plotly.graph_objects as go

from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILIES = {
    "raw": "Direct decision score",
    "screened_embedding": "Screened decision representation",
    "centroid": "Class-centroid margin",
    "nearest": "Nearest-example margin",
    "top5": "Top-five-example margin",
    "geometry_blend": "Fixed decision/geometry blend",
}


def evidence(root: Path) -> dict:
    folder = root / "reports/support_adaptation"
    manifest = json.loads((folder / "metadata.json").read_text())
    records = {}
    for name, sha in manifest["files"].items():
        if Path(name).name != name or not name.endswith(".json") or digest(folder / name) != sha:
            raise ValueError("Adaptation evidence contract differs")
        records[Path(name).stem] = json.loads((folder / name).read_text())
    return records


def build(root: Path) -> None:
    folder = root / "reports/support_adaptation"
    records = evidence(root)
    fixture = json.loads((folder / "metadata.json").read_text()).get("synthetic_fixture", False)
    scores = {row["model"]: row["metrics"]["rule_macro_auc"] for row in records["results"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "jigsaw-adaptation"})
    plot = go.Figure()
    figure, axis = plt.subplots(figsize=(11, 5.8))
    position = np.arange(len(FAMILIES))
    for i, (model, label, color) in enumerate(
        (("frozen", "Frozen 4B", "#7b8fa1"), ("adapted", "Support-adapted 4B", "#087f8c"))
    ):
        values = [scores[model + "_" + family] for family in FAMILIES]
        labels = [f"{v:.4f}" for v in values]
        plot.add_trace(
            go.Bar(
                name=label,
                x=values,
                y=list(FAMILIES.values()),
                orientation="h",
                marker_color=color,
                text=labels,
                textposition="outside",
            )
        )
        bars = axis.barh(position + (i - 0.5) * 0.36, values, height=0.36, label=label, color=color)
        axis.bar_label(bars, labels=labels, padding=4, fontsize=9)
    title = "Does learning from known supports improve novel-comment ranking?"
    subtitle = (
        f"{records['protocol']['rows']:,} novel comments · 2 development policies"
        " · not a Kaggle score"
    )
    if fixture:
        title, subtitle = "Software layout check", "SYNTHETIC FIXTURE · no experimental results"
    plot.update_layout(
        title=title + "<br><sup>" + subtitle + "</sup>",
        template="plotly_white",
        barmode="group",
        height=560,
        margin={"l": 260, "r": 60, "t": 95, "b": 70},
        xaxis={"title": "Policy-macro ROC AUC", "range": [0, 1.05]},
        yaxis={"autorange": "reversed"},
        legend={"orientation": "h", "y": -0.2},
    )
    axis.set_yticks(position, list(FAMILIES.values()))
    axis.invert_yaxis()
    axis.set(xlim=(0, 1.05), xlabel="Policy-macro ROC AUC")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.axvline(0.5, color="#aaa", linestyle="--", linewidth=1)
    axis.legend(loc="lower right")
    figure.suptitle(title, fontsize=14, x=0.03, ha="left", weight="bold")
    figure.text(0.03, 0.92, subtitle, fontsize=10, color="#52677d")
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    atomic_bytes(folder / "comparison.plotly.json", plot.to_json().encode())
    figure.savefig(folder / "comparison.svg", metadata={"Date": None}, bbox_inches="tight")
    (root / "logs").mkdir(exist_ok=True)
    figure.savefig(root / "logs/adaptation-comparison.png", dpi=130, bbox_inches="tight")
    plt.close(figure)
    atomic_json(
        folder / "figures.json",
        {
            "source_sha256": digest(Path(__file__)),
            "metadata_sha256": digest(folder / "metadata.json"),
            "files": {
                name: digest(folder / name) for name in ("comparison.svg", "comparison.plotly.json")
            },
        },
    )


def display_comparison(root: Path) -> None:
    from IPython.display import display

    evidence(root)
    folder = root / "reports/support_adaptation"
    if json.loads((folder / "metadata.json").read_text()).get("synthetic_fixture"):
        raise ValueError("Synthetic fixtures cannot be displayed as public research")
    manifest = json.loads((folder / "figures.json").read_text())
    if manifest["source_sha256"] != digest(Path(__file__)) or manifest["metadata_sha256"] != digest(
        folder / "metadata.json"
    ):
        raise ValueError("Adaptation figure source differs")
    for name, sha in manifest["files"].items():
        if Path(name).name != name or digest(folder / name) != sha:
            raise ValueError("Adaptation figure checksum differs")
    display(
        {
            "application/vnd.plotly.v1+json": json.loads(
                (folder / "comparison.plotly.json").read_text()
            ),
            "image/svg+xml": (folder / "comparison.svg").read_text(),
        },
        raw=True,
    )


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1])
