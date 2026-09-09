"""Read-only competition feature evidence and Plotly/static comparison figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import plotly.graph_objects as go

from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "lexical_reference": "Lexical reference",
    "rule_raw": "4B rule score",
    "support_raw": "4B example score",
    "rule_support_raw": "4B joint score",
    "rule_embedding": "Screened rule representation",
    "rule_support_embedding": "Screened joint representation",
    "all_features": "Screened combined features",
}


def evidence(root):
    folder = root / "reports/competition_features"
    manifest = json.loads((folder / "metadata.json").read_text())
    result = {}
    for name, sha in manifest["files"].items():
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("Unsafe public evidence path")
        if digest(folder / name) != sha:
            raise ValueError("Competition evidence checksum differs: " + name)
        result[Path(name).stem] = json.loads((folder / name).read_text())
    return result


def build(root):
    record = evidence(root)
    folder = root / "reports/competition_features"
    plot = go.Figure()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "jigsaw-competition"})
    fig, ax = plt.subplots(figsize=(11, 6))
    position = np.arange(len(LABELS))
    for i, (protocol, title, color) in enumerate(
        (
            ("seen_rule", "Familiar policy", "#7b8fa1"),
            ("heldout_rule", "Held-out policy", "#087f8c"),
        )
    ):
        scores = {
            row["model"]: row["metrics"]["rule_macro_auc"]
            for row in record["results"]
            if row["protocol"] == protocol
        }
        values = [scores[name] for name in LABELS]
        plot.add_trace(
            go.Bar(
                name=title,
                x=values,
                y=list(LABELS.values()),
                orientation="h",
                marker_color=color,
                text=[f"{v:.4f}" for v in values],
                textposition="outside",
            )
        )
        bars = ax.barh(position + (i - 0.5) * 0.36, values, height=0.36, label=title, color=color)
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=4, fontsize=9)
    title = "Competition rebuild: what richer context contributes"
    subtitle = "Original 2,029-row development set; these are not Kaggle scores"
    plot.update_layout(
        title=title + "<br><sup>" + subtitle + "</sup>",
        template="plotly_white",
        barmode="group",
        height=600,
        margin={"l": 250, "r": 55, "t": 95, "b": 65},
        xaxis={"title": "Policy-macro ROC AUC", "range": [0, 1.05]},
        yaxis={"autorange": "reversed"},
        legend={"orientation": "h", "y": -0.15},
    )
    ax.set_yticks(position, list(LABELS.values()))
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.05), xlabel="Policy-macro ROC AUC")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.axvline(0.5, color="#aaa", linestyle="--", linewidth=1)
    ax.legend(loc="lower right")
    fig.suptitle(title, fontsize=15, x=0.03, ha="left", weight="bold")
    fig.text(0.03, 0.92, subtitle, fontsize=10, color="#52677d")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    atomic_bytes(folder / "comparison.plotly.json", plot.to_json().encode())
    fig.savefig(folder / "comparison.svg", metadata={"Date": None}, bbox_inches="tight")
    (root / "logs").mkdir(exist_ok=True)
    fig.savefig(root / "logs/competition-comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    atomic_json(
        folder / "figures.json",
        {
            "source_metadata_sha256": digest(folder / "metadata.json"),
            "files": {
                name: digest(folder / name) for name in ("comparison.plotly.json", "comparison.svg")
            },
        },
    )
    return plot


def display_comparison(root):
    from IPython.display import display

    evidence(root)
    folder = root / "reports/competition_features"
    manifest = json.loads((folder / "figures.json").read_text())
    if manifest["source_metadata_sha256"] != digest(folder / "metadata.json"):
        raise ValueError("Competition figure has stale evidence")
    for name, sha in manifest["files"].items():
        if Path(name).name != name or digest(folder / name) != sha:
            raise ValueError("Competition figure checksum differs")
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
