"""Render the protected data boundary with matching Plotly and static views."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
import plotly.graph_objects as go

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jigsaw_rules.released import released_evidence
from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = {"boundary.svg", "boundary.plotly.json"}
ROLES = [
    ("research", "Additional research", "#16837b"),
    ("confirmation", "Reserved confirmation", "#277da8"),
    ("boundary_overlap", "Research overlap exclusions", "#d49b52"),
    ("historical_overlap", "Historical exposure exclusions", "#a6afb9"),
]
LABELS = {
    "advertising": "Advertising",
    "financial": "Financial advice",
    "legal": "Legal advice",
    "medical": "Medical advice",
    "promotion": "Illegal-activity promotion",
    "spoilers": "Spoilers",
}


def build(root: Path = ROOT) -> None:
    evidence = released_evidence(root)
    if evidence is None:
        raise ValueError("Prepare the released boundary before rendering")
    counts = pd.DataFrame(evidence["boundary"]["counts"])
    counts = counts.groupby(["rule_id", "role"]).rows.sum().unstack(fill_value=0)
    counts = counts.reindex(list(LABELS))
    title = "New research data without consuming confirmation"
    subtitle = "54,059 released rows · role assignment uses metadata and text only"
    plot = go.Figure()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(12, 5.6))
    left = pd.Series(0, index=counts.index)
    labels = [LABELS[name] for name in counts.index]
    for role, label, color in ROLES:
        values = counts.get(role, pd.Series(0, index=counts.index))
        plot.add_trace(
            go.Bar(
                x=values,
                y=labels,
                name=label,
                orientation="h",
                marker_color=color,
                hovertemplate="%{y}<br>%{x:,} rows<extra>" + label + "</extra>",
            )
        )
        ax.barh(labels, values, left=left, label=label, color=color, height=0.62)
        for index, value in enumerate(values):
            if value >= 1200:
                ax.text(
                    left.iloc[index] + value / 2,
                    index,
                    f"{value:,}",
                    va="center",
                    ha="center",
                    color="white",
                    fontsize=10,
                )
        left += values
    plot.update_layout(
        template="plotly_white",
        barmode="stack",
        height=530,
        title=title + "<br><sup>" + subtitle + "</sup>",
        margin={"l": 205, "r": 25, "t": 85, "b": 105},
        legend={"orientation": "h", "y": -0.19},
        xaxis_title="Rows",
        yaxis={"autorange": "reversed"},
    )
    ax.invert_yaxis()
    ax.set_title(title, loc="left", pad=38, fontsize=15, fontweight="bold", color="#172b4d")
    ax.text(0, 1.045, subtitle, transform=ax.transAxes, color="#52677d")
    ax.set_xlabel("Rows")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.17), ncol=2, frameon=False)
    fig.tight_layout()
    folder = root / "reports/released"
    atomic_bytes(folder / "boundary.plotly.json", plot.to_json().encode())
    fig.savefig(folder / "boundary.svg", bbox_inches="tight", facecolor="white")
    (root / "logs").mkdir(exist_ok=True)
    fig.savefig(
        root / "logs/released-boundary.png", dpi=150, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    atomic_json(
        folder / "figures.json",
        {
            "schema": 1,
            "builder_sha256": digest(Path(__file__)),
            "boundary_sha256": digest(folder / "boundary.json"),
            "files": {name: digest(folder / name) for name in sorted(FIGURES)},
        },
    )


def verify_figures(root: Path) -> dict:
    folder = root / "reports/released"
    manifest = json.loads((folder / "figures.json").read_text())
    if (
        manifest["schema"] != 1
        or set(manifest["files"]) != FIGURES
        or manifest["builder_sha256"] != digest(root / "scripts/build_release_report.py")
        or manifest["boundary_sha256"] != digest(folder / "boundary.json")
        or any(digest(folder / name) != sha for name, sha in manifest["files"].items())
    ):
        raise ValueError("Released-boundary figure contract differs")
    return manifest


def display_boundary(root: Path) -> None:
    from IPython.display import display

    released_evidence(root)
    verify_figures(root)
    folder = root / "reports/released"
    display(
        {
            "application/vnd.plotly.v1+json": json.loads(
                (folder / "boundary.plotly.json").read_text()
            ),
            "image/svg+xml": (folder / "boundary.svg").read_text(),
        },
        raw=True,
    )


if __name__ == "__main__":
    build()
