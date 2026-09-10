"""Hash-verified second-model evidence with Plotly and portable SVG rendering."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import plotly.graph_objects as go

from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "qwen": "Adapted Qwen · reused control",
    "phi_frozen": "Frozen Phi",
    "phi": "Support-adapted Phi",
    "blend": "Fixed 50/50 rank blend",
}


def evidence(root: Path) -> dict:
    folder = root / "reports/complementarity"
    manifest = json.loads((folder / "metadata.json").read_text())
    records = {}
    for name, sha in manifest["files"].items():
        if Path(name).name != name or not name.endswith(".json") or digest(folder / name) != sha:
            raise ValueError("Complementarity evidence checksum differs")
        records[Path(name).stem] = json.loads((folder / name).read_text())
    if set(records) != {
        "results",
        "uncertainty",
        "decision",
        "correlations",
        "protocol",
        "provenance",
        "inference",
    }:
        raise ValueError("Incomplete complementarity evidence")
    return records


def build(root: Path) -> None:
    records = evidence(root)
    folder = root / "reports/complementarity"
    values = [records["results"][key]["rule_macro_auc"] for key in LABELS]
    labels = [f"{value:.4f}" for value in values]
    colors = ["#7b8fa1", "#b5bdc7", "#4278a4", "#087f8c"]
    title = "Does a second model improve novel-comment ranking?"
    subtitle = (
        f"{records['protocol']['rows']} comments · 2 development policies · not a Kaggle score"
    )
    plot = go.Figure(
        go.Bar(
            x=values,
            y=list(LABELS.values()),
            orientation="h",
            marker_color=colors,
            text=labels,
            textposition="outside",
        )
    )
    plot.update_layout(
        title=title + "<br><sup>" + subtitle + "</sup>",
        template="plotly_white",
        height=430,
        margin={"l": 250, "r": 70, "t": 95, "b": 60},
        xaxis={"title": "Policy-macro ROC AUC", "range": [0, 1.05]},
        yaxis={"autorange": "reversed"},
        showlegend=False,
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "jigsaw-complementarity"})
    figure, axis = plt.subplots(figsize=(10.5, 4.5))
    bars = axis.barh(list(LABELS.values()), values, color=colors)
    axis.bar_label(bars, labels=labels, padding=5, fontsize=10)
    axis.invert_yaxis()
    axis.set(xlim=(0, 1.05), xlabel="Policy-macro ROC AUC")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.axvline(0.5, color="#aaa", linestyle="--", linewidth=1)
    figure.suptitle(title, fontsize=14, x=0.03, ha="left", weight="bold")
    figure.text(0.03, 0.90, subtitle, fontsize=10, color="#52677d")
    figure.tight_layout(rect=(0, 0, 1, 0.87))
    atomic_bytes(folder / "comparison.plotly.json", plot.to_json().encode())
    figure.savefig(folder / "comparison.svg", metadata={"Date": None}, bbox_inches="tight")
    (root / "logs").mkdir(exist_ok=True)
    figure.savefig(root / "logs/complementarity-comparison.png", dpi=130, bbox_inches="tight")
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
    folder = root / "reports/complementarity"
    record = json.loads((folder / "figures.json").read_text())
    if record["source_sha256"] != digest(Path(__file__)) or record["metadata_sha256"] != digest(
        folder / "metadata.json"
    ):
        raise ValueError("Complementarity figure source differs")
    for name, sha in record["files"].items():
        if Path(name).name != name or digest(folder / name) != sha:
            raise ValueError("Complementarity figure checksum differs")
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
