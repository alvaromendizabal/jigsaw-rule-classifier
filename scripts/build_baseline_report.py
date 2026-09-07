"""Render the published aggregate evidence; never reads or publishes raw comments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jigsaw_rules.runtime import Progress, digest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    directory = ROOT / "reports/baseline"
    with Progress(ROOT / "logs/public_report.jsonl", "publish_aggregate_chart"):
        metadata = json.loads((directory / "metadata.json").read_text())
        for name, expected in metadata["files"].items():
            if digest(directory / name) != expected:
                raise ValueError(f"Evidence checksum mismatch: {name}")
        records = json.loads((directory / "results.json").read_text())
        colors = ["#087f8c", "#b75b3c"]
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 11,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.spines.left": False,
                "axes.edgecolor": "#cad4d9",
                "text.color": "#1b303d",
                "axes.labelcolor": "#1b303d",
                "svg.hashsalt": "jigsaw-baseline",
            }
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), facecolor="#f5f7f8")
        fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.24, wspace=0.3)
        fig.text(
            0.07, 0.94, "JIGSAW  /  BASELINE EVIDENCE", fontsize=10, color=colors[0], weight="bold"
        )
        fig.text(0.07, 0.87, "Generalization is the next challenge", fontsize=22, weight="bold")
        fig.text(
            0.07,
            0.8,
            "2,029 labeled comments · 2 rules · local out-of-fold evaluation",
            color="#536775",
        )
        models = [("comment_only", "Comment only"), ("rule_examples", "Rule + examples")]
        for ax, protocol, title in zip(
            axes, ["seen_rule", "heldout_rule"], ["Familiar rules", "Held-out rule"], strict=True
        ):
            subset = {r["model"]: r["metrics"] for r in records if r["protocol"] == protocol}
            values = [subset[name]["rule_macro_auc"] for name, _ in models]
            bars = ax.bar([label for _, label in models], values, color=colors, width=0.57)
            ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=8, weight="bold")
            ax.axhline(0.5, color="#748691", linestyle="--", linewidth=1)
            ax.set(ylim=(0, 1), title=title, ylabel="Rule macro ROC AUC")
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.grid(axis="y", alpha=0.15)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
        fig.text(
            0.07,
            0.135,
            "Dashed line: chance ranking (0.5). Full 0–1 scale; no uncertainty interval estimated.",
            fontsize=10,
        )
        fig.text(
            0.07,
            0.087,
            "Two rules provide limited transfer evidence. These are not Kaggle leaderboard scores.",
            fontsize=10,
            color="#536775",
        )
        fig.savefig(
            directory / "comparison.svg", metadata={"Date": None}, facecolor=fig.get_facecolor()
        )
        fig.savefig(directory / "comparison.png", dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)


if __name__ == "__main__":
    main()
