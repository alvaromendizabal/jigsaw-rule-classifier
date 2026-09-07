"""Reproduce the public semantic comparison from reviewed aggregate evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from jigsaw_rules.runtime import Progress, digest

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "reports/semantic"
    with Progress(ROOT / "logs/public_semantic_report.jsonl", "semantic_public_report"):
        metadata = json.loads((directory / "metadata.json").read_text())
        for name, sha in metadata["files"].items():
            if digest(directory / name) != sha:
                raise ValueError(f"Published evidence checksum mismatch: {name}")
        baseline = json.loads((ROOT / "reports/baseline/results.json").read_text())
        semantic = json.loads((directory / "results.json").read_text())
        records = baseline + semantic
        models = [
            ("rule_examples", "Lexical\nreference"),
            ("semantic_margin", "Semantic\nmargin"),
            ("semantic_classifier", "Semantic\nclassifier"),
        ]
        colors = ["#7b8fa1", "#087f8c", "#4a58a7"]
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 11,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.spines.left": False,
                "svg.hashsalt": "jigsaw-semantic",
            }
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor="#f5f7f8")
        fig.subplots_adjust(left=0.07, right=0.98, top=0.7, bottom=0.26, wspace=0.28)
        fig.text(
            0.07,
            0.94,
            "JIGSAW / SEMANTIC GENERALIZATION",
            color=colors[1],
            fontsize=10,
            weight="bold",
        )
        fig.text(0.07, 0.865, "Beyond lexical overlap", fontsize=24, weight="bold", color="#1b303d")
        fig.text(
            0.07,
            0.795,
            "Frozen Qwen3 embeddings · same saved validation splits · 2,029 labeled rows",
            color="#536775",
        )
        for ax, protocol, title in zip(
            axes, ["seen_rule", "heldout_rule"], ["Familiar rules", "Held-out rule"], strict=True
        ):
            values = [
                next(
                    r["metrics"]["rule_macro_auc"]
                    for r in records
                    if r["model"] == name and r["protocol"] == protocol
                )
                for name, _ in models
            ]
            bars = ax.bar([label for _, label in models], values, color=colors, width=0.62)
            ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=7, weight="bold")
            ax.set(title=title, ylim=(0, 1), ylabel="Rule macro ROC AUC")
            ax.axhline(0.5, linestyle="--", color="#748691", linewidth=1)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.grid(axis="y", alpha=0.14)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
        fig.text(
            0.07,
            0.13,
            "Local out-of-fold results; not Kaggle leaderboard scores. "
            "Dashed line: chance ranking.",
            fontsize=10,
        )
        fig.text(
            0.07,
            0.08,
            "Only two rules are observed. Probability quality and paired intervals "
            "are reported separately.",
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
