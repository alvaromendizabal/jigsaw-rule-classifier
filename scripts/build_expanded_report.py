"""Matched interactive and static figures for the four-policy feature decision."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jigsaw_rules.expanded import expanded_evidence
from jigsaw_rules.retrieval import retrieval_evidence
from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("ablation", "policies", "screening", "comparison")
FILES = {f"{name}.{suffix}" for name in NAMES for suffix in ("svg", "plotly.json")}
FAMILIES = {
    "word": "Words",
    "character": "Character patterns",
    "structure": "Style and structure",
    "lexical": "Lexical support comparisons",
    "support_tokens": "Support-token interactions",
    "semantic": "Embedding coordinates",
    "semantic_scalar": "Semantic comparisons",
    "ranks": "Training-reference percentiles",
    "community": "Community and frequency",
    "target_context": "Cross-fitted target context",
}
MODELS = {
    "comment_only": "Comment-only reference",
    "rule_examples": "Rule/example reference",
    "word_screened": "Screened words",
    "word_character": "Words + character patterns",
    "word_semantic_scalar": "Words + semantic comparisons",
    "semantic_scalar_only": "Semantic comparisons only",
    "character_full": "Full character vocabulary",
    "word_character_nb": "Word/character NB weighting",
    "latent_joint_lexical": "Lexical SVD (128)",
    "latent_qwen_interaction": "Semantic interaction SVD (128)",
    "all_transfer": "All transfer families",
    "all_with_coordinates": "All + embedding coordinates",
    "all_with_metadata": "All + community/target context",
    "qwen_centroid": "Frozen semantic centroid",
    "semantic_retrieval": "Semantic comparisons + retrieval",
    "word_semantic_retrieval": "Words + semantics + retrieval",
}


def _save(root: Path, name: str, plot: go.Figure, fig, title: str, subtitle: str) -> None:
    metadata = json.loads((root / "reports/expanded/metadata.json").read_text())
    if metadata.get("software_test"):
        title = "SYNTHETIC LAYOUT TEST · " + title
        subtitle = "Authored software-test scores; not competition or development performance"
    plot.update_layout(
        template="plotly_white",
        height=550,
        title=title + "<br><sup>" + subtitle + "</sup>",
        font={"family": "Arial", "color": "#18324d", "size": 12},
        margin={"l": 235, "r": 35, "t": 90, "b": 65},
    )
    fig.suptitle(
        title,
        x=0.015,
        y=1.04,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color="#18324d",
    )
    fig.text(0.015, 1.02, subtitle, va="top", color="#52677d", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    folder = root / "reports/expanded"
    folder.mkdir(parents=True, exist_ok=True)
    atomic_bytes(folder / f"{name}.plotly.json", plot.to_json().encode())
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight", facecolor="white")
    (root / "logs").mkdir(exist_ok=True)
    fig.savefig(root / f"logs/expanded-{name}.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build(root: Path = ROOT) -> None:
    evidence = expanded_evidence(root)
    if evidence is None:
        raise ValueError("Complete and verify the expanded study before plotting")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    contrasts = pd.DataFrame(evidence["uncertainty"])
    additions = contrasts[
        (contrasts.protocol == "heldout_rule") & contrasts.contrast.str.startswith("add_")
    ].copy()
    additions["family"] = additions.contrast.str.removeprefix("add_")
    additions = additions.sort_values("observed_delta")
    labels = [FAMILIES[name] for name in additions.family]
    delta = additions.observed_delta.to_numpy()
    low = delta - additions.simultaneous_lower.to_numpy()
    high = additions.simultaneous_upper.to_numpy() - delta
    colors = ["#16837b" if x > 0 else "#b86554" for x in delta]
    plot = go.Figure(
        go.Bar(
            x=delta,
            y=labels,
            orientation="h",
            marker_color=colors,
            error_x={"type": "data", "array": high, "arrayminus": low},
            hovertemplate="%{y}<br>AUC change %{x:+.4f}<extra></extra>",
        )
    )
    plot.add_vline(x=0, line_color="#18324d", line_width=1)
    plot.update_xaxes(title="Policy-macro AUC change from screened words")
    fig, ax = plt.subplots(figsize=(11.6, 5.4))
    ax.barh(
        labels,
        delta,
        color=colors,
        xerr=np.vstack([low, high]),
        height=0.6,
        error_kw={"elinewidth": 1, "capsize": 3, "ecolor": "#52677d"},
    )
    ax.axvline(0, color="#18324d", lw=1)
    ax.set_xlabel("Policy-macro AUC change from screened words")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    _save(
        root,
        "ablation",
        plot,
        fig,
        "Which feature additions improve policy transfer?",
        "Four held-out policies · fixed classifier · simultaneous conditional 95% intervals",
    )

    records = {r["model"]: r for r in evidence["results"] if r["protocol"] == "heldout_rule"}
    retrieval = retrieval_evidence(root)
    if retrieval is not None:
        records.update(
            {r["model"]: r for r in retrieval["results"] if r["protocol"] == "heldout_rule"}
        )
    selected = [name for name in MODELS if name in records]
    ordered = sorted(selected, key=lambda n: records[n]["metrics"]["rule_macro_auc"])
    scores = [records[n]["metrics"]["rule_macro_auc"] for n in ordered]
    labels = [MODELS[n] for n in ordered]
    colors = ["#16837b" if n == "qwen_centroid" else "#277da8" for n in ordered]
    plot = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{x:.4f}" for x in scores],
            textposition="outside",
        )
    )
    plot.update_xaxes(range=[0, 1], title="Policy-macro ROC AUC")
    plot.add_vline(x=0.5, line_dash="dot", line_color="#52677d")
    fig, ax = plt.subplots(figsize=(11.6, 7.0))
    ax.barh(labels, scores, color=colors, height=0.62)
    for i, score in enumerate(scores):
        ax.text(score + 0.004, i, f"{score:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="#52677d", ls=":", lw=1)
    ax.set_xlabel("Policy-macro ROC AUC")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    _save(
        root,
        "comparison",
        plot,
        fig,
        "Representation quality under the same validation boundary",
        f"{evidence['audit']['development_rows']:,} development rows · four-policy transfer · "
        "exploratory development results",
    )

    chosen = [
        "rule_examples",
        "character_full",
        "word_semantic_scalar",
        "all_transfer",
        "qwen_centroid",
    ]
    rules = sorted(records["rule_examples"]["metrics"]["per_rule_auc"])
    names = ["Advertising", "Legal advice", "Medical advice", "Illegal-activity promotion"]
    z = np.array(
        [[records[name]["metrics"]["per_rule_auc"][rule] for rule in rules] for name in chosen]
    )
    labels = [MODELS[name] for name in chosen]
    plot = go.Figure(
        go.Heatmap(
            z=z,
            x=names,
            y=labels,
            zmin=0,
            zmax=1,
            zmid=0.5,
            colorscale="RdBu",
            text=z,
            texttemplate="%{text:.3f}",
            colorbar_title="ROC AUC",
        )
    )
    plot.update_yaxes(autorange="reversed")
    fig, ax = plt.subplots(figsize=(11.6, 4.8))
    heat = ax.imshow(z, vmin=0, vmax=1, cmap="RdBu", aspect="auto")
    ax.set_xticks(range(4), names)
    ax.set_yticks(range(len(labels)), labels)
    for i, j in np.ndindex(z.shape):
        ax.text(
            j,
            i,
            f"{z[i, j]:.3f}",
            ha="center",
            va="center",
            color="white" if z[i, j] < 0.25 or z[i, j] > 0.75 else "#18324d",
        )
    fig.colorbar(heat, ax=ax, shrink=0.75, label="ROC AUC")
    _save(
        root,
        "policies",
        plot,
        fig,
        "Does the gain survive each held-out policy?",
        "Each policy receives equal weight in the primary metric; averages can hide failures",
    )

    screens = pd.DataFrame(evidence["screening"]["families"])
    grouped = screens.groupby("family")[["candidates", "retained"]].mean().reindex(list(FAMILIES))
    labels = list(FAMILIES.values())
    plot = go.Figure()
    for column, label, color in [
        ("candidates", "Generated", "#a6afb9"),
        ("retained", "Retained", "#16837b"),
    ]:
        plot.add_trace(
            go.Bar(x=grouped[column], y=labels, name=label, orientation="h", marker_color=color)
        )
    plot.update_layout(barmode="group", legend={"orientation": "h", "y": -0.15})
    plot.update_xaxes(type="log", title="Mean columns per fold (log scale)")
    plot.update_yaxes(autorange="reversed")
    fig, ax = plt.subplots(figsize=(11.6, 5.7))
    y = np.arange(len(labels))
    ax.barh(y - 0.18, grouped.candidates, height=0.34, color="#a6afb9", label="Generated")
    ax.barh(y + 0.18, grouped.retained, height=0.34, color="#16837b", label="Retained")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Mean columns per fold (log scale)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.legend(frameon=False)
    _save(
        root,
        "screening",
        plot,
        fig,
        "A large candidate space becomes a smaller fold-specific bank",
        "Training-only screening · seven folds · final family selection remains separate",
    )
    folder = root / "reports/expanded"
    atomic_json(
        folder / "figures.json",
        {
            "schema": 1,
            "builder_sha256": digest(Path(__file__)),
            "evidence_sha256": digest(folder / "metadata.json"),
            "retrieval_sha256": digest(root / "reports/retrieval/metadata.json")
            if retrieval
            else None,
            "files": {name: digest(folder / name) for name in sorted(FILES)},
        },
    )


def verify_figures(root: Path) -> dict:
    folder = root / "reports/expanded"
    manifest = json.loads((folder / "figures.json").read_text())
    if (
        manifest["schema"] != 1
        or set(manifest["files"]) != FILES
        or manifest["builder_sha256"] != digest(root / "scripts/build_expanded_report.py")
        or manifest["evidence_sha256"] != digest(folder / "metadata.json")
        or manifest.get("retrieval_sha256")
        != (
            digest(root / "reports/retrieval/metadata.json")
            if (root / "reports/retrieval/metadata.json").exists()
            else None
        )
        or any(digest(folder / name) != sha for name, sha in manifest["files"].items())
    ):
        raise ValueError("Expanded-study figure contract differs")
    return manifest


def display_figure(root: Path, name: str) -> None:
    from IPython.display import display

    if name not in NAMES:
        raise ValueError("Unknown expanded-study figure")
    expanded_evidence(root)
    verify_figures(root)
    folder = root / "reports/expanded"
    display(
        {
            "application/vnd.plotly.v1+json": json.loads(
                (folder / f"{name}.plotly.json").read_text()
            ),
            "image/svg+xml": (folder / f"{name}.svg").read_text(),
        },
        raw=True,
    )


if __name__ == "__main__":
    build()
