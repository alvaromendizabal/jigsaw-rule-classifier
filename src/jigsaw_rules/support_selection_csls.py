"""Target-free CSLS-style support selection over cached adapted representations.

This module is a distinct experiment from the completed raw-cosine selector. It
never reads query targets, model predictions, or released labels. Matrix row i
remains aligned with plan training[i]. The only labels consumed are the legitimate
binary labels already present on eligible training/support rows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from jigsaw_rules.data import normalize


@dataclass(frozen=True)
class SelectorConfig:
    k: int = 16
    epsilon: float = 1e-12
    max_reuse_fraction: float = 0.25


def _unit(x: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("representations must be finite 2D arrays")
    scale = np.max(np.abs(a), axis=1)
    safe = np.where(scale > 0, scale, 1.0)
    b = a / safe[:, None]
    norms = np.sqrt(np.sum(b * b, axis=1))
    zero = (scale == 0) | (norms <= epsilon)
    out = np.divide(b, norms[:, None], out=np.zeros_like(b), where=norms[:, None] > 0)
    out[zero] = 0
    return out, zero


def _topk_mean(scores: np.ndarray, k: int, axis: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    n = scores.shape[axis]
    kk = min(k, n)
    if kk == n:
        return scores.mean(axis=axis)
    part = np.partition(scores, n - kk, axis=axis)
    sl = [slice(None)] * scores.ndim
    sl[axis] = slice(n - kk, None)
    return part[tuple(sl)].mean(axis=axis)


def same_rule_indices(fold: dict) -> np.ndarray:
    rule = normalize(fold["rule"])
    query_text = {normalize(r["body"]) for r in fold["queries"]}
    seen = set()
    indices = []
    for i, row in enumerate(fold["training"]):
        if set(row) != {"body", "rule", "rule_violation", "repeat"}:
            raise ValueError("unexpected training schema")
        if row["rule_violation"] not in (0, 1):
            raise ValueError("support label must be binary")
        key = (normalize(row["rule"]), normalize(row["body"]))
        if key in seen:
            raise ValueError("duplicate normalized training pair")
        seen.add(key)
        if key[1] in query_text:
            raise ValueError("query body leaked into training pool")
        if key[0] == rule:
            if row["repeat"] != 2:
                raise ValueError("same-rule support provenance differs from frozen plan")
            indices.append(i)
    if not indices:
        raise ValueError("no same-rule supports")
    labels = {fold["training"][i]["rule_violation"] for i in indices}
    if labels != {0, 1}:
        raise ValueError("both support classes are required")
    return np.asarray(indices, dtype=int)


def csls_scores(query: np.ndarray, support: np.ndarray, *, k: int) -> np.ndarray:
    """Compute CSLS(q,s)=2*cos(q,s)-r_q-r_s to reduce semantic hubs."""
    cosine = query @ support.T
    r_q = _topk_mean(cosine, k, axis=1)
    r_s = _topk_mean(cosine, k, axis=0)
    return 2.0 * cosine - r_q[:, None] - r_s[None, :]


def select_fold(
    fold: dict,
    train_vectors: np.ndarray,
    query_vectors: np.ndarray,
    config: SelectorConfig | None = None,
) -> dict:
    config = config or SelectorConfig()
    if train_vectors.shape[0] != len(fold["training"]):
        raise ValueError("training/vector row alignment differs")
    if query_vectors.shape[0] != len(fold["queries"]):
        raise ValueError("query/vector row alignment differs")
    train_u, train_zero = _unit(train_vectors, config.epsilon)
    query_u, query_zero = _unit(query_vectors, config.epsilon)
    if query_zero.any():
        raise ValueError("zero-norm query vector")
    pool = same_rule_indices(fold)
    labels = np.asarray([fold["training"][i]["rule_violation"] for i in pool], dtype=int)
    valid = ~train_zero[pool]
    if any(not np.any(valid & (labels == c)) for c in (0, 1)):
        raise ValueError("a support class has no nonzero vector")
    support = train_u[pool]
    raw = query_u @ support.T
    adjusted = csls_scores(query_u, support, k=config.k)
    selections = []
    reuse = {"raw": {0: Counter(), 1: Counter()}, "csls": {0: Counter(), 1: Counter()}}
    for qi, query_row in enumerate(fold["queries"]):
        record = {"query_index": qi, "row_id": query_row["row_id"]}
        for method, matrix in (("raw", raw), ("csls", adjusted)):
            picked = {}
            for label, name in ((1, "positive"), (0, "negative")):
                eligible = np.flatnonzero(valid & (labels == label))
                local = eligible[int(np.argmax(matrix[qi, eligible]))]
                index = int(pool[local])
                picked[name] = {
                    "training_index": index,
                    "score": float(matrix[qi, local]),
                    "cosine": float(raw[qi, local]),
                }
                reuse[method][label][index] += 1
            record[method] = picked
        record["pair_changed"] = any(
            record["raw"][name]["training_index"] != record["csls"][name]["training_index"]
            for name in ("positive", "negative")
        )
        selections.append(record)

    def summarize(method: str) -> dict:
        out = {}
        n = len(selections)
        for label, name in ((1, "positive"), (0, "negative")):
            counts = reuse[method][label]
            top = max(counts.values()) if counts else 0
            probs = np.asarray(list(counts.values()), dtype=float) / max(n, 1)
            entropy = (
                -float(np.sum(probs * np.log(np.maximum(probs, 1e-300)))) if len(probs) else 0.0
            )
            out[name] = {
                "distinct_supports": len(counts),
                "maximum_reuse_fraction": top / max(n, 1),
                "effective_support_count": float(np.exp(entropy)),
            }
        return out

    return {
        "queries": len(selections),
        "same_rule_supports": len(pool),
        "changed_pairs": int(sum(r["pair_changed"] for r in selections)),
        "changed_pair_fraction": float(np.mean([r["pair_changed"] for r in selections])),
        "raw": summarize("raw"),
        "csls": summarize("csls"),
        "selections": selections,
        "config": {
            "k": config.k,
            "epsilon": config.epsilon,
            "max_reuse_fraction": config.max_reuse_fraction,
        },
    }


def promotion_diagnostics(result: dict, max_reuse_fraction: float) -> dict:
    """Diagnostic gate only; it cannot promote a model or justify GPU inference alone."""
    csls = result["csls"]
    raw = result["raw"]
    reuse_better = all(
        csls[c]["maximum_reuse_fraction"] < raw[c]["maximum_reuse_fraction"]
        for c in ("positive", "negative")
    )
    within_cap = all(
        csls[c]["maximum_reuse_fraction"] <= max_reuse_fraction for c in ("positive", "negative")
    )
    return {
        "selection_changed": result["changed_pairs"] > 0,
        "reuse_strictly_improved_both_classes": reuse_better,
        "reuse_within_declared_cap_both_classes": within_cap,
        "eligible_for_blinded_relevance_review": bool(result["changed_pairs"] > 0 and reuse_better),
        "gpu_inference_authorized": False,
    }
