"""Target-free, CPU-only support selection. No model or prediction-array reads.

The caller supplies a hash-pinned adaptation plan. Matrix row i always refers to
training[i]; sorting the pool never reorders the matrix independently. Purging is
per validation fold, across *all* training rules in that fold, not across folds.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from jigsaw_rules.data import normalize

PROTOCOL = "new_rule_supplied_support_adaptation"
READ_ARRAYS = ("train_adapted_vectors", "query_adapted_vectors", "query_row_ids")


def _noop() -> None:
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(normalize(value))


def validate_plan(plan: dict) -> list[dict]:
    """Mirror the reviewed training/query schema and add explicit support checks.

    Same-rule entries have repeat=2 because the pinned build_study uses only
    supplied supports for the held-out rule. This field alone is not independent
    provenance proof: the plan checksum and reviewed producer are also required.
    """
    if set(plan) != {"schema", "protocol", "folds"}:
        raise ValueError("Unexpected plan schema")
    if plan["schema"] != 1 or plan["protocol"] != PROTOCOL:
        raise ValueError("Unexpected plan protocol")
    if not isinstance(plan["folds"], list) or not plan["folds"]:
        raise ValueError("At least one fold is required")
    seen_ids, seen_rules, reports = set(), set(), []
    for fi, fold in enumerate(plan["folds"]):
        if set(fold) != {"rule", "training", "queries", "audit"}:
            raise ValueError("Unexpected fold schema")
        if not _text(fold["rule"]) or not isinstance(fold["audit"], dict):
            raise ValueError("Invalid policy/audit metadata")
        rule = normalize(fold["rule"])
        if rule in seen_rules:
            raise ValueError("Duplicate fold policy")
        seen_rules.add(rule)
        if not fold["queries"] or not fold["training"]:
            raise ValueError("Empty query or training pool")
        query_text = set()
        for row in fold["queries"]:
            if set(row) != {"row_id", "body", "rule"}:
                raise ValueError("Query schema must exclude targets and predictions")
            if type(row["row_id"]) is not int or row["row_id"] in seen_ids:
                raise ValueError("Query IDs must be unique integers")
            if not _text(row["body"]) or row["rule"] != fold["rule"]:
                raise ValueError("Invalid query body or inconsistent policy")
            seen_ids.add(row["row_id"])
            query_text.add(normalize(row["body"]))
        keys, counts, same = set(), Counter(), []
        for i, row in enumerate(fold["training"]):
            if set(row) != {"body", "rule", "rule_violation", "repeat"}:
                raise ValueError("Unexpected adaptation training schema")
            if not _text(row["body"]) or not _text(row["rule"]):
                raise ValueError("Empty or nontext training fields")
            if type(row["rule_violation"]) is not int or row["rule_violation"] not in (0, 1):
                raise ValueError("Support labels must be binary integers")
            nr, nb = normalize(row["rule"]), normalize(row["body"])
            if type(row["repeat"]) is not int or row["repeat"] != (2 if nr == rule else 1):
                raise ValueError("Training provenance/repetition convention differs")
            if (nr, nb) in keys:
                raise ValueError("Duplicate or conflicting normalized training pair")
            if nb in query_text:
                raise ValueError("Query body occurs in a fitted source, including other rules")
            keys.add((nr, nb))
            if nr == rule:
                same.append(i)
                counts[row["rule_violation"]] += 1
        if set(counts) != {0, 1}:
            raise ValueError("Both same-rule supplied-support classes are required")
        reports.append(
            {
                "fold": fi,
                "training_rows": len(fold["training"]),
                "query_rows": len(fold["queries"]),
                "same_rule_supports": len(same),
                "support_class_counts": {str(k): counts[k] for k in (0, 1)},
                "query_schema_target_free": True,
                "per_fold_all_rule_purge_passed": True,
                "support_indices": same,
            }
        )
    return reports


def load_vectors(path: Path, fold: dict, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    """Open exactly three members; saved model scores and other arrays are unused."""
    with np.load(path, allow_pickle=False) as saved:
        if not set(READ_ARRAYS).issubset(saved.files):
            raise ValueError("Missing cached vector/ID arrays")
        # Even an unused extra target member indicates a wrong or polluted input.
        forbidden = {"query_targets", "query_labels", "labels", "rule_violation", "y_query"}
        if forbidden.intersection(saved.files):
            raise ValueError("Unexpected target array in representation input")
        train = saved["train_adapted_vectors"]
        query = saved["query_adapted_vectors"]
        ids = saved["query_row_ids"]
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("Unexpected query ID array")
    if not np.array_equal(ids, [r["row_id"] for r in fold["queries"]]):
        raise ValueError("Query vector/plan order mismatch")
    for name, array, rows in (
        ("training", train, fold["training"]),
        ("query", query, fold["queries"]),
    ):
        if array.ndim != 2 or array.shape != (len(rows), dimensions):
            raise ValueError(f"Unexpected {name} representation dimensions or row count")
        if array.dtype.kind != "f" or not np.isfinite(array).all():
            raise ValueError(f"Nonfinite or nonfloating {name} representations")
    return train, query


def normalized_vectors(matrix: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale rows before normalization, avoiding overflow from finite large values."""
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite vectors")
    values = np.asarray(matrix, dtype=np.float64)
    scale = np.max(np.abs(values), axis=1)
    safe = np.where(scale > 0, scale, 1.0)
    scaled = values / safe[:, None]
    lengths = np.sqrt(np.sum(scaled * scaled, axis=1))
    # Equivalent to norm <= epsilon without overflowing the norm itself.
    zero = (scale == 0) | (scale <= epsilon / np.maximum(lengths, 1.0))
    result = np.divide(
        scaled, lengths[:, None], out=np.zeros_like(scaled), where=lengths[:, None] > 0
    )
    result[zero] = 0
    return result, zero


def prepare_selector(
    fold: dict, train: np.ndarray, query: np.ndarray, epsilon: float = 1e-12
) -> dict:
    """Keep original indices while sorting texts exactly like reviewed lexical code."""
    train_unit, train_zero = normalized_vectors(train, epsilon)
    query_unit, query_zero = normalized_vectors(query, epsilon)
    if query_zero.any():
        raise ValueError("Zero-norm query vector: no semantic decision is defensible")
    rule = normalize(fold["rule"])
    pool = sorted(
        [i for i, r in enumerate(fold["training"]) if normalize(r["rule"]) == rule],
        key=lambda i: (normalize(fold["training"][i]["body"]), fold["training"][i]["body"]),
    )
    labels = np.array([fold["training"][i]["rule_violation"] for i in pool])
    if set(labels) != {0, 1}:
        raise ValueError("Both same-rule support classes are required")
    pool_array = np.asarray(pool, dtype=int)
    valid = ~train_zero[pool_array]
    if any(not np.any(valid & (labels == c)) for c in (0, 1)):
        raise ValueError("No nonzero semantic support for a required class")
    # This is exactly the historical lexical comparator, not a new tuned variant.
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    lexical_train = tfidf.fit_transform([fold["training"][i]["body"] for i in pool])
    return {
        "pool": pool_array,
        "labels": labels,
        "semantic_valid": valid,
        "train_unit": train_unit,
        "query_unit": query_unit,
        "tfidf": tfidf,
        "lexical_train": lexical_train,
        "zero_training_rows": int(train_zero.sum()),
        "zero_same_rule_supports": int((~valid).sum()),
    }


def select_batch(
    fold: dict, state: dict, start: int, end: int, tick: Callable[[], None] = _noop
) -> list[dict]:
    """No query labels, saved margins, model calls or query-fitted transformations."""
    tick()
    pool, labels = state["pool"], state["labels"]
    queries = fold["queries"][start:end]
    lexical_query = state["tfidf"].transform([r["body"] for r in queries])
    lexical = (lexical_query @ state["lexical_train"].T).toarray()
    semantic = state["query_unit"][start:end] @ state["train_unit"][pool].T
    tick()
    result = []
    for j, row in enumerate(queries):
        record = {"query_index": start + j, "row_id": row["row_id"]}
        for method, scores in (("lexical", lexical[j]), ("semantic", semantic[j])):
            selected = {}
            for label, key in ((1, "positive"), (0, "negative")):
                eligible = labels == label
                if method == "semantic":
                    eligible &= state["semantic_valid"]
                positions = np.flatnonzero(eligible)
                # pool is normalized-text sorted. Exact score ties choose its first row.
                pos = positions[int(np.argmax(scores[positions]))]
                index = int(pool[pos])
                selected[key] = {
                    "training_index": index,
                    "cosine": float(scores[pos]),
                    "support_id": fingerprint(
                        {
                            "rule": normalize(fold["rule"]),
                            "body": normalize(fold["training"][index]["body"]),
                        }
                    ),
                }
            selected["positive_minus_negative_cosine"] = (
                selected["positive"]["cosine"] - selected["negative"]["cosine"]
            )
            a, b = (selected[k]["training_index"] for k in ("positive", "negative"))
            selected["pair_semantic_cosine"] = float(
                state["train_unit"][a] @ state["train_unit"][b]
            )
            record[method] = selected
        record["pair_changed"] = any(
            record["semantic"][k]["support_id"] != record["lexical"][k]["support_id"]
            for k in ("positive", "negative")
        )
        result.append(record)
    return result


# Diagnostics only: these shallow indicators are NOT ground truth or selection features.
CUES = {
    "request": r"\?|\b(?:can|could|should) (?:i|we)\b|\b(?:any advice|how do i|what should)\b",
    "advice_offer": r"\b(?:you should|you must|you need to|i recommend|my advice)\b",
    "quotation": r'["“”]|(?m:^\s*>)',
    "negation": r"\b(?:no|not|never|without|cannot|can't|don't|doesn't)\b",
    "exception": r"\b(?:unless|except|however|but|although)\b",
    "personal_account": r"\b(?:i was|i had|in my experience|happened to me)\b",
    "commercial_cue": r"https?://|\b(?:buy|sale|discount|coupon|subscribe)\b",
}


def cues(text: str) -> dict[str, bool]:
    return {name: bool(re.search(pattern, text, re.IGNORECASE)) for name, pattern in CUES.items()}


def describe(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "min": float(a.min()),
        "median": float(np.median(a)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def summarize_fold(
    fi: int,
    fold: dict,
    rows: list[dict],
    state_summary: dict,
    warning_change: float,
    warning_reuse: float,
) -> dict:
    n = len(rows)
    changed = sum(r["pair_changed"] for r in rows)
    out = {
        "fold": fi,
        "queries": n,
        "changed_pairs": changed,
        "changed_pair_fraction": changed / n,
        **state_summary,
        "methods": {},
    }
    warnings = []
    if changed / n < warning_change:
        warnings.append("few_changed_pairs_diagnostic_warning")
    for method in ("lexical", "semantic"):
        m = {
            "balanced_pairs": n,
            "class_counts_per_pair": {"positive": 1, "negative": 1},
            "positive_minus_negative_cosine": describe(
                [r[method]["positive_minus_negative_cosine"] for r in rows]
            ),
            "positive_negative_pair_semantic_cosine": describe(
                [r[method]["pair_semantic_cosine"] for r in rows]
            ),
            "classes": {},
        }
        for sign in ("positive", "negative"):
            ids = [r[method][sign]["support_id"] for r in rows]
            counts = Counter(ids)
            mismatches = Counter()
            overlaps = []
            for row in rows:
                q = fold["queries"][row["query_index"]]["body"]
                s = fold["training"][row[method][sign]["training_index"]]["body"]
                qc, sc = cues(q), cues(s)
                mismatches.update({name: int(qc[name] != sc[name]) for name in CUES})
                qt, st = (
                    set(re.findall(r"\w+", normalize(q))),
                    set(re.findall(r"\w+", normalize(s))),
                )
                overlaps.append(len(qt & st) / max(1, len(qt | st)))
            most = max(counts.values()) / n
            probabilities = np.asarray(list(counts.values()), dtype=float) / n
            m["classes"][sign] = {
                "distinct_supports": len(counts),
                "maximum_reuse_fraction": most,
                "effective_support_count": float(
                    np.exp(-np.sum(probabilities * np.log(probabilities)))
                ),
                "selected_cosine": describe([r[method][sign]["cosine"] for r in rows]),
                "word_set_overlap": describe(overlaps),
                "cue_disagreement_fraction": {name: mismatches[name] / n for name in CUES},
            }
            if method == "semantic" and most > warning_reuse:
                warnings.append(f"high_{sign}_reuse_diagnostic_warning")
        out["methods"][method] = m
    out["warnings"] = warnings
    out["cue_diagnostics_are_not_intent_labels"] = True
    return out


def qualitative_sample(fold: dict, count: int, seed: int) -> list[int]:
    """Fixed, label/prediction-blind normalized-comment-group sampling."""
    unique = {}
    for i, row in enumerate(fold["queries"]):
        key = normalize(row["body"])
        unique.setdefault(key, i)
    ordered = sorted(
        unique,
        key=lambda body: (
            fingerprint({"seed": seed, "rule": normalize(fold["rule"]), "body": body}),
            body,
        ),
    )
    return [unique[k] for k in ordered[:count]]
