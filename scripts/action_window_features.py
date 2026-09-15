"""Fixed action-centered spans; no automatic rationale or span labels."""

from __future__ import annotations

import re

import numpy as np
from sklearn.model_selection import GroupKFold

from jigsaw_rules.data import normalize
from scripts.local_support_features import checked_labels
from scripts.multiprototype_features import validate_inputs
from scripts.passage_support_features import aggregate_passages, vectorizer_for

ACTION = re.compile(
    r"\b(?:should|must|recommend|advise|suggest|hire|sue|buy|subscribe|order|contact|"
    # Separate fixed alternatives without creating a line-length failure.
    r"could i|can i|may i|how do i|how can i|we offer|i offer|message me)\b",
    re.I,
)
QUALIFIER = re.compile(
    r"\b(?:not|never|unless|however|although|but|disclaimer|quoted|said|wrote|"
    # Lexical cues are approximate: no syntactic-scope claim is made.
    r"according to|if|without|not a lawyer|isn't|don't|cannot)\b|[\"“”]",
    re.I,
)
MEASURES = (
    "maximum_evidence",
    "mean_evidence",
    "dispersion",
    "top2_evidence",
    "first_evidence",
    "last_evidence",
    "ending_shift",
    "strongest_position",
)
NAMES = tuple(
    f"{family}/{label}/{measure}"
    for family in ("action_windows", "qualification_windows")
    for label in ("permitted", "violating", "margin")
    for measure in MEASURES
)
MODES = ("windows", "displaced", "whole_comment", "label_null")


def select_windows(text, family, *, radius=12, maximum=6):
    if not isinstance(text, str) or not text.strip() or len(text) > 50000:
        raise ValueError("nonempty text within the fixed budget required")
    if family not in ("action", "qualification"):
        raise ValueError("unknown span family")
    if radius != 12 or maximum != 6:
        raise ValueError("registered span budget differs")
    tokens = list(re.finditer(r"\S+", text))
    pattern = ACTION if family == "action" else QUALIFIER
    intervals = []
    for match in pattern.finditer(text):
        ix = next((i for i, t in enumerate(tokens) if t.end() > match.start()), len(tokens) - 1)
        left, right = max(0, ix - radius), min(len(tokens), ix + radius + 1)
        if intervals and left <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(right, intervals[-1][1]))
        else:
            intervals.append((left, right))
    hit = bool(intervals)
    if not intervals:
        intervals = [(0, len(tokens))]
    # Keep cues at both ends when the number of disjoint spans exceeds the cap.
    if len(intervals) > maximum:
        indices = np.linspace(0, len(intervals) - 1, maximum).round().astype(int)
        intervals = [intervals[i] for i in indices]
    spans = [text[tokens[a].start() : tokens[b - 1].end()] for a, b in intervals]
    shifted = []
    for a, b in intervals:
        width = b - a
        capacity = len(tokens) - width
        start = 0 if capacity == 0 else (a + max(1, len(tokens) // 2)) % (capacity + 1)
        shifted.append(text[tokens[start].start() : tokens[start + width - 1].end()])
    if [len(re.findall(r"\S+", v)) for v in spans] != [len(re.findall(r"\S+", v)) for v in shifted]:
        raise AssertionError("displaced control did not match token counts")
    stats = {
        "cue_present": hit,
        "spans": len(spans),
        "selected_fraction": sum(b - a for a, b in intervals) / len(tokens),
        "control_changed_fraction": float(
            np.mean([a != b for a, b in zip(spans, shifted, strict=True)])
        ),
    }
    return spans, shifted, stats


def feature_block(reference, labels, bodies, queries, query_bodies=None, *, seed=20260929):
    texts = np.asarray(list(bodies), dtype=object)
    qt = list(query_bodies) if query_bodies is not None else []
    if not qt or any(
        not isinstance(t, str) or not t.strip() or len(t) > 50000 for t in [*texts, *qt]
    ):
        raise ValueError("nonempty reference/query text within budget required")
    y = checked_labels(labels, len(texts))
    order = np.argsort([normalize(t) for t in texts], kind="stable")
    texts, y = texts[order], y[order]
    target_null = np.random.default_rng(seed + 1).permutation(y)
    encoder = vectorizer_for("word")
    try:
        ref = encoder.fit_transform(texts)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
        ref, encoder = None, None
    banks, diagnostics = {m: [] for m in MODES}, []
    for family in ("action", "qualification"):
        selected = [select_windows(t, family) for t in qt]
        groupings = {
            "windows": [v[0] for v in selected],
            "displaced": [v[1] for v in selected],
            "whole_comment": [[t] for t in qt],
        }
        similarity, missing = {}, {}
        for mode, grouped in groupings.items():
            flat = [v for group in grouped for v in group]
            if encoder is None:
                similarity[mode] = np.zeros((len(flat), len(texts)))
                missing[mode] = 1.0
            else:
                query = encoder.transform(flat)
                similarity[mode] = np.clip((query @ ref.T).toarray(), 0, 1)
                missing[mode] = float(np.mean(query.getnnz(axis=1) == 0))
        for mode in MODES:
            source = "windows" if mode == "label_null" else mode
            sim = similarity[source]
            target = target_null if mode == "label_null" else y
            classes = []
            for cls in (0, 1):
                values = sim[:, target == cls]
                near = np.sort(values, axis=1)[:, -min(5, values.shape[1]) :].mean(1)
                classes.append(aggregate_passages(near, [len(v) for v in groupings[source]]))
            banks[mode].extend([*classes, classes[1] - classes[0]])
            diagnostics.append(
                {
                    "mode": mode,
                    "family": family,
                    "reference_rows": len(texts),
                    "vocabulary_columns": 0 if encoder is None else len(encoder.vocabulary_),
                    "mean_windows": float(np.mean([len(v) for v in groupings[source]])),
                    "cue_present_fraction": float(np.mean([v[2]["cue_present"] for v in selected])),
                    "selected_token_fraction": float(
                        np.mean([v[2]["selected_fraction"] for v in selected])
                    ),
                    "zero_lexical_fraction": missing[source],
                    "new_span_labels": 0,
                    "displaced_changed_fraction": float(
                        np.mean([v[2]["control_changed_fraction"] for v in selected])
                    ),
                }
            )
    return {m: np.column_stack(v) for m, v in banks.items()}, diagnostics


def cross_fitted(reference, labels, bodies, rules, queries, query_bodies, *, seed=20260929):
    r, y, texts, q, qt, groups = validate_inputs(
        reference, labels, bodies, rules, queries, query_bodies
    )
    train = {m: np.empty((len(r), 48)) for m in MODES}
    coverage, diagnostics = np.zeros(len(r), dtype=int), []
    for index, (keep, held) in enumerate(GroupKFold(3).split(r, groups=groups)):
        bank, stats = feature_block(r[keep], y[keep], texts[keep], r[held], texts[held], seed=seed)
        for mode in MODES:
            train[mode][held] = bank[mode]
        coverage[held] += 1
        diagnostics.extend({"inner_fold": str(index), "self_overlap": 0, **s} for s in stats)
    if not np.all(coverage == 1):
        raise ValueError("crossfit coverage differs")
    query, stats = feature_block(r, y, texts, q, qt, seed=seed)
    diagnostics.extend({"inner_fold": "outer_query", "self_overlap": 0, **s} for s in stats)
    banks = {m: (train[m], query[m]) for m in MODES}
    if not all(np.isfinite(a).all() for pair in banks.values() for a in pair):
        raise ValueError("nonfinite span evidence")
    return banks, diagnostics
