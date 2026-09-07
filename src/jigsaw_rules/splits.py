"""Group comments, hold out rules, and purge validation bodies from training context."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from jigsaw_rules.data import EXAMPLES, normalize


def purged_split(df: pd.DataFrame, train_idx, valid_idx) -> tuple[np.ndarray, np.ndarray, int]:
    valid_idx = np.asarray(valid_idx)
    forbidden = set(df.iloc[valid_idx].body.map(normalize))
    candidate = df.iloc[train_idx]
    overlap = (
        candidate[["body", *EXAMPLES]]
        .apply(lambda c: c.map(normalize).isin(forbidden))
        .any(axis=1)
        .to_numpy()
    )
    keep = np.asarray(train_idx)[~overlap]
    if len(keep) == 0 or df.iloc[keep].rule_violation.nunique() < 2:
        raise ValueError("Purging left insufficient training classes; inspect the data audit")
    for _, group in df.iloc[valid_idx].groupby("rule"):
        if group.rule_violation.nunique() < 2:
            raise ValueError("Validation rule has one class; reduce folds or revise split")
    return keep, valid_idx, int(overlap.sum())


def make_splits(df: pd.DataFrame, protocol: str, folds: int, seed: int):
    if protocol == "heldout_rule":
        rules = sorted(df.rule.unique())
        if len(rules) < 2:
            raise ValueError("At least two rules required for held-out-rule validation")
        pairs = [(np.flatnonzero(df.rule != r), np.flatnonzero(df.rule == r)) for r in rules]
    elif protocol == "seen_rule":
        if folds < 2:
            raise ValueError("At least two folds required")
        strata = df.rule.astype(str) + "::" + df.rule_violation.astype(str)
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        pairs = list(splitter.split(df, strata, groups=df.body.map(normalize)))
    else:
        raise ValueError(f"Unknown validation protocol {protocol}")
    return [purged_split(df, train_idx, valid_idx) for train_idx, valid_idx in pairs]
