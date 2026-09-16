"""Shared plus policy-specific features, with a norm-matched shared-copy control."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from jigsaw_rules.data import normalize


def normalized_rules(values):
    rules = list(values)
    if not rules or any(not isinstance(v, str) or not v.strip() for v in rules):
        raise ValueError("nonempty rule strings required")
    return np.asarray([normalize(v) for v in rules], dtype=object)


class PolicyMap:
    """Learn only policy names from eligible training; no query labels or fitted targets."""

    def fit(self, rules):
        self.rules_ = tuple(sorted(set(normalized_rules(rules))))
        return self

    def augment(self, matrix, rules, *, mode):
        if not hasattr(self, "rules_"):
            raise ValueError("fit the policy map on training rules first")
        if mode not in {"condition", "copy"}:
            raise ValueError("unknown augmentation mode")
        x = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
        x.sum_duplicates()
        x.eliminate_zeros()
        names = normalized_rules(rules)
        if x.shape[0] != len(names) or not np.isfinite(x.data).all():
            raise ValueError("finite, aligned feature matrix required")
        lookup = {value: index for index, value in enumerate(self.rules_)}
        codes = np.asarray([lookup.get(v, -1) for v in names], dtype=int)
        known = codes >= 0
        coo = x.tocoo()
        eligible = known[coo.row]
        blocks = codes[coo.row[eligible]] if mode == "condition" else 0
        cols = coo.col[eligible] + blocks * x.shape[1]
        out = sparse.csr_matrix(
            (coo.data[eligible], (coo.row[eligible], cols)),
            shape=(x.shape[0], x.shape[1] * len(self.rules_)),
        )
        # Both arms have equal dimensions and row norms. Unknown policy => zero extra block.
        left = np.asarray(out.multiply(out).sum(axis=1)).ravel()
        right = np.asarray(x.multiply(x).sum(axis=1)).ravel() * known
        if not np.allclose(left, right, atol=1e-12, rtol=1e-12):
            raise AssertionError("augmentation row-norm invariant failed")
        return out, {
            "known_rows": int(known.sum()),
            "unknown_rows": int((~known).sum()),
            "columns": int(out.shape[1]),
            "nonzeros": int(out.nnz),
            "active_columns": int(np.unique(out.indices).size),
        }


def matched_blocks(matrix, rules, mapper):
    """Return two different representations, not two algorithms or tuned C values."""
    conditioned, a = mapper.augment(matrix, rules, mode="condition")
    copied, b = mapper.augment(matrix, rules, mode="copy")
    if conditioned.shape != copied.shape or conditioned.nnz != copied.nnz:
        raise AssertionError("copy/condition dimension or sparsity mismatch")
    return {"condition": conditioned, "copy": copied}, {"condition": a, "copy": b}
