"""Training-only, rule-specific lexical evidence with equal-row-norm controls."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from scripts.policy_features import normalized_rules


def checked_matrix(matrix):
    x = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    x.sum_duplicates()
    x.eliminate_zeros()
    if x.ndim != 2 or not x.shape[1] or not np.isfinite(x.data).all():
        raise ValueError("finite nonempty two-dimensional features required")
    if (x.data < 0).any():
        raise ValueError("nonnegative lexical features required")
    return x


def row_norms(x):
    return np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())


def reweight_preserving_norm(matrix, weight):
    """Redistribute feature mass without changing a row's family-specific norm."""
    x = checked_matrix(matrix)
    w = np.asarray(weight, dtype=float)
    if w.shape != (x.shape[1],) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("positive finite aligned weights required")
    out = x.multiply(w).tocsr()
    before, after = row_norms(x), row_norms(out)
    scale = np.divide(before, after, out=np.ones_like(before), where=after > 0)
    out = out.multiply(scale[:, None]).tocsr()
    if not np.allclose(row_norms(out), before, rtol=1e-12, atol=1e-12):
        raise ValueError("reweighting changed row norms")
    if out.nnz != x.nnz:
        raise ValueError("positive weighting changed sparsity")
    return out


def log_ratio(x, y, sample_weight, alpha):
    """Smoothed class-conditional document incidence, not document lengths."""
    binary = x.copy()
    binary.data[:] = 1.0
    counts = []
    for label in (0, 1):
        mask = y == label
        count = np.asarray(binary[mask].T @ sample_weight[mask]).ravel()
        count = count + alpha
        counts.append(count / count.sum())
    return np.log(counts[1]) - np.log(counts[0])


class RuleEvidence:
    """Supervised feature transform; query targets are never an argument."""

    def __init__(self, *, alpha=1.0, shrinkage=64.0, cap=3.0, seed=20260911):
        self.alpha = float(alpha)
        self.shrinkage = float(shrinkage)
        self.cap = float(cap)
        self.seed = int(seed)
        parameters = (self.alpha, self.shrinkage, self.cap)
        if not all(np.isfinite(v) and v > 0 for v in parameters):
            raise ValueError("smoothing, shrinkage and cap must be positive")

    def fit(self, matrix, targets, rules, sample_weight=None):
        x = checked_matrix(matrix)
        y = np.asarray(targets)
        names = normalized_rules(rules)
        weight = (
            np.ones(len(y), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        if y.shape != (x.shape[0],) or len(names) != len(y):
            raise ValueError("training rows, labels and rules must align")
        if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("both binary training classes required")
        if weight.shape != y.shape or not np.isfinite(weight).all() or (weight <= 0).any():
            raise ValueError("aligned positive training weights required")
        self.columns_ = x.shape[1]
        self.global_ = log_ratio(x, y, weight, self.alpha)
        self.local_, self.shrunk_, self.reliability_ = {}, {}, {}
        self.training_counts_ = []
        for rule in sorted(set(names)):
            mask = names == rule
            n0, n1 = (int(np.sum(y[mask] == c)) for c in (0, 1))
            # Reliability counts unique deduplicated pairs, not oversampled occurrences.
            smallest = min(n0, n1)
            trust = smallest / (smallest + self.shrinkage)
            if smallest:
                local = log_ratio(x[mask], y[mask], weight[mask], self.alpha)
            else:
                local = self.global_.copy()
            self.local_[rule] = local
            self.shrunk_[rule] = trust * local + (1 - trust) * self.global_
            self.reliability_[rule] = trust
            self.training_counts_.append(
                {
                    "policy": rule,
                    "permitted_pairs": n0,
                    "violating_pairs": n1,
                    "local_evidence_fraction": trust,
                    "fallback_global": not bool(smallest),
                }
            )
        return self

    def weights(self, rule, mode):
        if not hasattr(self, "columns_"):
            raise ValueError("fit the evidence transform first")
        if mode not in {"global", "rule", "permuted", "unshrunk"}:
            raise ValueError("unknown evidence weighting mode")
        name = normalized_rules([rule])[0]
        if mode == "global":
            ratio = self.global_
        elif mode == "unshrunk":
            ratio = self.local_.get(name, self.global_)
        else:
            ratio = self.shrunk_.get(name, self.global_)
        # Sign alone is reparameterizable by an unconstrained linear classifier.
        # Test magnitude-based relative importance, not an illusory sign advantage.
        weights = 1.0 + np.minimum(np.abs(ratio), self.cap)
        if mode == "permuted":
            weights = weights[np.random.default_rng(self.seed).permutation(len(weights))]
        return weights.copy()

    def transform(self, matrix, rules, *, mode):
        if not hasattr(self, "columns_"):
            raise ValueError("fit the evidence transform first")
        x = checked_matrix(matrix)
        names = normalized_rules(rules)
        if x.shape != (len(names), self.columns_):
            raise ValueError("query feature dimensions or rows differ")
        out = x.copy()
        # Assignment only touches existing nonzeros; it cannot create vocabulary.
        for rule in sorted(set(names)):
            rows = names == rule
            positions = np.repeat(rows, np.diff(x.indptr))
            w = self.weights(rule, mode)
            out.data[positions] *= w[x.indices[positions]]
        before, after = row_norms(x), row_norms(out)
        scale = np.divide(before, after, out=np.ones_like(before), where=after > 0)
        out = out.multiply(scale[:, None]).tocsr()
        error = float(np.max(np.abs(row_norms(out) - before), initial=0.0))
        if error > 1e-10 or out.nnz != x.nnz:
            raise ValueError("norm or sparsity preservation failed")
        return out, {
            "row_norm_max_abs_error": error,
            "columns": int(x.shape[1]),
            "nonzeros": int(x.nnz),
            "unknown_rule_rows": int(sum(r not in self.local_ for r in names)),
        }

    def profile(self, *, family, fold):
        rows = []
        for count in self.training_counts_:
            rule = count["policy"]
            for mode in ("global", "rule", "unshrunk"):
                w = self.weights(rule, mode)
                q = np.quantile(w, [0.0, 0.25, 0.5, 0.75, 1.0])
                rows.append(
                    {
                        **count,
                        "family": family,
                        "fold": fold,
                        "mode": mode,
                        "columns": self.columns_,
                        "minimum": float(q[0]),
                        "q25": float(q[1]),
                        "median": float(q[2]),
                        "q75": float(q[3]),
                        "maximum": float(q[4]),
                        "capped_columns": int(np.sum(w >= 1 + self.cap)),
                    }
                )
        return rows
