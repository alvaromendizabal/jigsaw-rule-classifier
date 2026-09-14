"""Reference-only covariance correction on frozen vectors."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.utils.extmath import randomized_svd

from jigsaw_rules.data import normalize
from scripts.local_support_features import BASIC, checked_labels, unit

MODES = ("centered", "deflated", "whitened", "random_dual", "label_null_dual")
FAMILIES = ("deflation", "whitening")
NAMES = tuple(f"{family}/{name}" for family in FAMILIES for name in BASIC)


def basic_summary(reference, labels, queries):
    r, q = unit(reference), unit(queries)
    y = checked_labels(labels, len(r))
    if r.shape[1] != q.shape[1]:
        raise ValueError("reference/query dimensions differ")
    cos = np.clip(q @ r.T, -1.0, 1.0)
    values = []
    for cls in (0, 1):
        keep = y == cls
        center = r[keep].mean(0)
        norm = np.linalg.norm(center)
        if norm <= 1e-12:
            raise ValueError("degenerate conditioned centroid")
        part = cos[:, keep]
        top = min(5, part.shape[1])
        values += [q @ (center / norm), part.max(1)]
        values += [np.partition(part, -top, axis=1)[:, -top:].mean(1)]
    out = np.column_stack(values)
    return np.column_stack([out, out[:, 3:] - out[:, :3]])


class ReferenceTransform:
    (
        "Fit unlabeled reference vectors only; never include held-out "
        # Keep narrative chunks separate for formatting.
        "vectors."
    )

    def fit(self, reference, *, seed=20260920, rank=32, remove=4, shrinkage=0.2):
        r = unit(reference)
        if len(r) < 6 or r.shape[1] < 3:
            raise ValueError("at least six references and three dimensions required")
        if not isinstance(rank, int) or rank < 1 or not isinstance(remove, int) or remove < 1:
            raise ValueError("positive integer spectral budgets required")
        if not 0 < shrinkage < 1:
            raise ValueError("shrinkage must be between zero and one")
        self.mean_ = r.mean(0)
        centered = r - self.mean_
        self.rank_ = min(rank, len(r) - 2, r.shape[1] - 1)
        self.remove_ = min(remove, self.rank_)
        _, singular, vt = randomized_svd(
            centered,
            n_components=self.rank_,
            n_iter=3,
            random_state=seed,
            flip_sign=True,
        )
        self.basis_ = vt.T
        self.eigen_ = singular**2 / (len(r) - 1)
        total = float(np.square(centered).sum() / (len(r) - 1))
        if not np.isfinite(total) or total <= 1e-12:
            raise ValueError("degenerate reference covariance")
        self.floor_ = max(shrinkage * total / r.shape[1], 1e-6)
        # Full-space regularized low-rank whitening; residual space is retained.
        self.gain_ = np.sqrt(self.floor_ / (self.floor_ + (1 - shrinkage) * self.eigen_))
        rng = np.random.default_rng(seed + 1)
        self.random_basis_, _ = np.linalg.qr(rng.normal(size=self.basis_.shape))
        self.stats_ = {
            "references": len(r),
            "dimensions": r.shape[1],
            "spectral_rank": self.rank_,
            "removed_directions": self.remove_,
            "shrinkage": shrinkage,
            "mean_vector_norm": float(np.linalg.norm(self.mean_)),
            "captured_variance_fraction": float(self.eigen_.sum() / total),
            "top_removed_variance_fraction": float(self.eigen_[: self.remove_].sum() / total),
            "minimum_relative_whitening_gain": float(self.gain_.min()),
        }
        return self

    def transform(self, values, mode):
        if not hasattr(self, "basis_"):
            raise ValueError("fit reference transform first")
        x = unit(values)
        if x.shape[1] != len(self.mean_):
            raise ValueError("reference/query dimensions differ")
        x = x - self.mean_
        if mode == "centered":
            return unit(x)
        basis = self.random_basis_ if mode.startswith("random_") else self.basis_
        if mode in ("deflated", "random_deflated"):
            u = basis[:, : self.remove_]
            x = x - (x @ u) @ u.T
        elif mode in ("whitened", "random_whitened"):
            x = x + ((x @ basis) * (self.gain_ - 1.0)) @ basis.T
        else:
            raise ValueError("unknown reference transform")
        return unit(x)


def mean_off_diagonal_cosine(values):
    x = unit(values)
    return float((np.square(x.sum(0)).sum() - len(x)) / (len(x) * (len(x) - 1)))


def feature_block(reference, labels, bodies, queries, *, seed=20260920):
    r, q = unit(reference), unit(queries)
    y = checked_labels(labels, len(r))
    texts = list(bodies)
    if len(texts) != len(r) or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("aligned nonempty reference text required")
    groups = np.asarray([normalize(t) for t in texts])
    if len(np.unique(groups)) != len(groups):
        raise ValueError("reference pool must use unique normalized texts")
    order = np.argsort(groups, kind="stable")
    r, y = r[order], y[order]
    transform = ReferenceTransform().fit(r, seed=seed)
    out, stats = {}, []
    labels_null = np.random.default_rng(seed + 2).permutation(y)
    for mode in ("centered", "deflated", "whitened", "random_deflated", "random_whitened"):
        rt, qt = transform.transform(r, mode), transform.transform(q, mode)
        out[mode] = basic_summary(rt, y, qt)
        if mode in ("deflated", "whitened"):
            out["null_" + mode] = basic_summary(rt, labels_null, qt)
        stats.append(
            {
                **transform.stats_,
                "mode": mode,
                "raw_reference_mean_cosine": mean_off_diagonal_cosine(r),
                "transformed_reference_mean_cosine": mean_off_diagonal_cosine(rt),
                "query_rows": len(q),
                "query_used_in_transform_fit": False,
            }
        )
    banks = {mode: out[mode] for mode in ("centered", "deflated", "whitened")}
    banks["dual"] = np.column_stack([out["deflated"], out["whitened"]])
    banks["random_dual"] = np.column_stack([out["random_deflated"], out["random_whitened"]])
    banks["label_null_dual"] = np.column_stack([out["null_deflated"], out["null_whitened"]])
    return banks, stats


def cross_fitted(reference, labels, bodies, rules, queries, query_bodies, *, seed=20260920):
    r, q = unit(reference), unit(queries)
    y = checked_labels(labels, len(r))
    texts, names = list(bodies), list(rules)
    if len(texts) != len(r) or len(names) != len(r):
        raise ValueError("reference metadata alignment differs")
    if any(not isinstance(t, str) or not t.strip() for t in texts + names + list(query_bodies)):
        raise ValueError("nonempty text/rule required")
    if len({normalize(v) for v in names}) != 1:
        raise ValueError("same rule references required")
    groups = np.asarray([normalize(t) for t in texts])
    if set(groups) & {normalize(t) for t in query_bodies}:
        raise ValueError("query text present in references")
    if len(query_bodies) != len(q) or len(np.unique(groups)) != len(groups):
        raise ValueError("unique aligned reference text required")
    modes = ("centered", "deflated", "whitened", "dual", "random_dual", "label_null_dual")
    tx = {m: np.empty((len(r), 18 if "dual" in m else 9)) for m in modes}
    coverage = np.zeros(len(r), dtype=int)
    metadata = []
    for index, (keep, held) in enumerate(GroupKFold(3).split(r, groups=groups)):
        if set(groups[keep]) & set(groups[held]):
            raise AssertionError("self group entered reference transform")
        block, stats = feature_block(r[keep], y[keep], np.asarray(texts)[keep], r[held], seed=seed)
        for mode in modes:
            tx[mode][held] = block[mode]
        coverage[held] += 1
        metadata.extend({"inner_fold": str(index), "self_text_overlap": 0, **s} for s in stats)
    if not np.all(coverage == 1):
        raise AssertionError("crossfit coverage differs")
    vx, stats = feature_block(r, y, texts, q, seed=seed)
    metadata.extend({"inner_fold": "outer_query", "self_text_overlap": 0, **s} for s in stats)
    banks = {m: (tx[m], vx[m]) for m in modes}
    if not all(np.isfinite(a).all() for pair in banks.values() for a in pair):
        raise ValueError("nonfinite conditioned feature bank")
    return banks, metadata
