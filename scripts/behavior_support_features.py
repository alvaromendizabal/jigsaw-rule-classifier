"""Behavior-conditioned, cross-fitted semantic evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold

from jigsaw_rules.data import normalize
from scripts.local_support_features import checked_labels, unit
from scripts.relational_features import relational_features

ROLE_COLUMNS = (
    "act_roles/self_request_legal_present",
    "act_roles/other_request_legal_present",
    "act_roles/directive_legal_present",
    "act_roles/offer_legal_present",
    "act_roles/experience_legal_present",
    "act_roles/reported_legal_present",
)
CONTEXT_COLUMNS = (
    "action_scope/quoted_directive_legal_present",
    "action_scope/directive_legal_negated_present",
    "action_scope/disclaimer_then_directive_legal_present",
    "link_intent/cta_url_present",
    "link_intent/owned_business_url_present",
    "link_intent/resource_url_present",
)
COLUMNS = ROLE_COLUMNS + CONTEXT_COLUMNS
STATISTICS = ("affinity_delta", "similarity_delta", "effective_delta", "minimum_coverage")
NAMES = tuple(f"conditional/{col}/{stat}" for col in COLUMNS for stat in STATISTICS)
MODES = ("conditioned", "permuted", "uniform", "descriptor_only")


def descriptor_matrix(bodies):
    texts = list(bodies)
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("nonempty observable body strings required")
    table = relational_features(pd.DataFrame({"body": texts}))
    values = table.loc[:, COLUMNS].to_numpy(float)
    if not np.isin(values, [0, 1]).all():
        raise ValueError("behavior descriptor contract changed")
    return values.astype(bool)


def _summary(log_kernel, similarity, labels, gate):
    log_gate = np.log(gate)
    density, means, effective, coverage = [], [], [], []
    for cls in (0, 1):
        keep = labels == cls
        lg = log_gate[:, keep]
        joint = log_kernel[:, keep] + lg
        total = logsumexp(joint, axis=1)
        density.append(total - logsumexp(lg, axis=1))
        weights = np.exp(joint - total[:, None])
        means.append((weights * similarity[:, keep]).sum(1))
        effective.append(2 * total - logsumexp(2 * joint, axis=1) - np.log(keep.sum()))
        coverage.append(gate[:, keep].mean(1))
    return np.column_stack(
        [
            density[1] - density[0],
            means[1] - means[0],
            effective[1] - effective[0],
            np.minimum(coverage[0], coverage[1]),
        ]
    )


def conditional_geometry(
    reference,
    labels,
    reference_descriptors,
    queries,
    query_descriptors,
    *,
    seed=20260921,
    temperature=0.1,
    backoff=0.2,
):
    r, q = unit(reference), unit(queries)
    y = checked_labels(labels, len(r))
    rd, qd = np.asarray(reference_descriptors), np.asarray(query_descriptors)
    if r.shape[1] != q.shape[1]:
        raise ValueError("reference/query dimensions differ")
    if rd.shape != (len(r), len(COLUMNS)) or qd.shape != (len(q), len(COLUMNS)):
        raise ValueError("descriptor dimensions differ")
    if not np.isin(rd, [0, 1]).all() or not np.isin(qd, [0, 1]).all():
        raise ValueError("binary behavior descriptors required")
    if not np.isfinite(temperature) or temperature <= 0 or not 0 < backoff < 1:
        raise ValueError("invalid temperature or backoff")
    similarity = np.clip(q @ r.T, -1, 1)
    log_kernel = (similarity - similarity.max(1, keepdims=True)) / temperature
    ones = np.ones_like(similarity)
    shuffled = rd[np.random.default_rng(seed).permutation(len(rd))]
    banks, diagnostics = {}, []
    for mode in MODES:
        out = np.empty((len(q), len(NAMES)))
        desc = shuffled if mode == "permuted" else rd
        lk = np.zeros_like(log_kernel) if mode == "uniform" else log_kernel
        sim = np.zeros_like(similarity) if mode == "uniform" else similarity
        baseline = _summary(lk, sim, y, ones)
        for j, col in enumerate(COLUMNS):
            matching = qd[:, j, None] == desc[None, :, j]
            gate = backoff + (1 - backoff) * matching
            values = _summary(lk, sim, y, gate)
            values[:, :3] -= baseline[:, :3]
            # Identical allocation; descriptor-only arm has no label-derived evidence.
            if mode == "descriptor_only":
                values[:] = 0
                values[:, 0] = qd[:, j]
            out[:, j * 4 : (j + 1) * 4] = values
            if mode == "conditioned":
                strict = np.column_stack([matching[:, y == cls].sum(1) for cls in (0, 1)])
                diagnostics.append(
                    {
                        "descriptor": col,
                        "references": len(r),
                        "query_rows": len(q),
                        "reference_prevalence": float(rd[:, j].mean()),
                        "query_prevalence": float(qd[:, j].mean()),
                        "zero_stratum_fraction": float((strict.min(1) == 0).mean()),
                        "mean_minimum_coverage": float(values[:, 3].mean()),
                        "backoff": backoff,
                    }
                )
        if not np.isfinite(out).all():
            raise ValueError("nonfinite conditional evidence")
        banks[mode] = out
    return banks, diagnostics


def cross_fitted(reference, labels, bodies, rules, queries, query_bodies, *, seed=20260921):
    r, q = unit(reference), unit(queries)
    y = checked_labels(labels, len(r))
    texts, query_texts, names = list(bodies), list(query_bodies), list(rules)
    if len(texts) != len(r) or len(names) != len(r) or len(query_texts) != len(q):
        raise ValueError("reference metadata alignment differs")
    if any(not isinstance(t, str) or not t.strip() for t in texts + names + query_texts):
        raise ValueError("nonempty text/rule required")
    if len({normalize(v) for v in names}) != 1:
        raise ValueError("same rule references required")
    groups = np.asarray([normalize(t) for t in texts])
    if set(groups) & {normalize(t) for t in query_texts}:
        raise ValueError("query text present in references")
    if len(np.unique(groups)) != len(groups):
        raise ValueError("unique reference text required")
    descriptors = descriptor_matrix(texts)
    query_descriptors = descriptor_matrix(query_texts)
    output = {mode: np.empty((len(r), len(NAMES))) for mode in MODES}
    coverage = np.zeros(len(r), dtype=int)
    metadata = []
    for index, (keep, held) in enumerate(GroupKFold(3).split(r, groups=groups)):
        if set(groups[keep]) & set(groups[held]):
            raise AssertionError("self group entered reference pool")
        keep = keep[np.argsort(groups[keep], kind="stable")]
        bank, stats = conditional_geometry(
            r[keep], y[keep], descriptors[keep], r[held], descriptors[held], seed=seed
        )
        for mode in MODES:
            output[mode][held] = bank[mode]
        coverage[held] += 1
        metadata.extend({"inner_fold": str(index), "self_text_overlap": 0, **s} for s in stats)
    if not np.all(coverage == 1):
        raise AssertionError("crossfit coverage differs")
    order = np.argsort(groups, kind="stable")
    bank, stats = conditional_geometry(
        r[order], y[order], descriptors[order], q, query_descriptors, seed=seed
    )
    metadata.extend({"inner_fold": "outer_query", "self_text_overlap": 0, **s} for s in stats)
    return {mode: (output[mode], bank[mode]) for mode in MODES}, metadata
