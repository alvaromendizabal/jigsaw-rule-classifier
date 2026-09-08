"""Paired comment-group bootstrap intervals conditional on the observed rules."""

import numpy as np
from sklearn.metrics import roc_auc_score


def paired_auc_comparisons(y, predictions, rules, groups, comparisons, *, draws=1000, seed=2025):
    """Joint group bootstrap with tie-exact weighted AUC and simultaneous intervals.

    The centered maximum absolute contrast error controls the family of supplied
    contrasts under this empirical bootstrap. It remains conditional on fixed OOF
    predictions and the observed policies; it is not nested selection validation.
    """
    y, rules, groups = map(np.asarray, (y, rules, groups))
    if y.ndim != 1 or rules.shape != y.shape or groups.shape != y.shape or draws < 100:
        raise ValueError("Joint bootstrap needs aligned arrays and at least 100 draws")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Joint bootstrap requires binary labels")
    prepared = {}
    for model, prediction in predictions.items():
        prediction = np.asarray(prediction)
        if prediction.shape != y.shape or not np.isfinite(prediction).all():
            raise ValueError("Joint bootstrap scores must be finite and aligned")
        parts = []
        for rule in sorted(set(rules)):
            indices = np.flatnonzero(rules == rule)
            order = indices[np.argsort(prediction[indices], kind="stable")]
            starts = np.r_[0, 1 + np.flatnonzero(np.diff(prediction[order]))]
            parts.append((order, starts))
        prepared[model] = parts
    if not comparisons or any(
        a not in predictions or b not in predictions for _, a, b in comparisons
    ):
        raise ValueError("Comparisons require named candidate and reference scores")

    def aucs(weights):
        scores = {}
        for model, parts in prepared.items():
            per_rule = []
            for order, starts in parts:
                positive = np.add.reduceat(weights[order] * y[order], starts)
                negative = np.add.reduceat(weights[order] * (1 - y[order]), starts)
                denominator = positive.sum() * negative.sum()
                if denominator == 0:
                    return None
                credit = np.cumsum(negative) - 0.5 * negative
                per_rule.append(float(positive @ credit / denominator))
            scores[model] = float(np.mean(per_rule))
        return np.asarray([scores[a] - scores[b] for _, a, b in comparisons])

    observed = aucs(np.ones(len(y)))
    if observed is None:
        raise ValueError("Every observed rule must have both classes")
    _, inverse = np.unique(groups, return_inverse=True)
    count = int(inverse.max()) + 1
    rng = np.random.default_rng(seed)
    samples = []
    attempts = 0
    while len(samples) < draws and attempts < 10 * draws:
        attempts += 1
        weights = np.bincount(rng.integers(count, size=count), minlength=count)[inverse]
        value = aucs(weights)
        if value is not None:
            samples.append(value)
    if len(samples) != draws:
        raise ValueError("Insufficient valid joint bootstrap draws")
    distribution = np.stack(samples)
    low, high = np.quantile(distribution, [0.025, 0.975], axis=0)
    radius = float(np.quantile(np.abs(distribution - observed).max(axis=1), 0.95))
    return [
        {
            "contrast": name,
            "candidate": a,
            "reference": b,
            "observed_delta": float(observed[i]),
            "ci_lower": float(low[i]),
            "ci_upper": float(high[i]),
            "simultaneous_lower": float(observed[i] - radius),
            "simultaneous_upper": float(observed[i] + radius),
            "draws": draws,
            "seed": seed,
            "comparisons": len(comparisons),
            "scope": "Fixed OOF predictions, normalized comment groups, observed rules only",
        }
        for i, (name, a, b) in enumerate(comparisons)
    ]


def paired_auc_interval(y, baseline, candidate, rules, groups, *, draws=500, seed=2025):
    y, baseline, candidate, rules, groups = map(np.asarray, (y, baseline, candidate, rules, groups))
    if y.ndim != 1 or any(x.shape != y.shape for x in [baseline, candidate, rules, groups]):
        raise ValueError("Bootstrap inputs must be aligned 1-D arrays")
    if (
        not len(y)
        or draws < 10
        or not np.isfinite(baseline).all()
        or not np.isfinite(candidate).all()
    ):
        raise ValueError("Bootstrap requires finite scores and at least ten draws")
    labels = sorted(set(rules))
    _, inverse = np.unique(groups, return_inverse=True)
    count = inverse.max() + 1
    rng = np.random.default_rng(seed)

    def difference(weights):
        differences = []
        for rule in labels:
            mask = (rules == rule) & (weights > 0)
            if len(set(y[mask])) != 2:
                return None
            differences.append(
                roc_auc_score(y[mask], candidate[mask], sample_weight=weights[mask])
                - roc_auc_score(y[mask], baseline[mask], sample_weight=weights[mask])
            )
        return float(np.mean(differences))

    observed = difference(np.ones(len(y)))
    if observed is None:
        raise ValueError("Every rule must contain both classes")
    values = []
    attempts = 0
    while len(values) < draws and attempts < draws * 10:
        attempts += 1
        weights = np.bincount(rng.integers(count, size=count), minlength=count)[inverse]
        value = difference(weights)
        if value is not None:
            values.append(value)
    if len(values) < draws:
        raise ValueError("Too few mixed-class group bootstrap samples")
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "observed_delta": observed,
        "ci_lower": float(low),
        "ci_upper": float(high),
        "confidence_level": 0.95,
        "draws": draws,
        "attempts": attempts,
        "seed": seed,
        "resampling_unit": "normalized comment body, shared across rules",
        "scope": (
            "Conditional on these rules and fixed OOF predictions; "
            "not uncertainty across unseen policies."
        ),
    }
