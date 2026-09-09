"""Small monotone calibration models and paired development-only acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import log_loss

from jigsaw_rules.metrics import evaluate


def probabilities(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Probabilities must be a nonempty finite vector")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must lie in [0, 1]")
    return values


@dataclass(frozen=True)
class MonotoneSigmoid:
    slope: float = 1.0
    intercept: float = 0.0
    clip: float = 1e-7
    fitted: bool = False

    def predict(self, values) -> np.ndarray:
        p = probabilities(values)
        if not self.fitted:
            return p.copy()
        if not np.isfinite([self.slope, self.intercept, self.clip]).all():
            raise ValueError("Calibration parameters must be finite")
        if self.slope <= 0 or not 0 < self.clip < 0.5:
            raise ValueError("Calibration must be monotone with a valid clipping bound")
        return expit(self.slope * logit(p.clip(self.clip, 1 - self.clip)) + self.intercept)

    @classmethod
    def fit(cls, values, target, spec: dict) -> MonotoneSigmoid:
        p, y = probabilities(values), np.asarray(target)
        if y.shape != p.shape or set(y.tolist()) != {0, 1}:
            raise ValueError("Calibration requires aligned labels from both classes")
        slope, intercept = spec["slope_bounds"], spec["intercept_bounds"]
        clip, ridge = spec["clip"], spec["l2"]
        if not 0 < slope[0] < slope[1] or not intercept[0] < intercept[1]:
            raise ValueError("Invalid calibration parameter bounds")
        if not 0 < clip < 0.5 or ridge < 0:
            raise ValueError("Invalid calibration clipping or regularization")
        x = logit(p.clip(clip, 1 - clip))

        def objective(parameters):
            a, b = parameters
            z = a * x + b
            loss = np.mean(np.logaddexp(0, z) - y * z)
            residual = expit(z) - y
            penalty = ridge * ((a - 1) ** 2 + b**2) / 2
            gradient = [np.mean(residual * x) + ridge * (a - 1), residual.mean() + ridge * b]
            return float(loss + penalty), np.asarray(gradient)

        result = minimize(
            objective,
            [1.0, 0.0],
            jac=True,
            method="L-BFGS-B",
            bounds=[slope, intercept],
            options={"maxiter": spec["max_iterations"], "ftol": spec["tolerance"]},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError("Calibration optimizer did not converge: " + str(result.message))
        return cls(float(result.x[0]), float(result.x[1]), clip, True)


def loss_interval(target, raw, calibrated, groups, *, draws, seed, quantiles) -> dict:
    """Positive difference means improved log loss; sample whole normalized bodies."""
    raw, calibrated = probabilities(raw), probabilities(calibrated)
    y, groups = np.asarray(target), np.asarray(groups)
    if raw.shape != calibrated.shape or y.shape != raw.shape or groups.shape != raw.shape:
        raise ValueError("Paired loss arrays must align")
    if set(y.tolist()) != {0, 1} or draws < 100:
        raise ValueError("Paired loss requires both classes and at least 100 draws")
    eps = np.finfo(float).eps
    r, c = raw.clip(eps, 1 - eps), calibrated.clip(eps, 1 - eps)
    improvement = -y * np.log(r) - (1 - y) * np.log1p(-r)
    improvement -= -y * np.log(c) - (1 - y) * np.log1p(-c)
    _, inverse = np.unique(groups, return_inverse=True)
    size = int(inverse.max()) + 1
    sums = np.bincount(inverse, weights=improvement)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sample = rng.integers(size, size=size)
        values.append(float(sums[sample].sum() / counts[sample].sum()))
    low, high = np.quantile(values, quantiles)
    return {
        "log_loss_improvement": float(improvement.mean()),
        "lower": float(low),
        "upper": float(high),
        "draws": draws,
        "seed": seed,
        "quantiles": list(quantiles),
        "scope": "Fixed development predictions, normalized bodies, two route decisions",
    }


def calibration_decision(frame, raw, calibrated, plan: dict) -> dict:
    before = evaluate(frame.rule_violation, raw, frame.rule)
    after = evaluate(frame.rule_violation, calibrated, frame.rule)
    from jigsaw_rules.data import normalize

    interval = loss_interval(
        frame.rule_violation,
        raw,
        calibrated,
        frame.body.map(normalize),
        draws=plan["bootstrap_draws"],
        seed=2025,
        quantiles=plan["interval_quantiles"],
    )
    policy_loss = {
        rule: float(log_loss(group.rule_violation, calibrated[indices], labels=[0, 1]))
        - float(log_loss(group.rule_violation, raw[indices], labels=[0, 1]))
        for rule, indices in frame.reset_index(drop=True).groupby("rule").indices.items()
        for group in [frame.iloc[indices]]
    }
    checks = {
        "practical_log_loss_gain": before["log_loss"] - after["log_loss"]
        >= plan["minimum_log_loss_improvement"],
        "paired_loss_interval": interval["lower"] > 0,
        "brier_tolerance": after["brier"] - before["brier"] <= plan["maximum_brier_increase"],
        "macro_auc_tolerance": before["rule_macro_auc"] - after["rule_macro_auc"]
        <= plan["maximum_macro_auc_drop"],
        "policy_auc_tolerance": all(
            before["per_rule_auc"][rule] - value <= plan["maximum_policy_auc_drop"]
            for rule, value in after["per_rule_auc"].items()
        ),
        "policy_log_loss_tolerance": max(policy_loss.values())
        <= plan["maximum_policy_log_loss_increase"],
    }
    return {
        "retain_calibration": all(checks.values()),
        "checks": {name: bool(value) for name, value in checks.items()},
        "raw": before,
        "calibrated": after,
        "loss_interval": interval,
        "policy_log_loss_changes": policy_loss,
    }
