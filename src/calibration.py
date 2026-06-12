"""Evaluation metrics (AUROC + calibration) and bootstrap confidence intervals.

The calibration slope and intercept use the standard regression-based
definitions (Cox; Van Calster et al., *J Clin Epidemiol* 2016; TRIPOD), computed
on the logit scale from the individual predictions rather than from a binned
reliability curve:

* **Calibration slope** – the coefficient ``b`` of a logistic regression of the
  observed outcome on the linear predictor ``LP = logit(p)``:
  ``logit(P(y=1)) = a + b * LP``.  Perfect calibration => ``b = 1``.
* **Calibration intercept (calibration-in-the-large)** – the intercept ``a`` of a
  logistic regression with the linear predictor entered as an *offset* (slope
  fixed at 1): ``logit(P(y=1)) = a + LP``.  Perfect calibration => ``a = 0``.

Both models are fit by Newton-Raphson / IRLS (a few iterations, vectorised) so
the 1000x bootstrap stays cheap.  A resample whose fit cannot be solved
(e.g. all-same outcome, perfect separation) returns NaN and is dropped from the
percentile CI, exactly as before.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from . import config as C


def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic (ties handled by average ranks).

    Numerically identical to ``sklearn.metrics.roc_auc_score`` but far cheaper
    inside the bootstrap loop.
    """
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _calibration_slope(y: np.ndarray, lp: np.ndarray,
                       max_iter: int = 25, tol: float = 1e-6):
    """Slope b of logit(P(y=1)) = a + b*LP, by IRLS. Returns NaN on failure."""
    # design matrix [1, LP]; warm-start at the perfectly-calibrated solution.
    Xd = np.column_stack([np.ones_like(lp), lp])
    beta = np.array([0.0, 1.0])
    for _ in range(max_iter):
        eta = np.clip(Xd @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-10, None)
        H = (Xd.T * w) @ Xd
        g = Xd.T @ (y - mu)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return np.nan
        beta = beta + delta
        if not np.all(np.isfinite(beta)):
            return np.nan
        if np.max(np.abs(delta)) < tol:
            break
    return float(beta[1])


def _calibration_in_the_large(y: np.ndarray, lp: np.ndarray,
                              max_iter: int = 25, tol: float = 1e-6):
    """Intercept a of logit(P(y=1)) = a + LP (LP as offset), by IRLS."""
    a = 0.0
    for _ in range(max_iter):
        eta = np.clip(a + lp, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = float(np.clip(mu * (1.0 - mu), 1e-10, None).sum())
        g = float((y - mu).sum())
        if w <= 0:
            return np.nan
        delta = g / w
        a += delta
        if not np.isfinite(a):
            return np.nan
        if abs(delta) < tol:
            break
    return float(a)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray, bins: int = C.CALIB_BINS):
    """Return the standard (slope, calibration-in-the-large intercept).

    ``bins`` is accepted for backward compatibility but unused: the slope and
    intercept are the regression-based estimands described in the module
    docstring, fit on the logit of the predicted probabilities.
    """
    y = np.asarray(y, dtype=float)
    lp = _logit(np.asarray(p, dtype=float))
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.nan
    slope = _calibration_slope(y, lp)
    intercept = _calibration_in_the_large(y, lp)
    return float(slope), float(intercept)


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """AUROC + calibration slope/intercept point estimates."""
    auroc = float(fast_auc(y, p))
    slope, intercept = calibration_slope_intercept(y, p)
    return {"AUROC": auroc, "CalibrationSlope": slope, "CalibrationIntercept": intercept}


def bootstrap_metrics(y: np.ndarray, p: np.ndarray,
                      n_boot: int = C.N_BOOTSTRAP,
                      seed: int = C.RANDOM_STATE) -> dict:
    """Point estimates and 95% bootstrap CIs for AUROC, slope and intercept.

    Returns a flat dict with ``<metric>``, ``<metric>_lower`` and
    ``<metric>_upper`` keys (percentile bootstrap, 1000 resamples by default).
    """
    y = np.asarray(y)
    p = np.asarray(p)
    out = point_metrics(y, p)

    rng = np.random.default_rng(seed)
    n = len(y)
    aucs, slopes, intercepts = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pb = y[idx], p[idx]
        a = fast_auc(yb, pb)
        if np.isnan(a):
            continue
        aucs.append(a)
        s, i = calibration_slope_intercept(yb, pb)
        if not (np.isnan(s) or np.isnan(i)):
            slopes.append(s)
            intercepts.append(i)

    def ci(vals):
        if not vals:
            return (np.nan, np.nan)
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    out["AUROC_lower"], out["AUROC_upper"] = ci(aucs)
    out["CalibrationSlope_lower"], out["CalibrationSlope_upper"] = ci(slopes)
    out["CalibrationIntercept_lower"], out["CalibrationIntercept_upper"] = ci(intercepts)
    return out
