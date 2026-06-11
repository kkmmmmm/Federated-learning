"""Evaluation metrics (AUROC + calibration) and bootstrap confidence intervals.

The calibration slope/intercept are computed exactly as in the published
analysis: a 10-bin reliability curve (sklearn ``calibration_curve``) followed by
an ordinary least-squares line fit of observed-vs-predicted, so the re-analysed
figures remain directly comparable to the originals.
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


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray, bins: int = C.CALIB_BINS):
    """Return (slope, intercept) of the OLS fit to the 10-bin reliability curve.

    Fast vectorised reproduction of ``sklearn.calibration.calibration_curve``
    with the default uniform-width strategy (equal-width bins over [0, 1], using
    only non-empty bins), then ``np.polyfit`` of observed-vs-predicted.  Matches
    the published calibration definition while avoiding sklearn's per-call
    overhead in the 1000x bootstrap loop.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    binids = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    bin_total = np.bincount(binids, minlength=bins).astype(float)
    bin_true = np.bincount(binids, weights=y.astype(float), minlength=bins)
    bin_pred = np.bincount(binids, weights=p, minlength=bins)
    nz = bin_total > 0
    if nz.sum() < 2:
        return np.nan, np.nan
    prob_true = bin_true[nz] / bin_total[nz]
    prob_pred = bin_pred[nz] / bin_total[nz]
    slope, intercept = np.polyfit(prob_pred, prob_true, 1)
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
