"""Centralized / local logistic-regression estimators and hyper-parameter search.

All hyper-parameter selection is done with cross-validation *inside the training
data only* (no test fold ever participates), which fixes the hyper-parameter
selection leak present in the original code.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config as C


@dataclass
class FittedModel:
    """A trained logistic model expressed on the *standardized* feature scale."""

    coef: np.ndarray        # shape (n_features,)
    intercept: float
    penalty: str
    C: float | None
    l1_ratio: float | None

    def decision(self, Xs: np.ndarray) -> np.ndarray:
        return Xs @ self.coef + self.intercept

    def predict_proba(self, Xs: np.ndarray) -> np.ndarray:
        return expit(self.decision(Xs))


def _base_estimator(penalty: str) -> LogisticRegression:
    if penalty == "none":
        return LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
    if penalty == "l1":
        return LogisticRegression(penalty="l1", solver="liblinear", max_iter=10000)
    if penalty == "l2":
        return LogisticRegression(penalty="l2", solver="lbfgs", max_iter=10000)
    if penalty == "elasticnet":
        # saga on standardized features converges well within a few thousand
        # iterations; the cap + tol keep the grid search tractable.
        return LogisticRegression(penalty="elasticnet", solver="saga",
                                  max_iter=3000, tol=1e-3)
    raise ValueError(penalty)


def fit_logreg(Xs: np.ndarray, y: np.ndarray, penalty: str,
               random_state: int = C.RANDOM_STATE) -> FittedModel:
    """Fit a logistic model on standardized features, selecting C (and l1_ratio)
    by stratified CV maximising AUROC.  ``Xs`` must already be standardized.
    """
    est = _base_estimator(penalty)
    if penalty == "none":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(Xs, y)
        return FittedModel(est.coef_[0].copy(), float(est.intercept_[0]),
                           penalty, None, None)

    grid = {"C": C.C_GRID}
    if penalty == "elasticnet":
        grid["l1_ratio"] = C.L1_RATIO_GRID
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs = GridSearchCV(est, grid, scoring=C.CV_SCORING, cv=cv, n_jobs=-1)
        gs.fit(Xs, y)

    # Deterministic resolution of the flat AUROC curve.  AUROC is rank-based and
    # nearly insensitive to the shrinkage strength, so its CV curve is flat across
    # a wide range of C; the plain argmax then picks an essentially arbitrary,
    # seed-dependent value, and a region that happens to land on a barely-penalised
    # fit shows inflated coefficients that distort the coefficient PCA.  Among all
    # hyper-parameters statistically tied for the best AUROC (within one standard
    # error of the maximum), we keep the *least*-regularised one (largest C, i.e.
    # closest to the maximum-likelihood fit).  Because penalised and unpenalised
    # models are indistinguishable in AUROC here, this retains the MLE solution and
    # puts every model on a common, unshrunk footing — so the coefficient PCA
    # reflects genuine regional differences rather than incidental shrinkage, and
    # the federated and centralized models stay on the same scale.
    best_params = _auroc_optimal_params(gs.cv_results_, penalty)
    final = _base_estimator(penalty)
    final.set_params(**best_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(Xs, y)
    return FittedModel(
        final.coef_[0].copy(), float(final.intercept_[0]), penalty,
        float(best_params["C"]),
        float(best_params.get("l1_ratio")) if penalty == "elasticnet" else None,
    )


def _auroc_optimal_params(cv_results: dict, penalty: str) -> dict:
    """Pick the least-regularised hyper-parameters among those tied for best AUROC.

    ``scoring`` is "higher is better" (AUROC), so the best mean score is the
    maximum and the acceptance threshold is ``best_mean - SE_of_best`` where the
    standard error uses the number of CV folds.  All candidates at or above this
    threshold are statistically indistinguishable from the AUROC maximiser; among
    them we keep the largest C (least shrinkage, closest to maximum likelihood),
    breaking remaining ties by the higher mean score, then — for Elastic Net — the
    less sparse (smaller l1_ratio) model.
    """
    means = np.asarray(cv_results["mean_test_score"], dtype=float)
    stds = np.asarray(cv_results["std_test_score"], dtype=float)
    params = cv_results["params"]
    best_idx = int(np.nanargmax(means))
    se_best = stds[best_idx] / np.sqrt(C.CV_FOLDS)
    threshold = means[best_idx] - se_best
    accepted = [i for i in range(len(means)) if means[i] >= threshold]

    def sort_key(i):
        p = params[i]
        return (p["C"], means[i], -p.get("l1_ratio", 0.0))

    return dict(params[max(accepted, key=sort_key)])


def c_to_alpha(C_value: float | None, n_total: int) -> float:
    """Translate sklearn's inverse-strength ``C`` into the SGD penalty ``alpha``.

    sklearn LogisticRegression minimises  C * sum_i loss_i + R(w)
    SGDClassifier minimises               (1/n) sum_i loss_i + alpha * R(w)
    Matching the two gives alpha = 1 / (C * n).  For an unpenalised model we use
    a negligible alpha so the FedAvg path still runs.
    """
    if C_value is None:
        return 1e-12
    return 1.0 / (C_value * n_total)
