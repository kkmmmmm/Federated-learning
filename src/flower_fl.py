"""Genuine Flower (flwr) FedAvg implementation of federated logistic regression.

Design
------
* Each region is one Flower ``NumPyClient`` (``LogRegClient``) holding only its
  own standardized data and a warm-started ``LogisticRegression`` as the local
  optimiser (lbfgs for none/L2, saga for L1/elastic-net).  Every client picks its
  **own** inverse-strength ``C`` (and ``l1_ratio``) by cross-validation on its own
  local data — the regularisation is a property of each client's local update, not
  borrowed from the pooled/centralized model.  The server only averages the
  shared coefficients, so federated learning never sees pooled patient data.
* Aggregation uses Flower's own ``flwr.server.strategy.FedAvg`` (sample-size
  weighted averaging of the regression coefficients) — the FedAvg rule of the
  manuscript.
* Communication rounds are driven in-process (the gRPC transport of a networked
  deployment is replaced by direct method calls; an equivalent networked
  server/client pair lives in ``flower_distributed/``), keeping the re-analysis
  reproducible and deterministic.

Earlier errors fixed here:
  1. *Not true FedAvg* — the client **warm-starts from the global model** each
     round and runs a bounded number of local iterations, instead of refitting
     from scratch and discarding the global parameters.
  2. *Inconsistent standardisation* — every client uses the **same global
     scaler** (federated standardisation), so coefficient averaging happens on a
     single common feature scale.

Global training log-loss / AUROC per round and wall-clock time are recorded to
characterise the FL convergence behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import warnings

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

import logging
logging.getLogger("flwr").setLevel(logging.ERROR)  # silence per-round info/warnings

import flwr as fl
from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg

from . import config as C
from .models import FittedModel


def _make_local_estimator(penalty: str, C_value: float | None,
                          l1_ratio: float | None, max_iter: int) -> LogisticRegression:
    common = dict(warm_start=True, max_iter=max_iter, fit_intercept=True)
    if penalty == "none":
        return LogisticRegression(penalty=None, solver="lbfgs", **common)
    if penalty == "l2":
        return LogisticRegression(penalty="l2", solver="lbfgs", C=C_value, **common)
    if penalty == "l1":
        return LogisticRegression(penalty="l1", solver="saga", C=C_value, **common)
    if penalty == "elasticnet":
        return LogisticRegression(penalty="elasticnet", solver="saga",
                                  C=C_value, l1_ratio=l1_ratio, **common)
    raise ValueError(penalty)


class LogRegClient(fl.client.NumPyClient):
    """A single region acting as a federated client."""

    def __init__(self, Xs: np.ndarray, y: np.ndarray, penalty: str,
                 C_value: float | None, l1_ratio: float | None, local_iters: int):
        self.Xs = Xs
        self.y = y
        self.n = len(y)
        self.clf = _make_local_estimator(penalty, C_value, l1_ratio, local_iters)
        self._init_estimator()

    def _init_estimator(self):
        self.clf.classes_ = np.array([0, 1])
        self.clf.coef_ = np.zeros((1, self.Xs.shape[1]))
        self.clf.intercept_ = np.zeros(1)
        self.clf.n_features_in_ = self.Xs.shape[1]

    def _set(self, params):
        coef, intercept = params
        self.clf.coef_ = coef.reshape(1, -1).astype(float)
        self.clf.intercept_ = np.atleast_1d(intercept).astype(float)
        self.clf.classes_ = np.array([0, 1])

    def get_parameters(self, config=None):
        return [self.clf.coef_.ravel(), self.clf.intercept_.copy()]

    def fit(self, parameters, config):
        self._set(parameters)              # warm start from the global model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.clf.fit(self.Xs, self.y)  # bounded local iterations
        return self.get_parameters(), self.n, {}

    def evaluate(self, parameters, config):
        self._set(parameters)
        p = expit(self.Xs @ self.clf.coef_.ravel() + self.clf.intercept_[0])
        return float(log_loss(self.y, p, labels=[0, 1])), self.n, {}


@dataclass
class FLHistory:
    rounds: list[int] = field(default_factory=list)
    train_logloss: list[float] = field(default_factory=list)
    train_auroc: list[float] = field(default_factory=list)
    converged_round: int | None = None
    wall_seconds: float = 0.0
    # Federated intercept-recalibration step appended after FedAvg convergence.
    recal_seconds: float = 0.0
    recal_logloss: float | None = None
    recal_auroc: float | None = None
    recal_delta: float | None = None


def _global_logloss_auroc(coef, intercept, Xs_all, y_all):
    p = expit(Xs_all @ coef + intercept)
    ll = log_loss(y_all, p, labels=[0, 1])
    try:
        auc = roc_auc_score(y_all, p)
    except ValueError:
        auc = np.nan
    return ll, auc


def run_fedavg(client_data: list[tuple[np.ndarray, np.ndarray]],
               penalty: str, client_C: list, client_l1: list,
               n_features: int,
               rounds: int = C.FL_ROUNDS,
               local_iters: int = C.FL_LOCAL_EPOCHS,
               record_history: bool = True):
    """Run FedAvg across the given (already standardized) client datasets.

    Returns ``(FittedModel, FLHistory)``.  ``client_data`` is a list of (Xs, y)
    per region; all share the same global scaler upstream.  ``client_C`` and
    ``client_l1`` give each client its own locally-selected regularisation
    (aligned with ``client_data``); the aggregated FL model therefore has no
    single global ``C`` (stored as ``None``).
    """
    clients = [LogRegClient(Xs, y, penalty, c, l1, local_iters)
               for (Xs, y), c, l1 in zip(client_data, client_C, client_l1)]
    strategy = FedAvg(fraction_fit=1.0, fraction_evaluate=1.0,
                      min_available_clients=len(clients))

    Xs_all = np.vstack([Xs for Xs, _ in client_data])
    y_all = np.concatenate([y for _, y in client_data])

    coef = np.zeros(n_features)
    intercept = np.array([0.0])

    hist = FLHistory()
    below = 0
    prev_ll = None
    if record_history:
        # round 0: the zero-initialised global model (start of FedAvg)
        ll0, auc0 = _global_logloss_auroc(coef, float(intercept[0]), Xs_all, y_all)
        hist.rounds.append(0)
        hist.train_logloss.append(ll0)
        hist.train_auroc.append(auc0)
        prev_ll = ll0
    t0 = time.perf_counter()

    for rnd in range(1, rounds + 1):
        params = [coef, intercept]
        results = []
        for cl in clients:
            new_params, n, _ = cl.fit(params, {})
            fr = FitRes(status=Status(Code.OK, ""),
                        parameters=ndarrays_to_parameters(new_params),
                        num_examples=n, metrics={})
            results.append((None, fr))
        agg_params, _ = strategy.aggregate_fit(rnd, results, [])
        coef, intercept = parameters_to_ndarrays(agg_params)
        coef = coef.ravel()
        intercept = np.atleast_1d(intercept)

        if record_history:
            ll, auc = _global_logloss_auroc(coef, float(intercept[0]), Xs_all, y_all)
            hist.rounds.append(rnd)
            hist.train_logloss.append(ll)
            hist.train_auroc.append(auc)
            if prev_ll is not None and abs(prev_ll - ll) < C.FL_CONV_TOL:
                below += 1
                if below >= C.FL_CONV_PATIENCE and hist.converged_round is None:
                    hist.converged_round = rnd
            else:
                below = 0
            prev_ll = ll

    hist.wall_seconds = time.perf_counter() - t0
    model = FittedModel(coef=coef, intercept=float(intercept[0]),
                        penalty=penalty, C=None, l1_ratio=None)
    return model, hist


def federated_intercept_recalibration(coef: np.ndarray, intercept: float,
                                      client_data: list[tuple[np.ndarray, np.ndarray]],
                                      max_iter: int = 50, tol: float = 1e-10) -> float:
    """Privacy-preserving calibration-in-the-large correction for a FedAvg model.

    FedAvg averages the per-region coefficients, so the aggregated model does not
    satisfy the pooled intercept score equation (mean predicted = mean observed):
    with a rare outcome this leaves a systematic calibration-in-the-large offset.
    This routine keeps the FedAvg **slopes fixed** and finds the single scalar
    shift ``delta`` for the intercept so that, summed across all clients, the
    total predicted probability equals the total number of events — i.e. it
    solves, in federated form, the one-parameter offset model
    ``logit(P(y=1)) = (intercept + delta) + slopes·x``.

    Privacy: each client only ever returns three scalars per Newton step — the
    sum of predicted probabilities, the sum of ``p*(1-p)`` and (once) the event
    count.  No patient-level data and no regularisation parameter ``C`` leaves a
    site, consistent with the federated-standardisation model (which already
    shares only counts/sums/sums-of-squares).  Returns ``delta``.
    """
    lps = [Xs @ coef + intercept for Xs, _ in client_data]
    Sy = float(sum(float(y.sum()) for _, y in client_data))   # per-site scalar
    delta = 0.0
    for _ in range(max_iter):
        Sp = 0.0
        Sw = 0.0
        for lp in lps:                       # in deployment: one scalar pair/site
            mu = expit(lp + delta)
            Sp += float(mu.sum())
            Sw += float((mu * (1.0 - mu)).sum())
        if Sw <= 0:
            break
        step = (Sy - Sp) / Sw
        delta += step
        if abs(step) < tol:
            break
    return float(delta)
