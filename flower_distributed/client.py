"""Networked Flower client (one per region) — corrected version.

Run one process per region (1..16), each pointing at the same server.  Mirrors
the in-process FedAvg in ``src/flower_fl.py`` but over real gRPC:

    python prepare_scaler.py                 # once: build the shared scaler
    python server.py                         # terminal 0
    python client.py --region 1              # terminal 1
    ...                                       # one terminal per region

Fixes vs the original GitHub clients:
  * warm-starts from the global model each round (true FedAvg) instead of
    refitting from scratch and discarding the global parameters;
  * uses the SHARED global scaler (built from aggregated count/sum/sumsq), so
    every client standardizes on one common feature scale.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import flwr as fl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C  # noqa: E402

warnings.filterwarnings("ignore")


def load_region(region: int):
    df = pd.read_csv(C.DATA_CSV)
    sub = df[df[C.REGION] == region]
    X = sub[C.FEATURES].to_numpy(float)
    y = sub[C.OUTCOME].to_numpy(int)
    scaler = json.load(open(os.path.join(os.path.dirname(__file__), "global_scaler.json")))
    mean = np.array(scaler["mean"]); std = np.array(scaler["std"])
    return (X - mean) / std, y


class RegionClient(fl.client.NumPyClient):
    def __init__(self, region: int, penalty: str, C_value, l1_ratio, local_iters):
        self.Xs, self.y = load_region(region)
        if penalty == "none":
            self.clf = LogisticRegression(penalty=None, solver="lbfgs",
                                          warm_start=True, max_iter=local_iters)
        elif penalty == "l2":
            self.clf = LogisticRegression(penalty="l2", solver="lbfgs", C=C_value,
                                          warm_start=True, max_iter=local_iters)
        elif penalty == "l1":
            self.clf = LogisticRegression(penalty="l1", solver="saga", C=C_value,
                                          warm_start=True, max_iter=local_iters)
        else:
            self.clf = LogisticRegression(penalty="elasticnet", solver="saga",
                                          C=C_value, l1_ratio=l1_ratio,
                                          warm_start=True, max_iter=local_iters)
        self.clf.classes_ = np.array([0, 1])
        self.clf.coef_ = np.zeros((1, self.Xs.shape[1]))
        self.clf.intercept_ = np.zeros(1)

    def _set(self, params):
        self.clf.coef_ = params[0].reshape(1, -1)
        self.clf.intercept_ = np.atleast_1d(params[1])
        self.clf.classes_ = np.array([0, 1])

    def get_parameters(self, config):
        return [self.clf.coef_.ravel(), self.clf.intercept_.copy()]

    def fit(self, parameters, config):
        self._set(parameters)
        self.clf.fit(self.Xs, self.y)
        return self.get_parameters(config), len(self.y), {}

    def evaluate(self, parameters, config):
        from sklearn.metrics import log_loss
        self._set(parameters)
        p = self.clf.predict_proba(self.Xs)[:, 1]
        return float(log_loss(self.y, p, labels=[0, 1])), len(self.y), {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", type=int, required=True)
    ap.add_argument("--server", default="127.0.0.1:8080")
    ap.add_argument("--penalty", default="l1", choices=C.PENALTIES)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--l1_ratio", type=float, default=0.5)
    ap.add_argument("--local_iters", type=int, default=5)
    a = ap.parse_args()
    client = RegionClient(a.region, a.penalty, a.C, a.l1_ratio, a.local_iters)
    fl.client.start_client(server_address=a.server, client=client.to_client())
