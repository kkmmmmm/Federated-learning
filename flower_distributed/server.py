"""Networked Flower server — FedAvg over the 16 regional clients (corrected).

    python server.py --rounds 60

Uses Flower's built-in ``FedAvg`` strategy (sample-size weighted averaging of
the regression coefficients).  Start this first, then launch the 16 clients.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C  # noqa: E402


def initial_parameters():
    coef = np.zeros((1, C.N_FEATURES))
    intercept = np.zeros(1)
    return fl.common.ndarrays_to_parameters([coef.ravel(), intercept])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=C.FL_ROUNDS)
    ap.add_argument("--address", default="0.0.0.0:8080")
    a = ap.parse_args()

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=C.N_REGIONS,
        min_evaluate_clients=C.N_REGIONS,
        min_available_clients=C.N_REGIONS,
        initial_parameters=initial_parameters(),
    )
    fl.server.start_server(
        server_address=a.address,
        config=fl.server.ServerConfig(num_rounds=a.rounds),
        strategy=strategy,
    )
