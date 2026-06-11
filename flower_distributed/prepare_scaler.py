"""Build the shared global scaler from federated summary statistics.

In a real deployment each client would transmit only (count, sum, sum-of-
squares) for its own data; the server aggregates them into a global mean/SD and
broadcasts it back.  This script performs that aggregation centrally for the
demo and writes ``global_scaler.json`` for the clients to load.  No raw records
leave a site in the federated protocol — only the three summary vectors.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C  # noqa: E402


def main():
    df = pd.read_csv(C.DATA_CSV)
    n = 0
    s1 = np.zeros(C.N_FEATURES)
    s2 = np.zeros(C.N_FEATURES)
    for r in C.REGIONS:                       # each region contributes summaries only
        X = df[df[C.REGION] == r][C.FEATURES].to_numpy(float)
        n += len(X)
        s1 += X.sum(0)
        s2 += (X ** 2).sum(0)
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 0)
    std = np.sqrt(var)
    std[std == 0] = 1.0
    out = os.path.join(os.path.dirname(__file__), "global_scaler.json")
    json.dump({"mean": mean.tolist(), "std": std.tolist(),
               "features": C.FEATURES}, open(out, "w"), indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
