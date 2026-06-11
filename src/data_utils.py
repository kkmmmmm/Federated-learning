"""Data loading, region splitting and (federated) standardisation utilities."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from . import config as C


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load the GUSTO-I CSV and keep only outcome, region and the 17 predictors."""
    df = pd.read_csv(path or C.DATA_CSV)
    keep = [C.OUTCOME, C.REGION] + C.FEATURES
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in data: {missing}")
    return df[keep].copy()


def region_arrays(df: pd.DataFrame, region: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for a single region as float arrays in FEATURES order."""
    sub = df[df[C.REGION] == region]
    X = sub[C.FEATURES].to_numpy(dtype=float)
    y = sub[C.OUTCOME].to_numpy(dtype=int)
    return X, y


def pooled_arrays(df: pd.DataFrame, regions) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) pooled across a list/iterable of regions."""
    sub = df[df[C.REGION].isin(list(regions))]
    X = sub[C.FEATURES].to_numpy(dtype=float)
    y = sub[C.OUTCOME].to_numpy(dtype=int)
    return X, y


# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #
@dataclass
class Scaler:
    """A plain mean/std standardiser (equivalent to sklearn StandardScaler).

    Implemented explicitly so the *federated* variant can be built from shared
    summary statistics only (count / sum / sum-of-squares), never raw data.
    """

    mean_: np.ndarray
    std_: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    @staticmethod
    def fit(X: np.ndarray) -> "Scaler":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        return Scaler(mean_=mean, std_=std)


@dataclass
class ClientStats:
    """Privacy-preserving summary statistics shared by one FL client."""

    n: int
    s1: np.ndarray   # sum of features
    s2: np.ndarray   # sum of squared features


def client_stats(X: np.ndarray) -> ClientStats:
    return ClientStats(n=len(X), s1=X.sum(axis=0), s2=(X ** 2).sum(axis=0))


def federated_scaler(stats: list[ClientStats]) -> Scaler:
    """Aggregate client summary statistics into a global mean/std scaler.

    This is the *federated standardisation* step: each client transmits only
    (n, sum, sum-of-squares); the server reconstructs the exact pooled mean and
    standard deviation without ever seeing raw records.  The result is
    numerically identical to standardising on the pooled data, so the L1/L2/
    elastic-net penalty is applied on a single common feature scale across all
    sites (a prerequisite for valid coefficient averaging).
    """
    n = sum(s.n for s in stats)
    s1 = np.sum([s.s1 for s in stats], axis=0)
    s2 = np.sum([s.s2 for s in stats], axis=0)
    mean = s1 / n
    var = s2 / n - mean ** 2
    var[var < 0] = 0.0
    std = np.sqrt(var)
    std[std == 0] = 1.0
    return Scaler(mean_=mean, std_=std)
