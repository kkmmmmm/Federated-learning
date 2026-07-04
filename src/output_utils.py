"""Map the pipeline's internal model labels to the manuscript's convention.

Internally the analysis stores two federated models side by side:

* ``FL``        -- the raw FedAvg model (no intercept correction), and
* ``FL_recal``  -- the same model after the federated intercept recalibration.

The manuscript defines **FL** as the *recalibrated* federated model ("Throughout,
FL denotes this recalibrated federated model").  Every published figure/table
must therefore show ``FL_recal`` as ``FL``.  To keep that mapping in one place —
so figures, Excel block workbooks and the PCA output can never diverge — all
final-output writers funnel their frames through the helpers below instead of
hard-coding source names.

For the calibration-intercept metric the *uncorrected* FedAvg model is also
shown (relabelled ``FedAvg``) so the calibration-in-the-large offset that the
recalibration removes stays visible; for every other metric FL and FL_recal are
identical (an intercept-only correction), so the uncorrected series is dropped.
"""
from __future__ import annotations

import pandas as pd


def remap_sources(df: pd.DataFrame, metric: str, col: str = "Source") -> pd.DataFrame:
    """Relabel FedAvg model rows for a final output frame.

    ``metric`` is one of ``"AUROC"``, ``"CalibrationSlope"`` or
    ``"CalibrationIntercept"``; ``col`` is the column holding the model name
    (``"Source"`` for between-region frames, ``"Model"`` for within-region).
    """
    df = df.copy()
    df[col] = df[col].astype(str)
    if metric == "CalibrationIntercept":
        # Keep the uncorrected FedAvg as its own series; promote FL_recal to FL.
        df.loc[df[col] == "FL", col] = "FedAvg"
        df.loc[df[col] == "FL_recal", col] = "FL"
    else:
        # FL and FL_recal are identical here; drop the uncorrected one.
        df = df[df[col] != "FL"].copy()
        df.loc[df[col] == "FL_recal", col] = "FL"
    return df


def remap_pca(df: pd.DataFrame, col: str = "Index") -> pd.DataFrame:
    """PCA output: drop the uncorrected FedAvg point, show FL_recal as FL."""
    df = df.copy()
    df[col] = df[col].astype(str)
    df = df[df[col] != "FL"].copy()
    df.loc[df[col] == "FL_recal", col] = "FL"
    return df
