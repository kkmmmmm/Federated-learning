"""Central configuration for the GUSTO-I federated-learning re-analysis.

This module is the single source of truth for the predictor set, the region
flag, the regularisation strategies and the experiment hyper-parameters used by
every other script in the package.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# By default we expect the raw GUSTO-I file next to the package.  Override with
# the environment variable GUSTO_CSV if it lives elsewhere.
DATA_CSV = os.environ.get(
    "GUSTO_CSV",
    os.path.join(os.path.dirname(PKG_DIR), "GUSTO_sterberg.csv"),
)
OUT_DIR = os.path.join(PKG_DIR, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
OUTCOME = "DAY30"          # 30-day mortality (0/1)
REGION = "REGL"            # region flag used to split data into 16 sites

# The 17 candidate predictors described in Table 1 of the manuscript.
# NB: the original GitHub code accidentally also fed ESAMP, GRPL and GRPS
# (sampling / treatment-group variables) into the model.  Those are NOT
# clinical predictors and are excluded here, restoring consistency with Table 1.
FEATURES = [
    "SEX",     # female sex (binary)
    "AGE",     # age, years (continuous)
    "KILLIP",  # Killip class I-IV (ordinal)
    "DIA",     # diabetes (binary)
    "HYP",     # hypotension, SBP<100 (binary)
    "HRT",     # tachycardia, pulse>80 (binary)
    "ANT",     # anterior infarct location (binary)
    "PMI",     # previous MI (binary)
    "HEI",     # height, cm (continuous)
    "WEI",     # weight, kg (continuous)
    "SMK",     # smoking history 1/2/3 (ordinal)
    "HTN",     # hypertension history (binary)
    "LIP",     # hypercholesterolemia (binary)
    "PAN",     # previous angina pectoris (binary)
    "FAM",     # family history of MI (binary)
    "STE",     # number of ECG leads with ST elevation, count (continuous)
    "TTR",     # time to relief of chest pain > 1 h (binary)
]
EXCLUDED = ["ESAMP", "GRPL", "GRPS"]   # explicitly dropped (not predictors)

N_FEATURES = len(FEATURES)
N_REGIONS = 16
REGIONS = list(range(1, N_REGIONS + 1))

# --------------------------------------------------------------------------- #
# Regularisation strategies
#   key  -> label used in tables/figures
# --------------------------------------------------------------------------- #
PENALTIES = ["none", "l1", "l2", "elasticnet"]
PENALTY_LABEL = {
    "none": "Without regularization",
    "l1": "L1",
    "l2": "L2",
    "elasticnet": "Elastic Net",
}

# Hyper-parameter grid (manuscript: C in 1e-3 .. 1e3; elastic-net l1_ratio).
import numpy as np  # noqa: E402

C_GRID = np.logspace(-3, 3, 7)
L1_RATIO_GRID = [0.1, 0.5, 0.9]
CV_FOLDS = 5
# AUROC is the CV optimization criterion; the winner is resolved with a
# one-standard-error rule (the least-regularized model within 1 SE of the best
# mean AUROC) in models._auroc_optimal_params. See Supplemental Methods.
CV_SCORING = "roc_auc"

# --------------------------------------------------------------------------- #
# Federated-learning (FedAvg) hyper-parameters
# --------------------------------------------------------------------------- #
FL_ROUNDS = 60              # max communication rounds
FL_LOCAL_EPOCHS = 1         # local epochs per round (full-batch passes)
FL_CONV_TOL = 1e-4          # |Δ global log-loss| threshold for convergence
FL_CONV_PATIENCE = 3        # consecutive rounds below tol -> "converged"

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
N_BOOTSTRAP = 1000          # bootstrap resamples for 95% CIs
CALIB_BINS = 10             # bins for calibration_curve (matches published figs)
RANDOM_STATE = 42

# Region highlighted in red across ALL between-region figures: the local model
# with the largest performance deviation (worst external AUROC) and the farthest
# outlier in coefficient space (Figure 4 PCA).  Used consistently for AUROC,
# calibration slope and calibration intercept so a single region's aberrant
# model explains both its poor discrimination and its poor calibration.
RED_REGION_AUROC = 10       # Figure 2 / S2-S4
RED_REGION_CALIB = 10       # Figures 3,4 / S5-S10 (was 16; under the standard
                            # calibration definition no single region dominates,
                            # so we highlight the overall outlier, Region 10)
