"""Regenerate the Excel figure-data workbooks and all figures from saved results.

Reads ``outputs/between_region.csv`` and ``outputs/results.xlsx`` (the PCA and
FL-convergence sheets) and rebuilds the block workbooks and publication figures
without re-running the expensive bootstrap/model fitting.

    python regen_figures.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

from src import config as C
from src import excel_export as X
from src import figures as F


@dataclass
class SimpleHist:
    rounds: list = field(default_factory=list)
    train_logloss: list = field(default_factory=list)
    train_auroc: list = field(default_factory=list)
    converged_round: int | None = None
    wall_seconds: float = 0.0
    recal_seconds: float = 0.0
    recal_logloss: float | None = None
    recal_auroc: float | None = None
    recal_delta: float | None = None


def main():
    between = pd.read_csv(os.path.join(C.OUT_DIR, "between_region.csv"))
    res = os.path.join(C.OUT_DIR, "results.xlsx")
    present = [p for p in C.PENALTIES if (between["Penalty"] == p).any()]

    # PCA from results.xlsx sheets
    suffix = {"none": "noreg", "l1": "L1", "l2": "L2", "elasticnet": "EN"}
    pca = {}
    for pen in present:
        try:
            pca[pen] = pd.read_excel(res, f"PCA_{suffix[pen]}")
        except Exception:
            pass

    # Convergence from results.xlsx
    conv = {}
    try:
        cdf = pd.read_excel(res, "FL_convergence")
        csum = pd.read_excel(res, "FL_convergence_summary").set_index("Penalty")
        for pen in present:
            sub = cdf[cdf["Penalty"] == pen].sort_values("Round")
            h = SimpleHist(rounds=sub["Round"].tolist(),
                           train_logloss=sub["TrainLogLoss"].tolist(),
                           train_auroc=sub["TrainAUROC"].tolist())
            if pen in csum.index:
                cr = csum.loc[pen, "ConvergedRound"]
                h.converged_round = None if pd.isna(cr) else int(cr)
                for src, dst in [("RecalSeconds", "recal_seconds"),
                                 ("RecalLogLoss", "recal_logloss"),
                                 ("RecalAUROC", "recal_auroc"),
                                 ("RecalDelta", "recal_delta")]:
                    if src in csum.columns and not pd.isna(csum.loc[pen, src]):
                        setattr(h, dst, float(csum.loc[pen, src]))
            conv[pen] = h
    except Exception as e:
        print("convergence reload skipped:", e)

    # Excel block workbooks
    X.write_block_workbook(between, "AUROC", os.path.join(C.OUT_DIR, "AUROC.xlsx"))
    X.write_block_workbook(between, "CalibrationSlope",
                           os.path.join(C.OUT_DIR, "calibration_slope.xlsx"))
    X.write_block_workbook(between, "CalibrationIntercept",
                           os.path.join(C.OUT_DIR, "calibration_intercept.xlsx"))
    if pca:
        X.write_pca_workbook(pca, os.path.join(C.OUT_DIR, "PCA.xlsx"))

    # Figures — the manuscript figure numbering and the FL_recal->FL remap are
    # centralized in src/figures.py, so this regeneration path is identical to
    # run_all.py's (FL = recalibrated federated model; the uncorrected FedAvg is
    # shown only on the calibration-intercept panels).
    F.render_between_panels(between, C.FIG_DIR, penalties=present)
    if pca:
        F.render_pca(pca, C.FIG_DIR)
    if conv:
        F.convergence_figure(conv, os.path.join(C.FIG_DIR, "FigureS1_FL_convergence.png"))
    print("Regenerated Excel workbooks and figures.")


if __name__ == "__main__":
    main()
