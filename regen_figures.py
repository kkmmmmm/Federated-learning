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

    # Figures
    os.makedirs(C.FIG_DIR, exist_ok=True)
    # ylim / major-tick matched to the original Excel charts
    # "FL" denotes the proposed method (FedAvg + intercept recalibration): a single
    # FL is shown in every figure; the uncorrected FedAvg appears only in Table S3.
    def remap(df, metric):
        df = df.copy(); df["Source"] = df["Source"].astype(str)
        if metric == "CalibrationIntercept":
            # keep the uncorrected FedAvg as a separate series so the
            # calibration-in-the-large offset that recalibration removes is visible
            df.loc[df.Source == "FL", "Source"] = "FedAvg"
            df.loc[df.Source == "FL_recal", "Source"] = "FL"
        else:
            df = df[df.Source != "FL"]
            df.loc[df.Source == "FL_recal", "Source"] = "FL"
        return df

    specs = [
        ("AUROC", C.RED_REGION_AUROC, "AUROC", (0.68, 0.88), 0.04, None, None,
         {"l1": "Figure1_AUROC_L1", "none": "FigureS1_AUROC_noreg",
          "l2": "FigureS2_AUROC_L2", "elasticnet": "FigureS3_AUROC_EN"}),
        ("CalibrationSlope", C.RED_REGION_CALIB, "Calibration slope", (0.5, 1.5), 0.25, 1.0, None,
         {"l1": "Figure2_slope_L1", "none": "FigureS4_slope_noreg",
          "l2": "FigureS5_slope_L2", "elasticnet": "FigureS6_slope_EN"}),
        ("CalibrationIntercept", C.RED_REGION_CALIB, "Calibration intercept", (-0.9, 0.9), 0.3, 0.0,
         ["FedAvg"],
         {"l1": "Figure3_intercept_L1", "none": "FigureS7_intercept_noreg",
          "l2": "FigureS8_intercept_L2", "elasticnet": "FigureS9_intercept_EN"}),
    ]
    for metric, red, ylab, ylim, major, ref, extra, names in specs:
        for pen in present:
            dfm = remap(between[between["Penalty"] == pen], metric)
            F.panel_figure(dfm, metric, red, ylab,
                           f"{ylab} ({C.PENALTY_LABEL[pen]})",
                           os.path.join(C.FIG_DIR, names[pen] + ".png"),
                           ref_line=ref, ylim=ylim, major=major, extra_sources=extra)
    if pca:
        pca_y = {}
        for pen, df in pca.items():
            df = df.copy(); df["Index"] = df["Index"].astype(str)
            df = df[df.Index != "FL"]
            df.loc[df.Index == "FL_recal", "Index"] = "FL"
            pca_y[pen] = df
        F.pca_figure(pca_y, os.path.join(C.FIG_DIR, "Figure4_PCA.png"))
    if conv:
        F.convergence_figure(conv, os.path.join(C.FIG_DIR, "FL_convergence.png"))
    print("Regenerated Excel workbooks and figures.")


if __name__ == "__main__":
    main()
