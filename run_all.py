"""Run the full GUSTO-I federated-learning re-analysis end to end.

Usage:
    python run_all.py                # full run (1000 bootstrap resamples)
    python run_all.py --boot 100     # faster run for testing
    python run_all.py --penalties l1 # subset of regularisation strategies

Outputs (under ``outputs/``):
    results.xlsx, AUROC.xlsx, calibration_slope.xlsx,
    calibration_intercept.xlsx, PCA.xlsx, figures/*.png|pdf
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src import config as C
from src import data_utils as D
from src import analysis as A
from src import excel_export as X
from src import figures as F


def region_counts(df):
    out = {}
    for r in C.REGIONS:
        sub = df[df[C.REGION] == r]
        out[r] = {"n": len(sub), "event": sub[C.OUTCOME].mean()}
    return out


def collect_coefficients(models: dict) -> pd.DataFrame:
    rows = []
    for pen, m in models.items():
        for i in C.REGIONS:
            mdl = m["local"][i][0]
            rows.append(dict(Penalty=pen, Model=f"Local_region{i}",
                             Intercept=mdl.intercept,
                             **dict(zip(C.FEATURES, mdl.coef))))
        # FL denotes the recalibrated federated model (fl_recal); the uncorrected
        # FedAvg is reported separately so its calibration-in-the-large offset (an
        # intercept-only difference) stays inspectable.
        for name, label in (("centralized", "Centralized"),
                            ("fl_recal", "FL"), ("fl", "FedAvg")):
            mdl = getattr(m["global_all"], name)
            rows.append(dict(Penalty=pen, Model=label,
                             Intercept=mdl.intercept,
                             **dict(zip(C.FEATURES, mdl.coef))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=C.N_BOOTSTRAP)
    ap.add_argument("--penalties", nargs="*", default=C.PENALTIES)
    args = ap.parse_args()

    os.makedirs(C.FIG_DIR, exist_ok=True)
    df = D.load_data()
    counts = region_counts(df)
    print(f"Loaded {len(df)} patients, {C.N_REGIONS} regions, "
          f"{C.N_FEATURES} predictors. Bootstrap={args.boot}")

    # ----- build all models (cached per penalty) -------------------------- #
    models = {}
    for pen in args.penalties:
        t0 = time.perf_counter()
        global_all = A.build_global_models(df, C.REGIONS, pen, record_history=True)
        global_loro = {r: A.build_global_models(df, [x for x in C.REGIONS if x != r], pen)
                       for r in C.REGIONS}
        local = {i: A.build_local_model(df, i, pen) for i in C.REGIONS}
        models[pen] = dict(global_all=global_all, global_loro=global_loro, local=local)
        print(f"  [{pen}] models built in {time.perf_counter()-t0:.1f}s "
              f"(FL conv round={global_all.fl_history.converged_round}, "
              f"{global_all.fl_history.wall_seconds:.2f}s)")

    # ----- analyses ------------------------------------------------------- #
    within = pd.concat([A.within_region(df, pen, models[pen]["global_loro"])
                        for pen in args.penalties], ignore_index=True)
    print("  within-region done")

    between_parts = []
    for pen in args.penalties:
        t0 = time.perf_counter()
        between_parts.append(A.between_region(df, pen, models[pen]["local"],
                                              models[pen]["global_loro"], n_boot=args.boot))
        print(f"  between-region [{pen}] done in {time.perf_counter()-t0:.1f}s")
    between = pd.concat(between_parts, ignore_index=True)

    pca = {pen: A.coefficient_pca(models[pen]["local"], models[pen]["global_all"])
           for pen in args.penalties}
    # Convergence experiment: 1 local iteration/round from zero init for a clean curve.
    conv = {pen: A.fl_convergence_run(df, pen) for pen in args.penalties}
    coeffs = collect_coefficients(models)

    # ----- save raw CSVs (so figures/excel can be regenerated) ------------ #
    within.to_csv(os.path.join(C.OUT_DIR, "within_region.csv"), index=False)
    between.to_csv(os.path.join(C.OUT_DIR, "between_region.csv"), index=False)

    # ----- Excel ---------------------------------------------------------- #
    X.write_block_workbook(between, "AUROC", os.path.join(C.OUT_DIR, "AUROC.xlsx"))
    X.write_block_workbook(between, "CalibrationSlope",
                           os.path.join(C.OUT_DIR, "calibration_slope.xlsx"))
    X.write_block_workbook(between, "CalibrationIntercept",
                           os.path.join(C.OUT_DIR, "calibration_intercept.xlsx"))
    X.write_pca_workbook(pca, os.path.join(C.OUT_DIR, "PCA.xlsx"))
    X.write_results_workbook(within, between, pca, conv, coeffs, counts,
                             os.path.join(C.OUT_DIR, "results.xlsx"))
    print("  Excel written")

    # ----- Figures -------------------------------------------------------- #
    # Figure numbering + the FL_recal->FL remap live in src/figures.py so this
    # path and regen_figures.py always emit identical, manuscript-consistent
    # figures (FL = recalibrated federated model; uncorrected FedAvg only on the
    # calibration-intercept panels).
    F.render_between_panels(between, C.FIG_DIR, penalties=args.penalties)
    F.render_pca(pca, C.FIG_DIR)
    F.convergence_figure(conv, os.path.join(C.FIG_DIR, "FigureS1_FL_convergence.png"))
    print("  Figures written. Done.")


if __name__ == "__main__":
    main()
