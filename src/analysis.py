"""End-to-end re-analysis: builds all models and produces the result tables.

Three learning paradigms are compared, each for four regularisation strategies
(none / L1 / L2 / elastic-net):

* **Local**     – a model trained on a single region.
* **Centralized** – a model trained on pooled patient-level data.
* **FL**        – a FedAvg model (Flower) that shares only coefficients.

Leak fix (reviewer comment)
---------------------------
For *within-region* validation the global (centralized & FL) models are now
trained with the **target region completely excluded** (the 15 other regions)
and applied to that region's held-out 1/5 test folds, instead of being trained
on the whole dataset (which had included the test data).  This is the same
leave-one-region-out (LORO) global model used for between-region validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config as C
from . import data_utils as D
from .models import FittedModel, fit_logreg
from .flower_fl import run_fedavg, FLHistory, federated_intercept_recalibration
from .calibration import point_metrics, bootstrap_metrics


# --------------------------------------------------------------------------- #
# Model containers
# --------------------------------------------------------------------------- #
@dataclass
class GlobalModels:
    scaler: D.Scaler
    centralized: FittedModel
    fl: FittedModel
    fl_history: FLHistory
    fl_recal: FittedModel = None   # FedAvg with a federated intercept recalibration


def _client_data(df, regions, scaler):
    cd = []
    for r in regions:
        X, y = D.region_arrays(df, r)
        cd.append((scaler.transform(X), y))
    return cd


def build_global_models(df: pd.DataFrame, regions: list[int], penalty: str,
                        record_history: bool = False) -> GlobalModels:
    """Centralized + FL global models trained on the given set of regions.

    The centralized model is fit on pooled standardized data; the FL model uses
    FedAvg over per-region clients.  Both share the *federated* scaler, so they
    sit on an identical feature scale.  Federated standardisation reproduces the
    exact pooled mean/SD, so the centralized fit is unaffected by it.
    """
    stats = [D.client_stats(D.region_arrays(df, r)[0]) for r in regions]
    scaler = D.federated_scaler(stats)

    X, y = D.pooled_arrays(df, regions)
    Xs = scaler.transform(X)
    centralized = fit_logreg(Xs, y, penalty)

    # Each FL client selects its own regularisation locally: cross-validation on
    # its own data at the shared (federated) feature scale.  The federation never
    # pools raw data and never reuses the centralized model's hyper-parameters.
    client_C, client_l1 = _client_hparams(df, regions, scaler, penalty)
    client_data = _client_data(df, regions, scaler)
    fl_model, hist = run_fedavg(client_data, penalty, client_C, client_l1,
                                C.N_FEATURES, record_history=record_history)

    # Federated calibration-in-the-large correction: keep the FedAvg slopes,
    # shift only the intercept so total predicted == total observed across the
    # training regions (sharing only per-site scalars, no raw data / no C).
    delta = federated_intercept_recalibration(fl_model.coef, fl_model.intercept,
                                              client_data)
    fl_recal = FittedModel(coef=fl_model.coef, intercept=fl_model.intercept + delta,
                           penalty=penalty, C=None, l1_ratio=None)
    return GlobalModels(scaler, centralized, fl_model, hist, fl_recal)


def _client_hparams(df, regions, scaler, penalty):
    """Per-client (C, l1_ratio), each chosen by CV on that region's own data
    standardized with the shared federated scaler (no pooled data, no centralized
    model)."""
    cs, l1s = [], []
    for r in regions:
        X, y = D.region_arrays(df, r)
        m = fit_logreg(scaler.transform(X), y, penalty)
        cs.append(m.C)
        l1s.append(m.l1_ratio)
    return cs, l1s


def fl_convergence_run(df: pd.DataFrame, penalty: str,
                       rounds: int = C.FL_ROUNDS, local_iters: int = 1) -> FLHistory:
    """Dedicated FedAvg run (all 16 regions) for the convergence figure.

    Uses one local iteration per round and logs from the zero initialisation so
    the round-by-round descent of the global training loss is visible.  Each
    client uses its own locally-selected regularisation.
    """
    import time
    from scipy.special import expit
    from sklearn.metrics import log_loss, roc_auc_score

    stats = [D.client_stats(D.region_arrays(df, r)[0]) for r in C.REGIONS]
    scaler = D.federated_scaler(stats)
    client_C, client_l1 = _client_hparams(df, C.REGIONS, scaler, penalty)
    client_data = _client_data(df, C.REGIONS, scaler)
    fl_model, hist = run_fedavg(client_data, penalty, client_C, client_l1, C.N_FEATURES,
                                rounds=rounds, local_iters=local_iters, record_history=True)

    # Final step of the method: one federated intercept-recalibration round
    # (clients return only aggregate scalars).  Record its wall-clock and the
    # resulting drop in the global training loss so the convergence figure and
    # the reported time reflect the *complete* procedure, not FedAvg alone.
    Xs_all = np.vstack([Xs for Xs, _ in client_data])
    y_all = np.concatenate([y for _, y in client_data])
    t0 = time.perf_counter()
    delta = federated_intercept_recalibration(fl_model.coef, fl_model.intercept,
                                              client_data)
    hist.recal_seconds = time.perf_counter() - t0
    p = expit(Xs_all @ fl_model.coef + (fl_model.intercept + delta))
    hist.recal_delta = float(delta)
    hist.recal_logloss = float(log_loss(y_all, p, labels=[0, 1]))
    hist.recal_auroc = float(roc_auc_score(y_all, p))
    return hist


def build_local_model(df: pd.DataFrame, region: int, penalty: str):
    """Local model + its own local scaler, trained on a region's full data."""
    X, y = D.region_arrays(df, region)
    scaler = D.Scaler.fit(X)
    model = fit_logreg(scaler.transform(X), y, penalty)
    return model, scaler


def _predict(model: FittedModel, scaler: D.Scaler, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(scaler.transform(X))


# --------------------------------------------------------------------------- #
# Within-region validation  (Table 2 / Table S2)  -- LEAK FIXED
# --------------------------------------------------------------------------- #
def within_region(df: pd.DataFrame, penalty: str,
                  global_loro: dict[int, GlobalModels]) -> pd.DataFrame:
    """Stratified 5-fold within-region validation.

    * Local: trained on the 4/5 training fold (own scaler, own CV), evaluated on
      the 1/5 test fold.
    * Centralized / FL: the LORO global model for that region (trained on the 15
      other regions) evaluated on the same 1/5 test folds — no leakage.
    """
    rows = []
    for r in C.REGIONS:
        X, y = D.region_arrays(df, r)
        skf = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True,
                              random_state=C.RANDOM_STATE)
        gm = global_loro[r]
        for fold, (tr, te) in enumerate(skf.split(X, y), 1):
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

            # Local model on the training fold.
            sc = D.Scaler.fit(Xtr)
            local = fit_logreg(sc.transform(Xtr), ytr, penalty)
            m_local = point_metrics(yte, local.predict_proba(sc.transform(Xte)))

            # LORO global models on the test fold (target region excluded).
            m_cen = point_metrics(yte, _predict(gm.centralized, gm.scaler, Xte))
            m_fl = point_metrics(yte, _predict(gm.fl, gm.scaler, Xte))
            m_flr = point_metrics(yte, _predict(gm.fl_recal, gm.scaler, Xte))

            for name, m in (("Local", m_local), ("Centralized", m_cen),
                            ("FL", m_fl), ("FL_recal", m_flr)):
                rows.append(dict(Region=r, Fold=fold, Model=name, Penalty=penalty,
                                 **m))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Between-region validation  (Figures 1-3 / S1-S9)
# --------------------------------------------------------------------------- #
def _between_one_region(df, r, penalty, local_models, global_loro, n_boot):
    X, y = D.region_arrays(df, r)
    rows = []
    for i in C.REGIONS:                       # foreign local models
        if i == r:
            continue
        model, sc = local_models[i]
        p = _predict(model, sc, X)
        rows.append(dict(ValidationRegion=r, Source=i, Penalty=penalty,
                         **bootstrap_metrics(y, p, n_boot=n_boot)))
    gm = global_loro[r]                       # LORO globals
    for name, mdl in (("Centralized", gm.centralized), ("FL", gm.fl),
                      ("FL_recal", gm.fl_recal)):
        p = _predict(mdl, gm.scaler, X)
        rows.append(dict(ValidationRegion=r, Source=name, Penalty=penalty,
                         **bootstrap_metrics(y, p, n_boot=n_boot)))
    return rows


def between_region(df: pd.DataFrame, penalty: str,
                   local_models: dict[int, tuple], global_loro: dict[int, GlobalModels],
                   n_boot: int = C.N_BOOTSTRAP, n_jobs: int = -1) -> pd.DataFrame:
    """Apply every foreign local model + the LORO globals to each region.

    Returns one row per (validation region, source model) with point estimates
    and 95% bootstrap CIs for AUROC, calibration slope and intercept.
    ``source`` is the training region (int), or ``'Centralized'`` / ``'FL'``.
    The 16 validation regions are bootstrapped in parallel.
    """
    from joblib import Parallel, delayed
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_between_one_region)(df, r, penalty, local_models, global_loro, n_boot)
        for r in C.REGIONS)
    return pd.DataFrame([row for part in parts for row in part])


# --------------------------------------------------------------------------- #
# Coefficient PCA  (Figure 4)
# --------------------------------------------------------------------------- #
def _to_global_coords(model: FittedModel, scaler: D.Scaler,
                      mu_g: np.ndarray, sd_g: np.ndarray) -> np.ndarray:
    """Re-express a model's (intercept, slopes) on a COMMON reference.

    Each model is fitted on its own standardisation (local models use their own
    region's mean/SD; global models use the federated scaler), so the raw
    intercepts/slopes are not directly comparable.  We map every model onto the
    same global standardisation: slopes per global-SD and the intercept as the
    log-odds at the *global mean* covariate profile.  This makes the intercept
    (calibration-in-the-large) comparable across models, so the recalibrated FL
    model is a distinct point that sits closer to the centralized model.
    """
    beta_raw = model.coef / scaler.std_
    b_raw = model.intercept - np.sum(model.coef * scaler.mean_ / scaler.std_)
    beta_g = beta_raw * sd_g
    b_g = b_raw + np.sum(beta_raw * mu_g)
    return np.concatenate([[b_g], beta_g])


def coefficient_pca(local_models: dict[int, tuple], global_all: GlobalModels):
    """PCA (PC1/PC2) of the FULL model parameters of the 19 models.

    Rows = 16 local + Centralized + FL + FL_recal; columns = the intercept
    (calibration-in-the-large, at the global-mean profile) plus the 17 slopes,
    all expressed on a common global standardisation and then column-standardised
    so the logit-scale intercept does not dominate.  Including the intercept lets
    the figure show FL's calibration-in-the-large offset (FL sits away from
    Centralized) and its removal by recalibration (FL_recal moves back towards
    Centralized).
    """
    from sklearn.decomposition import PCA

    mu_g, sd_g = global_all.scaler.mean_, global_all.scaler.std_
    labels = [str(i) for i in C.REGIONS] + ["Centralized", "FL", "FL_recal"]
    mat = [_to_global_coords(local_models[i][0], local_models[i][1], mu_g, sd_g)
           for i in C.REGIONS]
    mat.append(_to_global_coords(global_all.centralized, global_all.scaler, mu_g, sd_g))
    mat.append(_to_global_coords(global_all.fl, global_all.scaler, mu_g, sd_g))
    mat.append(_to_global_coords(global_all.fl_recal, global_all.scaler, mu_g, sd_g))
    mat = np.vstack(mat)
    Z = (mat - mat.mean(0)) / mat.std(0)        # column-standardise (18 params)
    pcs = PCA(n_components=2, random_state=C.RANDOM_STATE).fit_transform(Z)
    return pd.DataFrame({"Index": labels, "PC1": pcs[:, 0], "PC2": pcs[:, 1]})
