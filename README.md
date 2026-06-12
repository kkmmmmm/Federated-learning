# Federated learning for 30-day mortality in acute myocardial infarction (GUSTO-I)

Reproducible re-analysis code for the study comparing **local**, **centralized**
and **federated** logistic-regression models predicting 30-day mortality after
AMI, using the GUSTO-I trial data (40,830 patients across 16 regions).

The federated learning is implemented with the **[Flower](https://flower.ai)
framework** (FedAvg). 

---

## What was corrected

1. **True FedAvg.** The original clients refit a fresh model on local data every
   round and discarded the global parameters, so no information actually
   federated. Here each client **warm-starts from the global model** and runs a
   bounded number of local iterations before the server performs sample-size
   weighted averaging (`src/flower_fl.py`).

2. **Consistent standardisation.** The original clients each fitted their own
   `StandardScaler`, so coefficients were averaged across mismatched feature
   scales. We use **federated standardisation**: every client shares only
   `count / sum / sum-of-squares`; the server reconstructs the exact pooled
   mean/SD and all clients standardise on one common scale
   (`src/data_utils.py:federated_scaler`). This is essential because the L1/L2/
   elastic-net penalty must act on a common scale for coefficient averaging to
   be valid.

3. **Within-region information leak (reviewer comment).** The published
   within-region validation trained the centralized/FL global models on the
   *entire* dataset — including the held-out test fold — and then evaluated on
   that fold. We now use **leave-one-region-out (LORO) global models**: for each
   target region the global model is trained on the **15 other regions** and
   applied to the target region's held-out 1/5 test folds. No test data ever
   enters training (`src/analysis.py:within_region`).

4. **Hyper-parameter selection leak.** Regularisation strength `C` (and the
   elastic-net `l1_ratio`) is now chosen by cross-validation **inside the
   training data only**, never using a test fold (`src/models.py:fit_logreg`).

5. **Predictor set.** The original code fed 20 columns into the model, including
   `ESAMP`, `GRPL`, `GRPS` (sampling / treatment-group variables). We use the
   **17 clinical predictors** described in Table 1 of the manuscript.

The leak fix barely changes the within-region numbers (one region's 1/5 ≈ 1.25 %
of the data), confirming the original conclusions while removing the leak.

---

## Data

`GUSTO_sterberg.csv` (columns: `DAY30` outcome, `REGL` region 1–16, and the 17
predictors `SEX, AGE, KILLIP, DIA, HYP, HRT, ANT, PMI, HEI, WEI, SMK, HTN, LIP,
PAN, FAM, STE, TTR`). Place it one directory above the package, or set the
`GUSTO_CSV` environment variable.

The GUSTO-I dataset is publicly available from the Duke Clinical Research
Institute and can be loaded in R via
`load(url("http://hbiostat.org/data/repo/gusto.rda"))`.

---

## Install & run

```bash
pip install -r requirements.txt
python run_all.py                 # full run (1000 bootstrap resamples)
python run_all.py --boot 100      # faster
python run_all.py --penalties l1  # subset of regularisation strategies
```

Outputs are written to `outputs/`:

| File | Contents |
|------|----------|
| `results.xlsx` | Table 2 / S2 (within-region mean±SD AUROC), raw folds, between-region long format, FL convergence, coefficients |
| `AUROC.xlsx` | Figure 1 / S1–S3 source data, per-region block layout |
| `calibration_slope.xlsx` | Figure 2 / S4–S6 source data |
| `calibration_intercept.xlsx` | Figure 3 / S7–S9 source data |
| `PCA.xlsx` | Figure 4 source data |
| `figures/Figure1_AUROC_L1.png` … | Publication-format figures (PNG + PDF) |
| `figures/FL_convergence.png` | FedAvg log-loss / AUROC vs round |

---

## Three learning paradigms (× 4 regularisations: none / L1 / L2 / elastic-net)

* **Local** — trained on one region only.
* **Centralized** — trained on pooled patient-level data.
* **FL** — FedAvg over per-region clients, sharing only coefficients (Flower).

### Within-region validation (Table 2 / S2)
Stratified 5-fold CV per region. Local models train on 4/5 and test on 1/5; the
LORO global models (trained on the 15 other regions) are evaluated on the same
1/5 test folds.

### Between-region validation (Figures 1–3 / S1–S9)
Every local model is applied to every other region, alongside the LORO global
models. AUROC, calibration slope and calibration intercept are reported with
95% bootstrap confidence intervals (1000 resamples). Calibration metrics use the
10-bin reliability-curve fit of the original analysis, so figures stay directly
comparable.

### Coefficient similarity (Figure 4)
PCA of the standardized coefficients of the 18 full-data models
(16 local + Centralized + FL).

### FL convergence
Global training log-loss and AUROC are logged each communication round; the
converged round and wall-clock time are reported (`FL_convergence` sheets and
figure).

---

## Package layout

```
run_all.py                 orchestrates the whole analysis + exports
src/
  config.py                predictors, regions, penalties, hyper-parameters
  data_utils.py            loading, region split, federated standardisation
  models.py                centralized/local estimators + CV (leak-free)
  flower_fl.py             Flower FedAvg + convergence logging
  analysis.py              within/between-region, PCA, convergence
  calibration.py           AUROC + calibration + bootstrap CIs
  excel_export.py          Excel in the published figure-data layout
  figures.py               publication-format figures
flower_distributed/        true networked Flower deployment (server + 16 clients)
```

The in-process FedAvg in `src/flower_fl.py` and the networked deployment in
`flower_distributed/` implement the same algorithm; the former replaces gRPC
transport with direct calls for full reproducibility.
